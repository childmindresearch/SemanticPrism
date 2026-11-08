#!/usr/bin/env python3
"""
SemanticPrism Stage 4: Synthesis Pipeline (Powered by PrismRunner)
==================================================================
Drop-in replacement for SemanticPrism's Stage 4 Ontological Schema Synthesis pipeline
(https://github.com/childmindresearch/SemanticPrism), replacing Ollama / pydantic-ai
with PrismRunner's local HuggingFace batch inference engine.

Workflow Phases & Output Artifacts:
  - Phase 0 : Dynamic Target Resolution
              * outputs/03_topology/03_results/resolved_community_targets.json
  - Phase 1 : Community Path Schema Synthesis (Single Pass from Raw Source Sentence Text)
              * outputs/04_synthesis/schemas/community/<01_module_name>.py ...
  - Phase 2 : Global Schema Consolidation & AST Class Deduplication
              * outputs/04_synthesis/schemas/community/master_ontology.py
              * outputs/04_synthesis/schemas/community/comprehensive_ontology.py
              * outputs/04_synthesis/logs/schema_sanitization.log
              * outputs/04_synthesis/logs/synthesis_community_errors.log
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Union

# Prevent PyTorch CUDA allocator fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Ensure repository root (and sibling PrismRunner checkout if present) is importable
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_SIBLING_PRISMRUNNER = REPO_ROOT.parent / "PrismRunner"
if _SIBLING_PRISMRUNNER.exists() and str(_SIBLING_PRISMRUNNER) not in sys.path:
    sys.path.insert(0, str(_SIBLING_PRISMRUNNER))

from configs.AST_toggle import (
    MAX_BASEMODEL_FIELDS_PER_CLASS as AST_MAX_BASEMODEL_FIELDS_PER_CLASS,
    MAX_ENUM_MEMBERS_PER_CLASS as AST_MAX_ENUM_MEMBERS_PER_CLASS,
)
from model_config import CODER_MODEL as MASTER_CODER_MODEL
from src.synthesis import (
    GeneratedModule,
    TargetResolver,
    SynthesisPipeline,
    configure_synthesis_defaults,
    estimate_tokens,
    clean_python_code,
    deduplicate_consolidated_classes,
    sanitize_module_name,
    validate_python_syntax,
)

# ============================================================================
# GLOBAL CONFIGURATION VARIABLES
# ============================================================================
# Edit these variables directly to configure models, hardware, paths, and
# synthesis hyperparameters.
#
# NOTE:
#   - MODEL_NAME : Generative Causal/Coding LLM used by PrismRunner for Pydantic v2
#                  schema synthesis and ontology consolidation (loaded from CODER_MODEL
#                  in model_config.py).

# --- 1. Generative LLM & Hardware Settings (Defaults loaded from model_config.py) ---
MODEL_NAME: str = MASTER_CODER_MODEL                   # Causal/Coding LLM for Stage 4 (from CODER_MODEL in model_config.py)
DEVICE: str = "auto"                                   # "auto", "cuda", "mps", or "cpu"
TORCH_DTYPE: str = "auto"                              # "auto", "bfloat16", "float16", or "float32"
LOAD_IN_4BIT: bool = False                             # Enable 4-bit NF4 quantization (CUDA bitsandbytes)
LOAD_IN_8BIT: bool = False                             # Enable 8-bit quantization (CUDA bitsandbytes)
ATTN_IMPLEMENTATION: str = "auto"                      # "auto", "flash_attention_2", "sdpa", or "eager"
BATCH_SIZE: Union[int, str] = "auto"                   # "auto" (continuous device-relative) or explicit int
CONTEXT_WINDOW_CAP: Union[int, str] = "auto"           # "auto" = full native model context window (e.g. 262,144 tokens), or int cap
MAX_NEW_TOKENS: int = 16384                            # Max generated tokens per community schema module
CONSOLIDATION_MAX_NEW_TOKENS: int = 32768              # Max generated tokens for consolidated master ontology
TEMPERATURE: float = 0.0                               # 0.0 for deterministic greedy decoding
REPETITION_PENALTY: float = 1.08                       # Mild penalty to prevent greedy repetition loops
CONSTRAIN_JSON: bool = True                            # Force JSON '{' prefix & bracket-balanced early stopping
MANAGE_VRAM: bool = True                               # Flush and offload GPU/MPS memory after synthesis completes
DRY_RUN: bool = False                                  # Set True to preview prompts without running LLM generation

# --- 2. Input & Output Paths ---
OUTPUTS_DIR: str = "outputs"
REFINED_TRIPLETS_PATH: str = "outputs/02_refinement/refined_triplets.json"
TOPOLOGY_PARTITIONS_PATH: str = "outputs/03_topology/03_results/topology_partitions.json"
RESOLVED_TARGETS_OUTPUT_DIR: str = "outputs/03_topology/03_results"
SAVE_LLM_PAYLOADS: bool = True
PRINT_CHECKPOINTS: bool = True                       # Print real-time timestamped checkpoint messages to console
RESUME_EXISTING_SCHEMAS: bool = False                # If True, skips community targets whose .py files already exist

# --- 3. Target Qualification & Triplet Filtering ---
MIN_CLUSTER_SIZE: int = 15                           # Minimum entity count required for a community to receive its own schema
MAX_TRIPLETS_PER_TARGET: int = 1200                  # Maximum triplets per schema target payload (intra-community centrality cap)
TRIPLET_PAYLOAD_MODE: str = "strictly_intra_only"    # "strictly_intra_only" or "include_boundary"

# --- 4. Schema Cardinality Bounds (Configured in configs/AST_toggle.py) ---
MAX_ENUM_MEMBERS_PER_CLASS: int = AST_MAX_ENUM_MEMBERS_PER_CLASS         # Max distinct members allowed per Enum class (from configs/AST_toggle.py)
MAX_BASEMODEL_FIELDS_PER_CLASS: int = AST_MAX_BASEMODEL_FIELDS_PER_CLASS # Max fields allowed per Pydantic BaseModel class (from configs/AST_toggle.py)

# --- 5. Domain Term Scoring (C-TF-IDF) ---
TOP_DOMAIN_TERMS_COUNT: int = 5
DOMAIN_TERMS_IDF_EXPONENT: float = 1.5

# --- 6. Thematic Observation Evidence Scoring (Phase 1 Context Quotes) ---
MAX_SENTENCES_PER_TARGET: int = 200
MIN_SENTENCE_LENGTH: int = 15
OBSERVATION_OVERLAP_WEIGHT: float = 0.25
OBSERVATION_OVERLAP_MAX_CAP: int = 4


# ============================================================================
# CONFIGURATION & CLI ENTRYPOINT
# ============================================================================


def build_default_config() -> Dict[str, Any]:
    """Constructs the SemanticPrism Stage 4 configuration dictionary from global variables."""
    return {
        "directories": {
            "outputs": OUTPUTS_DIR,
        },
        "topology": {
            "semantic_clustering": {
                "triplet_payload_mode": TRIPLET_PAYLOAD_MODE,
            },
        },
        "synthesis": {
            "llm": {
                "model_name": MODEL_NAME,
                "temperature": TEMPERATURE,
            },
            "context_window_cap": CONTEXT_WINDOW_CAP,
            "save_llm_payloads": SAVE_LLM_PAYLOADS,
            "min_cluster_size": MIN_CLUSTER_SIZE,
            "max_triplets_per_target": MAX_TRIPLETS_PER_TARGET,
            "top_domain_terms_count": TOP_DOMAIN_TERMS_COUNT,
            "domain_terms_idf_exponent": DOMAIN_TERMS_IDF_EXPONENT,
            "max_sentences_per_target": MAX_SENTENCES_PER_TARGET,
            "min_sentence_length": MIN_SENTENCE_LENGTH,
            "observation_overlap_weight": OBSERVATION_OVERLAP_WEIGHT,
            "observation_overlap_max_cap": OBSERVATION_OVERLAP_MAX_CAP,
        },
    }


def load_json(filepath: str | Path, required: bool = True) -> Any:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        if required:
            print(f"CRITICAL ERROR: Required file {filepath} not found.")
        return None


def main() -> None:
    global DRY_RUN, RESUME_EXISTING_SCHEMAS, MODEL_NAME, DEVICE, BATCH_SIZE
    parser = argparse.ArgumentParser(
        description="SemanticPrism Stage 4: Ontological Schema Synthesis Pipeline (Powered by PrismRunner)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline in dry-run mode without loading the LLM weights.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing Phase 1 community schema .py files on disk.",
    )
    parser.add_argument("--model", type=str, default=None, help="Override MODEL_NAME")
    parser.add_argument("--device", type=str, default=None, help="Override DEVICE ('auto', 'cuda', 'mps', 'cpu')")
    parser.add_argument("--batch-size", type=str, default=None, help="Override BATCH_SIZE ('auto' or integer)")
    args, _ = parser.parse_known_args()
    if args.dry_run:
        DRY_RUN = True
    if args.resume:
        RESUME_EXISTING_SCHEMAS = True
    if args.model:
        MODEL_NAME = args.model
    if args.device:
        DEVICE = args.device
    if args.batch_size:
        BATCH_SIZE = int(args.batch_size) if args.batch_size.isdigit() else args.batch_size

    # Sync any edited global variables into src.synthesis module defaults & sanitizers
    configure_synthesis_defaults(
        model_name=MODEL_NAME,
        device=DEVICE,
        torch_dtype=TORCH_DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
        attn_implementation=ATTN_IMPLEMENTATION,
        batch_size=BATCH_SIZE,
        context_window_cap=CONTEXT_WINDOW_CAP,
        max_new_tokens=MAX_NEW_TOKENS,
        consolidation_max_new_tokens=CONSOLIDATION_MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        repetition_penalty=REPETITION_PENALTY,
        constrain_json=CONSTRAIN_JSON,
        manage_vram=MANAGE_VRAM,
        dry_run=DRY_RUN,
        outputs_dir=OUTPUTS_DIR,
        refined_triplets_path=REFINED_TRIPLETS_PATH,
        topology_partitions_path=TOPOLOGY_PARTITIONS_PATH,
        resolved_targets_output_dir=RESOLVED_TARGETS_OUTPUT_DIR,
        save_llm_payloads=SAVE_LLM_PAYLOADS,
        print_checkpoints=PRINT_CHECKPOINTS,
        resume_existing_schemas=RESUME_EXISTING_SCHEMAS,
        min_cluster_size=MIN_CLUSTER_SIZE,
        max_triplets_per_target=MAX_TRIPLETS_PER_TARGET,
        triplet_payload_mode=TRIPLET_PAYLOAD_MODE,
        max_enum_members_per_class=MAX_ENUM_MEMBERS_PER_CLASS,
        max_basemodel_fields_per_class=MAX_BASEMODEL_FIELDS_PER_CLASS,
        top_domain_terms_count=TOP_DOMAIN_TERMS_COUNT,
        domain_terms_idf_exponent=DOMAIN_TERMS_IDF_EXPONENT,
        max_sentences_per_target=MAX_SENTENCES_PER_TARGET,
        min_sentence_length=MIN_SENTENCE_LENGTH,
        observation_overlap_weight=OBSERVATION_OVERLAP_WEIGHT,
        observation_overlap_max_cap=OBSERVATION_OVERLAP_MAX_CAP,
    )

    print(f"=== SemanticPrism Stage 4: Synthesis Pipeline (Model: {MODEL_NAME}) ===")

    config = build_default_config()

    refined_triplets = load_json(REFINED_TRIPLETS_PATH)
    if not refined_triplets:
        print(f"CRITICAL ERROR: Cannot run Stage 4. Missing required Stage 2 output ({REFINED_TRIPLETS_PATH}).")
        print("Please run preceding stages first.")
        sys.exit(1)

    # Phase 0: Dynamic Target Resolution using CURRENT synthesis config
    print("   -> Phase 0: Resolving Synthesis Targets from Stage 3 Graph Partitions...")
    comm_file = Path(TOPOLOGY_PARTITIONS_PATH)
    if not comm_file.exists():
        comm_file = Path(OUTPUTS_DIR) / "03_topology" / "03_results" / "topology_partitions.json"
    comm_topology = load_json(comm_file, required=False)

    resolved_targets_path = Path(RESOLVED_TARGETS_OUTPUT_DIR) / "resolved_community_targets.json"
    if comm_topology:
        comm_target_file = TargetResolver.export_resolved_targets_json(
            comm_topology,
            "community",
            refined_triplets,
            config,
            Path(RESOLVED_TARGETS_OUTPUT_DIR),
        )
        comm_targets = load_json(comm_target_file)
    elif resolved_targets_path.exists():
        print(f"   -> Using pre-existing resolved targets at: {resolved_targets_path}")
        comm_targets = load_json(resolved_targets_path)
    else:
        print(
            f"CRITICAL ERROR: Cannot run Stage 4. Missing required Stage 3 community topology partition outputs ({comm_file})."
        )
        sys.exit(1)

    try:
        pipeline = SynthesisPipeline(config)
        pipeline.execute(
            comm_targets=comm_targets,
            refined_triplets=refined_triplets,
        )
        print("=== Stage 4 Synthesis Completed Successfully ===")
    except Exception as e:
        print(f"Error during Stage 4 Synthesis execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
