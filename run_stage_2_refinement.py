#!/usr/bin/env python3
"""
SemanticPrism Stage 2: Refinement & Taxonomic Hypernym Lifting (Powered by PrismRunner)
=======================================================================================
Performs the exact Stage 2 Refinement workflow from:
    https://github.com/childmindresearch/SemanticPrism

Workflow Steps:
  - Step 1 : Lexical Normalization & Term Frequency Weighting (from outputs/01_extraction/original_triplets.json)
  - Step 2 : Two-Tier Dimension-Aware SVO Vector Clustering:
             * Tier 1: UMAP manifold reduction + HDBSCAN micro-clustering + Spherical Fréchet Mean centroids
                       + Conservative Outlier Rescue (rescue_tau) + Singleton Protection
             * Tier 2: Cosine Agglomerative Macro-Clustering (tau_macro)
             (-> subject_clusters.json, predicate_clusters.json, object_clusters.json, umap_trustworthiness_*.jpg)
  - Step 3 : Concurrent Batch Taxonomic Hypernym Lifting via PrismRunner (with WAL SQLite resume cache):
             (-> refinement_state.db, subject_taxonomic_map.json, predicate_taxonomic_map.json,
                 object_taxonomic_map.json, taxonomic_map.json)
  - Step 4 : Contextual Sentence Vector Grounding & Edge Fidelity (gamma_e) Computation:
             (-> refined_triplets.json, contextual_grounding_diagnostics.json)

All LLM, embedding, clustering calibration, input/output paths, and lifting toggles are
configured as global variables at the top of this file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Union

# Prevent PyTorch CUDA allocator fragmentation when switching between embedding and LLM stages
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", message=r".*n_jobs value.*overridden to 1.*", category=UserWarning)

# Ensure repository root (and sibling PrismRunner checkout if present) is importable
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_SIBLING_PRISMRUNNER = REPO_ROOT.parent / "PrismRunner"
if _SIBLING_PRISMRUNNER.exists() and str(_SIBLING_PRISMRUNNER) not in sys.path:
    sys.path.insert(0, str(_SIBLING_PRISMRUNNER))

from model_config import (
    EMBEDDING_MODEL as MASTER_EMBEDDING_MODEL,
    MODEL_NAME as MASTER_MODEL_NAME,
)
from src.refinement import (
    RawTriple,
    TaxonomicVerification,
    PipelineRunContext,
    RefinementPipeline,
    run_fidelity_only,
)

# =============================================================================
# GLOBAL CONFIGURATION VARIABLES
# =============================================================================

# 1. LLM, Embedding & Hardware Settings (Defaults loaded from model_config.py)
#    - MODEL_NAME      : Generative Causal LLM used in Step 3 for Taxonomic Hypernym Lifting (e.g., "Qwen/Qwen3.6-27B")
#    - EMBEDDING_MODEL : SentenceTransformer vector encoder used in Step 2 (SVO clustering) & Step 4 (sentence fidelity)
#                        (e.g., "all-MiniLM-L6-v2" [384d] or "BAAI/bge-large-en-v1.5" [1024d])
MODEL_NAME: str = MASTER_MODEL_NAME            # Generative LLM for Step 3 Hypernym Lifting (from model_config.py)
EMBEDDING_MODEL: str = MASTER_EMBEDDING_MODEL  # Vector encoder for Step 2 & 4 (from model_config.py)
DEVICE: str = "auto"                           # "auto", "cuda", "mps", or "cpu"
TORCH_DTYPE: str = "auto"                      # "auto", "bfloat16", "float16", or "float32"
LOAD_IN_4BIT: bool = False                     # Enable 4-bit NF4 quantization (CUDA bitsandbytes)
LOAD_IN_8BIT: bool = False                     # Enable 8-bit quantization (CUDA bitsandbytes)
ATTN_IMPLEMENTATION: str = "auto"              # "auto", "flash_attention_2", "sdpa", or "eager"
BATCH_SIZE: Union[int, str] = "auto"           # "auto" (continuous device-relative) or explicit int (e.g., 8)
TEMPERATURE: float = 0.0                       # 0.0 for deterministic greedy decoding
CONSTRAIN_JSON: bool = True                    # Force JSON '{' prefix & bracket-balanced early stopping
MANAGE_VRAM: bool = True                       # Flush and offload GPU/MPS memory between embedding & LLM steps
DRY_RUN: bool = False                          # Set True to preview prompts without running LLM generation

# 2. Input / Output & Execution Settings
INPUT_DIR: str = "outputs/01_extraction"       # Directory containing Stage 1 original_triplets.json & master_themes.json
OUTPUTS_DIR: str = "outputs"                   # Base outputs directory (writes 02_refinement, 02_refinement/cache, logs, payload_text)
SAVE_LLM_PAYLOADS: bool = True                 # Save initial LLM prompts to outputs/payload_text for inspection
RESUME_MODE: str = "skip"                      # "skip" (reuse SQLite cache in refinement_state.db) or "overwrite"
PRINT_CHECKPOINTS: bool = True                 # Print real-time timestamped checkpoint progress to console
PLOT_TRUSTWORTHINESS: bool = True              # Generate UMAP Trustworthiness (T_k) vs n_components .jpg charts

# 3. Context, Token & SVO Taxonomic Lifting Settings
CONTEXT_WINDOW_CAP: Union[int, str] = "auto"   # "auto" (detect from model config via PrismRunner) or integer token cap
LIFT_MAX_NEW_TOKENS: int = 256                 # Max generated tokens per TaxonomicVerification JSON response
LIFT_SUBJECTS: bool = True                     # Abstract entity subjects into canonical hypernyms
LIFT_PREDICATES: bool = False                  # Abstract relational predicates into canonical hypernyms
LIFT_OBJECTS: bool = True                      # Abstract entity objects into canonical hypernyms


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================


def main() -> None:
    global MODEL_NAME, DEVICE, BATCH_SIZE, RESUME_MODE, DRY_RUN

    parser = argparse.ArgumentParser(
        description="SemanticPrism Stage 2: Refinement Pipeline (Vector Clustering & Taxonomic Lifting via PrismRunner)"
    )
    parser.add_argument(
        "--fidelity-only",
        action="store_true",
        help="Fast computation of sentence vectors and contextual fidelity on existing refined triplets (< 10s)",
    )
    parser.add_argument("--model", type=str, default=None, help="Override MODEL_NAME")
    parser.add_argument("--device", type=str, default=None, help="Override DEVICE ('auto', 'cuda', 'mps', 'cpu')")
    parser.add_argument("--batch-size", type=str, default=None, help="Override BATCH_SIZE ('auto' or integer)")
    parser.add_argument("--overwrite", action="store_true", help="Set RESUME_MODE='overwrite' to clear SQLite cache")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode without loading LLM weights")
    parser.add_argument("--live", action="store_true", help="Force live LLM execution (DRY_RUN = False)")
    args = parser.parse_args()

    if args.model:
        MODEL_NAME = args.model
    if args.device:
        DEVICE = args.device
    if args.batch_size:
        BATCH_SIZE = int(args.batch_size) if args.batch_size.isdigit() else args.batch_size
    if args.overwrite:
        RESUME_MODE = "overwrite"
    if args.dry_run:
        DRY_RUN = True
    elif args.live:
        DRY_RUN = False

    pipeline = RefinementPipeline(
        model_name=MODEL_NAME,
        embedding_model=EMBEDDING_MODEL,
        device=DEVICE,
        torch_dtype=TORCH_DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
        attn_implementation=ATTN_IMPLEMENTATION,
        batch_size=BATCH_SIZE,
        temperature=TEMPERATURE,
        constrain_json=CONSTRAIN_JSON,
        manage_vram=MANAGE_VRAM,
        dry_run=DRY_RUN,
        input_dir=INPUT_DIR,
        outputs_dir=OUTPUTS_DIR,
        save_llm_payloads=SAVE_LLM_PAYLOADS,
        resume_mode=RESUME_MODE,
        print_checkpoints=PRINT_CHECKPOINTS,
        plot_trustworthiness=PLOT_TRUSTWORTHINESS,
        context_window_cap=CONTEXT_WINDOW_CAP,
        lift_max_new_tokens=LIFT_MAX_NEW_TOKENS,
        lift_subjects=LIFT_SUBJECTS,
        lift_predicates=LIFT_PREDICATES,
        lift_objects=LIFT_OBJECTS,
    )
    out_dir = pipeline.out_dir

    if args.fidelity_only:
        run_fidelity_only(pipeline, out_dir)
        return

    print(f"=== SemanticPrism Stage 2: Refinement Pipeline (Model: {MODEL_NAME}) ===")
    stage_start = time.time()

    input_dir = Path(INPUT_DIR)
    if not input_dir.exists() and (REPO_ROOT / input_dir).exists():
        input_dir = REPO_ROOT / input_dir

    triplets_path = input_dir / "original_triplets.json"
    if not triplets_path.exists():
        if DRY_RUN:
            print(
                f"   -> [Dry Run] Stage 1 output '{triplets_path}' not yet generated; "
                "verified RefinementPipeline setup in dry-run mode."
            )
            print("=== Stage 2 Refinement (Dry Run) Completed Successfully ===")
            return
        print(f"CRITICAL ERROR: Required file '{triplets_path}' not found.")
        print("Stage 2 requires Stage 1 triple extraction outputs ('outputs/01_extraction/original_triplets.json').")
        print("Please run Stage 1 first ('python3 run_stage_1_extraction.py').")
        sys.exit(1)

    try:
        with open(triplets_path, "r", encoding="utf-8") as f:
            triplets_data = json.load(f)
            raw_triples = [RawTriple(**t) for t in triplets_data]
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load original triplets: {e}")
        sys.exit(1)

    try:
        pipeline.execute(raw_triples=raw_triples, dry_run=DRY_RUN)

        total_duration = time.time() - stage_start
        print("\n" + "=" * 60)
        print("STAGE 2 REFINEMENT PIPELINE COMPLETE!")
        print(f"Total Stage 2 Time: {total_duration:.2f} seconds")
        print(f"Outputs saved in '{out_dir}':")
        print(" - subject_clusters.json")
        print(" - predicate_clusters.json")
        print(" - object_clusters.json")
        print(" - subject_taxonomic_map.json")
        print(" - predicate_taxonomic_map.json")
        print(" - object_taxonomic_map.json")
        print(" - taxonomic_map.json")
        print(" - refined_triplets.json")
        print(" - contextual_grounding_diagnostics.json")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"Error during Stage 2 Refinement execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
