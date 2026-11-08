#!/usr/bin/env python3
"""
SemanticPrism Stage 4: Synthesis Pipeline (Powered by PrismRunner)
==================================================================
Drop-in replacement for SemanticPrism's Stage 4 Ontological Schema Synthesis pipeline
(https://github.com/childmindresearch/SemanticPrism), replacing Ollama / pydantic-ai
with PrismRunner's local HuggingFace batch inference engine.

Workflow Phases & Output Artifacts:
  - Phase 0 : Dynamic Target Resolution
              * outputs/03_topology/03_paths/community/resolved_community_targets.json
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
import ast
from collections import Counter, defaultdict
from datetime import datetime
import json
import keyword
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Prevent PyTorch CUDA allocator fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pydantic import BaseModel, Field, model_validator

# Ensure repository root is importable when executed directly
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prismrunner import (
    PrismPipeline,
    clear_device_memory,
)

try:
    import tiktoken
    _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")

    def estimate_tokens(text: str) -> int:
        """Counts exact tokens using tiktoken (cl100k_base)."""
        if not text:
            return 0
        return len(_TIKTOKEN_ENCODER.encode(text))
except ImportError:
    def estimate_tokens(text: str) -> int:
        """Heuristic-based estimation: 1 word ~ 1.3 tokens."""
        if not text:
            return 0
        return int(len(text.split()) * 1.3)


# ============================================================================
# GLOBAL CONFIGURATION VARIABLES
# ============================================================================
# Edit these variables directly to configure models, hardware, paths, and
# synthesis hyperparameters.
#
# NOTE:
#   - MODEL_NAME : Generative Causal/Coding LLM used by PrismRunner for Pydantic v2
#                  schema synthesis and ontology consolidation (e.g., "Qwen/Qwen3-Coder-30B-A3B-Instruct",
#                  "Qwen/Qwen3-Coder-Next", or "Qwen/Qwen2.5-Coder-32B-Instruct").

# --- 1. Generative LLM & Hardware Settings ---
MODEL_NAME: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"  # Causal/Coding LLM for Stage 4 (256K native context)
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
TOPOLOGY_PARTITIONS_PATH: str = "outputs/03_topology/03_paths/community/topology_partitions.json"
RESOLVED_TARGETS_OUTPUT_DIR: str = "outputs/03_topology/03_paths/community"
SAVE_LLM_PAYLOADS: bool = True
PRINT_CHECKPOINTS: bool = True                       # Print real-time timestamped checkpoint messages to console
RESUME_EXISTING_SCHEMAS: bool = False                # If True, skips community targets whose .py files already exist

# --- 3. Target Qualification & Triplet Filtering ---
MIN_CLUSTER_SIZE: int = 15                           # Minimum entity count required for a community to receive its own schema
MAX_TRIPLETS_PER_TARGET: int = 1200                  # Maximum triplets per schema target payload (intra-community centrality cap)
TRIPLET_PAYLOAD_MODE: str = "strictly_intra_only"    # "strictly_intra_only" or "include_boundary"

# --- 4. Schema Cardinality Bounds ---
MAX_ENUM_MEMBERS_PER_CLASS: int = 15                 # Max distinct members allowed per Enum class
MAX_BASEMODEL_FIELDS_PER_CLASS: int = 15             # Max fields allowed per Pydantic BaseModel class

# --- 5. Domain Term Scoring (C-TF-IDF) ---
TOP_DOMAIN_TERMS_COUNT: int = 5
DOMAIN_TERMS_IDF_EXPONENT: float = 1.5

# --- 6. Thematic Observation Evidence Scoring (Phase 1 Context Quotes) ---
MAX_SENTENCES_PER_TARGET: int = 200
MIN_SENTENCE_LENGTH: int = 15
OBSERVATION_OVERLAP_WEIGHT: float = 0.25
OBSERVATION_OVERLAP_MAX_CAP: int = 4


# ============================================================================
# STAGE 4 PYDANTIC SCHEMAS (Exact match with src/synthesis/schemas.py)
# ============================================================================

class GeneratedModule(BaseModel):
    module_name: str = Field(
        description=(
            "A specific 2-to-4 word snake_case module name derived directly from this cluster's highest-ranked "
            "top_domain_terms and primary BaseModel class (e.g., 'resource_allocation_workflow' or "
            "'incident_escalation_protocol'). NEVER use broad macro theme titles or generic suffixes like "
            "'_schema', '_module', or '_ontology'."
        )
    )
    source_code: str = Field(
        description=(
            "The raw Python source code containing the generated Pydantic BaseModels and Enums, including all necessary "
            "import statements. Every BaseModel class MUST include a descriptive triple-quoted docstring (\"\"\"...\"\"\") "
            f"and have at most {MAX_BASEMODEL_FIELDS_PER_CLASS} fields (decompose sub-entities into separate "
            "models instead of prefix-flattened fields, and use List/Enum fields instead of boolean flag lists). "
            f"Every Enum class MUST have at most {MAX_ENUM_MEMBERS_PER_CLASS} concise atomic members (<= 4 words each)."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_module_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            mod_name = (
                data.get("module_name")
                or data.get("name")
                or data.get("filename")
                or data.get("module")
                or "domain_schema"
            )
            src_code = (
                data.get("source_code")
                or data.get("code")
                or data.get("python_code")
                or data.get("content")
                or ""
            )
            out = dict(data)
            out["module_name"] = str(mod_name).strip()
            out["source_code"] = str(src_code).strip()
            return out
        return data


# ============================================================================
# STAGE 4 SYSTEM PROMPTS
# ============================================================================

SCHEMA_SYNTHESIS_PROMPT = f"""You are an Expert Domain Modeling and Ontological Architect.
You will receive an extracted thematic cluster of domain concepts and real-world source text observations.
Your job is to interpret the underlying domain workflows, entities, and attributes, and synthesize a clean, practical, and coherent Pydantic (v2) schema module.

INPUT STRUCTURE:
1. `target_domain_focus`: Overarching theme(s) defining this cluster (ordered by prominence, with 'Other' always placed last when present).
2. `top_domain_terms`: Key domain concepts and vocabulary defining this cluster. Use these to guide the naming of classes, fields, and core entities.
3. `thematic_observations`: Authentic source text sentences organized by theme (with 'Other' placed last). Study these narrative observations to understand real-world workflows, data types, entity interactions, and field constraints.

CRITICAL MODELING RULES:
1. **Synthesize Domain Workflows, Do Not Transcribe:** Do NOT mechanically turn every term into a separate class. Step back, understand the domain narrative, and design 1 or more intuitive, cohesive Pydantic `BaseModel` classes modeling the core subject matter.
2. **Strict BaseModel Field Budget (`<= {MAX_BASEMODEL_FIELDS_PER_CLASS}` Fields Maximum) & Anti-Bloat Rules:**
   - Every Pydantic `BaseModel` class MUST contain **between 4 and {MAX_BASEMODEL_FIELDS_PER_CLASS} fields maximum (`<= {MAX_BASEMODEL_FIELDS_PER_CLASS}` fields)**. Prioritize the most essential, high-level domain attributes.
   - **NO Prefix-Flattened Sub-Object Dumps:** NEVER flatten a related sub-entity into dozens of prefixed fields on a parent model (e.g., NEVER emit `sub_entity_name`, `sub_entity_description`, `sub_entity_status`, `sub_entity_cost`, etc.). Instead, either compose a separate compact `BaseModel` (`<= {MAX_BASEMODEL_FIELDS_PER_CLASS}` fields) referenced by a single field, or use 1–2 high-level summary fields.
   - **NO Boolean-Flag Lists:** NEVER turn list items, goals, capabilities, or roles into individual `Optional[bool]` fields (e.g., NEVER emit `outcome_a_achieved: Optional[bool]`, `outcome_b_achieved: Optional[bool]`, `outcome_c_achieved: Optional[bool]`). Group them into a single `Optional[List[str]]` or `Optional[List[Enum]]` field (e.g., `target_outcomes: Optional[List[str]]`).
3. **Field-Grounded Atomic Enums/Literals (NO Orphan or Sentence-Fragment Enums):**
   - Define an `Enum` class ONLY when it is directly referenced as the type of at least one field on a `BaseModel` in this module.
   - Every `Enum` member MUST be a short, atomic category label of **at most 4 words (`<= 3` underscores, `<= 35` characters)**, with **at most {MAX_ENUM_MEMBERS_PER_CLASS} members per Enum class**.
   - If `'Other'` / `OTHER` is included in any `Enum` or `Literal`, ALWAYS place `'Other'` last regardless of how many values are included.
   - NEVER encode full sentences, prepositional clauses, or multi-attribute combinations inside `Enum` values; decompose distinct dimensions into separate `BaseModel` fields.
4. **Class Docstrings, Field Descriptions & Optionality:**
   - Every `BaseModel` class MUST begin with a descriptive triple-quoted docstring (`\"\"\"...\"\"\"`) explaining what the model entails and its role in the domain.
   - Declare model attributes using `Field(default=None, description="...")`, providing a concise explanation of the domain meaning derived from `thematic_observations`.
   - Field names MUST be valid Python identifiers in `snake_case` (alphanumeric and underscores only, NEVER hyphens like `field-name`).
5. **Self-Contained & Valid Imports:** The source code must include all necessary imports:
   `from pydantic import BaseModel, Field`
   `from typing import Optional, List, Dict, Any, Union, Literal`
   `from enum import Enum`
6. **Strictly Isolated Module:** Model only the concepts and workflows substantiated by the provided data.
7. **ABSOLUTE BAN ON META-MODELING:** You MUST model the authentic real-world domain phenomena described INSIDE the observations. NEVER define models that describe the prompt itself, the JSON payload wrapper, or the synthesis framework. Specifically:
   - NEVER create classes named `ThemeObservation`, `ThematicObservation`, `DomainTerm`, or `GeneratedModule`.
   - NEVER create fields named `theme_name`, `observations`, or `source_code`.
   Any class that attempts to model the prompt structure itself is strictly invalid.

OUTPUT SCHEMA INSTRUCTIONS:
Return a JSON object conforming to the GeneratedModule schema:
- `module_name` (string): A specific 2-to-4 word `snake_case` name anchored to this cluster's `top_domain_terms` and primary `BaseModel` class (e.g., "resource_allocation_workflow"). Do NOT copy the broad macro theme header and do NOT append generic suffixes like "_schema", "_module", or "_model".
- `source_code` (string): The complete, executable Python code defining the Pydantic BaseModels (each with a descriptive `\"\"\"...\"\"\"` class docstring) and any field-referenced Enums."""


CONSOLIDATION_SYSTEM_PROMPT = f"""You are a Master Ontologist.
Merge, deduplicate, and consolidate the provided community Pydantic schema files into a single unified, standalone `master_ontology.py` module.

CRITICAL RULES:
1. **Deduplicate & Merge (Strict Class Name Uniqueness & `<= {MAX_BASEMODEL_FIELDS_PER_CLASS}` Fields Per BaseModel):** Identify classes modeling the same core concept across files. Merge their core fields into ONE single unified class using the most descriptive and comprehensive names, keeping **at most {MAX_BASEMODEL_FIELDS_PER_CLASS} high-value fields per `BaseModel`**.
   - NEVER define two classes with the exact same name in the file.
   - NEVER define an Enum and a BaseModel sharing the exact same class name.
   - Consolidate repetitive boolean flags or prefixed sub-attributes into clean `List[str]` / `Enum` fields or composed child models rather than bloating a single class past {MAX_BASEMODEL_FIELDS_PER_CLASS} fields.
2. **Atomic Enum Hygiene:** Embed and deduplicate `Enum` classes at the top of the file (`<= {MAX_ENUM_MEMBERS_PER_CLASS}` members per Enum, `<= 4` words / `<= 35` characters per member, with `OTHER = "Other"` always placed last when present). Discard any verbose sentence-fragment Enum members.
3. **Preserve Hierarchy & Relationships:** Maintain valid class inheritance and typed cross-model references.
4. **Preserve Class Docstrings & Field Descriptions:** Ensure every `BaseModel` class begins with a descriptive triple-quoted docstring (`\"\"\"...\"\"\"`) explaining what it entails, and retain attributes declared as `Field(default=None, description="...")`, preserving the most detailed `description` string across merged attributes.
5. **Valid Python Identifiers:** All attribute names must be valid Python identifiers in `snake_case` (alphanumeric and underscores only, NEVER hyphens).
6. **Self-Contained Imports:** Ensure all necessary Pydantic and typing imports are included at the top of the file.

OUTPUT SCHEMA INSTRUCTIONS:
Return a JSON object conforming to the GeneratedModule schema:
- `module_name`: Set strictly to "master_ontology".
- `source_code`: The complete, consolidated Python source code."""


SCHEMA_REFORMAT_SYSTEM_PROMPT = f"""You are an expert Ontological Schema Repair Assistant.
You will receive a Python module attempt and the specific validation or syntax error that occurred.
Your job is to repair the Python code to strictly conform to the expected GeneratedModule schema structure.

TARGET SCHEMA:
```json
{{
  "module_name": "string (snake_case identifier)",
  "source_code": "string (valid, executable Python source code)"
}}
```

REPAIR RULES:
1. Fix all Python syntax errors, indentation issues, or missing imports (`from pydantic import BaseModel, Field`, `from enum import Enum`, `from typing import ...`).
2. Ensure every `BaseModel` class includes a descriptive triple-quoted docstring (`\"\"\"...\"\"\"`), all Pydantic fields use valid syntax (`field_name: Optional[Type] = Field(default=None, description="...")`), and each `BaseModel` class has **at most {MAX_BASEMODEL_FIELDS_PER_CLASS} fields** (removing repetitive prefixed sub-fields or boolean flag series).
3. Ensure Enum member names use valid `UPPER_SNAKE_CASE` identifiers, inherit from `(str, Enum)`, and contain **at most {MAX_ENUM_MEMBERS_PER_CLASS} concise atomic members (`<= 4` words / `<= 35` characters each)**, placing `OTHER = "Other"` last if present. Strip any multi-clause phrase Enum members.
4. ABSOLUTE BAN ON META-MODELS: If the malformed input or code contains meta-models like `class ThemeObservation` or `class GeneratedModule` inside `source_code`, DISCARD them completely. Replace them with substantive domain Pydantic models based on the domain concepts and key terms.
5. Return pure JSON matching the GeneratedModule schema only. Do NOT output markdown explanations or conversational text."""


# ============================================================================
# TARGET RESOLVER (Exact match with src/topology/resolver.py)
# ============================================================================

class TargetResolver:
    """
    Target qualification and triplet filtering criteria used by Stage 4 for LLM synthesis.
    Provides methods to resolve qualified community targets and filter/cap triplet payloads
    based on local community centrality and intra-cluster edge priority.
    """

    @staticmethod
    def qualify_and_resolve_targets(
        topology: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Resolves qualified community targets (>= min_cluster_size).
        """
        min_cluster_size = (
            config.get("synthesis", {}).get("min_cluster_size", MIN_CLUSTER_SIZE)
            if config
            else MIN_CLUSTER_SIZE
        )

        targets = []
        for comm in topology.get("communities", []):
            nodes_in_comm = list(comm.get("nodes", []))
            if len(nodes_in_comm) >= min_cluster_size:
                c_id = comm.get("community_id")
                targets.append({
                    "type": "community",
                    "id": c_id,
                    "nodes": nodes_in_comm,
                    "merged_from": comm.get("merged_from", [c_id]),
                })

        targets.sort(key=lambda x: len(x["nodes"]), reverse=True)
        return targets

    @staticmethod
    def filter_and_cap_triplets(
        target_nodes: Any,
        triplets: Any,
        max_triplets_cap: int = MAX_TRIPLETS_PER_TARGET,
        config: Optional[Dict[str, Any]] = None,
        strictly_intra_only: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filters triplets incident to target_nodes and prioritizes them using a
        pure Local Community Centrality engine.
        """
        # Gracefully handle if caller passed (triplets, target_nodes) in reverse order
        if isinstance(target_nodes, list) and target_nodes and isinstance(target_nodes[0], dict):
            target_nodes, triplets = triplets, target_nodes

        target_nodes_norm = {str(n).lower().strip() for n in target_nodes}

        if config:
            sem_cfg = (
                config.get("topology", {}).get("semantic_clustering", {})
                if "topology" in config
                else config.get("semantic_clustering", {})
            )
        else:
            sem_cfg = {}

        if strictly_intra_only is None:
            strictly_intra = sem_cfg.get("triplet_payload_mode", TRIPLET_PAYLOAD_MODE) == "strictly_intra_only"
        else:
            strictly_intra = bool(strictly_intra_only)

        local_degree = Counter()
        edge_frequencies = Counter()

        for t in triplets:
            subj = str(t.get("subject", "")).lower().strip()
            obj = str(t.get("object", "")).lower().strip()
            pred = str(t.get("predicate", "")).strip()

            if not subj or not obj:
                continue

            is_subj_in = subj in target_nodes_norm
            is_obj_in = obj in target_nodes_norm

            if is_subj_in and is_obj_in:
                local_degree[subj] += 1
                local_degree[obj] += 1
                edge_frequencies[(subj, pred, obj)] += 1
            elif not strictly_intra and (is_subj_in or is_obj_in):
                edge_frequencies[(subj, pred, obj)] += 1

        scored_triplets = []
        for t in triplets:
            subj = str(t.get("subject", "")).lower().strip()
            obj = str(t.get("object", "")).lower().strip()

            if not subj or not obj:
                continue

            is_subj_in = subj in target_nodes_norm
            is_obj_in = obj in target_nodes_norm

            if strictly_intra:
                if not (is_subj_in and is_obj_in):
                    continue
            else:
                if not (is_subj_in or is_obj_in):
                    continue

            pred = str(t.get("predicate", "")).strip()
            freq = edge_frequencies.get((subj, pred, obj), 1)
            freq_boost = 1.0 + 0.1 * min(10, freq - 1)

            if is_subj_in and is_obj_in:
                local_sum = local_degree[subj] + local_degree[obj]
                rank_score = 1000.0 + (local_sum * freq_boost)
            else:
                internal_node = subj if is_subj_in else obj
                anchor_degree = local_degree[internal_node]
                rank_score = 1.0 + (anchor_degree * freq_boost)

            t_copy = dict(t)
            t_copy["triplet_rank_score"] = round(float(rank_score), 4)
            scored_triplets.append((rank_score, t_copy))

        scored_triplets.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored_triplets[:max_triplets_cap]]

    @staticmethod
    def export_resolved_targets_json(
        topology: Dict[str, Any],
        path_type: str = "community",
        refined_triplets: Optional[List[Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Executes target qualification and triplet centrality capping, and exports
        resolved_community_targets.json to output_dir with explicit rank scores.
        """
        if output_dir is None:
            output_dir = Path(RESOLVED_TARGETS_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        synth_cfg = config.get("synthesis", {}) if config else {}
        min_cluster_size = synth_cfg.get("min_cluster_size", MIN_CLUSTER_SIZE)
        max_triplets_cap = synth_cfg.get("max_triplets_per_target", MAX_TRIPLETS_PER_TARGET)

        targets = TargetResolver.qualify_and_resolve_targets(topology, config)

        target_payloads = []
        for i, target in enumerate(targets, start=1):
            target_type = target["type"]
            c_id = target["id"]
            target_nodes = set(target["nodes"])

            capped_triplets = TargetResolver.filter_and_cap_triplets(
                target_nodes,
                refined_triplets or [],
                max_triplets_cap=max_triplets_cap,
                config=config,
            )

            filename_suffix = f"community_{c_id}"
            output_module_name = f"{i:02d}_{filename_suffix}.py"
            rank_score = float(len(target_nodes))

            payload_item = {
                "target_index": i,
                "target_type": target_type,
                "target_id": c_id,
                "merged_from": target.get("merged_from", [c_id]),
                "output_module_name": output_module_name,
                "rank_score": rank_score,
                "nodes_count": len(target_nodes),
                "nodes": sorted(list(target_nodes)),
                "triplets_count": len(capped_triplets),
                "triplets": capped_triplets,
            }
            target_payloads.append(payload_item)

        audit_payload = {
            "path_type": path_type,
            "min_cluster_size": min_cluster_size,
            "max_triplets_per_target": max_triplets_cap,
            "targets_count": len(target_payloads),
            "targets": target_payloads,
        }

        target_file = output_dir / "resolved_community_targets.json"
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(audit_payload, f, indent=2)

        print(f"   [OK] Resolved Targets Saved: {target_file}")
        return target_file


# ============================================================================
# DOMAIN-AGNOSTIC STRUCTURAL AST SANITIZERS & DEDUPLICATION HELPERS
# ============================================================================

def _log_sanitization_detail(
    context_label: str,
    class_name: str,
    class_kind: str,
    kept_count: int,
    stripped_items: List[str],
    collapsed_items: Optional[List[str]] = None,
    extra_note: Optional[str] = None,
) -> None:
    """
    Logs detailed post-processing AST sanitization events (stripped or collapsed Enum members
    and BaseModel fields) to both the console and `outputs/04_synthesis/logs/schema_sanitization.log`.
    """
    if not stripped_items and not collapsed_items and not extra_note:
        return

    ts = datetime.now().strftime("%H:%M:%S")
    iso_ts = datetime.now().isoformat()
    header = (
        f"[{ts} | AST Sanitizer | {context_label}] {class_kind} '{class_name}' "
        f"(kept={kept_count}, stripped={len(stripped_items)}"
        f"{f', collapsed_groups={len(collapsed_items)}' if collapsed_items else ''})"
    )
    detail_lines: List[str] = [header]
    if extra_note:
        detail_lines.append(f"      * Note: {extra_note}")
    if collapsed_items:
        for c_item in collapsed_items:
            detail_lines.append(f"      ~ Collapsed: {c_item}")
    for s_item in stripped_items:
        detail_lines.append(f"      - Stripped : {s_item}")

    full_msg = "\n".join(detail_lines)
    if PRINT_CHECKPOINTS:
        print(full_msg, flush=True)

    try:
        log_dir = Path(OUTPUTS_DIR) / "04_synthesis" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        audit_log_path = log_dir / "schema_sanitization.log"
        with open(audit_log_path, "a", encoding="utf-8") as lf:
            lf.write(f"[{iso_ts}] {full_msg}\n")
    except Exception:
        pass


def _is_verbose_or_runaway_enum_member(member_name: str, member_val: Optional[str] = None) -> bool:
    """
    Domain-agnostic structural check: returns True if an Enum member identifier or string value
    exceeds atomic token/character bounds (> 4 tokens or > 35 characters), preventing sentence
    fragments or combinatorial permutations from becoming Enum members in any domain.
    """
    max_words = 4
    max_chars = 35
    ident = str(member_name).strip().upper()
    if not ident:
        return True
    words = [w for w in ident.split("_") if w]
    if len(words) > max_words or len(ident) > max_chars:
        return True
    if member_val is not None:
        val_str = str(member_val).strip()
        val_words = [w for w in re.split(r"[\s_-]+", val_str) if w]
        if len(val_words) > max_words + 1 or len(val_str) > max_chars + 5:
            return True
    return False


def _sanitize_ast_enums(code: str, context_label: str = "schema") -> str:
    """
    Domain-agnostic AST post-processor for Enum classes that:
      1. Strips non-atomic (> 4 token / > 35 char) Enum members.
      2. Deduplicates Enum member names and values within each class.
      3. Caps each Enum class at MAX_ENUM_MEMBERS_PER_CLASS (12).
      4. Rewrites any BaseModel field annotation referencing a dropped/empty Enum to `str`.
    """
    try:
        tree = ast.parse(code)
    except Exception:
        return code

    def is_enum_class(node: ast.ClassDef) -> bool:
        for b in node.bases:
            if isinstance(b, ast.Name) and "Enum" in b.id:
                return True
            if isinstance(b, ast.Attribute) and "Enum" in b.attr:
                return True
        return False

    new_body: List[ast.AST] = []
    dropped_enum_names: set = set()
    modified = False

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and is_enum_class(node):
            kept_stmts: List[ast.stmt] = []
            seen_names: set = set()
            seen_vals: set = set()
            member_count = 0
            stripped_details: List[str] = []

            for stmt in node.body:
                m_name = None
                m_val = None
                if isinstance(stmt, ast.Assign) and stmt.targets and isinstance(stmt.targets[0], ast.Name):
                    m_name = stmt.targets[0].id
                    if isinstance(stmt.value, ast.Constant):
                        m_val = stmt.value.value
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    m_name = stmt.target.id
                    if stmt.value and isinstance(stmt.value, ast.Constant):
                        m_val = stmt.value.value

                if m_name is not None:
                    reason: Optional[str] = None
                    if _is_verbose_or_runaway_enum_member(m_name, m_val):
                        reason = "verbose/non-atomic member (>4 words or >35 chars)"
                    elif m_name in seen_names:
                        reason = "duplicate member name"
                    elif m_val is not None and str(m_val).lower().strip() in seen_vals:
                        reason = "duplicate member value"
                    elif member_count >= MAX_ENUM_MEMBERS_PER_CLASS:
                        reason = f"exceeded MAX_ENUM_MEMBERS_PER_CLASS ({MAX_ENUM_MEMBERS_PER_CLASS})"

                    if reason is not None:
                        modified = True
                        stripped_details.append(f"{m_name} = {m_val!r} [reason: {reason}]")
                        continue

                    seen_names.add(m_name)
                    if m_val is not None:
                        seen_vals.add(str(m_val).lower().strip())
                    member_count += 1
                    kept_stmts.append(stmt)
                else:
                    kept_stmts.append(stmt)

            if member_count > 0:
                # Always place 'OTHER' / 'Other' last regardless of how many Enum members are included
                non_field_stmts: List[ast.stmt] = []
                regular_member_stmts: List[ast.stmt] = []
                other_member_stmts: List[ast.stmt] = []
                for k_stmt in kept_stmts:
                    k_name = None
                    k_val = None
                    if isinstance(k_stmt, ast.Assign) and k_stmt.targets and isinstance(k_stmt.targets[0], ast.Name):
                        k_name = k_stmt.targets[0].id
                        if isinstance(k_stmt.value, ast.Constant):
                            k_val = k_stmt.value.value
                    elif isinstance(k_stmt, ast.AnnAssign) and isinstance(k_stmt.target, ast.Name):
                        k_name = k_stmt.target.id
                        if k_stmt.value and isinstance(k_stmt.value, ast.Constant):
                            k_val = k_stmt.value.value
                    if k_name is None:
                        non_field_stmts.append(k_stmt)
                    elif k_name.strip().upper() == "OTHER" or (k_val is not None and str(k_val).strip().lower() == "other"):
                        other_member_stmts.append(k_stmt)
                    else:
                        regular_member_stmts.append(k_stmt)
                reordered_stmts = non_field_stmts + regular_member_stmts + other_member_stmts
                if reordered_stmts != kept_stmts:
                    modified = True
                node.body = reordered_stmts
                new_body.append(node)
                if stripped_details:
                    _log_sanitization_detail(
                        context_label=context_label,
                        class_name=node.name,
                        class_kind="Enum",
                        kept_count=member_count,
                        stripped_items=stripped_details,
                    )
            else:
                dropped_enum_names.add(node.name)
                modified = True
                _log_sanitization_detail(
                    context_label=context_label,
                    class_name=node.name,
                    class_kind="Enum",
                    kept_count=0,
                    stripped_items=stripped_details,
                    extra_note="Enum class had 0 valid members remaining and was removed (field references rewritten to 'str').",
                )
        else:
            new_body.append(node)

    if dropped_enum_names:
        class _DroppedEnumRewriter(ast.NodeTransformer):
            def visit_Name(self, n: ast.Name) -> ast.AST:
                if n.id in dropped_enum_names and isinstance(n.ctx, ast.Load):
                    return ast.copy_location(ast.Name(id="str", ctx=ast.Load()), n)
                return n

        rewriter = _DroppedEnumRewriter()
        new_body = [rewriter.visit(n) for n in new_body]

    if not modified:
        return code

    tree.body = new_body
    try:
        return ast.unparse(tree)
    except Exception:
        return code


def _sanitize_ast_models(code: str, context_label: str = "schema") -> str:
    """
    Domain-agnostic AST post-processor for Pydantic BaseModel classes that:
      1. Renames fields shadowing reserved Pydantic BaseModel attributes or Python keywords.
      2. Rewrites undefined/dangling non-builtin type annotations to `str`.
      3. Deduplicates fields within each BaseModel.
      4. Caps/collapses shared 2-token snake_case prefixes (prevents flattened sub-entity attribute dumps).
      5. Caps/collapses shared non-generic suffix tokens.
      6. Enforces MAX_BASEMODEL_FIELDS_PER_CLASS per BaseModel, prioritizing core/typed/collection fields.
      7. Guarantees every BaseModel has a descriptive triple-quoted docstring (\"\"\"...\"\"\").
      8. Ensures `from __future__ import annotations` and canonical top-level ordering (Imports -> Enums -> Topological BaseModels).
    """
    max_shared_prefix = 2
    max_shared_suffix = 2
    try:
        tree = ast.parse(code)
    except Exception:
        return code

    def is_enum_class(node: ast.ClassDef) -> bool:
        for b in node.bases:
            if isinstance(b, ast.Name) and "Enum" in b.id:
                return True
            if isinstance(b, ast.Attribute) and "Enum" in b.attr:
                return True
        return False

    exempt_suffixes = {
        "id", "name", "type", "types", "status", "level", "score", "range",
        "date", "time", "plan", "notes", "summary", "description", "details",
        "count", "rate", "list", "items", "category", "mode", "source",
    }

    pydantic_reserved_attrs = {
        "schema", "json", "copy", "dict", "validate", "fields",
        "model_config", "model_fields", "model_computed_fields",
        "model_extra", "model_fields_set", "model_dump",
        "model_dump_json", "model_validate", "model_json_schema",
        "model_construct", "model_copy", "model_rebuild",
        "update_forward_refs", "parse_obj", "parse_raw",
        "parse_file", "from_orm", "construct",
    }
    python_keywords = set(keyword.kwlist) | {"async", "await"}

    # Collect all defined or imported type names to guard against dangling type references
    known_type_names: set = {
        "str", "int", "float", "bool", "bytes", "list", "dict", "set", "tuple",
        "object", "None", "Any", "Optional", "List", "Dict", "Set", "Tuple",
        "Union", "Literal", "Callable", "Sequence", "Mapping", "Iterable",
        "Type", "ClassVar", "Final", "Annotated", "BaseModel", "Field", "Enum",
        "date", "datetime", "time", "timedelta", "UUID", "Decimal", "Path",
    }
    for top_node in tree.body:
        if isinstance(top_node, ast.ClassDef):
            known_type_names.add(top_node.name)
        elif isinstance(top_node, ast.Import):
            for alias in top_node.names:
                known_type_names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(top_node, ast.ImportFrom):
            for alias in top_node.names:
                known_type_names.add(alias.asname or alias.name)
        elif isinstance(top_node, ast.Assign):
            for t in top_node.targets:
                if isinstance(t, ast.Name):
                    known_type_names.add(t.id)

    class _DanglingTypeRewriter(ast.NodeTransformer):
        def __init__(self) -> None:
            self.replaced: List[str] = []

        def visit_Subscript(self, sub_node: ast.Subscript) -> ast.AST:
            # Do not rewrite string/identifier arguments inside Literal[...]
            if isinstance(sub_node.value, ast.Name) and sub_node.value.id == "Literal":
                return sub_node
            if isinstance(sub_node.value, ast.Attribute) and sub_node.value.attr == "Literal":
                return sub_node
            return self.generic_visit(sub_node)

        def visit_Name(self, name_node: ast.Name) -> ast.AST:
            if isinstance(name_node.ctx, ast.Load) and name_node.id not in known_type_names:
                self.replaced.append(name_node.id)
                return ast.copy_location(ast.Name(id="str", ctx=ast.Load()), name_node)
            return name_node

    modified = False
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or is_enum_class(node):
            continue

        non_field_prefix_stmts: List[ast.stmt] = []
        non_field_suffix_stmts: List[ast.stmt] = []
        field_entries: List[Tuple[int, str, ast.stmt, bool, List[str], str]] = []
        seen_fields: set = set()
        stripped_details: List[str] = []
        collapsed_details: List[str] = []

        for idx, stmt in enumerate(node.body):
            f_name: Optional[str] = None
            ann_str = ""
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                f_name = stmt.target.id
                # Rewrite any dangling/undefined type references in field annotation to `str`
                rewriter = _DanglingTypeRewriter()
                stmt.annotation = rewriter.visit(stmt.annotation)
                try:
                    ann_str = ast.unparse(stmt.annotation)
                except Exception:
                    ann_str = ""
                if rewriter.replaced:
                    modified = True
                    collapsed_details.append(
                        f"'{f_name}' type annotation -> '{ann_str}' "
                        f"[reason: rewrote undefined type reference(s) {sorted(set(rewriter.replaced))} to 'str']"
                    )
            elif isinstance(stmt, ast.Assign) and stmt.targets and isinstance(stmt.targets[0], ast.Name):
                f_name = stmt.targets[0].id

            if f_name is None:
                if not field_entries:
                    non_field_prefix_stmts.append(stmt)
                else:
                    non_field_suffix_stmts.append(stmt)
                continue

            norm_name = f_name.strip().lower()
            # Rename fields that shadow Pydantic BaseModel attributes or Python keywords
            if norm_name in pydantic_reserved_attrs or norm_name in python_keywords:
                new_f_name = f"{norm_name}_field"
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    stmt.target.id = new_f_name
                elif isinstance(stmt, ast.Assign) and stmt.targets and isinstance(stmt.targets[0], ast.Name):
                    stmt.targets[0].id = new_f_name
                modified = True
                collapsed_details.append(
                    f"'{f_name}' -> '{new_f_name}' [reason: renamed field shadowing reserved keyword/BaseModel attribute]"
                )
                f_name = new_f_name
                norm_name = new_f_name

            if norm_name in seen_fields:
                modified = True
                stripped_details.append(f"{f_name}: {ann_str or 'Any'} [reason: duplicate field name]")
                continue
            seen_fields.add(norm_name)

            is_bool = bool(re.search(r"\bbool\b", ann_str))
            parts = [p for p in norm_name.split("_") if p]
            field_entries.append((idx, norm_name, stmt, is_bool, parts, ann_str or "Any"))

        if not field_entries:
            continue

        # Pass 1: Structural prefix-entropy & suffix-repetition filtering
        raw_prefix2_counts: Counter = Counter(
            f"{e[4][0]}_{e[4][1]}" for e in field_entries if len(e[4]) >= 3
        )
        raw_suffix_counts: Counter = Counter(
            e[4][-1] for e in field_entries if len(e[4]) >= 2
        )
        prefix2_counts: Counter = Counter()
        suffix_counts: Counter = Counter()
        filtered_entries: List[Tuple[int, str, ast.stmt, bool, List[str], str]] = []

        for entry in field_entries:
            idx, norm_name, stmt, is_bool, parts, ann_str = entry

            # 1a. Shared 2-token prefix collapse/cap (collapses 3+ flattened sub-entity fields into a List[str] field)
            if len(parts) >= 3:
                prefix2 = f"{parts[0]}_{parts[1]}"
                if raw_prefix2_counts[prefix2] >= 3:
                    modified = True
                    if prefix2_counts[prefix2] == 0:
                        collapsed_name = prefix2 if prefix2.endswith("s") else f"{prefix2}s"
                        grouped_names = [
                            e[1] for e in field_entries if len(e[4]) >= 3 and f"{e[4][0]}_{e[4][1]}" == prefix2
                        ]
                        if collapsed_name not in seen_fields:
                            seen_fields.add(collapsed_name)
                            human_label = collapsed_name.replace("_", " ")
                            stmt = ast.parse(
                                f'{collapsed_name}: Optional[List[str]] = Field(default=None, description="Associated {human_label}.")'
                            ).body[0]
                            norm_name = collapsed_name
                            is_bool = False
                            parts = [p for p in collapsed_name.split("_") if p]
                            ann_str = "Optional[List[str]]"
                        prefix2_counts[prefix2] += 1
                        filtered_entries.append((idx, norm_name, stmt, is_bool, parts, ann_str))
                        collapsed_details.append(
                            f"{grouped_names} -> '{norm_name}: Optional[List[str]]' "
                            f"[reason: collapsed {len(grouped_names)} fields sharing prefix '{prefix2}_*']"
                        )
                    continue
                if prefix2_counts[prefix2] >= max_shared_prefix:
                    modified = True
                    stripped_details.append(
                        f"{norm_name}: {ann_str} [reason: exceeded shared prefix cap ({max_shared_prefix}) for '{prefix2}_*']"
                    )
                    continue
            else:
                prefix2 = None

            # 1b. Shared suffix collapse/cap (collapses 3+ repeated suffix/checklist fields into a List[str] field)
            suffix1 = parts[-1] if len(parts) >= 2 else None
            if suffix1 and suffix1 not in exempt_suffixes:
                if raw_suffix_counts[suffix1] >= 3:
                    modified = True
                    if suffix_counts[suffix1] == 0:
                        collapsed_name = suffix1 if suffix1.endswith("s") else f"{suffix1}s"
                        grouped_names = [
                            e[1] for e in field_entries if len(e[4]) >= 2 and e[4][-1] == suffix1
                        ]
                        if collapsed_name not in seen_fields:
                            seen_fields.add(collapsed_name)
                            human_label = collapsed_name.replace("_", " ")
                            stmt = ast.parse(
                                f'{collapsed_name}: Optional[List[str]] = Field(default=None, description="Associated {human_label}.")'
                            ).body[0]
                            norm_name = collapsed_name
                            is_bool = False
                            parts = [p for p in collapsed_name.split("_") if p]
                            ann_str = "Optional[List[str]]"
                        suffix_counts[suffix1] += 1
                        filtered_entries.append((idx, norm_name, stmt, is_bool, parts, ann_str))
                        collapsed_details.append(
                            f"{grouped_names} -> '{norm_name}: Optional[List[str]]' "
                            f"[reason: collapsed {len(grouped_names)} fields sharing suffix '*_{suffix1}']"
                        )
                    continue
                if suffix_counts[suffix1] >= max_shared_suffix:
                    modified = True
                    stripped_details.append(
                        f"{norm_name}: {ann_str} [reason: exceeded shared suffix cap ({max_shared_suffix}) for '*_{suffix1}']"
                    )
                    continue

            if prefix2:
                prefix2_counts[prefix2] += 1
            if suffix1 and suffix1 not in exempt_suffixes:
                suffix_counts[suffix1] += 1

            filtered_entries.append((idx, norm_name, stmt, is_bool, parts, ann_str))

        # Pass 2: Enforce hard cap of MAX_BASEMODEL_FIELDS_PER_CLASS (15)
        if len(filtered_entries) > MAX_BASEMODEL_FIELDS_PER_CLASS:
            modified = True
            scored = sorted(
                filtered_entries,
                key=lambda item: (
                    1 if item[3] else 0,                 # non-bool before bool
                    1 if (len(item[4]) >= 3 and raw_prefix2_counts[f"{item[4][0]}_{item[4][1]}"] > 1) else 0,
                    1 if len(item[4]) >= 3 else 0,       # 1-2 token core fields before 3+ token compound fields
                    item[0],                             # original declaration order
                ),
            )
            kept_top = scored[:MAX_BASEMODEL_FIELDS_PER_CLASS]
            dropped_overflow = sorted(scored[MAX_BASEMODEL_FIELDS_PER_CLASS:], key=lambda item: item[0])
            for d_item in dropped_overflow:
                stripped_details.append(
                    f"{d_item[1]}: {d_item[5]} [reason: exceeded MAX_BASEMODEL_FIELDS_PER_CLASS ({MAX_BASEMODEL_FIELDS_PER_CLASS})]"
                )
            kept_top.sort(key=lambda item: item[0])
            filtered_entries = kept_top

        # Guarantee every BaseModel class begins with a descriptive triple-quoted docstring
        has_docstring = (
            bool(non_field_prefix_stmts)
            and isinstance(non_field_prefix_stmts[0], ast.Expr)
            and isinstance(non_field_prefix_stmts[0].value, ast.Constant)
            and isinstance(non_field_prefix_stmts[0].value.value, str)
            and bool(str(non_field_prefix_stmts[0].value.value).strip())
        )
        if not has_docstring:
            human_cls = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", node.name).strip()
            sample_fields = [e[1].replace("_", " ") for e in filtered_entries[:4]]
            if sample_fields:
                synth_doc = f"Represents {human_cls} domain attributes, including {', '.join(sample_fields)}."
            else:
                synth_doc = f"Represents {human_cls} domain entity and its associated attributes."
            non_field_prefix_stmts.insert(0, ast.Expr(value=ast.Constant(value=synth_doc)))
            modified = True

        if stripped_details or collapsed_details:
            _log_sanitization_detail(
                context_label=context_label,
                class_name=node.name,
                class_kind="BaseModel",
                kept_count=len(filtered_entries),
                stripped_items=stripped_details,
                collapsed_items=collapsed_details,
            )

        if filtered_entries:
            node.body = non_field_prefix_stmts + [e[2] for e in filtered_entries] + non_field_suffix_stmts
        else:
            node.body = non_field_prefix_stmts + [ast.Pass()] + non_field_suffix_stmts
            modified = True

    # Canonical top-level ordering:
    # (__future__ (if any) -> standard imports -> constants/aliases -> Enums -> topologically ordered BaseModels)
    future_nodes: List[ast.AST] = []
    import_nodes: List[ast.AST] = []
    enum_nodes: List[ast.ClassDef] = []
    model_nodes: List[ast.ClassDef] = []
    other_top_nodes: List[ast.AST] = []

    for top_node in tree.body:
        if isinstance(top_node, ast.ImportFrom) and top_node.module == "__future__":
            future_nodes.append(top_node)
        elif isinstance(top_node, (ast.Import, ast.ImportFrom)):
            import_nodes.append(top_node)
        elif isinstance(top_node, ast.ClassDef):
            if is_enum_class(top_node):
                enum_nodes.append(top_node)
            else:
                model_nodes.append(top_node)
        else:
            other_top_nodes.append(top_node)

    # Topologically order BaseModel classes so referenced child BaseModels precede parent BaseModels
    if len(model_nodes) >= 1:
        model_map = {m.name: m for m in model_nodes}
        deps_by_model: Dict[str, List[str]] = {}
        for m in model_nodes:
            refs: List[str] = []
            for st in m.body:
                if isinstance(st, ast.AnnAssign):
                    for sub in ast.walk(st.annotation):
                        if isinstance(sub, ast.Name) and sub.id in model_map and sub.id != m.name and sub.id not in refs:
                            refs.append(sub.id)
            deps_by_model[m.name] = refs

        ordered_models: List[ast.ClassDef] = []
        visited_models: set = set()
        visiting_models: set = set()

        def _visit_model(m_name: str) -> None:
            if m_name in visited_models or m_name in visiting_models:
                return
            visiting_models.add(m_name)
            for dep in deps_by_model.get(m_name, []):
                if dep in model_map:
                    _visit_model(dep)
            visiting_models.remove(m_name)
            visited_models.add(m_name)
            ordered_models.append(model_map[m_name])

        for m in model_nodes:
            _visit_model(m.name)
        if [m.name for m in ordered_models] != [m.name for m in model_nodes]:
            modified = True
        model_nodes = ordered_models

        # Quote any remaining self-reference or cyclic forward reference so class creation never raises NameError
        defined_so_far: set = set()
        for m in model_nodes:
            class _ForwardRefQuoter(ast.NodeTransformer):
                def __init__(self, current_cls: str, ready: set) -> None:
                    self.current_cls = current_cls
                    self.ready = ready
                    self.quoted = False

                def visit_Name(self, n: ast.Name) -> ast.AST:
                    if isinstance(n.ctx, ast.Load) and n.id in model_map and (n.id == self.current_cls or n.id not in self.ready):
                        self.quoted = True
                        return ast.copy_location(ast.Constant(value=n.id), n)
                    return n

            quoter = _ForwardRefQuoter(m.name, defined_so_far)
            for st in m.body:
                if isinstance(st, ast.AnnAssign):
                    st.annotation = quoter.visit(st.annotation)
            if quoter.quoted:
                modified = True
            defined_so_far.add(m.name)

    new_tree_body = future_nodes + import_nodes + other_top_nodes + enum_nodes + model_nodes
    if new_tree_body != tree.body:
        tree.body = new_tree_body
        modified = True

    if not modified:
        return code

    try:
        return ast.unparse(tree)
    except Exception:
        return code


def clean_python_code(code: str, context_label: str = "schema") -> str:
    """Removes markdown formatting, invalid characters, and escapes that LLMs often inject."""
    if not code:
        return ""
    code = code.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]

    # Fix over-escaped quotes caused by LLM JSON serialization
    code = code.replace('\\"', '"')

    # Replace literal \n escape sequences that some LLMs emit outside raw strings
    code = code.replace("\\n", "\n")

    python_keywords = set(keyword.kwlist) | {"async", "await"}
    keyword_renames: List[str] = []

    def fix_line(line: str) -> str:
        # 1. Assignment '=' fix
        m = re.match(r"^(\s+)([a-zA-Z0-9_, ]+?)(\s*=\s*.*)$", line)
        if m:
            indent, var, rest = m.groups()
            stripped_var = var.strip()
            if ("," in stripped_var or " " in stripped_var) and not stripped_var.startswith(
                ("def ", "class ", "return ", "if ", "elif ")
            ):
                clean_var = re.sub(r"[, ]+", "_", stripped_var)
                line = f"{indent}{clean_var}{rest}"

        # 2. Type annotation ':' fix for hyphens & Python reserved keywords
        m_ann = re.match(r"^(\s+)([a-zA-Z_][a-zA-Z0-9_-]*)(\s*:\s*.*)$", line)
        if m_ann:
            indent, var, rest = m_ann.groups()
            clean_var = var.replace("-", "_") if "-" in var else var
            if clean_var in python_keywords:
                renamed_kw = f"{clean_var}_field"
                keyword_renames.append(f"'{clean_var}' -> '{renamed_kw}' [reason: Python reserved keyword used as field name]")
                clean_var = renamed_kw
            if clean_var != var:
                line = f"{indent}{clean_var}{rest}"
        return line

    code = "\n".join(fix_line(line_str) for line_str in code.splitlines())
    if keyword_renames:
        _log_sanitization_detail(
            context_label=context_label,
            class_name="<module>",
            class_kind="KeywordFieldFix",
            kept_count=len(keyword_renames),
            stripped_items=[],
            collapsed_items=keyword_renames,
        )

    # Strip rogue CJK / non-ASCII character leakage from bilingual models in comments or docstrings
    code = re.sub(r"[\u4e00-\u9fff]+", "", code)

    # Strip any accidental meta-modeling classes that mirror the prompt/JSON wrapper
    meta_matches = re.findall(
        r"(?m)^class\s+(ThemeObservation|ThematicObservation|GeneratedModule)\s*\([^)]*\):",
        code,
    )
    if meta_matches:
        code = re.sub(
            r"(?m)^class\s+(ThemeObservation|ThematicObservation|GeneratedModule)\s*\([^)]*\):.*?(?=(?:\nclass\s+|\n[A-Za-z0-9_]+\s*=|\Z))",
            "",
            code,
            flags=re.DOTALL,
        )
        for m_cls in meta_matches:
            _log_sanitization_detail(
                context_label=context_label,
                class_name=m_cls,
                class_kind="MetaModel",
                kept_count=0,
                stripped_items=[f"class {m_cls} [reason: banned prompt-wrapper meta-model]"],
                extra_note="Meta-model class mirroring prompt envelope was stripped completely.",
            )
    code = _sanitize_ast_enums(code.strip(), context_label=context_label)
    code = _sanitize_ast_models(code.strip(), context_label=context_label)
    return code.strip()


def deduplicate_consolidated_classes(code: str, context_label: str = "master_ontology") -> str:
    """
    Programmatic AST post-processor for consolidated ontologies (master_ontology.py and comprehensive_ontology.py).
    Merges duplicate Enums and BaseModels, disambiguates Enum vs BaseModel naming collisions,
    and enforces MAX_ENUM_MEMBERS_PER_CLASS and MAX_BASEMODEL_FIELDS_PER_CLASS.
    """
    clean_code = clean_python_code(code, context_label=context_label)
    try:
        tree = ast.parse(clean_code)
    except Exception:
        return clean_code

    def is_enum_class(node: ast.ClassDef) -> bool:
        for b in node.bases:
            if isinstance(b, ast.Name) and "Enum" in b.id:
                return True
            if isinstance(b, ast.Attribute) and "Enum" in b.attr:
                return True
        return False

    def get_assigned_name(stmt: ast.AST) -> Optional[str]:
        if isinstance(stmt, ast.Assign) and stmt.targets and isinstance(stmt.targets[0], ast.Name):
            return stmt.targets[0].id
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            return stmt.target.id
        return None

    enums_by_name: Dict[str, ast.ClassDef] = {}
    models_by_name: Dict[str, ast.ClassDef] = {}
    other_nodes: List[ast.AST] = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            name = node.name
            if is_enum_class(node):
                if name not in enums_by_name:
                    enums_by_name[name] = node
                else:
                    existing_members = {
                        get_assigned_name(stmt)
                        for stmt in enums_by_name[name].body
                        if get_assigned_name(stmt) is not None
                    }
                    added_members, skipped_members = [], []
                    for stmt in node.body:
                        m_name = get_assigned_name(stmt)
                        if m_name:
                            if m_name not in existing_members:
                                enums_by_name[name].body.append(stmt)
                                existing_members.add(m_name)
                                added_members.append(m_name)
                            else:
                                skipped_members.append(f"{m_name} [reason: duplicate member across merged Enum '{name}']")
                    _log_sanitization_detail(
                        context_label=context_label,
                        class_name=name,
                        class_kind="Enum (Consolidation Merge)",
                        kept_count=len(existing_members),
                        stripped_items=skipped_members,
                        extra_note=f"Merged duplicate Enum class '{name}' (added {len(added_members)} unique member(s)).",
                    )
            else:
                if name not in models_by_name:
                    models_by_name[name] = node
                else:
                    # If the existing model lacks a docstring and the merged one has a docstring, adopt it
                    existing_has_doc = (
                        bool(models_by_name[name].body)
                        and isinstance(models_by_name[name].body[0], ast.Expr)
                        and isinstance(models_by_name[name].body[0].value, ast.Constant)
                        and isinstance(models_by_name[name].body[0].value.value, str)
                    )
                    incoming_has_doc = (
                        bool(node.body)
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)
                    )
                    if not existing_has_doc and incoming_has_doc:
                        models_by_name[name].body.insert(0, node.body[0])

                    existing_fields = {
                        get_assigned_name(stmt)
                        for stmt in models_by_name[name].body
                        if get_assigned_name(stmt) is not None
                    }
                    added_fields, skipped_fields = [], []
                    for stmt in node.body:
                        f_name = get_assigned_name(stmt)
                        if f_name:
                            if f_name not in existing_fields:
                                models_by_name[name].body.append(stmt)
                                existing_fields.add(f_name)
                                added_fields.append(f_name)
                            else:
                                skipped_fields.append(f"{f_name} [reason: duplicate field across merged BaseModel '{name}']")
                    _log_sanitization_detail(
                        context_label=context_label,
                        class_name=name,
                        class_kind="BaseModel (Consolidation Merge)",
                        kept_count=len(existing_fields),
                        stripped_items=skipped_fields,
                        extra_note=f"Merged duplicate BaseModel class '{name}' (added {len(added_fields)} unique field(s)).",
                    )
        else:
            other_nodes.append(node)

    # Disambiguate Enum vs BaseModel name collisions
    for m_name in list(models_by_name.keys()):
        if m_name in enums_by_name:
            new_name = f"{m_name}Detail"
            model_node = models_by_name.pop(m_name)
            model_node.name = new_name
            models_by_name[new_name] = model_node
            _log_sanitization_detail(
                context_label=context_label,
                class_name=m_name,
                class_kind="BaseModel (Collision Rename)",
                kept_count=len(model_node.body),
                stripped_items=[],
                extra_note=f"Renamed BaseModel '{m_name}' -> '{new_name}' to disambiguate collision with Enum '{m_name}'.",
            )

    tree.body = other_nodes + list(enums_by_name.values()) + list(models_by_name.values())
    try:
        merged_code = ast.unparse(tree)
        merged_code = _sanitize_ast_enums(merged_code, context_label=f"{context_label} (merged)")
        merged_code = _sanitize_ast_models(merged_code, context_label=f"{context_label} (merged)")
        return merged_code
    except Exception:
        return clean_code


def sanitize_module_name(name: str, fallback: str = "domain_schema") -> str:
    """Normalizes a module_name string into a valid snake_case Python module identifier."""
    if not name:
        return fallback
    cleaned = re.sub(r"\.py$", "", str(name).strip().lower())
    cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned).strip("_")
    stripped_suffix = re.sub(r"_(?:schema|schemas|module|ontology|models|model)$", "", cleaned).strip("_")
    if stripped_suffix and stripped_suffix not in ("domain", "community", "master", "comprehensive"):
        cleaned = stripped_suffix
    return cleaned or fallback


def _camel_to_snake(name: str) -> str:
    """Converts a PascalCase / CamelCase identifier into snake_case."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", str(name).strip())
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    return re.sub(r"[^a-z0-9_]+", "_", s2).strip("_")


def _extract_ast_basemodel_names(code: str) -> List[str]:
    """Returns snake_case names of all non-Enum BaseModel classes defined in `code`."""
    try:
        tree = ast.parse(code)
    except Exception:
        return []

    def is_enum_class(node: ast.ClassDef) -> bool:
        for b in node.bases:
            if isinstance(b, ast.Name) and "Enum" in b.id:
                return True
            if isinstance(b, ast.Attribute) and "Enum" in b.attr:
                return True
        return False

    names: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not is_enum_class(node):
            snake = sanitize_module_name(_camel_to_snake(node.name), fallback="")
            if snake and snake not in names:
                names.append(snake)
    return names


def _resolve_unique_module_name(
    raw_mod_name: str,
    code: str,
    top_terms: Optional[List[str]],
    used_names: set,
    fallback: str,
) -> str:
    """
    Ensures each community `.py` file receives a distinct, specific snake_case module name.
    If the LLM's `raw_mod_name` is generic or collides with an already-used module name in
    `used_names`, disambiguates using the module's AST BaseModel class names and C-TF-IDF
    `top_domain_terms`.
    """
    candidate = sanitize_module_name(raw_mod_name, fallback=fallback)
    is_generic = candidate in ("domain_schema", "community_schema", fallback) or candidate.startswith("community_")
    model_names = _extract_ast_basemodel_names(code)
    term_slugs = [
        sanitize_module_name(t, fallback="")
        for t in (top_terms or [])
        if sanitize_module_name(t, fallback="")
    ]

    if not is_generic and candidate not in used_names:
        used_names.add(candidate)
        return candidate

    # 1. Try primary BaseModel class names directly (in snake_case)
    for m_name in model_names:
        if m_name not in used_names:
            used_names.add(m_name)
            return m_name

    # 2. Try combining the top distinctive C-TF-IDF term with the primary BaseModel name or candidate
    for t_slug in term_slugs:
        if model_names:
            t_tokens = [tok for tok in t_slug.split("_") if tok and tok not in model_names[0].split("_")]
            if t_tokens:
                combined_model = f"{'_'.join(t_tokens[:2])}_{model_names[0]}"
                if combined_model not in used_names:
                    used_names.add(combined_model)
                    return combined_model
        if not is_generic:
            t_tokens = [tok for tok in t_slug.split("_") if tok and tok not in candidate.split("_")]
            if t_tokens:
                combined_cand = f"{candidate}_{'_'.join(t_tokens[:2])}"
                if combined_cand not in used_names:
                    used_names.add(combined_cand)
                    return combined_cand
        if len(t_slug.split("_")) >= 2 and t_slug not in used_names:
            used_names.add(t_slug)
            return t_slug

    # 3. Try combining top two BaseModel names
    if len(model_names) >= 2:
        pair_name = f"{model_names[0]}_{model_names[1]}"
        if pair_name not in used_names:
            used_names.add(pair_name)
            return pair_name

    # 4. Deterministic fallback with target suffix
    base = model_names[0] if model_names else (term_slugs[0] if term_slugs else candidate)
    unique_cand = f"{base}_{fallback}" if base != fallback else fallback
    used_names.add(unique_cand)
    return unique_cand


def validate_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Checks whether cleaned Python code parses into a non-empty AST with at least one class."""
    if not code or not code.strip():
        return False, "Empty source_code returned."
    try:
        tree = ast.parse(code)
        has_class = any(isinstance(node, ast.ClassDef) for node in tree.body)
        if not has_class:
            return False, "Generated source_code contains no Python class definitions (BaseModel or Enum)."
        return True, None
    except SyntaxError as err:
        return False, f"SyntaxError: {err}"
    except Exception as err:
        return False, str(err)


# ============================================================================
# STAGE 4 SYNTHESIS PIPELINE (Powered by PrismRunner)
# ============================================================================

class SynthesisPipeline:
    """
    Stage 4 Synthesis Pipeline using PrismRunner for batch local LLM execution.
    Executes single-pass ontological synthesis from raw source sentence text:
      - Phase 1: Community Schema Synthesis (`<id>_<module>.py`)
      - Phase 2: Global Consolidation (`master_ontology.py` & `comprehensive_ontology.py`)
    """

    def __init__(self, config: Dict[str, Any], prism_pipeline: Optional[PrismPipeline] = None):
        self.config = config
        self.model_name = (
            config.get("synthesis", {}).get("llm", {}).get("model_name")
            or config.get("llm", {}).get("model_name")
            or MODEL_NAME
        )

        # Guard against accidentally passing a SentenceTransformer embedding model as MODEL_NAME
        _lower_llm = self.model_name.lower()
        if any(marker in _lower_llm for marker in ("bge-", "minilm", "e5-", "gte-", "nomic-embed", "sentence-transformers")):
            raise ValueError(
                f"MODEL_NAME is set to '{self.model_name}', which is an encoder-only embedding model, "
                f"not a generative causal LLM.\n"
                f"  -> Set MODEL_NAME to a causal/coding LLM (e.g. 'Qwen/Qwen3-Coder-Next', 'Qwen/Qwen3.6-27B', "
                f"or 'Qwen/Qwen2.5-Coder-32B-Instruct')."
            )

        if prism_pipeline is not None:
            self.prism = prism_pipeline
        else:
            self.prism = PrismPipeline(
                model_name=self.model_name,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                config={
                    "model": {
                        "name": self.model_name,
                        "device": DEVICE,
                        "torch_dtype": TORCH_DTYPE,
                        "load_in_4bit": LOAD_IN_4BIT,
                        "load_in_8bit": LOAD_IN_8BIT,
                        "attn_implementation": ATTN_IMPLEMENTATION,
                    },
                    "inference": {
                        "batch_size": BATCH_SIZE,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "temperature": TEMPERATURE,
                        "repetition_penalty": REPETITION_PENALTY,
                        "constrain_json": CONSTRAIN_JSON,
                    },
                },
            )

    def _checkpoint(self, phase: str, message: str, file_path: Optional[Path] = None) -> None:
        """Prints a formatted real-time checkpoint notification when PRINT_CHECKPOINTS is enabled."""
        if not PRINT_CHECKPOINTS:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        path_suffix = f" -> {file_path}" if file_path is not None else ""
        print(f"[{ts} | Checkpoint | {phase}] {message}{path_suffix}", flush=True)

    def _apply_context_window_cap(self) -> int:
        """Apply CONTEXT_WINDOW_CAP if specified, otherwise use the model's native context limit."""
        native_ctx = getattr(self.prism, "context_window", 16384) or 16384
        raw_cap = self.config.get("synthesis", {}).get("context_window_cap", CONTEXT_WINDOW_CAP)
        if isinstance(raw_cap, int) and raw_cap > 0:
            effective_ctx = min(native_ctx, raw_cap)
        else:
            effective_ctx = native_ctx
        self.prism.context_window = effective_ctx
        if self.prism.engine is not None:
            self.prism.engine.context_window = effective_ctx
        return effective_ctx

    def _ensure_engine_loaded(self, phase_label: str) -> None:
        """Ensures the PrismPipeline engine is loaded and GPU memory is cleared beforehand."""
        if DRY_RUN:
            return
        if self.prism.engine is None:
            if MANAGE_VRAM:
                clear_device_memory()
            self._checkpoint(phase_label, f"Loading LLM '{self.model_name}' onto device...")
            self.prism.init_engine(
                batch_size=BATCH_SIZE,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                repetition_penalty=REPETITION_PENALTY,
            )
            eff_ctx = self._apply_context_window_cap()
            self._checkpoint(
                phase_label,
                f"Model ready on device '{self.prism.device}' "
                f"(effective context window: {eff_ctx} tokens)",
            )

    def _unload_prism_pipeline(self, phase_label: str = "Cleanup") -> None:
        """Releases LLM weights and flushes GPU cache."""
        if getattr(self.prism, "engine", None) is not None or getattr(self.prism, "model", None) is not None:
            self._checkpoint(phase_label, f"Unloading LLM '{self.model_name}' and clearing VRAM...")
            self.prism.engine = None
            self.prism.model = None
            self.prism.tokenizer = None
        if MANAGE_VRAM:
            clear_device_memory()

    @staticmethod
    def _extract_generated_module(items: List[Any], fallback_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Extracts (module_name, cleaned_source_code) from parsed PrismRunner items."""
        for item in items:
            if isinstance(item, GeneratedModule):
                mod_name = sanitize_module_name(item.module_name, fallback=fallback_name)
                code = clean_python_code(item.source_code, context_label=mod_name or fallback_name)
                if code:
                    return mod_name, code
            elif isinstance(item, dict):
                try:
                    parsed = GeneratedModule.model_validate(item)
                    mod_name = sanitize_module_name(parsed.module_name, fallback=fallback_name)
                    code = clean_python_code(parsed.source_code, context_label=mod_name or fallback_name)
                    if code:
                        return mod_name, code
                except Exception:
                    pass
        return None, None

    def _run_module_batch(
        self,
        items: List[Dict[str, Any]],
        system_prompt: str,
        phase_label: str,
        default_module_name: str = "domain_schema",
        allow_reformat_repair: bool = True,
        max_new_tokens: int = MAX_NEW_TOKENS,
        on_valid_module_ready: Optional[Callable[[Dict[str, Any], str, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes a batch of prompt payloads through PrismRunner using the `GeneratedModule` schema.
        Validates the resulting Python source code with `ast.parse` and automatically runs a
        batch repair pass with `SCHEMA_REFORMAT_SYSTEM_PROMPT` for any items that failed validation
        or produced syntax errors.

        Each item in `items` must contain:
          - `id`: unique string identifier
          - `payload`: user prompt string
          - `fallback_module_name` (optional): fallback snake_case module name
        Returns a list of dicts (aligned with `items`) with keys:
          - `id`, `success`, `module_name`, `source_code`, `error`, `raw_text`
        """
        if not items:
            return []

        self._ensure_engine_loaded(phase_label)

        rows = [
            {"uid": str(item["id"]), "text": str(item["payload"])}
            for item in items
        ]
        meta_by_uid = {str(item["id"]): (idx, item) for idx, item in enumerate(items)}

        module_template = {
            "version": "1.0",
            "task": "schema_synthesis",
            "system_prompt": system_prompt,
            "prompt": "{{ text }}\n\n{schema_block}",
        }

        parsed_outputs: List[Dict[str, Any]] = [
            {
                "id": item["id"],
                "success": False,
                "module_name": item.get("fallback_module_name", default_module_name),
                "source_code": "",
                "error": "Not yet processed",
                "raw_text": "",
            }
            for item in items
        ]
        notified_uids: set = set()
        lock = threading.Lock()

        def _on_batch_start(batch_idx: int, total_batches: int, batch_items: List[Dict[str, Any]]) -> None:
            ids_preview = [str(it.get("record_id") or "") for it in batch_items[:4]]
            self._checkpoint(
                phase_label,
                f"Running PrismRunner batch {batch_idx}/{total_batches} ({len(batch_items)} item(s): {ids_preview})",
            )

        def _on_chunk_processed(chunk_res: Dict[str, Any]) -> None:
            uid = str(chunk_res.get("record_id") or chunk_res.get("id") or "")
            if uid not in meta_by_uid:
                return
            idx, item = meta_by_uid[uid]
            fallback_name = item.get("fallback_module_name", default_module_name)
            p_items = chunk_res.get("parsed_items") or chunk_res.get("extracted_items") or []
            raw_out = str(chunk_res.get("raw_output") or "")
            mod_name, cleaned_code = self._extract_generated_module(p_items, fallback_name)

            with lock:
                if uid in notified_uids:
                    return
                if cleaned_code:
                    is_valid, syntax_err = validate_python_syntax(cleaned_code)
                    if is_valid:
                        notified_uids.add(uid)
                        parsed_outputs[idx] = {
                            "id": item["id"],
                            "success": True,
                            "module_name": mod_name or fallback_name,
                            "source_code": cleaned_code,
                            "error": None,
                            "raw_text": raw_out,
                        }
                        if on_valid_module_ready is not None:
                            on_valid_module_ready(item, mod_name or fallback_name, cleaned_code)
                        return
                    err_reason = syntax_err or "Invalid Python syntax in generated source_code."
                    malformed = cleaned_code or raw_out
                else:
                    err_reason = str(chunk_res.get("parse_error") or "Constrained JSON validation failed.")
                    malformed = raw_out

                parsed_outputs[idx] = {
                    "id": item["id"],
                    "success": False,
                    "module_name": fallback_name,
                    "source_code": "",
                    "error": err_reason,
                    "raw_text": malformed,
                }

        results = self.prism.run(
            task="extract",
            input_data=rows,
            text_column="text",
            id_column="uid",
            schema_cls=GeneratedModule,
            prompt_template=module_template,
            system_prompt=system_prompt,
            chunk=False,
            as_array=False,
            dedup=False,
            batch_size=BATCH_SIZE,
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            repetition_penalty=REPETITION_PENALTY,
            constrain_json=CONSTRAIN_JSON,
            dry_run=DRY_RUN,
            on_batch_start=_on_batch_start,
            on_chunk_processed=_on_chunk_processed,
        )

        if DRY_RUN:
            for idx, item in enumerate(items):
                fallback_name = item.get("fallback_module_name", default_module_name)
                stub_code = (
                    "from enum import Enum\n"
                    "from typing import Optional\n"
                    "from pydantic import BaseModel, Field\n\n\n"
                    f"class DryRunModel_{idx + 1}(BaseModel):\n"
                    '    summary: Optional[str] = Field(default=None, description="Dry-run placeholder schema.")\n'
                )
                parsed_outputs[idx] = {
                    "id": item["id"],
                    "success": True,
                    "module_name": fallback_name,
                    "source_code": stub_code,
                    "error": None,
                    "raw_text": "",
                }
                if on_valid_module_ready is not None:
                    on_valid_module_ready(item, fallback_name, stub_code)
            return parsed_outputs

        # Ensure any items not processed by streaming callback are finalized
        for idx, (item, res) in enumerate(zip(items, results)):
            uid = str(item["id"])
            if uid not in notified_uids and not parsed_outputs[idx]["success"]:
                _on_chunk_processed(
                    {
                        "record_id": uid,
                        "parsed_items": res.get("extracted_items") or res.get("parsed_items") or [],
                        "raw_output": (res.get("raw_outputs") or [""])[0] if res.get("raw_outputs") else "",
                        "parse_error": res.get("warning"),
                    }
                )

        # Collect any failed items for the Schema Reformat Repair Agent pass
        repair_candidates: List[Tuple[int, Dict[str, Any], str, str]] = [
            (idx, items[idx], out["raw_text"], str(out["error"]))
            for idx, out in enumerate(parsed_outputs)
            if not out["success"]
        ]

        if repair_candidates and allow_reformat_repair:
            self._checkpoint(
                f"{phase_label} Repair",
                f"Running Schema Reformat Agent via PrismRunner on {len(repair_candidates)} failed item(s)...",
            )
            repair_rows = []
            for idx, item, malformed, err_reason in repair_candidates:
                reformat_prompt = (
                    f"Original Schema Generation Task context was:\n{item['payload'][:4000]}\n\n"
                    f"The malformed Python output from the failed attempts was:\n{malformed[:4000]}\n\n"
                    f"The parsing validation error that occurred was:\n{err_reason}\n\n"
                    f"Please correct and repair this Python code so it strictly maps to the schemas.GeneratedModule structure. "
                    f"The output must consist strictly of the Pydantic GeneratedModule schema (with 'module_name' and 'source_code') "
                    f"and must contain no conversational text, notes, markdown code blocks, or other extraneous words. "
                    f"IMPORTANT: Do NOT generate meta-models representing the prompt envelope (e.g. NEVER output 'class ThemeObservation' or 'class GeneratedModule'). "
                    f"Synthesize authentic domain models reflecting the key terms and observations."
                )
                repair_rows.append({"uid": str(item["id"]), "text": reformat_prompt})

            reformat_template = {
                "version": "1.0",
                "task": "schema_reformat",
                "system_prompt": SCHEMA_REFORMAT_SYSTEM_PROMPT,
                "prompt": "{{ text }}\n\n{schema_block}",
            }

            repair_results = self.prism.run(
                task="extract",
                input_data=repair_rows,
                text_column="text",
                id_column="uid",
                schema_cls=GeneratedModule,
                prompt_template=reformat_template,
                system_prompt=SCHEMA_REFORMAT_SYSTEM_PROMPT,
                chunk=False,
                as_array=False,
                dedup=False,
                batch_size=BATCH_SIZE,
                max_new_tokens=max_new_tokens,
                temperature=TEMPERATURE,
                repetition_penalty=REPETITION_PENALTY,
                constrain_json=CONSTRAIN_JSON,
                dry_run=DRY_RUN,
            )

            for (idx, item, _, _), rep_res in zip(repair_candidates, repair_results):
                fallback_name = item.get("fallback_module_name", default_module_name)
                rep_items = rep_res.get("extracted_items") or rep_res.get("parsed_items") or []
                mod_name, cleaned_code = self._extract_generated_module(rep_items, fallback_name)
                if cleaned_code:
                    parsed_outputs[idx] = {
                        "id": item["id"],
                        "success": True,
                        "module_name": mod_name or fallback_name,
                        "source_code": cleaned_code,
                        "error": None,
                        "raw_text": (rep_res.get("raw_outputs") or [""])[0] if rep_res.get("raw_outputs") else "",
                    }
                    print(f"            -> Reformat successful for {item['id']}!")
                    if on_valid_module_ready is not None:
                        on_valid_module_ready(item, mod_name or fallback_name, cleaned_code)
                else:
                    parsed_outputs[idx]["error"] = str(rep_res.get("warning") or parsed_outputs[idx]["error"])

        return parsed_outputs

    def execute(
        self,
        comm_targets: Optional[Dict[str, Any]] = None,
        refined_triplets: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        print("[Synthesis] Starting Stage 4 Pipeline...")
        try:
            if comm_targets:
                print("   ==> Executing Community Path Schema Synthesis...")
                self.execute_path_synthesis(
                    comm_targets,
                    "community",
                    refined_triplets or [],
                )
            else:
                print("   [Warning] Community resolved targets file not found. Skipping synthesis.")
        finally:
            self._unload_prism_pipeline("Stage 4 Complete")

        print("[Synthesis] Stage 4 Pipeline complete.")

    def execute_path_synthesis(
        self,
        targets_data: Dict[str, Any],
        path_type: str,
        refined_triplets: List[dict],
    ) -> None:
        path_label = "Community Workflow" if path_type == "community" else f"Path: {path_type.title()}"
        print(f"   -> Starting Single-Pass Synthesis for [{path_label}] (Raw Source Sentence Text)...")

        synth_cfg = self.config.get("synthesis", {})
        min_cluster_size = synth_cfg.get("min_cluster_size", MIN_CLUSTER_SIZE)
        print(f"      -> Entity Threshold (min_cluster_size): {min_cluster_size}")

        # Setup logging
        outputs_dir = self.config.get("directories", {}).get("outputs", OUTPUTS_DIR)
        log_dir = Path(outputs_dir) / "04_synthesis" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"synthesis_{path_type}_errors.log"

        def log_error(msg: str) -> None:
            print(msg)
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"[{datetime.now().isoformat()}] {msg}\n")

        # Single-pass output directory
        output_base_dir = Path(outputs_dir) / "04_synthesis" / "schemas" / path_type
        if not RESUME_EXISTING_SCHEMAS and output_base_dir.exists():
            shutil.rmtree(output_base_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)

        # Payload dumping directory
        save_llm_payloads = synth_cfg.get("save_llm_payloads", SAVE_LLM_PAYLOADS)
        payload_base_dir = Path(outputs_dir) / "payload_text" / "stage_04_synthesis" / path_type
        if save_llm_payloads:
            if not RESUME_EXISTING_SCHEMAS and payload_base_dir.exists():
                shutil.rmtree(payload_base_dir)
            payload_base_dir.mkdir(parents=True, exist_ok=True)

        def save_payload_file(filename: str, payload_str: str) -> None:
            if save_llm_payloads:
                try:
                    with open(payload_base_dir / filename, "w", encoding="utf-8") as pf:
                        pf.write(payload_str)
                except Exception as err:
                    log_error(f"[Synthesis {path_type}] Error saving payload {filename}: {err}")

        targets = targets_data.get("targets", [])

        # =====================================================================
        # PHASE 1: Community Schema Generation (Single Pass from Raw Source Text)
        # =====================================================================
        print(f"   -> [{path_label}] Phase 1: Community Schema Generation...")
        print(f"      -> Target Composition: {len(targets)} qualified schema targets")

        max_triplets_per_target = int(synth_cfg.get("max_triplets_per_target", MAX_TRIPLETS_PER_TARGET))
        max_sentences_per_target = int(synth_cfg.get("max_sentences_per_target", MAX_SENTENCES_PER_TARGET))
        top_domain_terms_count = int(synth_cfg.get("top_domain_terms_count", TOP_DOMAIN_TERMS_COUNT))
        min_sentence_length = int(synth_cfg.get("min_sentence_length", MIN_SENTENCE_LENGTH))
        domain_terms_idf_exponent = float(synth_cfg.get("domain_terms_idf_exponent", DOMAIN_TERMS_IDF_EXPONENT))
        observation_overlap_weight = float(synth_cfg.get("observation_overlap_weight", OBSERVATION_OVERLAP_WEIGHT))
        observation_overlap_max_cap = int(synth_cfg.get("observation_overlap_max_cap", OBSERVATION_OVERLAP_MAX_CAP))
        print(
            f"      -> Payload Limits: max_triplets_per_target={max_triplets_per_target}, "
            f"max_sentences_per_target={max_sentences_per_target}, top_domain_terms={top_domain_terms_count}"
        )

        # Precompute cross-cluster presence (n_targets) for C-TF-IDF so "strong" concepts that
        # connect to or appear across multiple clusters are penalized by DOMAIN_TERMS_IDF_EXPONENT,
        # even though Leiden allocates each node to a single cluster's `nodes` list.
        node_to_cluster: Dict[str, int] = {}
        term_clusters: Dict[str, set] = defaultdict(set)
        cluster_quote_blobs: List[Tuple[int, str]] = []

        for c_idx, t in enumerate(targets):
            for n in t.get("nodes", []):
                nk = str(n).lower().strip()
                if nk:
                    node_to_cluster[nk] = c_idx
                    term_clusters[nk].add(c_idx)
            t_quotes = {
                str(tr.get("source_quote", "")).lower()
                for tr in t.get("triplets", [])
                if tr.get("source_quote")
            }
            for tr in t.get("triplets", []):
                s_tr = str(tr.get("subject", "")).lower().strip()
                o_tr = str(tr.get("object", "")).lower().strip()
                if s_tr:
                    term_clusters[s_tr].add(c_idx)
                if o_tr:
                    term_clusters[o_tr].add(c_idx)
            if t_quotes:
                cluster_quote_blobs.append((c_idx, "\n".join(t_quotes)))

        # 1. Track cross-cluster graph edges in refined_triplets
        for tr in refined_triplets or []:
            s = str(tr.get("subject", "")).lower().strip()
            o = str(tr.get("object", "")).lower().strip()
            c_s = node_to_cluster.get(s)
            c_o = node_to_cluster.get(o)
            if s and c_o is not None:
                term_clusters[s].add(c_o)
            if o and c_s is not None:
                term_clusters[o].add(c_s)

        # 2. Track cross-cluster lexical presence in other clusters' source quotes
        for nk in list(node_to_cluster.keys()):
            if len(nk) <= 2:
                continue
            for c_idx, blob in cluster_quote_blobs:
                if c_idx not in term_clusters[nk] and nk in blob:
                    term_clusters[nk].add(c_idx)

        global_target_counts: Dict[str, int] = {
            nk: max(1, len(clusters)) for nk, clusters in term_clusters.items()
        }

        closed_class_pronouns = {
            "someone", "something", "anyone", "anything",
            "everyone", "everything", "other", "none",
        }

        def extract_top_domain_terms(
            target_nodes: List[str],
            triplets: List[dict],
            top_k: int = TOP_DOMAIN_TERMS_COUNT,
        ) -> List[str]:
            target_nodes_norm = {str(n).lower().strip(): str(n) for n in target_nodes}
            local_deg = Counter()
            for tr in triplets:
                s = str(tr.get("subject", "")).lower().strip()
                o = str(tr.get("object", "")).lower().strip()
                if s in target_nodes_norm:
                    local_deg[s] += 1
                if o in target_nodes_norm:
                    local_deg[o] += 1

            scored = []
            for nk, orig in target_nodes_norm.items():
                if nk in closed_class_pronouns or len(nk) <= 2 or nk.isdigit() or not any(c.isalnum() for c in nk):
                    continue
                n_targets = global_target_counts.get(nk, 1)
                deg = local_deg.get(nk, 0)
                score = (deg + 1.0) / (n_targets ** domain_terms_idf_exponent)
                scored.append((score, deg, orig))

            scored.sort(key=lambda x: x[0], reverse=True)

            seen_stems = set()
            cleaned_terms = []
            for _, _, orig in scored:
                base = orig.lower().strip()
                stem = base[:-1] if base.endswith("s") else base
                if stem in seen_stems or base in seen_stems:
                    continue
                seen_stems.add(stem)
                seen_stems.add(base)
                cleaned_terms.append(orig)
                if len(cleaned_terms) >= top_k:
                    break
            return cleaned_terms

        def extract_thematic_observations(
            triplets: List[dict],
            target_nodes: List[str],
            max_sentences: int = MAX_SENTENCES_PER_TARGET,
        ) -> Dict[str, List[str]]:
            target_nodes_norm = {str(n).lower().strip() for n in target_nodes}
            quote_info: Dict[str, Dict[str, Any]] = {}

            for tr in triplets:
                quote = str(tr.get("source_quote", "")).strip()
                if not quote or len(quote) < min_sentence_length:
                    continue

                theme = tr.get("theme_association", "General Domain Observations") or "General Domain Observations"
                fid = float(tr.get("contextual_fidelity", 0.50))

                if quote not in quote_info:
                    quote_info[quote] = {"theme": theme, "fidelities": [fid]}
                else:
                    quote_info[quote]["fidelities"].append(fid)

            scored_quotes = []
            for q, info in quote_info.items():
                avg_fid = sum(info["fidelities"]) / len(info["fidelities"])
                q_lower = q.lower()
                node_overlap = sum(1 for n in target_nodes_norm if n in q_lower)
                sentence_score = avg_fid * (
                    1.0 + observation_overlap_weight * min(observation_overlap_max_cap, node_overlap)
                )
                scored_quotes.append((sentence_score, q, info["theme"]))

            scored_quotes.sort(key=lambda x: x[0], reverse=True)
            selected = scored_quotes[:max_sentences]

            theme_groups = defaultdict(list)
            for _, text_item, theme in selected:
                theme_groups[theme].append(text_item)

            # Always place 'Other' last regardless of how many theme groups are included
            ordered_theme_groups: Dict[str, List[str]] = {}
            other_entries: List[Tuple[str, List[str]]] = []
            for th_key, quotes_list in theme_groups.items():
                if str(th_key).strip().lower() == "other":
                    other_entries.append((th_key, quotes_list))
                else:
                    ordered_theme_groups[th_key] = quotes_list
            for th_key, quotes_list in other_entries:
                ordered_theme_groups[th_key] = quotes_list

            return ordered_theme_groups

        def format_phase1_payload(
            target_id: Any,
            target_nodes: List[str],
            subset: List[dict],
            merged_from: Optional[List[Any]] = None,
        ) -> Tuple[str, List[str]]:
            capped_subset = subset[:max_triplets_per_target] if max_triplets_per_target > 0 else subset
            top_terms = extract_top_domain_terms(target_nodes, capped_subset, top_k=top_domain_terms_count)
            observations = extract_thematic_observations(
                capped_subset, target_nodes, max_sentences=max_sentences_per_target
            )

            theme_counts = Counter(
                str(tr.get("theme_association")).strip()
                for tr in capped_subset
                if tr.get("theme_association") and str(tr.get("theme_association")).strip()
            )
            # Order themes by frequency descending, and ALWAYS place 'Other' last regardless of how many values are included
            non_other_themes = [
                theme_name
                for theme_name, _ in theme_counts.most_common()
                if theme_name.lower() != "other"
            ]
            other_themes = [
                "Other" if theme_name.lower() == "other" else theme_name
                for theme_name, _ in theme_counts.most_common()
                if theme_name.lower() == "other"
            ]
            ordered_focus_themes = non_other_themes + (["Other"] if other_themes else [])

            if not ordered_focus_themes and observations:
                obs_non_other = [k for k in observations.keys() if str(k).strip().lower() != "other"]
                obs_other = ["Other"] if any(str(k).strip().lower() == "other" for k in observations.keys()) else []
                ordered_focus_themes = obs_non_other + obs_other

            if len(ordered_focus_themes) == 1:
                domain_focus: Union[str, List[str]] = ordered_focus_themes[0]
            elif len(ordered_focus_themes) > 1:
                domain_focus = ", ".join(ordered_focus_themes)
            else:
                domain_focus = f"Community #{target_id}"

            payload_dict: Dict[str, Any] = {
                "community_id": target_id,
                "target_domain_focus": domain_focus,
                "top_domain_terms": top_terms,
                "thematic_observations": observations,
            }
            if merged_from and len(merged_from) > 1:
                payload_dict["consolidated_sub_communities"] = merged_from
            return json.dumps(payload_dict, indent=2), top_terms

        phase1_batch_items: List[Dict[str, Any]] = []
        used_module_names: set = set()
        active_sys_prompt = SCHEMA_SYNTHESIS_PROMPT

        for i, target in enumerate(targets, start=1):
            target_nodes = set(target["nodes"])
            target_type = target.get("target_type", target.get("type", "community"))
            target_id = target.get("target_id", target.get("id"))
            merged_from = target.get("merged_from", [target_id])

            subset = list(target.get("triplets", []))
            if max_triplets_per_target > 0:
                subset = subset[:max_triplets_per_target]
            if subset:
                existing_files = list(output_base_dir.glob(f"{i:02d}_*.py")) if RESUME_EXISTING_SCHEMAS else []
                if existing_files:
                    existing_stem = re.sub(r"^\d+_", "", existing_files[0].stem)
                    if existing_stem:
                        used_module_names.add(existing_stem)
                    print(f"         -> [Resume] Skipping Target {i}/{len(targets)} ({existing_files[0].name} exists)")
                else:
                    print(
                        f"         -> Queuing Target {i}/{len(targets)} ({target_type} {target_id}) "
                        f"(Size: {len(target_nodes)} nodes, {len(subset)}/{max_triplets_per_target} max triplets)..."
                    )
                    user_payload, top_terms = format_phase1_payload(
                        target_id, list(target_nodes), subset, merged_from=merged_from
                    )
                    save_payload_file(
                        f"phase1_target_{i:02d}_{target_id}_payload.txt",
                        f"=== SYSTEM PROMPT ===\n{active_sys_prompt}\n\n=== USER PAYLOAD ===\n{user_payload}",
                    )
                    phase1_batch_items.append({
                        "id": f"target_{i:02d}_{target_id}",
                        "target_index": i,
                        "target_type": target_type,
                        "target_id": target_id,
                        "target_dir": output_base_dir,
                        "payload": user_payload,
                        "top_terms": top_terms,
                        "fallback_module_name": f"community_{target_id}",
                    })

        if phase1_batch_items:
            saved_phase1_ids: set = set()

            def _save_phase1_module(item: Dict[str, Any], mod_name: str, code: str) -> None:
                if item["id"] in saved_phase1_ids:
                    return
                saved_phase1_ids.add(item["id"])
                i = item["target_index"]
                target_dir = item["target_dir"]
                resolved_mod_name = _resolve_unique_module_name(
                    raw_mod_name=mod_name,
                    code=code,
                    top_terms=item.get("top_terms"),
                    used_names=used_module_names,
                    fallback=item.get("fallback_module_name", f"community_{item['target_id']}"),
                )
                filename = f"{i:02d}_{resolved_mod_name}.py"
                out_path = target_dir / filename
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(code)
                self._checkpoint(
                    "Phase 1: Community Schemas",
                    f"Saved Target {i} schema module",
                    out_path,
                )

            phase1_results = self._run_module_batch(
                items=phase1_batch_items,
                system_prompt=active_sys_prompt,
                phase_label="Phase 1: Community Schemas",
                default_module_name="community_schema",
                allow_reformat_repair=True,
                on_valid_module_ready=_save_phase1_module,
            )
            for item, res in zip(phase1_batch_items, phase1_results):
                i = item["target_index"]
                if res["success"] and res["source_code"]:
                    _save_phase1_module(item, res["module_name"], res["source_code"])
                else:
                    log_error(
                        f"[Synthesis {path_type}] Error generating schema {i} "
                        f"({item['target_type']} {item['target_id']}): {res['error']}"
                    )

        # =====================================================================
        # PHASE 2: Global Consolidation (Master & Comprehensive Ontology)
        # =====================================================================
        print(f"   -> [{path_label}] Phase 2: Global Consolidation...")

        def strip_inline_comments(code: str) -> str:
            """Strips inline '#' comments and extra blank lines while preserving triple-quoted class docstrings."""
            code = re.sub(r"(?m)^\s*#.*$", "", code)
            code = re.sub(r"\n\s*\n", "\n", code)
            return code.strip()

        self._ensure_engine_loaded("Phase 2: Global Consolidation")
        synthesis_cap = self._apply_context_window_cap()
        sys_tokens = estimate_tokens(CONSOLIDATION_SYSTEM_PROMPT)
        max_payload_tokens = max(500, synthesis_cap - sys_tokens - CONSOLIDATION_MAX_NEW_TOKENS)

        def run_consolidation_single(payload: str, label: str) -> str:
            res_list = self._run_module_batch(
                items=[{"id": label, "payload": payload, "fallback_module_name": "master_ontology"}],
                system_prompt=CONSOLIDATION_SYSTEM_PROMPT,
                phase_label=f"Phase 2: Consolidation ({label})",
                default_module_name="master_ontology",
                allow_reformat_repair=True,
                max_new_tokens=CONSOLIDATION_MAX_NEW_TOKENS,
            )
            if res_list and res_list[0]["success"]:
                return res_list[0]["source_code"]
            raise RuntimeError(res_list[0]["error"] if res_list else "Consolidation failed")

        def process_consolidation_sync(files_list: List[dict], schema_dir: Path) -> str:
            payload = "\n".join([f"--- File: {f['name']} ---\n{f['content']}\n" for f in files_list])
            save_payload_file(
                "phase2_consolidation_payload.txt",
                f"=== SYSTEM PROMPT ===\n{CONSOLIDATION_SYSTEM_PROMPT}\n\n=== USER PAYLOAD ===\n{payload}",
            )
            if estimate_tokens(payload) <= max_payload_tokens:
                return run_consolidation_single(payload, schema_dir.name)

            for f in files_list:
                f["content"] = strip_inline_comments(f["content"])
            payload = "\n".join([f"--- File: {f['name']} ---\n{f['content']}\n" for f in files_list])
            if estimate_tokens(payload) <= max_payload_tokens:
                return run_consolidation_single(payload, schema_dir.name)

            sub_batches, current_batch = [], []
            for f in files_list:
                test_batch = current_batch + [f]
                test_payload = "\n".join([f"--- File: {x['name']} ---\n{x['content']}\n" for x in test_batch])
                if estimate_tokens(test_payload) > max_payload_tokens and current_batch:
                    sub_batches.append(current_batch)
                    current_batch = [f]
                else:
                    current_batch.append(f)
            if current_batch:
                sub_batches.append(current_batch)

            intermediate_results = []
            for idx, batch in enumerate(sub_batches, start=1):
                res = process_consolidation_sync(batch, schema_dir)
                intermediate_results.append({"name": f"intermediate_{idx}.py", "content": res})

            return process_consolidation_sync(intermediate_results, schema_dir)

        def consolidate_schemas(schema_dir: Path) -> None:
            py_files = sorted([
                f for f in schema_dir.glob("*.py")
                if f.name not in ("master_ontology.py", "comprehensive_ontology.py", "__init__.py")
            ])
            if not py_files:
                return

            files_list = []
            for pf in py_files:
                with open(pf, "r", encoding="utf-8") as f:
                    files_list.append({"name": pf.name, "content": f.read()})

            try:
                master_code = process_consolidation_sync(files_list, schema_dir)
                master_code = deduplicate_consolidated_classes(master_code)
                out_master = schema_dir / "master_ontology.py"
                with open(out_master, "w", encoding="utf-8") as f:
                    f.write(master_code)
                self._checkpoint("Phase 2: Consolidation", "Saved master_ontology.py", out_master)

                comp_path = schema_dir / "comprehensive_ontology.py"
                with open(comp_path, "w", encoding="utf-8") as f:
                    f.write(master_code)
                self._checkpoint("Phase 2: Consolidation", "Saved comprehensive_ontology.py", comp_path)
            except Exception as e:
                log_error(f"[Synthesis {path_type}] Error generating master_ontology: {e}")

        consolidate_schemas(output_base_dir)

        # Finalization (__init__.py & optional ruff format if ruff is installed)
        with open(output_base_dir / "__init__.py", "w", encoding="utf-8") as f:
            f.write("# Synthesized Community Schemas\n")
        ruff_bin = shutil.which("ruff")
        if ruff_bin:
            try:
                subprocess.run([ruff_bin, "format", str(output_base_dir)], capture_output=True, text=True, check=False)
            except Exception as e:
                log_error(f"[Synthesis {path_type}] Error running ruff on {output_base_dir}: {e}")

        print(f"   [OK] [{path_label}] Synthesis complete: {output_base_dir}")


# ============================================================================
# CONFIGURATION & CLI ENTRYPOINT (Matches run_stage_4_synthesis.py)
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
    global DRY_RUN, RESUME_EXISTING_SCHEMAS
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
    args, _ = parser.parse_known_args()
    if args.dry_run:
        DRY_RUN = True
    if args.resume:
        RESUME_EXISTING_SCHEMAS = True

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
        comm_file = Path(OUTPUTS_DIR) / "03_topology" / "community" / "topology_partitions.json"
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
