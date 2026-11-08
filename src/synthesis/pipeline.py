"""
Stage 4 Synthesis Pipeline: Target Resolution, Schema Synthesis, and Consolidation.
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from datetime import datetime
import json
import keyword
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Ensure repository root (and sibling PrismRunner checkout if present) is importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_SIBLING_PRISMRUNNER = REPO_ROOT.parent / "PrismRunner"
if _SIBLING_PRISMRUNNER.exists() and str(_SIBLING_PRISMRUNNER) not in sys.path:
    sys.path.insert(0, str(_SIBLING_PRISMRUNNER))

import configs.AST_toggle as ast_cfg
from prismrunner import (
    PrismPipeline,
    clear_device_memory,
)
from src.synthesis.schemas import GeneratedModule
from src.synthesis.prompts import (
    get_schema_synthesis_prompt,
    get_consolidation_system_prompt,
    get_schema_reformat_system_prompt,
    SCHEMA_SYNTHESIS_PROMPT,
    CONSOLIDATION_SYSTEM_PROMPT,
    SCHEMA_REFORMAT_SYSTEM_PROMPT,
)
from src.synthesis.sanitizers import (
    estimate_tokens,
    configure_sanitizer_defaults,
    _log_sanitization_detail,
    _is_verbose_or_runaway_enum_member,
    _sanitize_ast_enums,
    _sanitize_ast_models,
    clean_python_code,
    deduplicate_consolidated_classes,
    sanitize_module_name,
    _camel_to_snake,
    _extract_ast_basemodel_names,
    _resolve_unique_module_name,
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

# --- 4. Schema Cardinality Bounds (Configured in configs/AST_toggle.py) ---
MAX_ENUM_MEMBERS_PER_CLASS: int = ast_cfg.MAX_ENUM_MEMBERS_PER_CLASS         # Max distinct members allowed per Enum class
MAX_BASEMODEL_FIELDS_PER_CLASS: int = ast_cfg.MAX_BASEMODEL_FIELDS_PER_CLASS # Max fields allowed per Pydantic BaseModel class

# --- 5. Domain Term Scoring (C-TF-IDF) ---
TOP_DOMAIN_TERMS_COUNT: int = 5
DOMAIN_TERMS_IDF_EXPONENT: float = 1.5

# --- 6. Thematic Observation Evidence Scoring (Phase 1 Context Quotes) ---
MAX_SENTENCES_PER_TARGET: int = 200
MIN_SENTENCE_LENGTH: int = 15
OBSERVATION_OVERLAP_WEIGHT: float = 0.25
OBSERVATION_OVERLAP_MAX_CAP: int = 4



def configure_synthesis_defaults(**overrides: Any) -> None:
    """Sync module-level defaults and sanitizer/prompt settings from runner overrides."""
    g = globals()
    for k, v in overrides.items():
        upper_k = k.upper()
        if upper_k in g and v is not None:
            g[upper_k] = v
    configure_sanitizer_defaults(
        max_enum_members_per_class=g["MAX_ENUM_MEMBERS_PER_CLASS"],
        max_basemodel_fields_per_class=g["MAX_BASEMODEL_FIELDS_PER_CLASS"],
        outputs_dir=g["OUTPUTS_DIR"],
        print_checkpoints=g["PRINT_CHECKPOINTS"],
    )
    g["SCHEMA_SYNTHESIS_PROMPT"] = get_schema_synthesis_prompt(
        max_basemodel_fields=g["MAX_BASEMODEL_FIELDS_PER_CLASS"],
        max_enum_members=g["MAX_ENUM_MEMBERS_PER_CLASS"],
    )
    g["CONSOLIDATION_SYSTEM_PROMPT"] = get_consolidation_system_prompt(
        max_basemodel_fields=g["MAX_BASEMODEL_FIELDS_PER_CLASS"],
        max_enum_members=g["MAX_ENUM_MEMBERS_PER_CLASS"],
    )
    g["SCHEMA_REFORMAT_SYSTEM_PROMPT"] = get_schema_reformat_system_prompt(
        max_basemodel_fields=g["MAX_BASEMODEL_FIELDS_PER_CLASS"],
        max_enum_members=g["MAX_ENUM_MEMBERS_PER_CLASS"],
    )

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

# STAGE 4 SYNTHESIS PIPELINE (Powered by PrismRunner)
# ============================================================================

class SynthesisPipeline:
    """
    Stage 4 Synthesis Pipeline using PrismRunner for batch local LLM execution.
    Executes single-pass ontological synthesis from raw source sentence text:
      - Phase 1: Community Schema Synthesis (`<id>_<module>.py`)
      - Phase 2: Global Consolidation (`master_ontology.py` & `comprehensive_ontology.py`)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        prism_pipeline: Optional[PrismPipeline] = None,
        **overrides: Any,
    ):
        if overrides:
            configure_synthesis_defaults(**overrides)
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
