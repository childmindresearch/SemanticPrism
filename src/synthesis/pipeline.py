"""
Stage 4 Synthesis Pipeline: Target Resolution, Schema Synthesis, and Consolidation.
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from datetime import datetime
import gc
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
import traceback
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
    detect_and_merge_semantic_duplicate_classes,
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
MODEL_NAME: str = "Qwen/Qwen2.5-Coder-14B-Instruct"    # Causal/Coding LLM for Stage 4 (~29 GB bf16, 128K native context)
DEVICE: str = "auto"                                   # "auto", "cuda", "mps", or "cpu"
TORCH_DTYPE: str = "auto"                              # "auto", "bfloat16", "float16", or "float32"
LOAD_IN_4BIT: bool = False                             # Enable 4-bit NF4 quantization (CUDA bitsandbytes)
LOAD_IN_8BIT: bool = False                             # Enable 8-bit quantization (CUDA bitsandbytes)
ATTN_IMPLEMENTATION: str = "auto"                      # "auto", "flash_attention_2", "sdpa", or "eager"
BATCH_SIZE: Union[int, str] = "auto"                   # "auto" (continuous device-relative) or explicit int
CONTEXT_WINDOW_CAP: Union[int, str] = "auto"           # "auto" = full native model context window (e.g. 262,144 tokens), or int cap
MAX_NEW_TOKENS: int = 4096                             # Max generated tokens per community schema module
CONSOLIDATION_MAX_NEW_TOKENS: int = 8192               # Max generated tokens for consolidated master ontology
TEMPERATURE: float = 0.0                               # 0.0 for deterministic greedy decoding
REPETITION_PENALTY: float = 1.08                       # Mild penalty to prevent greedy repetition loops
CONSTRAIN_JSON: bool = False                           # Keep False for Python code synthesis so '{'/'}'/docstrings inside source_code do not trigger early stopping
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
    def resolve_effective_min_cluster_size(topology: Dict[str, Any], configured_min_size: int = MIN_CLUSTER_SIZE) -> int:
        """
        Adapts min_cluster_size to corpus scale so smaller corpora (e.g., 5-20 documents)
        do not have most or all of their communities filtered out by a static 15-node floor.
        """
        raw_hubs_set = set(topology.get("global_hubs", []))
        comm_sizes = [
            len([n for n in comm.get("nodes", []) if n not in raw_hubs_set])
            for comm in topology.get("communities", [])
        ]
        comm_sizes = [s for s in comm_sizes if s > 0]
        if not comm_sizes:
            return configured_min_size

        total_comm_nodes = sum(comm_sizes)
        if total_comm_nodes >= 500:
            return configured_min_size

        effective_min = max(4, min(configured_min_size, int(total_comm_nodes * 0.03)))
        qualifying = [s for s in comm_sizes if s >= effective_min]
        viable = sorted([s for s in comm_sizes if s >= 4], reverse=True)
        if len(qualifying) < min(4, len(viable)) and viable:
            target_idx = min(5, len(viable)) - 1
            effective_min = min(effective_min, viable[target_idx])

        return max(3, effective_min)

    @staticmethod
    def qualify_and_resolve_targets(
        topology: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Resolves qualified community targets (>= effective min_cluster_size).
        """
        configured_min_size = (
            config.get("synthesis", {}).get("min_cluster_size", MIN_CLUSTER_SIZE)
            if config
            else MIN_CLUSTER_SIZE
        )
        min_cluster_size = TargetResolver.resolve_effective_min_cluster_size(topology, configured_min_size)
        raw_hubs_set = set(topology.get("global_hubs", []))

        targets = []
        for comm in topology.get("communities", []):
            nodes_in_comm = [n for n in comm.get("nodes", []) if n not in raw_hubs_set]
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
        configured_min_size = synth_cfg.get("min_cluster_size", MIN_CLUSTER_SIZE)
        min_cluster_size = TargetResolver.resolve_effective_min_cluster_size(topology, configured_min_size)
        max_triplets_cap = synth_cfg.get("max_triplets_per_target", MAX_TRIPLETS_PER_TARGET)

        # Preserve pre-computed architectural_tiers from Stage 3 if available on disk
        existing_tiers_by_cid: Dict[Any, Any] = {}
        tiers_file = output_dir / "community_architectural_tiers.json"
        if tiers_file.exists():
            try:
                with open(tiers_file, "r", encoding="utf-8") as tf:
                    for item in json.load(tf):
                        if isinstance(item, dict) and "community_id" in item:
                            existing_tiers_by_cid[item["community_id"]] = item
            except Exception:
                pass

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
            if c_id in existing_tiers_by_cid:
                payload_item["architectural_tiers"] = existing_tiers_by_cid[c_id]
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

def _get_gpu_vram_summary() -> str:
    """Returns a concise summary of active accelerator memory (free/total VRAM) for diagnostics."""
    try:
        import torch
        if torch.cuda.is_available():
            dev_idx = torch.cuda.current_device()
            dev_name = torch.cuda.get_device_name(dev_idx)
            free_b, total_b = torch.cuda.mem_get_info(dev_idx)
            return f"CUDA GPU '{dev_name}' ({free_b / (1024**3):.1f} GB free / {total_b / (1024**3):.1f} GB total)"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple Silicon (MPS Unified Memory)"
    except Exception:
        pass
    return "CPU Host Memory"


def _purge_failed_model_vram(prism_obj: Any, exc: Optional[BaseException] = None) -> str:
    """
    Completely releases partially loaded model tensors from GPU/CPU memory after a failed
    `from_pretrained` call by clearing exception traceback frames (`traceback.clear_frames`)
    before invoking garbage collection and CUDA/MPS cache flushing.
    """
    if prism_obj is not None:
        prism_obj.engine = None
        prism_obj.model = None
        prism_obj.tokenizer = None
    if exc is not None and getattr(exc, "__traceback__", None) is not None:
        try:
            traceback.clear_frames(exc.__traceback__)
        except Exception:
            pass
    gc.collect()
    gc.collect()
    clear_device_memory()
    return _get_gpu_vram_summary()


def _validate_no_meta_or_disk_offload(prism_obj: Any, model_label: str) -> None:
    """
    Verifies that the loaded model has no parameters on the PyTorch 'meta' (disk-offloaded)
    device, which would otherwise fail during `model.generate()` with
    `RuntimeError: Tensor.item() cannot be called on meta tensors`.
    """
    dev_str = str(getattr(prism_obj, "device", "") or "")
    if dev_str.startswith("meta"):
        raise RuntimeError(
            f"Model '{model_label}' spilled onto the 'meta' (disk offload) device due to insufficient GPU VRAM."
        )
    model = getattr(prism_obj, "model", None)
    if model is not None:
        hf_map = getattr(model, "hf_device_map", None)
        if isinstance(hf_map, dict):
            offloaded_modules = [k for k, v in hf_map.items() if str(v) in ("disk", "meta")]
            if offloaded_modules:
                raise RuntimeError(
                    f"Model '{model_label}' had {len(offloaded_modules)} module(s) offloaded to disk/meta "
                    f"(e.g. {offloaded_modules[:3]}) due to insufficient GPU VRAM."
                )
        for param in model.parameters():
            if getattr(param, "is_meta", False) or (getattr(param, "device", None) is not None and param.device.type == "meta"):
                raise RuntimeError(
                    f"Model '{model_label}' has parameters on the 'meta' device due to insufficient GPU VRAM."
                )


class SynthesisPipeline:
    """
    Stage 4 Synthesis Pipeline using PrismRunner for batch local LLM execution.
    Executes single-pass ontological synthesis from raw source sentence text:
      - Phase 1: Community Schema Synthesis (`<id>_<module>.py`)
      - Phase 2: Global Consolidation (`master_ontology.py`)
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
        self.configured_model_name: str = self.model_name
        self.fallback_used: bool = False
        self.fallback_reason: Optional[str] = None
        self.fallback_history: List[str] = []
        self.active_quantization: str = (
            "4-bit NF4" if LOAD_IN_4BIT else ("8-bit" if LOAD_IN_8BIT else f"native ({TORCH_DTYPE})")
        )

        # Guard against accidentally passing a SentenceTransformer embedding model as MODEL_NAME
        _lower_llm = self.model_name.lower()
        if any(marker in _lower_llm for marker in ("bge-", "minilm", "e5-", "gte-", "nomic-embed", "sentence-transformers")):
            raise ValueError(
                f"MODEL_NAME is set to '{self.model_name}', which is an encoder-only embedding model, "
                f"not a generative causal LLM.\n"
                f"  -> Set MODEL_NAME to a causal/coding LLM (e.g. 'Qwen/Qwen2.5-Coder-14B-Instruct', 'Qwen/Qwen3.6-27B', "
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

    def _save_model_execution_metadata(self, eff_ctx: int) -> None:
        """Saves Stage 4 model provenance & fallback details to `outputs/04_synthesis/logs/model_execution_metadata.json`."""
        outputs_dir = self.config.get("directories", {}).get("outputs", OUTPUTS_DIR)
        log_dir = Path(outputs_dir) / "04_synthesis" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        meta_path = log_dir / "model_execution_metadata.json"
        meta_payload = {
            "stage": "stage_4_synthesis",
            "configured_model": self.configured_model_name,
            "active_model": self.model_name,
            "quantization_or_dtype": self.active_quantization,
            "device": str(getattr(self.prism, "device", DEVICE)),
            "effective_context_window": eff_ctx,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "fallback_history": self.fallback_history,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(meta_payload, mf, indent=2)
        except Exception:
            pass

        if self.fallback_used:
            err_log = log_dir / "synthesis_community_errors.log"
            try:
                with open(err_log, "a", encoding="utf-8") as lf:
                    lf.write(
                        f"[{datetime.now().isoformat()}] [MODEL FALLBACK NOTE] "
                        f"Configured CODER_MODEL='{self.configured_model_name}' -> "
                        f"Active Model='{self.model_name}' ({self.active_quantization}). "
                        f"Reason: {self.fallback_reason}\n"
                    )
            except Exception:
                pass

    def _build_schema_provenance_header(self) -> str:
        """Builds a concise header comment block documenting the active synthesis model (and any fallback note)."""
        lines = [
            f"# Synthesized by SemanticPrism Stage 4 | Active Model: {self.model_name} ({self.active_quantization})"
        ]
        if self.fallback_used and self.fallback_reason:
            lines.append(
                f"# MODEL FALLBACK NOTE: Configured CODER_MODEL '{self.configured_model_name}' was replaced by "
                f"'{self.model_name}' ({self.active_quantization})."
            )
        return "\n".join(lines) + "\n"

    def _ensure_engine_loaded(self, phase_label: str) -> None:
        """
        Ensures the PrismPipeline engine is loaded onto a valid accelerator device (never 'meta'/disk).
        If the primary CODER_MODEL exceeds available GPU VRAM, explicitly purges traceback frames
        to free 100% of partially allocated GPU tensors before executing the fallback chain.
        """
        if DRY_RUN:
            self._save_model_execution_metadata(16384)
            return
        if self.prism.engine is not None:
            return

        if MANAGE_VRAM:
            clear_device_memory()

        vram_info = _get_gpu_vram_summary()
        self._checkpoint(phase_label, f"Loading LLM '{self.model_name}' [{vram_info}]...")

        def _try_load(candidate_model: str, use_4bit: bool) -> None:
            self.prism.config.setdefault("model", {})["name"] = candidate_model
            self.prism.config["model"]["load_in_4bit"] = use_4bit
            self.prism.init_engine(
                model_name=candidate_model,
                batch_size=BATCH_SIZE,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                repetition_penalty=REPETITION_PENALTY,
            )
            _validate_no_meta_or_disk_offload(self.prism, candidate_model)

        try:
            _try_load(self.model_name, LOAD_IN_4BIT)
        except Exception as load_err:
            primary_err_msg = str(load_err).strip()
            vram_after = _purge_failed_model_vram(self.prism, load_err)
            del load_err
            self.fallback_history.append(
                f"Primary model '{self.configured_model_name}' (4bit={LOAD_IN_4BIT}) failed: {primary_err_msg}"
            )
            print(
                f"\n   [Warning] Primary CODER_MODEL '{self.configured_model_name}' could not fit on device: {primary_err_msg}\n"
                f"   [VRAM Cleanup] Purged partial model tensors from memory -> {vram_after}",
                flush=True,
            )

            loaded = False
            lower_cfg = self.configured_model_name.lower()
            is_moe = "moe" in lower_cfg or "-a3b" in lower_cfg

            # Fallback Tier 1: Try 4-bit NF4 quantization if primary model is a dense transformer
            if not LOAD_IN_4BIT and not is_moe:
                print(
                    f"   [Fallback Tier 1] Retrying '{self.configured_model_name}' with 4-bit quantization (load_in_4bit=True)...",
                    flush=True,
                )
                try:
                    _try_load(self.configured_model_name, use_4bit=True)
                    loaded = True
                    self.model_name = self.configured_model_name
                    self.fallback_used = True
                    self.active_quantization = "4-bit NF4 (auto-fallback)"
                    self.fallback_reason = (
                        f"Configured CODER_MODEL '{self.configured_model_name}' exceeded GPU VRAM in default precision "
                        f"({primary_err_msg}); automatically loaded in 4-bit NF4 quantization."
                    )
                except Exception as q4_err:
                    q4_msg = str(q4_err).strip()
                    vram_after = _purge_failed_model_vram(self.prism, q4_err)
                    del q4_err
                    self.fallback_history.append(
                        f"Tier 1 (4-bit '{self.configured_model_name}') failed: {q4_msg}"
                    )
                    print(
                        f"   [Warning] 4-bit load attempt failed ({q4_msg}). VRAM after cleanup: {vram_after}",
                        flush=True,
                    )

            # Fallback Tier 2: Try MODEL_NAME from model_config.py (proven in Stages 1 & 2) or Qwen2.5-Coder-14B / 7B
            if not loaded:
                fallback_candidates: List[Tuple[str, str]] = []
                try:
                    from model_config import MODEL_NAME as MASTER_STAGE_MODEL
                    if MASTER_STAGE_MODEL and MASTER_STAGE_MODEL != self.configured_model_name:
                        fallback_candidates.append(
                            (MASTER_STAGE_MODEL, "pipeline MODEL_NAME from model_config.py")
                        )
                except Exception:
                    pass
                for coder_fb, desc in (
                    ("Qwen/Qwen2.5-Coder-14B-Instruct", "14B dense Coder fallback (~29 GB VRAM)"),
                    ("Qwen/Qwen2.5-Coder-7B-Instruct", "7B lightweight Coder fallback (~15 GB VRAM)"),
                ):
                    if coder_fb != self.configured_model_name and all(c[0] != coder_fb for c in fallback_candidates):
                        fallback_candidates.append((coder_fb, desc))

                for fb_model, fb_desc in fallback_candidates:
                    print(
                        f"   [Fallback Tier 2] Attempting fallback model '{fb_model}' ({fb_desc})...",
                        flush=True,
                    )
                    try:
                        _try_load(fb_model, use_4bit=LOAD_IN_4BIT)
                        loaded = True
                        self.model_name = fb_model
                        self.fallback_used = True
                        self.active_quantization = (
                            "4-bit NF4" if LOAD_IN_4BIT else ("8-bit" if LOAD_IN_8BIT else f"native ({TORCH_DTYPE})")
                        )
                        self.fallback_reason = (
                            f"Configured CODER_MODEL '{self.configured_model_name}' failed to load ({primary_err_msg}); "
                            f"automatically fell back to '{fb_model}' ({fb_desc})."
                        )
                        break
                    except Exception as fb_err:
                        fb_msg = str(fb_err).strip()
                        vram_after = _purge_failed_model_vram(self.prism, fb_err)
                        del fb_err
                        self.fallback_history.append(f"Fallback '{fb_model}' failed: {fb_msg}")
                        print(
                            f"   [Warning] Fallback model '{fb_model}' failed ({fb_msg}). VRAM after cleanup: {vram_after}",
                            flush=True,
                        )

            if not loaded:
                raise RuntimeError(
                    f"Stage 4 could not load configured CODER_MODEL '{self.configured_model_name}' or any fallback model.\n"
                    f"History:\n  - " + "\n  - ".join(self.fallback_history) + "\n"
                    f"Troubleshooting: Edit CODER_MODEL in 'model_config.py' to a model matching your GPU VRAM "
                    f"(e.g. 'Qwen/Qwen2.5-Coder-14B-Instruct' for 48GB+ or 'Qwen/Qwen2.5-Coder-7B-Instruct' for 16-24GB)."
                )

            print(
                "\n" + "-" * 72 + "\n"
                "   [STAGE 4 MODEL FALLBACK NOTICE]\n"
                f"   • Configured Model : {self.configured_model_name}\n"
                f"   • Active Model Used: {self.model_name} [{self.active_quantization}]\n"
                f"   • Fallback Reason  : {self.fallback_reason}\n"
                "   • Tip              : To avoid fallback retries on future runs, set CODER_MODEL in\n"
                f"                        'model_config.py' directly to '{self.model_name}'.\n"
                + "-" * 72 + "\n",
                flush=True,
            )

        eff_ctx = self._apply_context_window_cap()
        self._save_model_execution_metadata(eff_ctx)
        self._checkpoint(
            phase_label,
            f"Model '{self.model_name}' ({self.active_quantization}) ready on device '{self.prism.device}' "
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
    def _extract_module_from_raw_text(raw_text: str, fallback_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Multi-strategy fallback parser that extracts `(module_name, cleaned_source_code)` directly
        from raw LLM text whenever PrismRunner's JSON parser fails (e.g., due to unescaped triple-quoted
        docstrings, unescaped quotes inside Field descriptions, literal newlines inside JSON strings,
        or direct ```python ... ``` markdown code fences emitted by Coder models).
        """
        if not raw_text or not raw_text.strip():
            return None, None

        # Strip <think>...</think> blocks if any
        text = re.sub(r"<think>[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE)
        if "</think>" in text.lower():
            idx = text.lower().rfind("</think>")
            text = text[idx + len("</think>") :]
        text = text.strip()
        if not text:
            return None, None

        # Extract module_name hint if present anywhere in the JSON-like text
        m_mod = re.search(r'"module_name"\s*:\s*["\']([^"\'\r\n]+)["\']', text)
        extracted_mod = sanitize_module_name(m_mod.group(1), fallback=fallback_name) if m_mod else fallback_name

        # Strategy 1: Repair JSON string control characters & triple-quotes and run json.loads
        stripped_fences = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        stripped_fences = re.sub(r"\s*```$", "", stripped_fences, flags=re.MULTILINE).strip()
        first_brace, last_brace = stripped_fences.find("{"), stripped_fences.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_cand = stripped_fences[first_brace : last_brace + 1]
            json_cand = re.sub(r",\s*([\]}])", r"\1", json_cand)
            if '"""' in json_cand:
                json_cand = re.sub(r'(?<!\\)"""', r'\"\"\"', json_cand)

            # Escape literal newlines/tabs inside JSON string literals
            escaped_chars: List[str] = []
            in_str = False
            esc = False
            for ch in json_cand:
                if esc:
                    escaped_chars.append(ch)
                    esc = False
                    continue
                if ch == "\\" and in_str:
                    escaped_chars.append(ch)
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    escaped_chars.append(ch)
                    continue
                if in_str and ch == "\n":
                    escaped_chars.append("\\n")
                    continue
                if in_str and ch == "\r":
                    escaped_chars.append("\\r")
                    continue
                if in_str and ch == "\t":
                    escaped_chars.append("\\t")
                    continue
                escaped_chars.append(ch)

            for attempt_str in (json_cand, "".join(escaped_chars)):
                try:
                    loaded = json.loads(attempt_str)
                    if isinstance(loaded, dict) and loaded.get("source_code"):
                        mod_val = sanitize_module_name(str(loaded.get("module_name") or extracted_mod), fallback=fallback_name)
                        code_val = clean_python_code(str(loaded["source_code"]), context_label=mod_val)
                        if code_val:
                            return mod_val, code_val
                except Exception:
                    pass

        # Strategy 2: Direct delimiter slicing of "source_code": "..." (immune to unescaped inner quotes)
        m_src = re.search(r'"source_code"\s*:\s*("""|\'\'\'|"|\')', stripped_fences)
        if m_src:
            quote_delim = m_src.group(1)
            start_pos = m_src.end()
            m_after = re.search(r'["\']\s*,\s*"module_name"\s*:', stripped_fences[start_pos:])
            if m_after:
                raw_code_slice = stripped_fences[start_pos : start_pos + m_after.start()]
            else:
                end_brace = stripped_fences.rfind("}")
                tail = (
                    stripped_fences[start_pos:end_brace].rstrip()
                    if end_brace > start_pos
                    else stripped_fences[start_pos:].rstrip()
                )
                if tail.endswith(","):
                    tail = tail[:-1].rstrip()
                if tail.endswith(quote_delim):
                    raw_code_slice = tail[: -len(quote_delim)]
                elif tail.endswith(('"""', "'''", '"', "'")):
                    raw_code_slice = tail[:-1]
                else:
                    raw_code_slice = tail

            code_val = clean_python_code(raw_code_slice, context_label=extracted_mod)
            if code_val and ("class " in code_val or "import " in code_val):
                if not m_mod:
                    bms = _extract_ast_basemodel_names(code_val)
                    if bms:
                        extracted_mod = sanitize_module_name(bms[0], fallback=fallback_name)
                return extracted_mod, code_val

        # Strategy 3: Direct ```python ... ``` code fence or raw Python module extraction
        py_blocks = re.findall(r"```(?:python|py)\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        if not py_blocks:
            py_blocks = [
                blk for blk in re.findall(r"```\s*([\s\S]*?)```", text)
                if "class " in blk and ("BaseModel" in blk or "Enum" in blk)
            ]
        if not py_blocks and "class " in text and "BaseModel" in text and not text.lstrip().startswith("{"):
            py_blocks = [text]

        for blk in py_blocks:
            code_val = clean_python_code(blk, context_label=extracted_mod)
            if code_val and "class " in code_val:
                if not m_mod:
                    bms = _extract_ast_basemodel_names(code_val)
                    if bms:
                        extracted_mod = sanitize_module_name(bms[0], fallback=fallback_name)
                return extracted_mod, code_val

        return None, None

    @classmethod
    def _extract_generated_module(
        cls,
        items: List[Any],
        fallback_name: str,
        raw_text: str = "",
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extracts (module_name, cleaned_source_code) from parsed PrismRunner items or raw LLM output."""
        best_invalid: Tuple[Optional[str], Optional[str]] = (None, None)
        for item in items:
            if isinstance(item, GeneratedModule):
                mod_name = sanitize_module_name(item.module_name, fallback=fallback_name)
                code = clean_python_code(item.source_code, context_label=mod_name or fallback_name)
                if code:
                    is_valid, _ = validate_python_syntax(code)
                    if is_valid:
                        return mod_name, code
                    if best_invalid[1] is None:
                        best_invalid = (mod_name, code)
            elif isinstance(item, dict):
                try:
                    parsed = GeneratedModule.model_validate(item)
                    mod_name = sanitize_module_name(parsed.module_name, fallback=fallback_name)
                    code = clean_python_code(parsed.source_code, context_label=mod_name or fallback_name)
                    if code:
                        is_valid, _ = validate_python_syntax(code)
                        if is_valid:
                            return mod_name, code
                        if best_invalid[1] is None:
                            best_invalid = (mod_name, code)
                except Exception:
                    pass

        if raw_text:
            raw_mod, raw_code = cls._extract_module_from_raw_text(raw_text, fallback_name=fallback_name)
            if raw_code:
                is_valid, _ = validate_python_syntax(raw_code)
                if is_valid or best_invalid[1] is None:
                    return raw_mod, raw_code

        return best_invalid

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
        eff_ctx = self._apply_context_window_cap()
        eff_max_new_tokens = (
            min(max_new_tokens, max(1024, int(eff_ctx * 0.35)))
            if not DRY_RUN
            else max_new_tokens
        )

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
            mod_name, cleaned_code = self._extract_generated_module(
                p_items,
                fallback_name,
                raw_text=raw_out,
            )

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

        try:
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
                max_new_tokens=eff_max_new_tokens,
                temperature=TEMPERATURE,
                repetition_penalty=REPETITION_PENALTY,
                constrain_json=CONSTRAIN_JSON,
                dry_run=DRY_RUN,
                on_batch_start=_on_batch_start,
                on_chunk_processed=_on_chunk_processed,
            )
        except Exception as batch_err:
            self._checkpoint(
                phase_label,
                f"Warning: Batch run raised {type(batch_err).__name__} ({batch_err}); retrying with batch_size=1 and reduced token cap...",
            )
            if MANAGE_VRAM:
                clear_device_memory()
            eff_max_new_tokens = min(2048, eff_max_new_tokens)
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
                batch_size=1,
                max_new_tokens=eff_max_new_tokens,
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
                max_new_tokens=eff_max_new_tokens,
                temperature=TEMPERATURE,
                repetition_penalty=REPETITION_PENALTY,
                constrain_json=CONSTRAIN_JSON,
                dry_run=DRY_RUN,
            )

            for (idx, item, _, _), rep_res in zip(repair_candidates, repair_results):
                fallback_name = item.get("fallback_module_name", default_module_name)
                rep_items = rep_res.get("extracted_items") or rep_res.get("parsed_items") or []
                rep_raw_out = (rep_res.get("raw_outputs") or [""])[0] if rep_res.get("raw_outputs") else str(rep_res.get("raw_output") or "")
                mod_name, cleaned_code = self._extract_generated_module(
                    rep_items,
                    fallback_name,
                    raw_text=rep_raw_out,
                )
                if cleaned_code:
                    is_valid, syntax_err = validate_python_syntax(cleaned_code)
                    if is_valid:
                        parsed_outputs[idx] = {
                            "id": item["id"],
                            "success": True,
                            "module_name": mod_name or fallback_name,
                            "source_code": cleaned_code,
                            "error": None,
                            "raw_text": rep_raw_out,
                        }
                        print(f"            -> Reformat successful for {item['id']}!")
                        if on_valid_module_ready is not None:
                            on_valid_module_ready(item, mod_name or fallback_name, cleaned_code)
                    else:
                        parsed_outputs[idx]["error"] = syntax_err or str(rep_res.get("warning") or parsed_outputs[idx]["error"])
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

        # Load upstream Stage 2 taxonomic maps (surface hyponym -> lifted hypernym) if available
        refinement_dir = Path(self.outputs_dir) / "02_refinement"
        global_hypernym_to_hyponyms: Dict[str, List[str]] = defaultdict(list)
        for tax_filename in ("taxonomic_map.json", "subject_taxonomic_map.json", "object_taxonomic_map.json"):
            tax_path = refinement_dir / tax_filename
            if tax_path.exists():
                try:
                    raw_tax = json.loads(tax_path.read_text(encoding="utf-8"))
                    if isinstance(raw_tax, dict):
                        for surf_term, lifted_term in raw_tax.items():
                            s_clean = str(surf_term).strip()
                            l_clean = str(lifted_term).strip()
                            if not s_clean or not l_clean:
                                continue
                            if s_clean.lower() == l_clean.lower():
                                continue
                            if len(s_clean.split()) <= 4 and len(s_clean) <= 35:
                                l_key = l_clean.lower()
                                if s_clean.lower() not in {x.lower() for x in global_hypernym_to_hyponyms[l_key]}:
                                    global_hypernym_to_hyponyms[l_key].append(s_clean)
                except Exception:
                    pass

        def _is_concise_categorical_value(val: str) -> bool:
            v = str(val).strip()
            if not v or len(v) <= 1 or len(v) > 35:
                return False
            v_lower = v.lower()
            if v_lower in closed_class_pronouns or v_lower.isdigit():
                return False
            if not any(ch.isalpha() for ch in v):
                return False
            words = [w for w in re.split(r"[\s_-]+", v) if w]
            return 1 <= len(words) <= 4

        def extract_candidate_categorical_enums(
            target_nodes: List[str],
            triplets: List[dict],
            max_categories: int = 10,
            max_values_per_category: int = 12,
        ) -> Dict[str, List[str]]:
            """
            Mines optional, corpus-derived categorical value sets from:
              1. Stage 2 taxonomic hyponym -> hypernym groupings observed in the community.
              2. Recurring (subject, predicate) -> multiple concise object groupings in community triplets.
            Returned candidates serve as reference context alongside thematic_observations.
            """
            target_nodes_norm = {str(n).lower().strip(): str(n).strip() for n in target_nodes if str(n).strip()}
            quote_blob_lower = " ".join(str(tr.get("source_quote", "")).lower() for tr in triplets)

            # 1. Collect surface hyponyms per canonical hypernym in this community
            hypernym_surface_counts: Dict[str, Counter] = defaultdict(Counter)
            for tr in triplets:
                for canon_key, surf_key in (("subject", "surface_subject"), ("object", "surface_object")):
                    canon_val = str(tr.get(canon_key, "")).strip().lower()
                    surf_val = str(tr.get(surf_key) or "").strip()
                    if canon_val in target_nodes_norm and _is_concise_categorical_value(surf_val):
                        if surf_val.lower() != canon_val:
                            hypernym_surface_counts[canon_val][surf_val] += 2

            for nk in target_nodes_norm:
                for hyponym in global_hypernym_to_hyponyms.get(nk, []):
                    if _is_concise_categorical_value(hyponym) and hyponym.lower() != nk:
                        # Prefer hyponyms that actually appear in this community's quotes
                        weight = 2 if hyponym.lower() in quote_blob_lower else 1
                        hypernym_surface_counts[nk][hyponym] += weight

            # 2. Collect multi-valued (subject, predicate) -> concise objects within community triplets
            sp_object_counts: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
            for tr in triplets:
                s_raw = str(tr.get("subject", "")).strip()
                p_raw = str(tr.get("predicate", "")).strip()
                o_raw = str(tr.get("object", "")).strip()
                s_low = s_raw.lower()
                if s_low in target_nodes_norm and _is_concise_categorical_value(o_raw) and o_raw.lower() != s_low:
                    if 1 <= len(p_raw.split()) <= 3:
                        sp_object_counts[(target_nodes_norm[s_low], p_raw)][o_raw] += 1

            scored_candidates: List[Tuple[float, str, List[str]]] = []

            for nk, surf_counter in hypernym_surface_counts.items():
                # Deduplicate case-insensitively while ordering by frequency
                seen_lower: set = set()
                distinct_vals: List[str] = []
                for val, _ in surf_counter.most_common():
                    vl = val.lower().strip()
                    if vl not in seen_lower:
                        seen_lower.add(vl)
                        distinct_vals.append(val)
                    if len(distinct_vals) >= max_values_per_category:
                        break
                if 2 <= len(distinct_vals) <= max_values_per_category:
                    in_quote_hits = sum(1 for v in distinct_vals if v.lower() in quote_blob_lower)
                    score = float(sum(surf_counter.values())) + 1.5 * in_quote_hits
                    scored_candidates.append((score, target_nodes_norm[nk], distinct_vals))

            for (subj_label, pred_label), obj_counter in sp_object_counts.items():
                seen_lower = set()
                distinct_objs: List[str] = []
                for val, _ in obj_counter.most_common():
                    vl = val.lower().strip()
                    if vl not in seen_lower:
                        seen_lower.add(vl)
                        distinct_objs.append(val)
                    if len(distinct_objs) >= max_values_per_category:
                        break
                if 3 <= len(distinct_objs) <= max_values_per_category:
                    cat_key = f"{subj_label} [{pred_label}]"
                    score = float(sum(obj_counter.values())) + len(distinct_objs)
                    scored_candidates.append((score, cat_key, distinct_objs))

            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            candidate_enums: Dict[str, List[str]] = {}
            for _, cat_label, vals in scored_candidates[:max_categories]:
                if cat_label not in candidate_enums:
                    candidate_enums[cat_label] = vals
            return candidate_enums

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
            candidate_enums = extract_candidate_categorical_enums(target_nodes, capped_subset)
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
            }
            if candidate_enums:
                payload_dict["candidate_categorical_enums"] = candidate_enums
            payload_dict["thematic_observations"] = observations
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
                header = self._build_schema_provenance_header()
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(f"{header}\n{code.lstrip()}")
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
        # PHASE 2: Global Consolidation (AST-First + Targeted Semantic Merge)
        # =====================================================================
        print(f"   -> [{path_label}] Phase 2: Global Consolidation (AST-First + Targeted Semantic Merge)...")

        def _targeted_llm_cluster_merge(cluster_label: str, snippet_code: str, primary_name: str) -> Optional[str]:
            """
            Targeted LLM merge callback invoked ONLY when 2+ cross-community BaseModel classes
            exhibit high lexical + field overlap. Leaves all non-overlapping classes untouched.
            """
            if DRY_RUN:
                return None
            self._ensure_engine_loaded("Phase 2: Targeted Semantic Merge")
            synthesis_cap = self._apply_context_window_cap()
            eff_merge_tokens = min(
                MAX_NEW_TOKENS,
                max(1024, int(synthesis_cap * 0.25)),
            )
            merge_prompt = (
                f"Consolidate the following overlapping Pydantic BaseModel classes into a SINGLE unified "
                f"BaseModel class named `{primary_name}` (preserving any referenced Enum classes).\n\n"
                f"--- Overlapping Classes Snippet ---\n{snippet_code}\n"
            )
            save_payload_file(
                f"phase2_{cluster_label}_payload.txt",
                f"=== SYSTEM PROMPT ===\n{CONSOLIDATION_SYSTEM_PROMPT}\n\n=== USER PAYLOAD ===\n{merge_prompt}",
            )
            res_list = self._run_module_batch(
                items=[{"id": cluster_label, "payload": merge_prompt, "fallback_module_name": "master_ontology"}],
                system_prompt=CONSOLIDATION_SYSTEM_PROMPT,
                phase_label=f"Phase 2: Targeted Merge ({primary_name})",
                default_module_name="master_ontology",
                allow_reformat_repair=True,
                max_new_tokens=eff_merge_tokens,
            )
            if res_list and res_list[0]["success"] and res_list[0]["source_code"]:
                return res_list[0]["source_code"]
            return None

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

            header = self._build_schema_provenance_header()

            # Remove any legacy consolidated files inside schemas/community/ to keep community/ strictly for individual modules
            for legacy_name in ("master_ontology.py", "comprehensive_ontology.py"):
                legacy_file = schema_dir / legacy_name
                if legacy_file.exists():
                    try:
                        legacy_file.unlink()
                    except Exception:
                        pass

            # Step 2A: Lossless Deterministic AST Union across all community modules
            combined_raw = "\n\n".join(f["content"] for f in files_list)
            ast_union_code = deduplicate_consolidated_classes(
                combined_raw,
                context_label="master_ontology",
            )

            # Step 2B: Targeted Cross-Community Semantic Duplicate Merge -> schemas/master_ontology.py
            try:
                master_code, merge_notes = detect_and_merge_semantic_duplicate_classes(
                    comprehensive_code=ast_union_code,
                    llm_merge_fn=_targeted_llm_cluster_merge,
                    context_label="master_ontology",
                )
                if merge_notes:
                    for note in merge_notes:
                        self._checkpoint("Phase 2: Consolidation", note)
                else:
                    self._checkpoint(
                        "Phase 2: Consolidation",
                        f"All {len(py_files)} community schema module(s) are semantically distinct; preserving 100% in master_ontology.py",
                    )
            except Exception as e:
                log_error(
                    f"[Synthesis {path_type}] Targeted semantic duplicate resolution encountered error ({e}); "
                    "using lossless AST union for master_ontology.py."
                )
                master_code = ast_union_code

            try:
                schemas_root_dir = schema_dir.parent
                schemas_root_dir.mkdir(parents=True, exist_ok=True)
                out_master = schemas_root_dir / "master_ontology.py"
                with open(out_master, "w", encoding="utf-8") as f:
                    f.write(f"{header}\n{master_code.lstrip()}")
                self._checkpoint("Phase 2: Consolidation", "Saved master_ontology.py", out_master)
            except Exception as e:
                log_error(f"[Synthesis {path_type}] Error saving master_ontology.py: {e}")

        consolidate_schemas(output_base_dir)

        # Finalization (__init__.py & optional ruff format if ruff is installed)
        with open(output_base_dir / "__init__.py", "w", encoding="utf-8") as f:
            f.write(self._build_schema_provenance_header())
        ruff_bin = shutil.which("ruff")
        if ruff_bin:
            try:
                subprocess.run([ruff_bin, "format", str(output_base_dir)], capture_output=True, text=True, check=False)
            except Exception as e:
                log_error(f"[Synthesis {path_type}] Error running ruff on {output_base_dir}: {e}")

        print(f"   [OK] [{path_label}] Synthesis complete: {output_base_dir}")


# ============================================================================
