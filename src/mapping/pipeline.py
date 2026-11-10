"""
Stage 5 Pipeline Orchestrator: Corpus Loading, Chunking, Resume Tracking,
Verbatim Chunk Consolidation, and Two-Pass Hardware-Batched Extraction via PrismRunner.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from enum import Enum
import gc
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

# Prevent PyTorch CUDA allocator fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
from pydantic import BaseModel

# Ensure repository root and sibling PrismRunner are importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_SIBLING_PRISMRUNNER = REPO_ROOT.parent / "PrismRunner"
if _SIBLING_PRISMRUNNER.exists() and str(_SIBLING_PRISMRUNNER) not in sys.path:
    sys.path.insert(0, str(_SIBLING_PRISMRUNNER))

from prismrunner import (
    ContextBudget,
    DatasetLoader,
    PrismPipeline,
    clear_device_memory,
)

from src.mapping.aggregator import DataAggregator
from src.mapping.prompts import (
    build_extraction_prompt,
    build_triage_prompt,
    build_unified_extraction_prompt,
)
from src.mapping.schemas import (
    SchemaMeta,
    TriageResult,
    build_unified_envelope_model,
    get_child_schemas,
    get_top_level_schemas,
    load_schemas,
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
# LOGGING & CHECKPOINT UTILITIES
# ============================================================================

logger = logging.getLogger("SemanticPrism5")


def configure_logging(outputs_dir: Union[str, Path] = "outputs") -> Path:
    """Configures file and console logging under outputs/logs/."""
    out_base = Path(outputs_dir)
    log_dir = out_base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "stage_05_extraction.log"

    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return log_file


def log_checkpoint(phase: str, message: str, print_checkpoints: bool = True) -> None:
    """Prints a formatted timestamped checkpoint message to stdout and logs it."""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts} | Checkpoint | {phase}] {message}"
    if print_checkpoints:
        print(line, flush=True)
    logger.info(f"[{phase}] {message}")


# ============================================================================
# PHASE 2: CORPUS LOADING & WORD-WINDOW CHUNKING
# ============================================================================

def load_input_documents(
    input_source: Union[str, Path, pd.DataFrame, List[Dict[str, Any]]],
    id_column: str = "unique_val",
    text_column: str = "report",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Loads input documents from a `.parquet`/`.pq`, `.csv`, `.json`/`.jsonl`,
    pandas DataFrame, list of dicts, or a directory of `.txt` files.
    Returns a standardized list of `{"id": str, "text": str}` dicts.
    """
    documents: List[Dict[str, Any]] = []

    if isinstance(input_source, pd.DataFrame):
        df = input_source
        if limit is not None and limit > 0:
            df = df.head(limit)
        for idx, row in df.iterrows():
            doc_id = str(row[id_column]) if id_column in df.columns and pd.notna(row[id_column]) else f"doc_{idx}"
            text_val = str(row[text_column]) if text_column in df.columns and pd.notna(row[text_column]) else ""
            if text_val.strip():
                documents.append({"id": doc_id, "text": text_val})
        return documents

    if isinstance(input_source, list):
        items = input_source[:limit] if (limit is not None and limit > 0) else input_source
        for idx, item in enumerate(items):
            if isinstance(item, dict):
                doc_id = str(item.get(id_column, f"doc_{idx}"))
                text_val = str(item.get(text_column, ""))
            else:
                doc_id = f"doc_{idx}"
                text_val = str(item)
            if text_val.strip():
                documents.append({"id": doc_id, "text": text_val})
        return documents

    path = Path(input_source)
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_dir():
        txt_files = sorted(path.glob("*.txt"))
        if limit is not None and limit > 0:
            txt_files = txt_files[:limit]
        for txt_file in txt_files:
            text_val = txt_file.read_text(encoding="utf-8", errors="replace").strip()
            if text_val:
                documents.append({"id": txt_file.stem, "text": text_val})
        return documents

    # Delegate standard file formats (.parquet, .pq, .csv, .json, .jsonl, .txt) to PrismRunner's DatasetLoader
    records = DatasetLoader.load(
        file_path_or_data=path,
        text_column=text_column,
        id_column=id_column,
        limit=limit,
    )
    for idx, rec in enumerate(records):
        if isinstance(rec, dict):
            doc_id = str(rec.get("id", f"doc_{idx}"))
            text_val = str(rec.get("text", ""))
        else:
            doc_id = str(getattr(rec, "id", f"doc_{idx}"))
            text_val = str(getattr(rec, "text", ""))
        if text_val.strip():
            documents.append({"id": doc_id, "text": text_val})

    return documents


def chunk_text(
    text: str,
    chunk_size_words: int = 1500,
    chunk_overlap_words: int = 100,
) -> List[str]:
    """
    Adaptive document chunker:
      - Whole-Document Fast-Path: Any document with `<= chunk_size_words` words (all small and
        medium documents) is returned intact as `[text.strip()]`, preserving all original
        newlines, paragraphs, bullet lists, and section formatting.
      - Sentence-Boundary Windowing + Tail Absorption: For very long documents exceeding
        `chunk_size_words`, splits on sentence boundaries within a bounded lookahead window
        and absorbs tiny trailing remainders so no tiny orphan tail chunks are created.
    """
    if not text or not text.strip():
        return []

    stripped = text.strip()
    words = stripped.split()
    if len(words) <= chunk_size_words:
        return [stripped]

    sentence_endings = (".", "!", "?", '."', '!"', '?"', ".'", "!'", "?'", ".)", "!)")
    max_lookahead = min(80, max(10, chunk_size_words // 10))
    tail_absorption_words = max(chunk_overlap_words, chunk_size_words // 10)

    chunks: List[str] = []
    start = 0
    total_words = len(words)

    while start < total_words:
        base_end = min(start + chunk_size_words, total_words)
        end = base_end

        if end < total_words:
            # If the remaining tail after base_end is small, absorb it into this chunk
            if (total_words - base_end) <= tail_absorption_words:
                end = total_words
            else:
                lookahead_limit = min(total_words, base_end + max_lookahead)
                while end < lookahead_limit and not words[end - 1].endswith(sentence_endings):
                    end += 1
                if end == lookahead_limit and not words[end - 1].endswith(sentence_endings):
                    end = base_end
                if (total_words - end) <= tail_absorption_words:
                    end = total_words

        chunks.append(" ".join(words[start:end]))
        if end >= total_words:
            break
        start = max(start + 1, end - chunk_overlap_words)

    return chunks


# ============================================================================
# PHASE 3B: RESUME TRACKER & VERBATIM CHUNK CONSOLIDATION
# ============================================================================

class SafeJSONEncoder(json.JSONEncoder):
    """JSON encoder that safely serializes Enums, Pydantic models, sets, and paths."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, BaseModel):
            return obj.model_dump(mode="json")
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def sanitize_filename(name: str) -> str:
    """Sanitizes a string to be safe for filesystem paths and filenames."""
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", str(name))


class ResumeTracker:
    """
    Tracks extraction progress at the document, schema, and individual chunk level
    under `outputs/05_extraction/extracts/<sanitized_doc_id>/`.
    """

    def __init__(self, extracts_dir: Path, mode: str = "skip"):
        self.extracts_dir = extracts_dir
        self.mode = mode
        self.extracts_dir.mkdir(parents=True, exist_ok=True)

    def get_doc_dir(self, doc_id: str) -> Path:
        return self.extracts_dir / sanitize_filename(doc_id)

    def get_manifest_path(self, doc_id: str) -> Path:
        return self.get_doc_dir(doc_id) / ".completed.json"

    def get_output_file(self, doc_id: str, schema_meta: SchemaMeta) -> Path:
        safe_doc_id = sanitize_filename(doc_id)
        filename = f"{safe_doc_id}__{schema_meta.module_name}__{schema_meta.name}.json"
        return self.get_doc_dir(doc_id) / filename

    def get_chunk_cache_dir(self, doc_id: str, schema_meta: SchemaMeta) -> Path:
        return self.get_doc_dir(doc_id) / "chunks" / sanitize_filename(schema_meta.qualified_name)

    def get_chunk_file(self, doc_id: str, schema_meta: SchemaMeta, chunk_idx: int) -> Path:
        safe_doc_id = sanitize_filename(doc_id)
        return self.get_chunk_cache_dir(doc_id, schema_meta) / f"{safe_doc_id}_chunk_{chunk_idx}.json"

    def is_chunk_completed(self, doc_id: str, schema_meta: SchemaMeta, chunk_idx: int) -> bool:
        if self.mode == "overwrite":
            return False
        return self.get_chunk_file(doc_id, schema_meta, chunk_idx).exists()

    def load_chunk_result(self, doc_id: str, schema_meta: SchemaMeta, chunk_idx: int) -> Optional[BaseModel]:
        chunk_file = self.get_chunk_file(doc_id, schema_meta, chunk_idx)
        if not chunk_file.exists():
            return None
        try:
            data = json.loads(chunk_file.read_text(encoding="utf-8"))
            if not data:
                return None
            return schema_meta.schema_cls.model_validate(data)
        except Exception as e:
            logger.warning(f"Corrupted chunk cache {chunk_file}, will re-extract: {e}")
            return None

    def save_chunk_result(
        self,
        doc_id: str,
        schema_meta: SchemaMeta,
        chunk_idx: int,
        instance: Optional[BaseModel],
    ) -> Path:
        chunk_file = self.get_chunk_file(doc_id, schema_meta, chunk_idx)
        chunk_file.parent.mkdir(parents=True, exist_ok=True)
        if instance is None or not has_extracted_content(instance):
            payload_str = json.dumps(None)
        else:
            payload_str = json.dumps(instance.model_dump(mode="json"), indent=2, cls=SafeJSONEncoder)
        chunk_file.write_text(payload_str, encoding="utf-8")
        return chunk_file

    def get_triage_chunk_file(self, doc_id: str, chunk_idx: int) -> Path:
        safe_doc_id = sanitize_filename(doc_id)
        return self.get_doc_dir(doc_id) / "chunks" / "_triage" / f"{safe_doc_id}_triage_chunk_{chunk_idx}.json"

    def is_triage_chunk_completed(self, doc_id: str, chunk_idx: int) -> bool:
        if self.mode == "overwrite":
            return False
        return self.get_triage_chunk_file(doc_id, chunk_idx).exists()

    def load_triage_chunk_result(self, doc_id: str, chunk_idx: int) -> Optional[List[str]]:
        triage_file = self.get_triage_chunk_file(doc_id, chunk_idx)
        if not triage_file.exists():
            return None
        try:
            data = json.loads(triage_file.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "relevant_schemas" in data and isinstance(data["relevant_schemas"], list):
                return [str(x).strip() for x in data["relevant_schemas"] if str(x).strip()]
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
            return []
        except Exception:
            return None

    def save_triage_chunk_result(
        self,
        doc_id: str,
        chunk_idx: int,
        relevant_schemas: List[str],
    ) -> Path:
        triage_file = self.get_triage_chunk_file(doc_id, chunk_idx)
        triage_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"doc_id": str(doc_id), "chunk_idx": int(chunk_idx), "relevant_schemas": list(relevant_schemas)}
        triage_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return triage_file

    def get_completed_schemas(self, doc_id: str) -> Set[str]:
        if self.mode == "overwrite":
            return set()
        manifest = self.get_manifest_path(doc_id)
        if not manifest.exists():
            return set()
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            return set(data.get("completed_schemas", []))
        except Exception:
            return set()

    def is_schema_completed(self, doc_id: str, schema_meta: SchemaMeta) -> bool:
        if self.mode == "overwrite":
            return False
        completed = self.get_completed_schemas(doc_id)
        out_file = self.get_output_file(doc_id, schema_meta)
        return schema_meta.qualified_name in completed and out_file.exists()

    def mark_schema_completed(self, doc_id: str, schema_meta: SchemaMeta) -> None:
        doc_dir = self.get_doc_dir(doc_id)
        doc_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.get_manifest_path(doc_id)

        completed = self.get_completed_schemas(doc_id)
        completed.add(schema_meta.qualified_name)

        data = {
            "doc_id": str(doc_id),
            "completed_schemas": sorted(list(completed)),
        }
        manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def save_extraction(
        self,
        doc_id: str,
        schema_meta: SchemaMeta,
        extracted_data: Any,
    ) -> Path:
        out_file = self.get_output_file(doc_id, schema_meta)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(
            json.dumps(extracted_data, indent=2, cls=SafeJSONEncoder),
            encoding="utf-8",
        )
        self.mark_schema_completed(doc_id, schema_meta)
        return out_file


def _is_empty_value(val: Any) -> bool:
    """Checks if an extracted field value is null or empty."""
    if val is None:
        return True
    if isinstance(val, (str, list, dict, set, tuple)) and len(val) == 0:
        return True
    if isinstance(val, BaseModel):
        return not has_extracted_content(val)
    return False


def has_extracted_content(instance: BaseModel) -> bool:
    """
    Returns True if the Pydantic model instance has at least one non-null, non-empty field.
    """
    for field_name in instance.model_fields:
        val = getattr(instance, field_name, None)
        if isinstance(val, list):
            non_empty_items = [item for item in val if not _is_empty_value(item)]
            if non_empty_items:
                return True
        elif not _is_empty_value(val):
            return True
    return False


def _canonical_key(item: Any) -> str:
    """Generates a deterministic canonical JSON string key for deduplicating list items."""
    if isinstance(item, BaseModel):
        return json.dumps(item.model_dump(mode="json"), sort_keys=True, cls=SafeJSONEncoder)
    if isinstance(item, (dict, list)):
        return json.dumps(item, sort_keys=True, cls=SafeJSONEncoder)
    if isinstance(item, Enum):
        return str(item.value)
    return str(item)


def merge_pydantic_instances(
    instances: List[BaseModel],
    schema_cls: Type[BaseModel],
) -> Optional[BaseModel]:
    """
    Consolidates multiple chunk-level Pydantic model instances into a single document-level
    instance WITHOUT mathematical alteration, averaging, or string modification:
      - `list` fields: concatenates and deduplicates items while preserving exact extraction order.
      - `dict` fields: merges keys in order without mutating values.
      - Nested `BaseModel` fields: recursively consolidates child fields.
      - Scalar fields (`str`, `int`, `float`, `bool`, `Enum`): retains the first non-null
        extracted value verbatim.
    """
    valid_instances = [inst for inst in instances if inst is not None and has_extracted_content(inst)]
    if not valid_instances:
        return None

    if len(valid_instances) == 1:
        return valid_instances[0]

    merged_data: Dict[str, Any] = {}

    for field_name, field_info in schema_cls.model_fields.items():
        field_values = [
            getattr(inst, field_name, None)
            for inst in valid_instances
            if getattr(inst, field_name, None) is not None
        ]

        if not field_values:
            merged_data[field_name] = None
            continue

        first_val = field_values[0]

        # Case 1: List fields -> concatenate & deduplicate preserving order
        if isinstance(first_val, list):
            combined_list: List[Any] = []
            seen_keys: Set[str] = set()
            for val_list in field_values:
                if not isinstance(val_list, list):
                    continue
                for item in val_list:
                    if _is_empty_value(item):
                        continue
                    key = _canonical_key(item)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        combined_list.append(item)
            merged_data[field_name] = combined_list

        # Case 2: Dict fields -> merge keys in order
        elif isinstance(first_val, dict):
            combined_dict: Dict[Any, Any] = {}
            for val_dict in field_values:
                if isinstance(val_dict, dict):
                    for k, v in val_dict.items():
                        if k not in combined_dict and not _is_empty_value(v):
                            combined_dict[k] = v
            merged_data[field_name] = combined_dict

        # Case 3: Nested BaseModel -> recursively merge
        elif isinstance(first_val, BaseModel):
            child_cls = type(first_val)
            nested_models = [v for v in field_values if isinstance(v, BaseModel)]
            merged_data[field_name] = merge_pydantic_instances(nested_models, child_cls)

        # Case 4: Scalar (str, int, float, bool, Enum) -> retain first non-null value verbatim
        else:
            chosen_scalar = None
            for v in field_values:
                if not _is_empty_value(v):
                    chosen_scalar = v
                    break
            merged_data[field_name] = chosen_scalar

    try:
        return schema_cls.model_validate(merged_data)
    except Exception:
        return schema_cls(**merged_data)


# ============================================================================
# MAIN STAGE 5 PIPELINE ORCHESTRATOR (POWERED BY PRISMRUNNER)
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


class SemanticPrismStage5Pipeline:
    """
    End-to-end Stage 5 Schema-Driven Extraction and Relational/Wide-Table Aggregation
    pipeline powered by PrismRunner.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
        device: str = "auto",
        torch_dtype: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        attn_implementation: str = "auto",
        batch_size: Union[int, str] = "auto",
        temperature: float = 0.0,
        repetition_penalty: float = 1.05,
        constrain_json: bool = True,
        context_window_cap: Union[int, str] = "auto",
        max_new_tokens: int = 2048,
        triage_max_new_tokens: int = 256,
        chunk_size_words: int = 1500,
        chunk_overlap_words: int = 100,
        enable_triage: bool = True,
        triage_threshold: int = 6,
        max_schemas_per_envelope: int = 6,
        fallback_on_empty_triage: bool = False,
        schema_path: Union[str, Path] = "outputs/04_synthesis/schemas/master_ontology.py",
        target_class_name: Optional[str] = None,
        extract_mode: str = "unified",
        outputs_dir: Union[str, Path] = "outputs",
        save_llm_payloads: bool = True,
        resume_mode: str = "skip",
        manage_vram: bool = True,
        print_checkpoints: bool = True,
        generate_csv: bool = True,
        generate_sqlite: bool = True,
        generate_parquet: bool = True,
        db_filename: str = "aggregated_data.db",
        multi_table_relational: bool = True,
        consolidated_wide_table: bool = True,
        wide_table_name: str = "consolidated_wide",
        include_source_text: bool = True,
        drop_empty_records: bool = True,
        drop_empty_tables: bool = True,
    ):
        self.model_name = model_name
        self.configured_model_name: str = model_name
        self.fallback_used: bool = False
        self.fallback_reason: Optional[str] = None
        self.fallback_history: List[str] = []
        self.active_quantization: str = (
            "4-bit NF4" if load_in_4bit else ("8-bit" if load_in_8bit else f"native ({torch_dtype})")
        )
        self.device = device
        self.torch_dtype = torch_dtype
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.attn_implementation = attn_implementation
        self.batch_size = batch_size
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.constrain_json = constrain_json
        self.context_window_cap = context_window_cap
        self.max_new_tokens = max_new_tokens
        self.triage_max_new_tokens = triage_max_new_tokens
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words
        self.enable_triage = enable_triage
        self.triage_threshold = triage_threshold
        self.max_schemas_per_envelope = max(1, int(max_schemas_per_envelope))
        self.fallback_on_empty_triage = fallback_on_empty_triage
        self.schema_path = Path(schema_path)
        self.target_class_name = target_class_name
        self.extract_mode = extract_mode
        self.outputs_dir = Path(outputs_dir)
        self.save_llm_payloads = save_llm_payloads
        self.resume_mode = resume_mode
        self.manage_vram = manage_vram
        self.print_checkpoints = print_checkpoints

        # Aggregation settings
        self.generate_csv = generate_csv
        self.generate_sqlite = generate_sqlite
        self.generate_parquet = generate_parquet
        self.db_filename = db_filename
        self.multi_table_relational = multi_table_relational
        self.consolidated_wide_table = consolidated_wide_table
        self.wide_table_name = wide_table_name
        self.include_source_text = include_source_text
        self.drop_empty_records = drop_empty_records
        self.drop_empty_tables = drop_empty_tables

        # Standard Stage 5 directory layout under outputs/05_extraction
        self.stage5_dir = self.outputs_dir / "05_extraction"
        self.extracts_dir = self.stage5_dir / "extracts"
        self.tables_dir = self.stage5_dir / "aggregated_tables"
        self.payload_dir = self.outputs_dir / "payload_text" / "stage_05_extraction"

        self.extracts_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        if self.save_llm_payloads:
            self.payload_dir.mkdir(parents=True, exist_ok=True)

        configure_logging(self.outputs_dir)
        self.tracker = ResumeTracker(self.extracts_dir, mode=self.resume_mode)

        # Initialize PrismRunner pipeline
        self.prism = PrismPipeline(
            model_name=self.model_name,
            device=self.device,
            batch_size=self.batch_size,
            config={
                "model": {
                    "name": self.model_name,
                    "device": self.device,
                    "torch_dtype": self.torch_dtype,
                    "load_in_4bit": self.load_in_4bit,
                    "load_in_8bit": self.load_in_8bit,
                    "attn_implementation": self.attn_implementation,
                },
                "inference": {
                    "batch_size": self.batch_size,
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "repetition_penalty": self.repetition_penalty,
                    "constrain_json": self.constrain_json,
                },
            },
        )
        if isinstance(self.context_window_cap, int) and self.context_window_cap > 0:
            self.prism.config.setdefault("model", {})["context_window_cap"] = self.context_window_cap

        self._saved_payload_keys: Set[str] = set()

    def _log(self, phase: str, message: str) -> None:
        log_checkpoint(phase, message, print_checkpoints=self.print_checkpoints)

    def _maybe_save_payload(self, key: str, system_prompt: str, user_prompt: str) -> None:
        """Saves the first rendered prompt per schema/triage pass to `outputs/payload_text/stage_05_extraction/`."""
        if not self.save_llm_payloads or key in self._saved_payload_keys:
            return
        self._saved_payload_keys.add(key)
        safe_key = sanitize_filename(key)
        out_path = self.payload_dir / f"{safe_key}.txt"
        content = (
            f"=== SYSTEM PROMPT ===\n{system_prompt}\n\n"
            f"=== USER PROMPT ===\n{user_prompt}\n"
        )
        out_path.write_text(content, encoding="utf-8")

    def resolve_target_schemas(self) -> List[SchemaMeta]:
        """
        Loads schemas from `self.schema_path` (single `.py` file or directory of `.py` files)
        and filters them according to `self.target_class_name` and `self.extract_mode`.
        """
        all_schemas = load_schemas(
            self.schema_path,
            target_class=self.target_class_name,
            outputs_dir=self.outputs_dir,
        )
        top_level = get_top_level_schemas(all_schemas)
        children = get_child_schemas(all_schemas)

        self._log(
            "Phase 1: Schema Discovery",
            f"Loaded {len(all_schemas)} total BaseModel(s) from '{self.schema_path}' "
            f"({len(top_level)} top-level, {len(children)} child sub-models).",
        )

        if self.target_class_name:
            target_schemas = all_schemas
        elif self.extract_mode in ("unified", "top_level"):
            target_schemas = top_level if top_level else all_schemas
        elif self.extract_mode == "child":
            target_schemas = children
        else:
            target_schemas = all_schemas

        if not target_schemas:
            raise ValueError(
                f"No target schemas selected from '{self.schema_path}' with extract_mode='{self.extract_mode}'."
            )

        schema_names_preview = ", ".join(s.qualified_name for s in target_schemas[:10])
        if len(target_schemas) > 10:
            schema_names_preview += f", ... (+{len(target_schemas) - 10} more)"
        self._log(
            "Phase 1: Schema Discovery",
            f"Selected {len(target_schemas)} target schema(s) for extraction ({self.extract_mode}): [{schema_names_preview}]",
        )
        return target_schemas

    def _save_model_execution_metadata(self) -> None:
        """Saves Stage 5 model provenance & fallback details to `outputs/05_extraction/model_execution_metadata.json`."""
        meta_path = self.stage5_dir / "model_execution_metadata.json"
        meta_payload = {
            "stage": "stage_5_mapping",
            "configured_model": self.configured_model_name,
            "active_model": self.model_name,
            "quantization_or_dtype": self.active_quantization,
            "device": str(getattr(self.prism, "device", self.device)),
            "effective_context_window": getattr(self.prism, "context_window", 32768),
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

    def _ensure_engine_loaded(self, dry_run: bool = False) -> None:
        """
        Ensures the PrismPipeline engine is loaded onto a valid accelerator device (never 'meta'/disk).
        If the primary model exceeds available GPU VRAM, explicitly purges traceback frames
        to free 100% of partially allocated GPU tensors before executing the fallback chain.
        """
        if dry_run:
            self._save_model_execution_metadata()
            return
        if self.prism.engine is not None:
            return
        if self.manage_vram:
            clear_device_memory()
        vram_info = _get_gpu_vram_summary()
        self._log("Phase 3: Engine Init", f"Loading LLM '{self.model_name}' [{vram_info}]...")

        def _try_load(candidate_model: str, use_4bit: bool) -> None:
            self.prism.config.setdefault("model", {})["name"] = candidate_model
            self.prism.config["model"]["load_in_4bit"] = use_4bit
            self.prism.init_engine(
                model_name=candidate_model,
                batch_size=self.batch_size,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
            )
            _validate_no_meta_or_disk_offload(self.prism, candidate_model)

        try:
            _try_load(self.model_name, self.load_in_4bit)
        except Exception as load_err:
            primary_err_msg = str(load_err).strip()
            vram_after = _purge_failed_model_vram(self.prism, load_err)
            del load_err
            self.fallback_history.append(
                f"Primary model '{self.configured_model_name}' (4bit={self.load_in_4bit}) failed: {primary_err_msg}"
            )
            self._log(
                "Phase 3: Engine Init",
                f"Warning: Primary model '{self.configured_model_name}' could not fit on device ({primary_err_msg}). "
                f"VRAM after cleanup: {vram_after}",
            )

            loaded = False
            lower_cfg = self.configured_model_name.lower()
            is_moe = "moe" in lower_cfg or "-a3b" in lower_cfg

            if not self.load_in_4bit and not is_moe:
                self._log(
                    "Phase 3: Engine Init",
                    f"Retrying '{self.configured_model_name}' with 4-bit quantization (load_in_4bit=True)...",
                )
                try:
                    _try_load(self.configured_model_name, use_4bit=True)
                    loaded = True
                    self.model_name = self.configured_model_name
                    self.fallback_used = True
                    self.active_quantization = "4-bit NF4 (auto-fallback)"
                    self.fallback_reason = (
                        f"Configured model '{self.configured_model_name}' exceeded GPU VRAM in default precision "
                        f"({primary_err_msg}); automatically loaded in 4-bit NF4 quantization."
                    )
                except Exception as q4_err:
                    q4_msg = str(q4_err).strip()
                    vram_after = _purge_failed_model_vram(self.prism, q4_err)
                    del q4_err
                    self.fallback_history.append(f"4-bit '{self.configured_model_name}' failed: {q4_msg}")
                    self._log("Phase 3: Engine Init", f"4-bit load attempt failed ({q4_msg}). VRAM: {vram_after}")

            if not loaded:
                for fb_model, fb_desc in (
                    ("Qwen/Qwen2.5-14B-Instruct", "14B dense fallback (~29 GB VRAM)"),
                    ("Qwen/Qwen2.5-7B-Instruct", "7B dense fallback (~15 GB VRAM)"),
                    ("Qwen/Qwen2.5-3B-Instruct", "3B lightweight fallback (~6 GB VRAM)"),
                ):
                    if fb_model == self.configured_model_name:
                        continue
                    self._log("Phase 3: Engine Init", f"Attempting fallback model '{fb_model}' ({fb_desc})...")
                    try:
                        _try_load(fb_model, use_4bit=self.load_in_4bit)
                        loaded = True
                        self.model_name = fb_model
                        self.fallback_used = True
                        self.active_quantization = (
                            "4-bit NF4" if self.load_in_4bit else ("8-bit" if self.load_in_8bit else f"native ({self.torch_dtype})")
                        )
                        self.fallback_reason = (
                            f"Configured model '{self.configured_model_name}' failed to load ({primary_err_msg}); "
                            f"automatically fell back to '{fb_model}' ({fb_desc})."
                        )
                        break
                    except Exception as fb_err:
                        fb_msg = str(fb_err).strip()
                        vram_after = _purge_failed_model_vram(self.prism, fb_err)
                        del fb_err
                        self.fallback_history.append(f"Fallback '{fb_model}' failed: {fb_msg}")
                        self._log("Phase 3: Engine Init", f"Fallback '{fb_model}' failed ({fb_msg}). VRAM: {vram_after}")

            if not loaded:
                raise RuntimeError(
                    f"Stage 5 could not load configured model '{self.configured_model_name}' or any fallback model.\n"
                    f"History:\n  - " + "\n  - ".join(self.fallback_history)
                )

            self._log(
                "STAGE 5 MODEL FALLBACK NOTICE",
                f"Configured='{self.configured_model_name}' -> Active='{self.model_name}' [{self.active_quantization}]. "
                f"Reason: {self.fallback_reason}",
            )

        self._save_model_execution_metadata()

    def _safe_chunk_document(self, text: str) -> List[str]:
        """
        Splits document text into overlapping word chunks while enforcing `ContextBudget`
        safety against the model's context window.
        """
        ctx_window = (
            self.context_window_cap
            if isinstance(self.context_window_cap, int) and self.context_window_cap > 0
            else getattr(self.prism, "context_window", 32768) or 32768
        )
        try:
            budget = ContextBudget.from_tokenizer(
                tokenizer=getattr(self.prism, "tokenizer", None),
                context_limit=ctx_window,
                max_new_tokens=self.max_new_tokens,
            )
            max_chunk_tokens = budget.max_input_tokens
        except ValueError:
            max_chunk_tokens = max(256, ctx_window // 2)
        # Convert token cap to approximate word cap (1 word ~ 1.35 tokens)
        safe_word_cap = max(50, int(max_chunk_tokens / 1.35))
        effective_words = min(self.chunk_size_words, safe_word_cap)
        effective_overlap = min(self.chunk_overlap_words, max(0, effective_words // 3))
        return chunk_text(text, chunk_size_words=effective_words, chunk_overlap_words=effective_overlap)

    def _run_triage_pass(
        self,
        chunk_records: List[Dict[str, Any]],
        target_schemas: List[SchemaMeta],
        dry_run: bool = False,
    ) -> Dict[Tuple[str, int], List[SchemaMeta]]:
        """
        Executes Pass 3a (Fast Multi-Schema Relevance Triage) across all pending chunks
        in a single hardware-batched PrismRunner pass.
        """
        chunk_to_schemas: Dict[Tuple[str, int], List[SchemaMeta]] = {}
        use_triage = (
            self.enable_triage
            and len(target_schemas) >= self.triage_threshold
            and self.target_class_name is None
            and not (self.extract_mode == "unified" and len(target_schemas) <= self.max_schemas_per_envelope)
        )

        if not use_triage:
            for rec in chunk_records:
                doc_id = rec["doc_id"]
                chunk_idx = rec["chunk_idx"]
                pending_for_doc = rec["pending_schemas"]
                chunk_to_schemas[(doc_id, chunk_idx)] = pending_for_doc
            return chunk_to_schemas

        # Only triage chunks that have at least one uncached schema and no cached triage file
        triage_inputs: List[Dict[str, Any]] = []
        triage_meta_by_id: Dict[str, Dict[str, Any]] = {}
        cached_triage_hits = 0
        for rec in chunk_records:
            rec_id = rec["id"]
            doc_id = rec["doc_id"]
            chunk_idx = rec["chunk_idx"]
            pending_for_doc: List[SchemaMeta] = rec["pending_schemas"]
            uncached = [
                s for s in pending_for_doc
                if not self.tracker.is_chunk_completed(doc_id, s, chunk_idx)
            ]
            if not uncached:
                # All pending schemas for this chunk already have cached chunk JSONs
                chunk_to_schemas[(doc_id, chunk_idx)] = pending_for_doc
            elif self.tracker.is_triage_chunk_completed(doc_id, chunk_idx):
                cached_names = self.tracker.load_triage_chunk_result(doc_id, chunk_idx) or []
                cached_names_lower = {n.strip().lower() for n in cached_names if n.strip()}
                matched = [
                    s for s in pending_for_doc
                    if s.name.lower() in cached_names_lower or s.qualified_name.lower() in cached_names_lower
                ]
                if not matched and self.fallback_on_empty_triage:
                    matched = pending_for_doc
                chunk_to_schemas[(doc_id, chunk_idx)] = matched
                cached_triage_hits += 1
            else:
                triage_inputs.append({"id": rec_id, "text": rec["text"]})
                triage_meta_by_id[rec_id] = rec

        if cached_triage_hits > 0:
            self._log(
                "Phase 3a: Schema Triage",
                f"Resumed {cached_triage_hits} already-triaged chunk(s) from disk cache.",
            )

        if not triage_inputs:
            return chunk_to_schemas

        triage_sys_prompt = build_triage_prompt(target_schemas)
        triage_instruction = (
            "Identify all candidate schemas from the system prompt that have explicit factual "
            "information present in the text excerpt above. Return their exact class names in `relevant_schemas`."
        )

        if triage_inputs:
            self._maybe_save_payload(
                "_SCHEMA_RELEVANCE_TRIAGE",
                triage_sys_prompt,
                f"Document Excerpt:\n{triage_inputs[0]['text']}\n\nInstruction:\n{triage_instruction}",
            )

        total_triage_chunks = len(triage_inputs)
        self._log(
            "Phase 3a: Schema Triage",
            f"Running batched relevance triage across {total_triage_chunks} chunk(s) "
            f"against {len(target_schemas)} candidate schemas...",
        )

        triage_start_ts = time.time()
        triage_done_count = 0

        def _on_triage_batch_start(batch_idx: int, total_batches: int, batch_records: List[Any]) -> None:
            self._log(
                "Phase 3a: Schema Triage",
                f"Batch {batch_idx + 1}/{total_batches} starting (batch_size={len(batch_records)} chunk(s))...",
            )

        def _on_triage_chunk_done(annotated_chunk: Dict[str, Any]) -> None:
            nonlocal triage_done_count
            rec_id = str(annotated_chunk.get("record_id") or annotated_chunk.get("id") or "")
            rec = triage_meta_by_id.get(rec_id)
            if not rec:
                return
            triage_done_count += 1
            doc_id = rec["doc_id"]
            chunk_idx = rec["chunk_idx"]
            pending_for_doc: List[SchemaMeta] = rec["pending_schemas"]
            parsed_items = annotated_chunk.get("parsed_items") or annotated_chunk.get("extracted_items") or []
            selected_names_lower: Set[str] = set()
            for item in parsed_items:
                if isinstance(item, dict):
                    names = item.get("relevant_schemas", [])
                elif isinstance(item, TriageResult):
                    names = item.relevant_schemas
                else:
                    names = []
                if isinstance(names, list):
                    for n in names:
                        if isinstance(n, str) and n.strip():
                            selected_names_lower.add(n.strip().lower())
            matched = [
                s for s in pending_for_doc
                if s.name.lower() in selected_names_lower or s.qualified_name.lower() in selected_names_lower
            ]
            if not matched and self.fallback_on_empty_triage:
                matched = pending_for_doc
            chunk_to_schemas[(doc_id, chunk_idx)] = matched
            triage_path = self.tracker.save_triage_chunk_result(
                doc_id, chunk_idx, [s.name for s in matched]
            )
            elapsed = max(0.001, time.time() - triage_start_ts)
            rate = triage_done_count / elapsed
            remaining = max(0, total_triage_chunks - triage_done_count)
            eta_s = int(remaining / rate) if rate > 0 else 0
            pct = (100.0 * triage_done_count) / max(1, total_triage_chunks)
            matched_str = ", ".join(s.name for s in matched) if matched else "none"
            self._log(
                "Phase 3a: Schema Triage",
                f"Chunk {triage_done_count}/{total_triage_chunks} ({pct:.1f}%) | "
                f"Doc '{doc_id}' chunk #{chunk_idx} -> [{matched_str}] "
                f"| {rate:.2f} ch/s | ETA {eta_s}s -> {triage_path.name}",
            )

        triage_results = self.prism.run(
            input_data=triage_inputs,
            id_column="id",
            text_column="text",
            task="extract",
            schema_cls=TriageResult,
            system_prompt=triage_sys_prompt,
            instruction=triage_instruction,
            max_new_tokens=self.triage_max_new_tokens,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            batch_size=self.batch_size,
            chunk=False,
            as_array=False,
            constrain_json=self.constrain_json,
            dedup=False,
            dry_run=dry_run,
            on_batch_start=_on_triage_batch_start,
            on_chunk_processed=_on_triage_chunk_done,
        )

        if dry_run:
            for rec_id, rec in triage_meta_by_id.items():
                chunk_to_schemas[(rec["doc_id"], rec["chunk_idx"])] = rec["pending_schemas"]
            return chunk_to_schemas

        # Map any remaining returned results not already handled by streaming callback
        results_by_id = {
            str(res.get("record_id") or res.get("id") or ""): res
            for res in triage_results
        }
        total_selected = 0

        for rec_id, rec in triage_meta_by_id.items():
            doc_id = rec["doc_id"]
            chunk_idx = rec["chunk_idx"]
            pending_for_doc: List[SchemaMeta] = rec["pending_schemas"]
            if (doc_id, chunk_idx) not in chunk_to_schemas:
                res = results_by_id.get(rec_id, {})
                extracted_items = res.get("extracted_items") or res.get("parsed_items") or []
                selected_names_lower: Set[str] = set()

                for item in extracted_items:
                    if isinstance(item, dict):
                        names = item.get("relevant_schemas", [])
                    elif isinstance(item, TriageResult):
                        names = item.relevant_schemas
                    else:
                        names = []
                    if isinstance(names, list):
                        for n in names:
                            if isinstance(n, str) and n.strip():
                                selected_names_lower.add(n.strip().lower())

                matched = [
                    s for s in pending_for_doc
                    if s.name.lower() in selected_names_lower
                    or s.qualified_name.lower() in selected_names_lower
                ]

                if not matched and self.fallback_on_empty_triage:
                    matched = pending_for_doc

                chunk_to_schemas[(doc_id, chunk_idx)] = matched
                self.tracker.save_triage_chunk_result(doc_id, chunk_idx, [s.name for s in matched])

            total_selected += len(chunk_to_schemas[(doc_id, chunk_idx)])

        total_possible = len(triage_inputs) * len(target_schemas)
        skipped_calls = max(0, total_possible - total_selected)
        self._log(
            "Phase 3a: Schema Triage",
            f"Triage complete: selected {total_selected}/{total_possible} (chunk, schema) pairs "
            f"(pruned {skipped_calls} irrelevant schema extraction call(s)).",
        )
        return chunk_to_schemas

    def run_extraction(
        self,
        documents: List[Dict[str, Any]],
        target_schemas: List[SchemaMeta],
        dry_run: bool = False,
    ) -> None:
        """
        Executes Two-Pass Hardware-Batched Extraction across all corpus documents and target schemas:
          1. Chunk documents & filter already-completed `(doc_id, schema)` pairs.
          2. Run Pass 3a (Fast Triage) across pending chunks.
          3. Run Pass 3b (Targeted Single-Schema Deep Extraction) batched per schema with
             streaming chunk-level disk caching.
          4. Run Pass 3c (Exact Verbatim Consolidation) via `merge_pydantic_instances` and
             write `<doc_id>__<module>__<Class>.json`.
        """
        # Prepare per-document chunks and pending schemas
        doc_chunks_map: Dict[str, List[str]] = {}
        doc_pending_schemas: Dict[str, List[SchemaMeta]] = {}
        all_chunk_records: List[Dict[str, Any]] = []
        doc_word_counts: List[int] = []
        whole_doc_count = 0

        skipped_doc_schemas = 0
        for doc in documents:
            doc_id = str(doc["id"])
            text = str(doc["text"])
            pending = [
                s for s in target_schemas
                if not self.tracker.is_schema_completed(doc_id, s)
            ]
            skipped_doc_schemas += len(target_schemas) - len(pending)
            if not pending:
                continue

            chunks = self._safe_chunk_document(text)
            if not chunks:
                for s in pending:
                    self.tracker.save_extraction(doc_id, s, [])
                continue

            doc_words = len(text.split())
            doc_word_counts.append(doc_words)
            if len(chunks) == 1:
                whole_doc_count += 1

            doc_chunks_map[doc_id] = chunks
            doc_pending_schemas[doc_id] = pending

            for chunk_idx, ch_text in enumerate(chunks):
                all_chunk_records.append(
                    {
                        "id": f"{doc_id}::chunk_{chunk_idx}",
                        "text": ch_text,
                        "doc_id": doc_id,
                        "chunk_idx": chunk_idx,
                        "pending_schemas": pending,
                    }
                )

        if doc_word_counts:
            sorted_wc = sorted(doc_word_counts)
            min_w = sorted_wc[0]
            med_w = sorted_wc[len(sorted_wc) // 2]
            max_w = sorted_wc[-1]
            whole_pct = (100.0 * whole_doc_count) / max(1, len(doc_word_counts))
            self._log(
                "Phase 2: Corpus Chunking",
                f"Prepared {len(all_chunk_records)} chunk(s) across {len(doc_word_counts)} pending doc(s) "
                f"(words/doc: min={min_w}, median={med_w}, max={max_w}; "
                f"{whole_doc_count}/{len(doc_word_counts)} [{whole_pct:.1f}%] kept as whole single-chunk docs).",
            )

        if skipped_doc_schemas > 0:
            self._log(
                "Phase 3: Resume Check",
                f"Skipped {skipped_doc_schemas} already-completed (document, schema) pair(s) on disk.",
            )

        if not all_chunk_records:
            self._log("Phase 3: Extraction", "All documents and schemas are already completed.")
            return

        self._ensure_engine_loaded(dry_run=dry_run)

        # Pass 3a: Fast Relevance Triage
        chunk_to_schemas = self._run_triage_pass(
            chunk_records=all_chunk_records,
            target_schemas=target_schemas,
            dry_run=dry_run,
        )

        # Group required (doc_id, chunk_idx, chunk_text) items by SchemaMeta
        schema_by_qname: Dict[str, SchemaMeta] = {s.qualified_name: s for s in target_schemas}
        chunk_text_lookup: Dict[Tuple[str, int], str] = {
            (rec["doc_id"], rec["chunk_idx"]): rec["text"]
            for rec in all_chunk_records
        }

        # Store chunk-level BaseModel results in memory: (doc_id, qualified_name) -> List[BaseModel]
        extracted_chunk_models: Dict[Tuple[str, str], List[BaseModel]] = defaultdict(list)
        schema_work_queues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        unified_bundle_queues: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
        remaining_doc_schema_chunks: Dict[Tuple[str, str], int] = defaultdict(int)
        finalized_doc_schemas: Set[Tuple[str, str]] = set()
        consolidated_count = 0
        non_empty_count = 0

        def _compute_adaptive_max_new_tokens(work_list: List[Dict[str, Any]], num_schemas_in_call: int) -> int:
            """
            Adaptively sizes `max_new_tokens` for corpora with smaller documents while
            scaling up smoothly for multi-schema unified envelopes on larger excerpts.
            """
            if not work_list:
                return self.max_new_tokens
            word_Lengths = sorted(len(w["text"].split()) for w in work_list)
            p90_words = word_Lengths[min(len(word_Lengths) - 1, int(0.9 * len(word_Lengths)))]
            if p90_words <= 250:
                estimated = 512 + (192 * num_schemas_in_call)
                return min(self.max_new_tokens, max(768, estimated))
            if num_schemas_in_call > 1:
                return max(self.max_new_tokens, min(3072, 512 * num_schemas_in_call))
            return self.max_new_tokens

        def _finalize_doc_schema_if_ready(doc_id_val: str, s_meta_val: SchemaMeta) -> None:
            nonlocal consolidated_count, non_empty_count
            key = (doc_id_val, s_meta_val.qualified_name)
            if key in finalized_doc_schemas:
                return
            if remaining_doc_schema_chunks.get(key, 0) > 0:
                return
            finalized_doc_schemas.add(key)
            chunk_instances = extracted_chunk_models.get(key, [])
            merged_doc_model = merge_pydantic_instances(chunk_instances, s_meta_val.schema_cls)
            if merged_doc_model is not None and has_extracted_content(merged_doc_model):
                serialized = merged_doc_model.model_dump(mode="json")
                out_f = self.tracker.save_extraction(doc_id_val, s_meta_val, serialized)
                non_empty_count += 1
                status_label = "populated"
            else:
                out_f = self.tracker.save_extraction(doc_id_val, s_meta_val, [])
                status_label = "empty"
            consolidated_count += 1
            self._log(
                "Phase 3c: Document Saved",
                f"Consolidated doc '{doc_id_val}' for schema '{s_meta_val.name}' ({status_label}) -> {out_f.name}",
            )

        for (doc_id, chunk_idx), selected_schemas in chunk_to_schemas.items():
            ch_text = chunk_text_lookup[(doc_id, chunk_idx)]
            uncached_for_chunk: List[SchemaMeta] = []
            for s_meta in selected_schemas:
                if self.tracker.is_chunk_completed(doc_id, s_meta, chunk_idx):
                    cached_inst = self.tracker.load_chunk_result(doc_id, s_meta, chunk_idx)
                    if cached_inst is not None and has_extracted_content(cached_inst):
                        extracted_chunk_models[(doc_id, s_meta.qualified_name)].append(cached_inst)
                else:
                    remaining_doc_schema_chunks[(doc_id, s_meta.qualified_name)] += 1
                    uncached_for_chunk.append(s_meta)

            if not uncached_for_chunk:
                continue

            if self.extract_mode == "unified" and len(uncached_for_chunk) > 1:
                max_per_env = max(1, self.max_schemas_per_envelope)
                for b_start in range(0, len(uncached_for_chunk), max_per_env):
                    sub_bundle = uncached_for_chunk[b_start : b_start + max_per_env]
                    if len(sub_bundle) == 1:
                        only_s = sub_bundle[0]
                        schema_work_queues[only_s.qualified_name].append(
                            {
                                "id": f"{doc_id}::chunk_{chunk_idx}::{only_s.qualified_name}",
                                "text": ch_text,
                                "doc_id": doc_id,
                                "chunk_idx": chunk_idx,
                                "qualified_name": only_s.qualified_name,
                            }
                        )
                    else:
                        qname_sig = tuple(s.qualified_name for s in sub_bundle)
                        unified_bundle_queues[qname_sig].append(
                            {
                                "id": f"{doc_id}::chunk_{chunk_idx}::bundle_{b_start // max_per_env}",
                                "text": ch_text,
                                "doc_id": doc_id,
                                "chunk_idx": chunk_idx,
                                "schemas": sub_bundle,
                            }
                        )
            else:
                for s_meta in uncached_for_chunk:
                    schema_work_queues[s_meta.qualified_name].append(
                        {
                            "id": f"{doc_id}::chunk_{chunk_idx}::{s_meta.qualified_name}",
                            "text": ch_text,
                            "doc_id": doc_id,
                            "chunk_idx": chunk_idx,
                            "qualified_name": s_meta.qualified_name,
                        }
                    )

        if not dry_run:
            # Immediately finalize any (doc_id, schema) pairs that require 0 LLM chunk calls
            for doc_id, pending_list in doc_pending_schemas.items():
                for s_meta in pending_list:
                    if remaining_doc_schema_chunks.get((doc_id, s_meta.qualified_name), 0) == 0:
                        _finalize_doc_schema_if_ready(doc_id, s_meta)

        # Pass 3b (Part 1): Unified Envelope Single-Pass Multi-Schema Extraction
        total_unified_bundles = len(unified_bundle_queues)
        for b_idx, (qname_sig, bundle_work_items) in enumerate(unified_bundle_queues.items(), start=1):
            if not bundle_work_items:
                continue
            bundle_schemas = [schema_by_qname[qn] for qn in qname_sig]
            envelope_cls, field_to_schema = build_unified_envelope_model(bundle_schemas, bundle_index=b_idx)
            sys_prompt = build_unified_extraction_prompt(field_to_schema)
            schema_names_str = ", ".join(f"`{s.name}`" for s in bundle_schemas)
            instruction = (
                f"Extract all information matching the {len(bundle_schemas)} domain schemas "
                f"({schema_names_str}) from the document excerpt above into the unified JSON object. "
                f"Use exact verbatim mentions and set any absent fields or absent domain schemas to null."
            )

            # Sort work items by word count so similar-length documents batch together with minimal padding
            sorted_work_items = sorted(bundle_work_items, key=lambda w: len(w["text"].split()))
            effective_max_tokens = _compute_adaptive_max_new_tokens(sorted_work_items, len(bundle_schemas))

            self._maybe_save_payload(
                f"unified_envelope_bundle_{b_idx}",
                sys_prompt,
                f"Document Excerpt:\n{sorted_work_items[0]['text']}\n\nInstruction:\n{instruction}",
            )

            total_bundle_chunks = len(sorted_work_items)
            bundle_label = "+".join(s.name for s in bundle_schemas)
            self._log(
                "Phase 3b: Unified Extraction",
                f"[Bundle {b_idx}/{total_unified_bundles}] Extracting {len(bundle_schemas)} schemas simultaneously "
                f"({bundle_label}) across {total_bundle_chunks} chunk(s) (max_new_tokens={effective_max_tokens})...",
            )

            work_meta_by_id = {w["id"]: w for w in sorted_work_items}
            prism_inputs = [{"id": w["id"], "text": w["text"]} for w in sorted_work_items]
            bundle_start_ts = time.time()
            bundle_done_chunks = 0

            def _on_unified_batch_start(
                batch_idx: int,
                total_batches: int,
                batch_records: List[Any],
                _b_idx: int = b_idx,
                _label: str = bundle_label,
            ) -> None:
                self._log(
                    "Phase 3b: Unified Extraction",
                    f"[Bundle {_b_idx}/{total_unified_bundles} '{_label}'] "
                    f"Batch {batch_idx + 1}/{total_batches} starting (batch_size={len(batch_records)} chunk(s))...",
                )

            def _on_unified_chunk_done(
                annotated_chunk: Dict[str, Any],
                _b_idx: int = b_idx,
                _total_chunks: int = total_bundle_chunks,
                _field_map: Dict[str, SchemaMeta] = field_to_schema,
                _lookup: Dict[str, Dict[str, Any]] = work_meta_by_id,
            ) -> None:
                nonlocal bundle_done_chunks
                rec_id = str(annotated_chunk.get("record_id") or annotated_chunk.get("id") or "")
                w_info = _lookup.get(rec_id, {})
                c_doc_id = w_info.get("doc_id")
                c_idx = w_info.get("chunk_idx")
                parsed_items = annotated_chunk.get("parsed_items", [])
                if c_doc_id is None or c_idx is None:
                    return
                bundle_done_chunks += 1
                valid_envelopes = [m for m in parsed_items if isinstance(m, BaseModel)]

                populated_schema_names: List[str] = []
                for f_name, s_meta in _field_map.items():
                    sub_instances: List[BaseModel] = []
                    for env_m in valid_envelopes:
                        sub_val = getattr(env_m, f_name, None)
                        if isinstance(sub_val, BaseModel) and has_extracted_content(sub_val):
                            sub_instances.append(sub_val)
                    merged_sub_inst = (
                        merge_pydantic_instances(sub_instances, s_meta.schema_cls)
                        if sub_instances
                        else None
                    )
                    self.tracker.save_chunk_result(c_doc_id, s_meta, c_idx, merged_sub_inst)
                    if merged_sub_inst is not None:
                        extracted_chunk_models[(c_doc_id, s_meta.qualified_name)].append(merged_sub_inst)
                        populated_schema_names.append(s_meta.name)

                elapsed = max(0.001, time.time() - bundle_start_ts)
                rate = bundle_done_chunks / elapsed
                rem = max(0, _total_chunks - bundle_done_chunks)
                eta_s = int(rem / rate) if rate > 0 else 0
                pct = (100.0 * bundle_done_chunks) / max(1, _total_chunks)
                pop_summary = (
                    f"populated: {', '.join(populated_schema_names)}"
                    if populated_schema_names
                    else f"0/{len(_field_map)} populated"
                )
                self._log(
                    "Phase 3b: Unified Extraction",
                    f"[Bundle {_b_idx}/{total_unified_bundles} | "
                    f"Chunk {bundle_done_chunks}/{_total_chunks} ({pct:.1f}%) | "
                    f"{rate:.2f} ch/s | ETA {eta_s}s] "
                    f"Doc '{c_doc_id}' chunk #{c_idx} ({pop_summary})",
                )

                for s_meta in _field_map.values():
                    key = (c_doc_id, s_meta.qualified_name)
                    remaining_doc_schema_chunks[key] = max(0, remaining_doc_schema_chunks.get(key, 1) - 1)
                    if remaining_doc_schema_chunks[key] == 0 and not dry_run:
                        _finalize_doc_schema_if_ready(c_doc_id, s_meta)

            self.prism.run(
                input_data=prism_inputs,
                id_column="id",
                text_column="text",
                task="extract",
                schema_cls=envelope_cls,
                system_prompt=sys_prompt,
                instruction=instruction,
                max_new_tokens=effective_max_tokens,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
                batch_size=self.batch_size,
                chunk=False,
                as_array=False,
                constrain_json=self.constrain_json,
                dedup=False,
                dry_run=dry_run,
                on_batch_start=_on_unified_batch_start,
                on_chunk_processed=_on_unified_chunk_done,
            )

        # Pass 3b (Part 2): Targeted Single-Schema Deep Extraction
        total_schemas_with_work = len(schema_work_queues)
        for s_idx, (qname, work_items) in enumerate(schema_work_queues.items(), start=1):
            s_meta = schema_by_qname[qname]
            sys_prompt = build_extraction_prompt(s_meta.schema_cls)
            instruction = (
                f"Extract all information matching `{s_meta.name}` from the document excerpt above. "
                f"Use exact verbatim mentions and set any absent fields to null."
            )

            sorted_work_items = sorted(work_items, key=lambda w: len(w["text"].split()))
            effective_max_tokens = _compute_adaptive_max_new_tokens(sorted_work_items, 1)

            if sorted_work_items:
                self._maybe_save_payload(
                    s_meta.name,
                    sys_prompt,
                    f"Document Excerpt:\n{sorted_work_items[0]['text']}\n\nInstruction:\n{instruction}",
                )

            total_schema_chunks = len(sorted_work_items)
            self._log(
                "Phase 3b: Deep Extraction",
                f"[{s_idx}/{total_schemas_with_work}] Extracting schema '{s_meta.qualified_name}' "
                f"across {total_schema_chunks} relevant chunk(s) (max_new_tokens={effective_max_tokens})...",
            )

            work_meta_by_id = {w["id"]: w for w in sorted_work_items}
            prism_inputs = [{"id": w["id"], "text": w["text"]} for w in sorted_work_items]
            schema_start_ts = time.time()
            schema_done_chunks = 0

            def _on_extract_batch_start(
                batch_idx: int,
                total_batches: int,
                batch_records: List[Any],
                _s_idx: int = s_idx,
                _meta: SchemaMeta = s_meta,
            ) -> None:
                self._log(
                    "Phase 3b: Deep Extraction",
                    f"[{_s_idx}/{total_schemas_with_work} '{_meta.name}'] "
                    f"Batch {batch_idx + 1}/{total_batches} starting (batch_size={len(batch_records)} chunk(s))...",
                )

            def _on_chunk_done(
                annotated_chunk: Dict[str, Any],
                _s_idx: int = s_idx,
                _total_chunks: int = total_schema_chunks,
                _meta: SchemaMeta = s_meta,
                _lookup: Dict[str, Dict[str, Any]] = work_meta_by_id,
            ) -> None:
                nonlocal schema_done_chunks
                rec_id = str(annotated_chunk.get("record_id") or annotated_chunk.get("id") or "")
                w_info = _lookup.get(rec_id, {})
                c_doc_id = w_info.get("doc_id")
                c_idx = w_info.get("chunk_idx")
                parsed_items = annotated_chunk.get("parsed_items", [])
                if c_doc_id is None or c_idx is None:
                    return
                schema_done_chunks += 1
                valid_models = [
                    m for m in parsed_items
                    if isinstance(m, BaseModel) and has_extracted_content(m)
                ]
                merged_chunk_inst = (
                    merge_pydantic_instances(valid_models, _meta.schema_cls)
                    if valid_models
                    else None
                )
                chunk_file = self.tracker.save_chunk_result(c_doc_id, _meta, c_idx, merged_chunk_inst)
                if merged_chunk_inst is not None:
                    extracted_chunk_models[(c_doc_id, _meta.qualified_name)].append(merged_chunk_inst)

                elapsed = max(0.001, time.time() - schema_start_ts)
                rate = schema_done_chunks / elapsed
                rem = max(0, _total_chunks - schema_done_chunks)
                eta_s = int(rem / rate) if rate > 0 else 0
                pct = (100.0 * schema_done_chunks) / max(1, _total_chunks)
                state_str = "populated" if merged_chunk_inst is not None else "empty"
                self._log(
                    "Phase 3b: Deep Extraction",
                    f"[{_s_idx}/{total_schemas_with_work} '{_meta.name}' | "
                    f"Chunk {schema_done_chunks}/{_total_chunks} ({pct:.1f}%) | "
                    f"{rate:.2f} ch/s | ETA {eta_s}s] "
                    f"Doc '{c_doc_id}' chunk #{c_idx} ({state_str}) -> {chunk_file.name}",
                )

                key = (c_doc_id, _meta.qualified_name)
                remaining_doc_schema_chunks[key] = max(0, remaining_doc_schema_chunks.get(key, 1) - 1)
                if remaining_doc_schema_chunks[key] == 0 and not dry_run:
                    _finalize_doc_schema_if_ready(c_doc_id, _meta)

            self.prism.run(
                input_data=prism_inputs,
                id_column="id",
                text_column="text",
                task="extract",
                schema_cls=s_meta.schema_cls,
                system_prompt=sys_prompt,
                instruction=instruction,
                max_new_tokens=effective_max_tokens,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
                batch_size=self.batch_size,
                chunk=False,
                as_array=False,
                constrain_json=self.constrain_json,
                dedup=False,
                dry_run=dry_run,
                on_batch_start=_on_extract_batch_start,
                on_chunk_processed=_on_chunk_done,
            )

        if dry_run:
            self._log("Phase 3: Dry Run", "Dry run complete; skipping consolidation and file writes.")
            return

        # Pass 3c: Final sweep for any remaining (doc_id, SchemaMeta) pairs not already finalized in streaming callback
        for doc_id, pending_list in doc_pending_schemas.items():
            for s_meta in pending_list:
                remaining_doc_schema_chunks[(doc_id, s_meta.qualified_name)] = 0
                _finalize_doc_schema_if_ready(doc_id, s_meta)

        self._log(
            "Phase 3c: Consolidation",
            f"Consolidated {consolidated_count} (document, schema) output(s) "
            f"({non_empty_count} populated with verbatim data).",
        )

    def run(
        self,
        input_source: Union[str, Path, pd.DataFrame, List[Dict[str, Any]]] = "inputs/testdocs.parquet",
        id_column: str = "unique_val",
        text_column: str = "report",
        limit: Optional[int] = None,
        skip_extraction: bool = False,
        skip_aggregation: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs the complete Stage 5 pipeline:
          - Phase 1: Schema DAG Discovery
          - Phase 2: Corpus Document Loading
          - Phase 3: Batched Triage, Extraction & Verbatim Consolidation
          - Phase 4: Relational & Consolidated Wide-Table Aggregation
        """
        source_texts: Dict[str, str] = {}
        try:
            documents = load_input_documents(
                input_source=input_source,
                id_column=id_column,
                text_column=text_column,
                limit=limit,
            )
            source_texts = {str(d["id"]): str(d["text"]) for d in documents}
            self._log("Phase 2: Corpus Loader", f"Loaded {len(documents)} document(s) from '{input_source}'.")
        except Exception as e:
            if not skip_extraction:
                raise
            logger.warning(f"Could not load source documents for aggregation ({e}); proceeding without source_text.")
            documents = []

        if not skip_extraction:
            try:
                target_schemas = self.resolve_target_schemas()
            except (FileNotFoundError, ValueError) as e:
                if dry_run:
                    self._log(
                        "Phase 1: Schema Discovery (Dry Run)",
                        f"Schema path '{self.schema_path}' not yet generated ({e}); dry-run verified pipeline & corpus setup.",
                    )
                    target_schemas = []
                else:
                    raise

            if target_schemas:
                try:
                    self.run_extraction(
                        documents=documents,
                        target_schemas=target_schemas,
                        dry_run=dry_run,
                    )
                finally:
                    if self.manage_vram and (
                        getattr(self.prism, "engine", None) is not None
                        or getattr(self.prism, "model", None) is not None
                    ):
                        self.prism.engine = None
                        self.prism.model = None
                        self.prism.tokenizer = None
                        clear_device_memory()
                        self._log("VRAM Management", "Unloaded LLM and cleared device memory.")

        agg_summary: Dict[str, Any] = {}
        if not skip_aggregation and not dry_run:
            aggregator = DataAggregator(
                extracts_dir=self.extracts_dir,
                tables_dir=self.tables_dir,
                generate_csv=self.generate_csv,
                generate_sqlite=self.generate_sqlite,
                generate_parquet=self.generate_parquet,
                db_filename=self.db_filename,
                multi_table_relational=self.multi_table_relational,
                consolidated_wide_table=self.consolidated_wide_table,
                wide_table_name=self.wide_table_name,
                include_source_text=self.include_source_text,
                drop_empty_records=self.drop_empty_records,
                drop_empty_tables=self.drop_empty_tables,
                print_checkpoints=self.print_checkpoints,
            )
            agg_summary = aggregator.run(source_texts=source_texts)

        self._log("Stage 5 Complete", f"Outputs saved under '{self.stage5_dir}'.")
        return agg_summary
