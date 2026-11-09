"""
Stage 5 Pipeline Orchestrator: Corpus Loading, Chunking, Resume Tracking,
Verbatim Chunk Consolidation, and Two-Pass Hardware-Batched Extraction via PrismRunner.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
import json
import logging
import os
from pathlib import Path
import re
import sys
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
from src.mapping.prompts import build_extraction_prompt, build_triage_prompt
from src.mapping.schemas import (
    SchemaMeta,
    TriageResult,
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
    chunk_size_words: int = 300,
    chunk_overlap_words: int = 20,
) -> List[str]:
    """
    Splits a document's text into overlapping word chunks.
    """
    if not text or not text.strip():
        return []

    words = text.split()
    if len(words) <= chunk_size_words:
        return [" ".join(words)]

    chunks: List[str] = []
    step = max(1, chunk_size_words - chunk_overlap_words)

    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size_words]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size_words >= len(words):
            break

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

@contextmanager
def _with_hf_offload_folder(offload_dir: Path):
    """Temporarily injects `offload_folder` into HuggingFace `from_pretrained` calls when `device_map` is used."""
    offload_dir.mkdir(parents=True, exist_ok=True)
    patched_classes = []
    try:
        import transformers as _tf
        for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForMultimodalLM"):
            target_cls = getattr(_tf, cls_name, None)
            if target_cls is not None and hasattr(target_cls, "from_pretrained"):
                orig_bound = target_cls.from_pretrained
                orig_func = getattr(orig_bound, "__func__", None)
                if orig_func is not None:
                    def _make_patched(fn):
                        def _patched(cls, *args, **kwargs):
                            if kwargs.get("device_map") is not None and "offload_folder" not in kwargs:
                                kwargs["offload_folder"] = str(offload_dir)
                            return fn(cls, *args, **kwargs)
                        return classmethod(_patched)
                    target_cls.from_pretrained = _make_patched(orig_func)
                    patched_classes.append((target_cls, orig_func))
        yield
    finally:
        for target_cls, orig_func in patched_classes:
            target_cls.from_pretrained = classmethod(orig_func)


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
        max_new_tokens: int = 4096,
        triage_max_new_tokens: int = 1024,
        chunk_size_words: int = 300,
        chunk_overlap_words: int = 20,
        enable_triage: bool = True,
        triage_threshold: int = 3,
        fallback_on_empty_triage: bool = False,
        schema_path: Union[str, Path] = "outputs/04_synthesis/schemas/community/master_ontology.py",
        target_class_name: Optional[str] = None,
        extract_mode: str = "top_level",
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
        elif self.extract_mode == "top_level":
            target_schemas = top_level
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

    def _ensure_engine_loaded(self, dry_run: bool = False) -> None:
        """Ensures the PrismPipeline engine is loaded with automatic 4-bit and offload_folder resilience."""
        if dry_run or self.prism.engine is not None:
            return
        if self.manage_vram:
            clear_device_memory()
        self._log("Phase 3: Engine Init", f"Loading LLM '{self.model_name}' onto device...")
        offload_dir = self.outputs_dir / ".offload_cache"
        try:
            self.prism.init_engine(
                batch_size=self.batch_size,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
            )
        except Exception as load_err:
            err_str = str(load_err).lower()
            is_mem_or_offload = any(
                k in err_str for k in ("offload_folder", "offloaded to the disk", "out of memory", "cuda error")
            )
            loaded = False
            if is_mem_or_offload and not self.prism.config.get("model", {}).get("load_in_4bit", False):
                self._log(
                    "Phase 3: Engine Init",
                    f"Model '{self.model_name}' exceeded device memory in default precision ({load_err}); "
                    "retrying with 4-bit quantization (load_in_4bit=True)...",
                )
                if self.manage_vram:
                    clear_device_memory()
                self.prism.config.setdefault("model", {})["load_in_4bit"] = True
                try:
                    with _with_hf_offload_folder(offload_dir):
                        self.prism.init_engine(
                            batch_size=self.batch_size,
                            max_new_tokens=self.max_new_tokens,
                            temperature=self.temperature,
                            repetition_penalty=self.repetition_penalty,
                        )
                    loaded = True
                except Exception as q4_err:
                    self._log("Phase 3: Engine Init", f"4-bit load attempt failed ({q4_err}).")
                    self.prism.config["model"]["load_in_4bit"] = self.load_in_4bit
                    if self.manage_vram:
                        clear_device_memory()
            if not loaded:
                self._log("Phase 3: Engine Init", f"Retrying '{self.model_name}' with offload_folder={offload_dir}...")
                if self.manage_vram:
                    clear_device_memory()
                with _with_hf_offload_folder(offload_dir):
                    self.prism.init_engine(
                        batch_size=self.batch_size,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        repetition_penalty=self.repetition_penalty,
                    )

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
        )

        if not use_triage:
            for rec in chunk_records:
                doc_id = rec["doc_id"]
                chunk_idx = rec["chunk_idx"]
                pending_for_doc = rec["pending_schemas"]
                chunk_to_schemas[(doc_id, chunk_idx)] = pending_for_doc
            return chunk_to_schemas

        # Only triage chunks that have at least one uncached schema
        triage_inputs: List[Dict[str, Any]] = []
        triage_meta_by_id: Dict[str, Dict[str, Any]] = {}
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
            else:
                triage_inputs.append({"id": rec_id, "text": rec["text"]})
                triage_meta_by_id[rec_id] = rec

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

        self._log(
            "Phase 3a: Schema Triage",
            f"Running batched relevance triage across {len(triage_inputs)} chunk(s) "
            f"against {len(target_schemas)} candidate schemas...",
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
        )

        if dry_run:
            for rec_id, rec in triage_meta_by_id.items():
                chunk_to_schemas[(rec["doc_id"], rec["chunk_idx"])] = rec["pending_schemas"]
            return chunk_to_schemas

        # Map returned results by chunk record id ("record_id" or "id")
        results_by_id = {
            str(res.get("record_id") or res.get("id") or ""): res
            for res in triage_results
        }
        total_selected = 0

        for rec_id, rec in triage_meta_by_id.items():
            doc_id = rec["doc_id"]
            chunk_idx = rec["chunk_idx"]
            pending_for_doc: List[SchemaMeta] = rec["pending_schemas"]

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
            total_selected += len(matched)

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

        for (doc_id, chunk_idx), selected_schemas in chunk_to_schemas.items():
            ch_text = chunk_text_lookup[(doc_id, chunk_idx)]
            for s_meta in selected_schemas:
                if self.tracker.is_chunk_completed(doc_id, s_meta, chunk_idx):
                    cached_inst = self.tracker.load_chunk_result(doc_id, s_meta, chunk_idx)
                    if cached_inst is not None and has_extracted_content(cached_inst):
                        extracted_chunk_models[(doc_id, s_meta.qualified_name)].append(cached_inst)
                else:
                    schema_work_queues[s_meta.qualified_name].append(
                        {
                            "id": f"{doc_id}::chunk_{chunk_idx}::{s_meta.qualified_name}",
                            "text": ch_text,
                            "doc_id": doc_id,
                            "chunk_idx": chunk_idx,
                            "qualified_name": s_meta.qualified_name,
                        }
                    )

        # Pass 3b: Targeted Single-Schema Deep Extraction
        total_schemas_with_work = len(schema_work_queues)
        for s_idx, (qname, work_items) in enumerate(schema_work_queues.items(), start=1):
            s_meta = schema_by_qname[qname]
            sys_prompt = build_extraction_prompt(s_meta.schema_cls)
            instruction = (
                f"Extract all information matching `{s_meta.name}` from the document excerpt above. "
                f"Use exact verbatim mentions and set any absent fields to null."
            )

            if work_items:
                self._maybe_save_payload(
                    s_meta.name,
                    sys_prompt,
                    f"Document Excerpt:\n{work_items[0]['text']}\n\nInstruction:\n{instruction}",
                )

            self._log(
                "Phase 3b: Deep Extraction",
                f"[{s_idx}/{total_schemas_with_work}] Extracting schema '{s_meta.qualified_name}' "
                f"across {len(work_items)} relevant chunk(s)...",
            )

            work_meta_by_id = {w["id"]: w for w in work_items}
            prism_inputs = [{"id": w["id"], "text": w["text"]} for w in work_items]

            def _on_chunk_done(
                annotated_chunk: Dict[str, Any],
                _meta: SchemaMeta = s_meta,
                _lookup: Dict[str, Dict[str, Any]] = work_meta_by_id,
            ) -> None:
                rec_id = str(annotated_chunk.get("record_id") or annotated_chunk.get("id") or "")
                w_info = _lookup.get(rec_id, {})
                c_doc_id = w_info.get("doc_id")
                c_idx = w_info.get("chunk_idx")
                parsed_items = annotated_chunk.get("parsed_items", [])
                if c_doc_id is None or c_idx is None:
                    return
                valid_models = [
                    m for m in parsed_items
                    if isinstance(m, BaseModel) and has_extracted_content(m)
                ]
                merged_chunk_inst = (
                    merge_pydantic_instances(valid_models, _meta.schema_cls)
                    if valid_models
                    else None
                )
                self.tracker.save_chunk_result(c_doc_id, _meta, c_idx, merged_chunk_inst)
                if merged_chunk_inst is not None:
                    extracted_chunk_models[(c_doc_id, _meta.qualified_name)].append(merged_chunk_inst)

            self.prism.run(
                input_data=prism_inputs,
                id_column="id",
                text_column="text",
                task="extract",
                schema_cls=s_meta.schema_cls,
                system_prompt=sys_prompt,
                instruction=instruction,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                repetition_penalty=self.repetition_penalty,
                batch_size=self.batch_size,
                chunk=False,
                as_array=False,
                constrain_json=self.constrain_json,
                dedup=False,
                dry_run=dry_run,
                on_chunk_processed=_on_chunk_done,
            )

        if dry_run:
            self._log("Phase 3: Dry Run", "Dry run complete; skipping consolidation and file writes.")
            return

        # Pass 3c: Exact Verbatim Chunk Consolidation per (doc_id, SchemaMeta)
        consolidated_count = 0
        non_empty_count = 0
        for doc_id, pending_list in doc_pending_schemas.items():
            for s_meta in pending_list:
                chunk_instances = extracted_chunk_models.get((doc_id, s_meta.qualified_name), [])
                merged_doc_model = merge_pydantic_instances(chunk_instances, s_meta.schema_cls)

                if merged_doc_model is not None and has_extracted_content(merged_doc_model):
                    serialized = merged_doc_model.model_dump(mode="json")
                    self.tracker.save_extraction(doc_id, s_meta, serialized)
                    non_empty_count += 1
                else:
                    self.tracker.save_extraction(doc_id, s_meta, [])
                consolidated_count += 1

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
