#!/usr/bin/env python3
"""
SemanticPrism Stage 5: Schema-Driven Extraction & Relational Aggregation (Powered by PrismRunner)
=================================================================================================
Performs the Stage 5 SchemaMap extraction and relational/wide-table aggregation workflow,
powered by PrismRunner's local HuggingFace batch inference engine.

Supported Schema Input Modes (SCHEMA_PATH):
  1. Single .py file with multiple Pydantic BaseModel schemas:
     e.g., "outputs/04_synthesis/schemas/community/master_ontology.py"
  2. Directory of multiple .py schema files:
     e.g., "outputs/04_synthesis/schemas/community"

Workflow Phases & Output Artifacts:
  - Phase 1 : Dynamic Schema DAG Discovery (Top-Level vs. Child BaseModel Classification)
  - Phase 2 : Corpus Loading & Overlapping Word-Window Chunking
  - Phase 3 : Two-Pass Hardware-Batched Extraction via PrismRunner
              * Pass 3a: Fast Multi-Schema Relevance Triage (TriageResult)
              * Pass 3b: Targeted Single-Schema Deep Extraction + Chunk-Level Caching
              * Pass 3c: Exact Verbatim Chunk Consolidation (merge_pydantic_instances)
              * outputs/05_extraction/extracts/<doc_id>/<doc_id>__<module>__<Class>.json
              * outputs/05_extraction/extracts/<doc_id>/chunks/<module>.<Class>/<doc_id>_chunk_<i>.json
              * outputs/05_extraction/extracts/<doc_id>/.completed.json
              * outputs/payload_text/stage_05_extraction/<Class>.txt
  - Phase 4 : Relational & Consolidated Wide-Table Aggregation
              * outputs/05_extraction/aggregated_tables/<Schema>.csv & <Schema>_<child>.csv
              * outputs/05_extraction/aggregated_tables/consolidated_wide.csv
              * outputs/05_extraction/aggregated_tables/consolidated_wide.parquet
              * outputs/05_extraction/aggregated_tables/aggregated_data.db (SQLite)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback
from typing import Optional, Union

# Prevent PyTorch CUDA allocator fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Ensure repo root and sibling PrismRunner are importable
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_SIBLING_PRISMRUNNER = REPO_ROOT.parent / "PrismRunner"
if _SIBLING_PRISMRUNNER.exists() and str(_SIBLING_PRISMRUNNER) not in sys.path:
    sys.path.insert(0, str(_SIBLING_PRISMRUNNER))

from model_config import (
    DOCUMENT_LIMIT as MASTER_DOCUMENT_LIMIT,
    ID_COLUMN as MASTER_ID_COLUMN,
    INPUT_PATH as MASTER_INPUT_PATH,
    MODEL_NAME as MASTER_MODEL_NAME,
    TEXT_COLUMN as MASTER_TEXT_COLUMN,
)
from src.mapping import (
    DataAggregator,
    ResumeTracker,
    SchemaMeta,
    SemanticPrismStage5Pipeline,
    TriageResult,
    build_extraction_prompt,
    build_triage_prompt,
    chunk_text,
    load_input_documents,
    load_schemas,
    merge_pydantic_instances,
)

# ============================================================================
# GLOBAL CONFIGURATION VARIABLES
# ============================================================================
# Edit these variables directly to configure models, hardware, paths,
# schema input mode, triage, extraction chunking, and table aggregation.

# --- 1. Generative LLM & Hardware Settings (Defaults loaded from model_config.py) ---
MODEL_NAME: str = MASTER_MODEL_NAME                    # Local HuggingFace causal LLM for triage & extraction
DEVICE: str = "auto"                                   # "auto", "cuda", "mps", or "cpu"
TORCH_DTYPE: str = "auto"                              # "auto", "bfloat16", "float16", or "float32"
LOAD_IN_4BIT: bool = False                             # Enable 4-bit NF4 quantization (CUDA bitsandbytes)
LOAD_IN_8BIT: bool = False                             # Enable 8-bit quantization (CUDA bitsandbytes)
ATTN_IMPLEMENTATION: str = "auto"                      # "auto", "flash_attention_2", "sdpa", or "eager"
BATCH_SIZE: Union[int, str] = "auto"                   # "auto" (continuous device-relative) or explicit int
TEMPERATURE: float = 0.0                               # 0.0 for deterministic greedy decoding
REPETITION_PENALTY: float = 1.05                       # Mild penalty to prevent greedy repetition loops
CONSTRAIN_JSON: bool = True                            # Force JSON '{' prefix & bracket-balanced early stopping
MANAGE_VRAM: bool = True                               # Flush and offload GPU/MPS memory after extraction completes
DRY_RUN: bool = False                                  # Set True to preview prompts without running LLM generation

# --- 2. Corpus Input Settings (Defaults loaded from model_config.py) ---
INPUT_PATH: str = MASTER_INPUT_PATH                    # .parquet, .csv, .json, .jsonl, DataFrame, or dir of .txt files
ID_COLUMN: str = MASTER_ID_COLUMN                      # Document ID column name
TEXT_COLUMN: str = MASTER_TEXT_COLUMN                  # Document text column name
DOCUMENT_LIMIT: Optional[int] = MASTER_DOCUMENT_LIMIT  # Optional cap on number of documents to process (None = all)

# --- 3. Schema Input Settings (Supports Single .py File OR Directory of .py Files) ---
# Option 1 (Single file): "outputs/04_synthesis/schemas/community/master_ontology.py" or "outputs/04_synthesis/schemas/community/comprehensive_ontology.py"
# Option 2 (Directory)  : "outputs/04_synthesis/schemas/community"
SCHEMA_PATH: str = "outputs/04_synthesis/schemas/community/master_ontology.py"
TARGET_CLASS_NAME: Optional[str] = None                # Optional single BaseModel class name filter (None = all eligible)
EXTRACT_MODE: str = "top_level"                        # "top_level" (recommended), "child", or "all"

# --- 4. Context Window, Chunking & Triage Settings ---
CONTEXT_WINDOW_CAP: Union[int, str] = "auto"           # "auto" = full native model context window, or int cap
MAX_NEW_TOKENS: int = 4096                             # Max generated tokens per schema extraction call
TRIAGE_MAX_NEW_TOKENS: int = 1024                      # Max generated tokens for fast schema relevance triage
CHUNK_SIZE_WORDS: int = 300                            # Target word count per text chunk
CHUNK_OVERLAP_WORDS: int = 20                          # Overlapping words between consecutive chunks
ENABLE_TRIAGE: bool = True                             # Run fast triage pass to skip irrelevant schemas per chunk
TRIAGE_THRESHOLD: int = 3                              # Only run triage when candidate schema count >= threshold
FALLBACK_ON_EMPTY_TRIAGE: bool = False                 # False = skip chunk if triage selects 0 schemas; True = run all

# --- 5. Output, Caching & Logging Settings ---
OUTPUTS_DIR: str = "outputs"                           # Base outputs directory (writes to outputs/05_extraction)
SAVE_LLM_PAYLOADS: bool = True                         # Save first rendered prompt per schema to outputs/payload_text/stage_05_extraction
RESUME_MODE: str = "skip"                              # "skip" (resume from existing extracts/chunks) or "overwrite"
PRINT_CHECKPOINTS: bool = True                         # Print timestamped progress checkpoints to console

# --- 6. Relational & Consolidated Wide-Table Aggregation Settings ---
GENERATE_CSV: bool = True                              # Export normalized relational CSVs and consolidated_wide.csv
GENERATE_SQLITE: bool = True                           # Export SQLite database (aggregated_data.db)
GENERATE_PARQUET: bool = True                          # Export consolidated_wide.parquet
DB_FILENAME: str = "aggregated_data.db"                # SQLite database filename inside outputs/05_extraction/aggregated_tables
MULTI_TABLE_RELATIONAL: bool = True                    # Generate normalized parent/child relational tables
CONSOLIDATED_WIDE_TABLE: bool = True                   # Generate 1-row-per-document consolidated_wide table
WIDE_TABLE_NAME: str = "consolidated_wide"             # Base name for consolidated wide table (.csv, .parquet, SQLite table)
INCLUDE_SOURCE_TEXT: bool = True                       # Include full document source_text column in consolidated_wide
DROP_EMPTY_RECORDS: bool = True                        # Omit all-null records when building relational/wide tables
DROP_EMPTY_TABLES: bool = True                         # Omit relational tables that have 0 extracted rows


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SemanticPrism Stage 5: Schema-Driven Extraction & Relational/Wide-Table Aggregation (Powered by PrismRunner)"
    )
    parser.add_argument(
        "--schema-path",
        type=str,
        default=SCHEMA_PATH,
        help="Path to a single .py schema file (e.g. master_ontology.py) OR a directory of .py schema files.",
    )
    parser.add_argument(
        "--target-class",
        type=str,
        default=TARGET_CLASS_NAME,
        help="Optional specific Pydantic BaseModel class name to extract.",
    )
    parser.add_argument(
        "--extract-mode",
        type=str,
        choices=["top_level", "child", "all"],
        default=EXTRACT_MODE,
        help="Which schema hierarchy tier to extract ('top_level' recommended).",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=INPUT_PATH,
        help="Path to input corpus (.parquet, .csv, .json, .jsonl, or directory of .txt files).",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default=ID_COLUMN,
        help="Column name for document IDs.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default=TEXT_COLUMN,
        help="Column name for document text.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DOCUMENT_LIMIT,
        help="Maximum number of documents to process.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="HuggingFace model ID or local path for PrismRunner.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help="Compute device ('auto', 'cuda', 'mps', 'cpu').",
    )
    parser.add_argument(
        "--batch-size",
        type=str,
        default=str(BATCH_SIZE),
        help="Inference batch size ('auto' or integer).",
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default=OUTPUTS_DIR,
        help="Base outputs directory (writes to <outputs-dir>/05_extraction).",
    )
    parser.add_argument(
        "--no-triage",
        action="store_true",
        help="Disable fast multi-schema relevance triage and run all schemas on every chunk.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extraction files instead of resuming.",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip LLM extraction and only run relational/wide-table aggregation on existing extracts.",
    )
    parser.add_argument(
        "--skip-aggregation",
        action="store_true",
        help="Run LLM extraction only and skip table aggregation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=DRY_RUN,
        help="Preview payloads and pipeline setup without loading or executing the LLM.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Force live LLM execution even if DRY_RUN is True.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_size_val: Union[int, str] = (
        int(args.batch_size) if str(args.batch_size).isdigit() else args.batch_size
    )
    is_dry_run = False if args.live else bool(args.dry_run)
    resume_mode = "overwrite" if args.overwrite else RESUME_MODE

    try:
        pipeline = SemanticPrismStage5Pipeline(
            model_name=args.model,
            device=args.device,
            torch_dtype=TORCH_DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
            attn_implementation=ATTN_IMPLEMENTATION,
            batch_size=batch_size_val,
            temperature=TEMPERATURE,
            repetition_penalty=REPETITION_PENALTY,
            constrain_json=CONSTRAIN_JSON,
            context_window_cap=CONTEXT_WINDOW_CAP,
            max_new_tokens=MAX_NEW_TOKENS,
            triage_max_new_tokens=TRIAGE_MAX_NEW_TOKENS,
            chunk_size_words=CHUNK_SIZE_WORDS,
            chunk_overlap_words=CHUNK_OVERLAP_WORDS,
            enable_triage=not args.no_triage and ENABLE_TRIAGE,
            triage_threshold=TRIAGE_THRESHOLD,
            fallback_on_empty_triage=FALLBACK_ON_EMPTY_TRIAGE,
            schema_path=args.schema_path,
            target_class_name=args.target_class,
            extract_mode=args.extract_mode,
            outputs_dir=args.outputs_dir,
            save_llm_payloads=SAVE_LLM_PAYLOADS,
            resume_mode=resume_mode,
            manage_vram=MANAGE_VRAM,
            print_checkpoints=PRINT_CHECKPOINTS,
            generate_csv=GENERATE_CSV,
            generate_sqlite=GENERATE_SQLITE,
            generate_parquet=GENERATE_PARQUET,
            db_filename=DB_FILENAME,
            multi_table_relational=MULTI_TABLE_RELATIONAL,
            consolidated_wide_table=CONSOLIDATED_WIDE_TABLE,
            wide_table_name=WIDE_TABLE_NAME,
            include_source_text=INCLUDE_SOURCE_TEXT,
            drop_empty_records=DROP_EMPTY_RECORDS,
            drop_empty_tables=DROP_EMPTY_TABLES,
        )
        pipeline.run(
            input_source=args.input_path,
            id_column=args.id_column,
            text_column=args.text_column,
            limit=args.limit,
            skip_extraction=args.skip_extraction,
            skip_aggregation=args.skip_aggregation,
            dry_run=is_dry_run,
        )
    except Exception as e:
        print(f"\n[ERROR] Stage 5 pipeline failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

