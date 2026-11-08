#!/usr/bin/env python3
"""
SemanticPrism Stage 1: Full Extraction Pipeline (Powered by PrismRunner)
========================================================================
Performs the exact Stage 1 Extraction workflow from:
    https://github.com/childmindresearch/SemanticPrism

Workflow Phases:
  - Phase 1   : Global Theme Discovery (per-document chunks -> themes/{doc}_themes.json)
  - Phase 1.5 : Theme Aggregation (-> all_themes.json)
  - Phase 1.8 : Empirical Vector Theme Clustering (Dynamic kNN/Leiden -> clustering_summary.json)
  - Phase 2   : Master Theme Synthesis (Map-Reduce -> cluster_candidates.json, master_themes.json, theme_mapping_clusters.json)
  - Phase 3   : Global SVO Triple Extraction with Master Theme Association (-> triples/{doc}_triplets.json)
  - Phase 3.5 : Triple Aggregation (-> original_triplets.json, triplet_counts.csv)

All LLM, input file/columns, output locations, chunking, and clustering parameters
are configured as global variables at the top of this file.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Optional, Union

# Ensure repo root and sibling PrismRunner are importable
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_SIBLING_PRISMRUNNER = REPO_ROOT.parent / "PrismRunner"
if _SIBLING_PRISMRUNNER.exists() and str(_SIBLING_PRISMRUNNER) not in sys.path:
    sys.path.insert(0, str(_SIBLING_PRISMRUNNER))

from model_config import (
    DOCUMENT_LIMIT as MASTER_DOCUMENT_LIMIT,
    EMBEDDING_MODEL as MASTER_EMBEDDING_MODEL,
    ID_COLUMN as MASTER_ID_COLUMN,
    INPUT_PATH as MASTER_INPUT_PATH,
    MODEL_NAME as MASTER_MODEL_NAME,
    TEXT_COLUMN as MASTER_TEXT_COLUMN,
)
from src.extraction import (
    Theme,
    ThemeDiscoveryResult,
    ClusterCandidateSynthesis,
    MasterThemeSynthesisResult,
    RawTriple,
    TripleExtractionResult,
    ThemeClusteringEngine,
    PipelineRunContext,
    SemanticPrismStage1Pipeline,
)

# =============================================================================
# GLOBAL CONFIGURATION VARIABLES
# =============================================================================

# --- 1. LLM, Embedding & Hardware Settings (Defaults loaded from model_config.py) ---
MODEL_NAME: str = MASTER_MODEL_NAME                     # Generative LLM for Theme Discovery, Master Synthesis & Triple Extraction
THEME_EMBEDDING_MODEL: str = MASTER_EMBEDDING_MODEL     # SentenceTransformer model for Phase 2 theme clustering
DEVICE: str = "auto"                                    # "auto", "cuda", "mps", or "cpu"
TORCH_DTYPE: str = "auto"                               # "auto", "bfloat16", "float16", or "float32"
LOAD_IN_4BIT: bool = False                              # Enable 4-bit NF4 quantization (CUDA bitsandbytes)
LOAD_IN_8BIT: bool = False                              # Enable 8-bit quantization (CUDA bitsandbytes)
ATTN_IMPLEMENTATION: str = "auto"                       # "auto", "flash_attention_2", "sdpa", or "eager"
BATCH_SIZE: Union[int, str] = "auto"                    # "auto" (hardware-aware) or explicit integer (e.g., 8)
CONTEXT_WINDOW_CAP: Union[int, str] = "auto"            # "auto" (native model context limit) or integer token cap
TEMPERATURE: float = 0.0                                # 0.0 for deterministic greedy decoding
CONSTRAIN_JSON: bool = True                             # Force JSON opening bracket & bracket-balanced early stopping
MANAGE_VRAM: bool = True                                # Purge GPU/MPS cache between major stage transitions
DRY_RUN: bool = False                                   # Set True to preview prompts without running LLM generation

# --- 2. Input Dataset & Output Settings (Defaults loaded from model_config.py) ---
INPUT_PATH: str = MASTER_INPUT_PATH                     # Path to input file (.parquet, .csv, .json, .jsonl) or folder of .txt files
ID_COLUMN: str = MASTER_ID_COLUMN                       # Column containing unique document IDs (SemanticPrism default: "unique_val")
TEXT_COLUMN: str = MASTER_TEXT_COLUMN                   # Column containing raw document text (SemanticPrism default: "report")
DOCUMENT_LIMIT: Optional[int] = MASTER_DOCUMENT_LIMIT   # Optional cap on number of documents to process (None = all)
OUTPUTS_DIR: str = "outputs"                            # Root outputs directory (writes to outputs/01_extraction, outputs/logs, etc.)
SAVE_LLM_PAYLOADS: bool = True                          # Save sample LLM prompts to outputs/payload_text
RESUME_MODE: str = "skip"                               # "skip" (reuse existing per-doc outputs) or "overwrite" (re-run all)
PRINT_CHECKPOINTS: bool = True                          # Print real-time chunk, document, and stage checkpoint progress to console

# --- 3. Chunking, Token & Theme Clustering Settings ---
THEME_CHUNK_MAX_WORDS: int = 2000                       # Max words per chunk during Phase 1 Theme Discovery
TRIPLE_CHUNK_MAX_WORDS: int = 400                       # Max words per chunk during Phase 3 SVO Triple Extraction
THEME_MAX_NEW_TOKENS: int = 512                         # Max generated tokens per theme discovery chunk
MASTER_SYNTHESIS_MAX_NEW_TOKENS: int = 512              # Max generated tokens for cluster map/reduce synthesis
TRIPLE_MAX_NEW_TOKENS: int = 768                        # Max generated tokens per triple extraction chunk
MAP_MAX_THEMES_PER_CLUSTER: int = 150                   # Max themes sent to Map specialist per cluster
MANIFOLD_EDGE_THRESHOLD: float = 0.35                   # Minimum cosine similarity for kNN theme graph edges
MICRO_CLUSTER_RATIO: float = 0.01                       # Clusters smaller than 1% of N are absorbed into nearest dominant centroid


# ==============================================================================
# CLI ENTRYPOINT
# ==============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="SemanticPrism Stage 1: Theme & Triple Extraction")
    parser.add_argument("--dry-run", action="store_true", help="Run pipeline in dry-run mode (mock LLM calls)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Disable resume mode and overwrite existing Stage 1 outputs from scratch",
    )
    parser.add_argument("--model", type=str, default=None, help="Override MODEL_NAME")
    parser.add_argument("--device", type=str, default=None, help="Override DEVICE (e.g., cuda, mps, cpu)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override BATCH_SIZE")
    args = parser.parse_args()

    dry_run = True if args.dry_run else DRY_RUN
    resume_mode = False if args.overwrite else RESUME_MODE
    model_name = args.model if args.model else MODEL_NAME
    device = args.device if args.device else DEVICE
    batch_size = args.batch_size if args.batch_size else BATCH_SIZE

    try:
        pipeline = SemanticPrismStage1Pipeline(
            model_name=model_name,
            theme_embedding_model=THEME_EMBEDDING_MODEL,
            device=device,
            torch_dtype=TORCH_DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
            attn_implementation=ATTN_IMPLEMENTATION,
            batch_size=batch_size,
            context_window_cap=CONTEXT_WINDOW_CAP,
            temperature=TEMPERATURE,
            constrain_json=CONSTRAIN_JSON,
            manage_vram=MANAGE_VRAM,
            dry_run=dry_run,
            input_path=INPUT_PATH,
            id_column=ID_COLUMN,
            text_column=TEXT_COLUMN,
            document_limit=DOCUMENT_LIMIT,
            outputs_dir=OUTPUTS_DIR,
            save_llm_payloads=SAVE_LLM_PAYLOADS,
            resume_mode=resume_mode,
            print_checkpoints=PRINT_CHECKPOINTS,
            theme_chunk_max_words=THEME_CHUNK_MAX_WORDS,
            triple_chunk_max_words=TRIPLE_CHUNK_MAX_WORDS,
            theme_max_new_tokens=THEME_MAX_NEW_TOKENS,
            master_synthesis_max_new_tokens=MASTER_SYNTHESIS_MAX_NEW_TOKENS,
            triple_max_new_tokens=TRIPLE_MAX_NEW_TOKENS,
            map_max_themes_per_cluster=MAP_MAX_THEMES_PER_CLUSTER,
            manifold_edge_threshold=MANIFOLD_EDGE_THRESHOLD,
            micro_cluster_ratio=MICRO_CLUSTER_RATIO,
        )
        pipeline.run()
    except Exception as e:
        print(f"CRITICAL ERROR: Stage 1 Pipeline failed with unhandled exception: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
