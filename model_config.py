"""
SemanticPrism Master Model & Input Configuration
================================================
Centralized configuration file for selecting the generative LLMs, embedding model,
and source document dataset used across the SemanticPrism pipeline:

  - MODEL_NAME      : Generative LLM used in Stage 1 (Extraction), Stage 2 (Refinement), and Stage 5 (Schema Mapping)
  - EMBEDDING_MODEL : SentenceTransformer embedding model used in Stage 1, Stage 2, and Stage 3
  - CODER_MODEL     : Generative Coding LLM used in Stage 4 (Pydantic Schema Synthesis)
  - INPUT_PATH      : Source dataset path (.parquet, .csv, .json, .jsonl, or directory of .txt files)
  - ID_COLUMN       : Column name containing unique document IDs
  - TEXT_COLUMN     : Column name containing raw document text
  - DOCUMENT_LIMIT  : Optional cap on number of documents to process (None = process all)

Recommended Models by GPU VRAM Tier (bfloat16 / float16):
  - 16–24 GB VRAM : MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  (~15 GB) | CODER_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"  (~15 GB)
  - 48–60 GB VRAM : MODEL_NAME = "Qwen/Qwen3.6-27B"          (~54 GB) | CODER_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct" (~29 GB)
  - 80 GB+ VRAM   : MODEL_NAME = "Qwen/Qwen3.6-27B"          (~54 GB) | CODER_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct" (~65 GB)

Note on Automatic Fallback:
  If CODER_MODEL in Stage 4 exceeds available GPU VRAM (or spills weights onto the 'meta'/disk device),
  Stage 4 automatically purges partial GPU allocations, attempts 4-bit quantization (for dense models),
  and falls back to MODEL_NAME (or "Qwen/Qwen2.5-Coder-7B-Instruct"). Any fallback is logged to
  outputs/04_synthesis/logs/model_execution_metadata.json, outputs/04_synthesis/logs/synthesis_community_errors.log,
  and the final outputs/pipeline_summary.txt report.
"""

from __future__ import annotations

from typing import Optional

# =============================================================================
# 1. MODEL SELECTION
# =============================================================================
# Default High-Accuracy Server Configuration (48–60 GB+ GPU VRAM):
MODEL_NAME: str = "Qwen/Qwen3.6-27B"                    # Stages 1, 2 & 5: Theme Discovery, Triple Extraction, Taxonomic Lifting & Schema Mapping (~54 GB bf16)
EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"         # Stages 1, 2 & 3: Vector Clustering & Grounding (~1.3 GB)
CODER_MODEL: str = "Qwen/Qwen2.5-Coder-14B-Instruct"    # Stage 4: Pydantic v2 Schema Synthesis (~29 GB bf16; auto-falls back to MODEL_NAME if VRAM is exceeded)

# -----------------------------------------------------------------------------
# Copy-and-Paste Presets for Smaller Machines (uncomment to override above):
# -----------------------------------------------------------------------------
# --- Lightweight / Laptop / Single 8–12 GB GPU ---
# MODEL_NAME: str = "Qwen/Qwen2.5-3B-Instruct"
# EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
# CODER_MODEL: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

# --- Mid-Range Workstation / Single 16–24 GB GPU ---
# MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
# EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
# CODER_MODEL: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

# --- Ultra-Large Server / 80 GB+ GPU (A100-80GB / H100) ---
# MODEL_NAME: str = "Qwen/Qwen3.6-27B"
# EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
# CODER_MODEL: str = "Qwen/Qwen2.5-Coder-32B-Instruct"

# =============================================================================
# 2. SOURCE DOCUMENT INPUT SETTINGS (STAGES 1 & 5)
# =============================================================================
INPUT_PATH: str = "inputs/testdocs.parquet"             # Path to input file (.parquet, .csv, .json, .jsonl) or folder of .txt files
ID_COLUMN: str = "unique_val"                           # Column containing unique document IDs
TEXT_COLUMN: str = "report"                             # Column containing raw document text
DOCUMENT_LIMIT: Optional[int] = None                    # Optional cap on number of documents to process (None = all)

