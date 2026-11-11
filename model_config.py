"""
SemanticPrism Master Model & Input Configuration (`model_config.py`)
====================================================================
Edit this single file to choose your AI models and point SemanticPrism to your documents.

Quick Guide to the 4 Sections Below:
  1. Hugging Face Token (Optional) : Speeds up model downloads and avoids rate limits.
  2. AI Model Selection            : Picks the AI models based on your computer's GPU memory (VRAM).
  3. Stage 1 Discovery Dataset     : The documents used in Stages 1–4 to discover topics and build schemas.
  4. Stage 5 Extraction Dataset    : The documents used in Stage 5 to extract structured CSV/Parquet/SQLite tables
                                     (can be the exact same dataset as Stage 1, or a separate/larger batch of files).

Choosing Models for Your Hardware (GPU Memory / VRAM):
  - Laptop / Small GPU (8–12 GB)   : Uncomment the "Lightweight" preset in Section 2.
  - Workstation GPU (16–24 GB)     : Uncomment the "Mid-Range" preset in Section 2.
  - Server GPU (48–60 GB)          : Uncomment the "Standard Server" preset in Section 2 (or rely on automatic 4-bit compression).
  - Large Server GPU (80 GB+, default): Uses the default 27B + 32B models as-is.

Automatic Memory Safety:
  If `CODER_MODEL` in Stage 4 is too large for your GPU memory, SemanticPrism automatically
  clears GPU memory, tries a compressed (4-bit) version, or switches to a smaller backup model
  and notes the change in `outputs/pipeline_summary.txt`.
"""

from __future__ import annotations

import os
from typing import Optional

# =============================================================================
# 1. HUGGING FACE HUB AUTHENTICATION (OPTIONAL)
# =============================================================================
# Optional: Paste a Hugging Face token (e.g., "hf_xxxxxxxxxxxxxxxxxxxx") for faster
# model downloads. Leave as None to use public access or your terminal's HF_TOKEN variable.
HF_TOKEN: Optional[str] = None

if HF_TOKEN and HF_TOKEN.strip():
    os.environ["HF_TOKEN"] = HF_TOKEN.strip()
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN.strip()

# =============================================================================
# 2. AI MODEL SELECTION
# =============================================================================
# - MODEL_NAME      : Reads text, finds facts, and extracts structured data (Stages 1, 2 & 5)
# - EMBEDDING_MODEL : Converts concepts into numbers to group similar ideas together (Stages 1, 2 & 3)
# - CODER_MODEL     : Writes clean Python data schemas from discovered topics (Stage 4)

# Default High-Accuracy Server Preset (80 GB+ GPU VRAM, or 48 GB+ with auto 4-bit compression):
MODEL_NAME: str = "Qwen/Qwen3.6-27B"                    # Main text & extraction LLM (~54 GB memory)
EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"         # Concept similarity model (~1.3 GB memory)
CODER_MODEL: str = "Qwen/Qwen2.5-Coder-32B-Instruct"    # Schema code-writing LLM (~65 GB bf16 / ~19 GB 4-bit; auto-falls back if needed)

# -----------------------------------------------------------------------------
# Ready-to-Use Hardware Presets (remove the '#' on the 3 model lines to use):
# -----------------------------------------------------------------------------
# --- Lightweight / Laptop / Single 8–12 GB GPU ---
# MODEL_NAME: str = "Qwen/Qwen2.5-3B-Instruct"
# EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
# CODER_MODEL: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

# --- Mid-Range Workstation / Single 16–24 GB GPU ---
# MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
# EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
# CODER_MODEL: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

# --- Standard Server / 48–60 GB GPU ---
# MODEL_NAME: str = "Qwen/Qwen3.6-27B"
# EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
# CODER_MODEL: str = "Qwen/Qwen2.5-Coder-14B-Instruct"

# =============================================================================
# 3. STAGE 1 DISCOVERY DATASET (USED TO LEARN TOPICS & BUILD SCHEMAS)
# =============================================================================
# Supports .parquet, .csv, .json, .jsonl files, or a folder of plain .txt files.
INPUT_PATH: str = "inputs/testdocs.parquet"             # File or folder of documents to analyze in Stage 1
ID_COLUMN: str = "unique_val"                           # Column name containing each document's unique ID
TEXT_COLUMN: str = "report"                             # Column name containing the document text
DOCUMENT_LIMIT: Optional[int] = None                    # Max documents to read (None = read all documents)

# =============================================================================
# 4. STAGE 5 EXTRACTION DATASET & SCHEMA SETTINGS (USED TO POPULATE TABLES)
# =============================================================================
# Can point to the same dataset as Stage 1 above, or a new/full dataset you want to extract into tables.
STAGE5_INPUT_PATH: str = "inputs/testdocs.parquet"      # File or folder of documents to extract in Stage 5
STAGE5_ID_COLUMN: str = "unique_val"                    # Column name containing each document's unique ID
STAGE5_TEXT_COLUMN: str = "report"                      # Column name containing the document text
STAGE5_DOCUMENT_LIMIT: Optional[int] = None             # Max documents to extract (None = extract all documents)

# Which Python schema file (or folder of .py schema files) Stage 5 should use:
STAGE5_SCHEMA_PATH: str = "outputs/04_synthesis/schemas/master_ontology.py"

# Extraction mode options:
#   - "unified"   (Recommended): Extracts all main schemas together in a single pass per document (fastest & most accurate).
#   - "top_level"              : Extracts each main schema in its own separate pass (automatically includes nested sub-items).
#   - "child"                  : Extracts only nested sub-schemas.
#   - "all"                    : Extracts both main schemas and nested sub-schemas separately.
STAGE5_EXTRACT_MODE: str = "unified"
