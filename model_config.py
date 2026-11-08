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
"""

from __future__ import annotations

from typing import Optional

# =============================================================================
# 1. MODEL SELECTION
# =============================================================================
MODEL_NAME: str = "Qwen/Qwen2.5-3B-Instruct"          # Stages 1, 2 & 5: Theme Discovery, Triple Extraction, Taxonomic Lifting & Schema Mapping (Recommended: "Qwen/Qwen3.6-27B")
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"             # Stages 1, 2 & 3: Vector Clustering & Grounding (Recommended: "BAAI/bge-large-en-v1.5")
CODER_MODEL: str = "Qwen/Qwen2.5-Coder-7B-Instruct"   # Stage 4: Pydantic v2 Schema Synthesis (Recommended: "Qwen/Qwen3-Coder-30B-A3B-Instruct")

# =============================================================================
# 2. SOURCE DOCUMENT INPUT SETTINGS (STAGES 1 & 5)
# =============================================================================
INPUT_PATH: str = "inputs/testdocs.parquet"           # Path to input file (.parquet, .csv, .json, .jsonl) or folder of .txt files
ID_COLUMN: str = "unique_val"                         # Column containing unique document IDs
TEXT_COLUMN: str = "report"                           # Column containing raw document text
DOCUMENT_LIMIT: Optional[int] = None                  # Optional cap on number of documents to process (None = all)

