#!/usr/bin/env python3
"""
SemanticPrism Stage 2: Refinement & Taxonomic Hypernym Lifting (Powered by PrismRunner)
=======================================================================================
Performs the exact Stage 2 Refinement workflow from:
    https://github.com/childmindresearch/SemanticPrism

Workflow Steps:
  - Step 1 : Lexical Normalization & Term Frequency Weighting (from outputs/01_extraction/original_triplets.json)
  - Step 2 : Two-Tier Dimension-Aware SVO Vector Clustering:
             * Tier 1: UMAP manifold reduction + HDBSCAN micro-clustering + Spherical Fréchet Mean centroids
                       + Conservative Outlier Rescue (rescue_tau) + Singleton Protection
             * Tier 2: Cosine Agglomerative Macro-Clustering (tau_macro)
             (-> subject_clusters.json, predicate_clusters.json, object_clusters.json, umap_trustworthiness_*.jpg)
  - Step 3 : Concurrent Batch Taxonomic Hypernym Lifting via PrismRunner (with WAL SQLite resume cache):
             (-> refinement_state.db, subject_taxonomic_map.json, predicate_taxonomic_map.json,
                 object_taxonomic_map.json, taxonomic_map.json)
  - Step 4 : Contextual Sentence Vector Grounding & Edge Fidelity (gamma_e) Computation:
             (-> refined_triplets.json, contextual_grounding_diagnostics.json)

All LLM, embedding, clustering calibration, input/output paths, and lifting toggles are
configured as global variables at the top of this file.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

# Prevent PyTorch CUDA allocator fragmentation when switching between embedding and LLM stages
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", message=r".*n_jobs value.*overridden to 1.*", category=UserWarning)

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

# Ensure repository root is importable when executed directly
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prismrunner import (
    ContextBudget,
    PrismPipeline,
    clear_device_memory,
    get_optimal_device,
    resolve_dynamic_batch_size,
)

# =============================================================================
# GLOBAL CONFIGURATION VARIABLES
# =============================================================================

# 1. LLM, Embedding & Hardware Settings
#    - MODEL_NAME      : Generative Causal LLM used in Step 3 for Taxonomic Hypernym Lifting (e.g., "Qwen/Qwen3.6-27B")
#    - EMBEDDING_MODEL : SentenceTransformer vector encoder used in Step 2 (SVO clustering) & Step 4 (sentence fidelity)
#                        (e.g., "all-MiniLM-L6-v2" [384d] or "BAAI/bge-large-en-v1.5" [1024d])
MODEL_NAME: str = "Qwen/Qwen2.5-3B-Instruct"   # Generative LLM for Step 3 Hypernym Lifting (e.g., "Qwen/Qwen3.6-27B")
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"      # Vector encoder for Step 2 & 4 (e.g., "BAAI/bge-large-en-v1.5")
DEVICE: str = "auto"                           # "auto", "cuda", "mps", or "cpu"
TORCH_DTYPE: str = "auto"                      # "auto", "bfloat16", "float16", or "float32"
LOAD_IN_4BIT: bool = False                     # Enable 4-bit NF4 quantization (CUDA bitsandbytes)
LOAD_IN_8BIT: bool = False                     # Enable 8-bit quantization (CUDA bitsandbytes)
ATTN_IMPLEMENTATION: str = "auto"              # "auto", "flash_attention_2", "sdpa", or "eager"
BATCH_SIZE: Union[int, str] = "auto"           # "auto" (continuous device-relative) or explicit int (e.g., 8)
TEMPERATURE: float = 0.0                       # 0.0 for deterministic greedy decoding
CONSTRAIN_JSON: bool = True                    # Force JSON '{' prefix & bracket-balanced early stopping
MANAGE_VRAM: bool = True                       # Flush and offload GPU/MPS memory between embedding & LLM steps
DRY_RUN: bool = False                          # Set True to preview prompts without running LLM generation

# 2. Input / Output & Execution Settings
INPUT_DIR: str = "outputs/01_extraction"       # Directory containing Stage 1 original_triplets.json & master_themes.json
OUTPUTS_DIR: str = "outputs"                   # Base outputs directory (writes 02_refinement, 02_refinement/cache, logs, payload_text)
SAVE_LLM_PAYLOADS: bool = True                 # Save initial LLM prompts to outputs/payload_text for inspection
RESUME_MODE: str = "skip"                      # "skip" (reuse SQLite cache in refinement_state.db) or "overwrite"
PRINT_CHECKPOINTS: bool = True                 # Print real-time timestamped checkpoint progress to console
PLOT_TRUSTWORTHINESS: bool = True              # Generate UMAP Trustworthiness (T_k) vs n_components .jpg charts

# 3. Context, Token & SVO Taxonomic Lifting Settings
CONTEXT_WINDOW_CAP: Union[int, str] = "auto"   # "auto" (detect from model config via PrismRunner) or integer token cap
LIFT_MAX_NEW_TOKENS: int = 256                 # Max generated tokens per TaxonomicVerification JSON response
LIFT_SUBJECTS: bool = True                     # Abstract entity subjects into canonical hypernyms
LIFT_PREDICATES: bool = False                  # Abstract relational predicates into canonical hypernyms
LIFT_OBJECTS: bool = True                      # Abstract entity objects into canonical hypernyms


# =============================================================================
# 1. SEMANTICPRISM STAGE 2 PYDANTIC SCHEMAS
# =============================================================================

class RawTriple(BaseModel):
    """Represents a single Subject-Predicate-Object triple from Stage 1."""

    subject: str = Field(description="The clear, concise entity acting as the subject.")
    predicate: str = Field(description="The active verb or relationship connecting the subject to the object.")
    object: str = Field(description="The target entity, concept, or condition affected by the predicate.")
    theme_association: str = Field(default="Other", description="Mapped Master Theme title or 'Other'.")
    source_quote: str = Field(default="", description="Verbatim sentence from the source text.")
    source_document: Optional[str] = Field(default=None, description="Identifier of the source document.")

    @model_validator(mode="before")
    @classmethod
    def normalize_triple_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            subj = data.get("subject") or data.get("head") or data.get("source") or data.get("entity_1") or ""
            pred = data.get("predicate") or data.get("relation") or data.get("relationship") or data.get("verb") or ""
            obj = data.get("object") or data.get("tail") or data.get("target") or data.get("entity_2") or ""
            theme = data.get("theme_association") or data.get("theme") or data.get("master_theme") or "Other"
            quote = data.get("source_quote") or data.get("quote") or data.get("evidence") or data.get("sentence") or ""
            out = dict(data)
            out.update({
                "subject": str(subj).strip(),
                "predicate": str(pred).strip(),
                "object": str(obj).strip(),
                "theme_association": str(theme).strip() or "Other",
                "source_quote": str(quote).strip(),
                "source_document": data.get("source_document"),
            })
            return out
        return data


class TaxonomicVerification(BaseModel):
    """
    Validates whether a cluster of terms shares a formal hypernym, forcing the creation of a
    hypernym regardless of heterogeneity, and provides a confidence score of accuracy.
    """

    formal_hypernym: str = Field(
        description="The formal taxonomic hypernym encompassing all terms in the cluster."
    )
    excluded_opposite: str = Field(
        description="A category this hypernym strictly IS NOT, providing axiomatic negative entailment."
    )
    confidence_score: float = Field(
        default=0.8,
        description="A score from 0.0 to 1.0 evaluating how accurately this label represents all members of the cluster.",
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="before")
    @classmethod
    def remap_hypernym_fields(cls, data: Any) -> Any:
        if isinstance(data, str) and data.strip():
            return {
                "formal_hypernym": data.strip(),
                "excluded_opposite": "Unrelated Entity",
                "confidence_score": 0.8,
            }
        if isinstance(data, dict):
            hyp = (
                data.get("formal_hypernym")
                or data.get("hypernym")
                or data.get("canonical_hypernym")
                or data.get("parent_category")
                or data.get("category")
                or data.get("label")
                or data.get("name")
                or ""
            )
            opp = (
                data.get("excluded_opposite")
                or data.get("opposite")
                or data.get("negative_category")
                or data.get("excluded")
                or data.get("not_category")
                or "Unrelated Category"
            )
            conf_raw = (
                data.get("confidence_score")
                if data.get("confidence_score") is not None
                else data.get("confidence", data.get("score", 0.8))
            )
            try:
                conf = float(conf_raw)
                if conf > 1.0 and conf <= 100.0:
                    conf = conf / 100.0
                conf = max(0.0, min(1.0, conf))
            except (ValueError, TypeError):
                conf = 0.8
            return {
                "formal_hypernym": str(hyp).strip(),
                "excluded_opposite": str(opp).strip() or "Unrelated Category",
                "confidence_score": conf,
            }
        return data

    @field_validator("formal_hypernym")
    @classmethod
    def validate_non_empty_hypernym(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("formal_hypernym cannot be empty")
        return v.strip()


# =============================================================================
# 2. SEMANTICPRISM STAGE 2 PROMPTS
# =============================================================================

TAXONOMIC_LIFTING_SYSTEM_PROMPT = """You are a Taxonomy and Lexical Categorization Expert.
You will receive a semantic cluster of related words and their top prototypical fallback terms.
Your task is to identify the lowest common parent category ("Hypernym") that uniformly encompasses all cluster members.

### FIELD INSTRUCTIONS:

1. `formal_hypernym` (string):
   - Provide the **Lowest Common Ancestor (LCA)** category (1 to 3 words, Title Case noun).
   - Do NOT over-generalize or skip hierarchy levels:
     * `["teacher", "professor"]` -> `Educator` (NOT `Person`)
     * `["postgreSQL", "MySQL"]` -> `Relational Database` (NOT `Software`)
     * `["car", "truck", "van"]` -> `Motor Vehicle` (NOT `Physical Object`)
   - **Centroid Anchoring**: If members are near-synonyms or variants of the centroid fallbacks (e.g. `["person", "persons", "individual"]`), keep the hypernym tightly anchored to the top fallback (`Person`).
   - **Generic Preservation**: Keep generic terms generic (`person` -> `Person`, NOT domain jargon like `Clinical Subject`).

2. `excluded_opposite` (string):
   - Name a plausible, distinct category in the same domain that this cluster strictly is NOT (axiomatic negative boundary).
   - Examples:
     * For `Motor Vehicle` -> `Aircraft` or `Pedestrian`
     * For `Relational Database` -> `Document Store` or `Operating System`
     * For `Pharmacological Agent` -> `Diagnostic Procedure`

3. `confidence_score` (float: 0.0 to 1.0):
   - `0.80 - 1.00`: Cluster members are tightly cohesive and cleanly fit a single immediate parent.
   - `0.50 - 0.79`: Cluster members are moderately diverse but share a clear functional category.
   - `0.00 - 0.49`: Cluster members are noisy or heterogeneous; relying heavily on centroid fallbacks.

### OUTPUT RULES:
- Output pure JSON matching the requested schema.
- Do NOT output schema definitions, markdown explanations, or commentary."""

TAXONOMIC_LIFTING_USER_PROMPT = """Analyze the following cluster of terms and deduce their formal taxonomic hypernym:

{payload_json}"""

PREDICATE_LIFTING_SYSTEM_PROMPT = """You are a Taxonomy and Relational Verb Categorization Expert.
You will receive a semantic cluster of related relational predicates (verbs/verb phrases) and their top prototypical fallback terms.
Your task is to identify the lowest common parent relational predicate ("Hypernym") that uniformly encompasses all cluster members.

### FIELD INSTRUCTIONS:

1. `formal_hypernym` (string):
   - Provide the **Lowest Common Ancestor (LCA)** active relational verb phrase (1 to 3 words, lowercase active verb phrase).
   - Do NOT nominalize verbs into nouns and do NOT over-generalize into vague verbs like `relates to` or `affects`:
     * `["administered", "dosed", "prescribed"]` -> `administers` (NOT `Medication Administration` or `affects`)
     * `["depends on", "relies upon", "requires"]` -> `requires` (NOT `Dependency`)
     * `["monitored", "tracked", "observed"]` -> `monitors` (NOT `Observation`)
   - **Centroid Anchoring**: If members are near-synonyms or inflectional variants of the centroid fallbacks, keep the predicate tightly anchored to the active present-tense form of the top fallback.

2. `excluded_opposite` (string):
   - Name a plausible, distinct relational predicate in the same domain that this cluster strictly is NOT (axiomatic negative boundary).

3. `confidence_score` (float: 0.0 to 1.0):
   - `0.80 - 1.00`: Cluster members are tightly cohesive and cleanly fit a single relational predicate.
   - `0.50 - 0.79`: Cluster members are moderately diverse but share a clear relational meaning.
   - `0.00 - 0.49`: Cluster members are noisy or heterogeneous; relying heavily on centroid fallbacks.

### OUTPUT RULES:
- Output pure JSON matching the requested schema.
- Do NOT output schema definitions, markdown explanations, or commentary."""


# =============================================================================
# 3. STAGE 2 PIPELINE CONTEXT & REFINEMENT ORCHESTRATOR
# =============================================================================

class PipelineRunContext:
    """Stores global state safely for the Stage 2 Refinement pipeline."""

    def __init__(self, master_domain: str = "General Subjects"):
        self.master_domain = master_domain
        self.term_frequencies: Dict[str, int] = defaultdict(int)


class RefinementPipeline:
    """
    Executes the complete SemanticPrism Stage 2 Refinement workflow (two-tier vector clustering,
    taxonomic hypernym lifting via PrismRunner, and contextual sentence fidelity grounding).
    """

    def __init__(
        self,
        context: Optional[PipelineRunContext] = None,
        prism_pipeline: Optional[PrismPipeline] = None,
    ):
        base_out = Path(OUTPUTS_DIR)
        self.in_dir = Path(INPUT_DIR)
        self.out_dir = base_out / "02_refinement"
        self.cache_dir = self.out_dir / "cache"
        self.log_dir = base_out / "logs"
        self.payload_dir = base_out / "payload_text"

        for d in (self.out_dir, self.cache_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)
        if SAVE_LLM_PAYLOADS:
            self.payload_dir.mkdir(parents=True, exist_ok=True)

        resolved_domain = self._resolve_master_domain()
        self.context = context or PipelineRunContext(master_domain=resolved_domain)
        if not self.context.master_domain or self.context.master_domain == "auto":
            self.context.master_domain = resolved_domain

        self.embedding_model = None
        self.embedding_model_name = EMBEDDING_MODEL

        # Guard against putting an encoder-only embedding model into MODEL_NAME instead of EMBEDDING_MODEL
        _lower_llm = str(MODEL_NAME).lower()
        if any(pat in _lower_llm for pat in ("bge-", "minilm", "mpnet", "sentence-transformers/", "e5-", "nomic-embed")):
            raise ValueError(
                f"MODEL_NAME is set to '{MODEL_NAME}', which is an encoder-only embedding model (not a generative LLM).\n"
                f"In SemanticPrism2.py:\n"
                f"  - Set EMBEDDING_MODEL = \"{MODEL_NAME}\"  (for Step 2 SVO clustering & Step 4 sentence vectors)\n"
                f"  - Set MODEL_NAME      = \"Qwen/Qwen3.6-27B\" (or another generative LLM for Step 3 Taxonomic Lifting)"
            )

        # Clear previous error log if starting fresh
        log_file = self.log_dir / "stage_02_refinement_errors.json"
        if log_file.exists() and RESUME_MODE != "skip":
            log_file.unlink()

        self.resume = str(RESUME_MODE).lower().strip() == "skip"
        mode_str = "skip (resume existing cache)" if self.resume else "overwrite (clear cache & restart)"
        print(f"[Refinement] Pipeline Resume Mode: {mode_str}\n")

        self.db_path = self.out_dir / "refinement_state.db"
        if not self.resume and self.db_path.exists():
            print(f"[Refinement] Starting fresh run: deleting existing cache database at {self.db_path}")
            try:
                self.db_path.unlink()
                for ext in (".db-wal", ".db-shm"):
                    extra_file = Path(str(self.db_path) + ext)
                    if extra_file.exists():
                        extra_file.unlink()
            except Exception as e:
                print(f"[Refinement] Warning: Could not delete cache database: {e}")

        self._db_lock = threading.Lock()
        self._init_db()

        if prism_pipeline is not None:
            self.prism = prism_pipeline
        else:
            self.prism = PrismPipeline(
                model_name=MODEL_NAME,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                config={
                    "model": {
                        "name": MODEL_NAME,
                        "device": DEVICE,
                        "torch_dtype": TORCH_DTYPE,
                        "load_in_4bit": LOAD_IN_4BIT,
                        "load_in_8bit": LOAD_IN_8BIT,
                        "attn_implementation": ATTN_IMPLEMENTATION,
                    },
                    "inference": {
                        "batch_size": BATCH_SIZE,
                        "temperature": TEMPERATURE,
                        "constrain_json": CONSTRAIN_JSON,
                    },
                },
            )

    def _resolve_master_domain(self) -> str:
        """Resolve master_domain from INPUT_DIR/master_themes.json (falling back to 'General Subjects')."""
        candidates = [self.in_dir / "master_themes.json", REPO_ROOT / self.in_dir / "master_themes.json"]
        for master_path in candidates:
            if master_path.exists():
                try:
                    data = json.loads(master_path.read_text(encoding="utf-8"))
                    dom = data.get("master_domain")
                    if dom and str(dom).strip():
                        return str(dom).strip()
                except Exception:
                    pass
        return "General Subjects"

    @staticmethod
    def _log_checkpoint(phase: str, message: str) -> None:
        """Print timestamped checkpoint progress to the console when PRINT_CHECKPOINTS is enabled."""
        if PRINT_CHECKPOINTS:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts} | Checkpoint | {phase}] {message}", flush=True)

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS taxonomic_lifting (
                    cluster_key TEXT PRIMARY KEY,
                    mapped_term TEXT
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _db_get_cached_lift(self, cluster_key: str) -> Optional[str]:
        with self._db_lock:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            try:
                cur = conn.cursor()
                cur.execute("SELECT mapped_term FROM taxonomic_lifting WHERE cluster_key = ?", (cluster_key,))
                row = cur.fetchone()
                return str(row[0]) if (row and row[0]) else None
            finally:
                conn.close()

    def _db_save_lift(self, cluster_key: str, mapped_term: str) -> None:
        with self._db_lock:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute(
                    "INSERT OR REPLACE INTO taxonomic_lifting (cluster_key, mapped_term) VALUES (?, ?)",
                    (cluster_key, mapped_term),
                )
                conn.commit()
            finally:
                conn.close()

    def _log_error(self, record: Dict[str, Any]) -> None:
        """Append error records to outputs/logs/stage_02_refinement_errors.json."""
        try:
            log_path = self.log_dir / "stage_02_refinement_errors.json"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[Refinement] Failed to write error log: {e}")

    def _save_payload_text(self, filename: str, content: str) -> None:
        """Export initial LLM prompt payload to outputs/payload_text/ when SAVE_LLM_PAYLOADS is enabled."""
        if not SAVE_LLM_PAYLOADS:
            return
        self.payload_dir.mkdir(parents=True, exist_ok=True)
        target_path = self.payload_dir / filename
        if target_path.exists():
            return
        try:
            target_path.write_text(content, encoding="utf-8")
        except Exception as e:
            print(f"Warning: Failed to save payload text file {filename}: {e}")

    @staticmethod
    def _nlp_preprocess(text: str) -> str:
        """Lexical normalization matching SemanticPrism Stage 2."""
        if not text:
            return text
        original = text
        text = text.replace("_", " ")
        text = re.sub(r"[<>/\\|\[\]{}]", "", text)
        text = text.lower()
        cleaned = " ".join(text.split())
        return cleaned if cleaned else original

    # =========================================================================
    # EMBEDDING MODEL MANAGEMENT & MODEL-AWARE VECTOR CACHING
    # =========================================================================

    def _load_embedding_model(self) -> None:
        if self.embedding_model is not None:
            return
        from sentence_transformers import SentenceTransformer

        models_dir = Path("models/embeddings")
        models_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                cache_folder=str(models_dir),
                local_files_only=True,
            )
        except Exception:
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                cache_folder=str(models_dir),
                local_files_only=False,
            )

    def _unload_embedding_model(self) -> None:
        """Offload and delete SentenceTransformer from GPU/MPS memory before LLM inference."""
        if self.embedding_model is not None:
            try:
                self.embedding_model.to("cpu")
            except Exception:
                pass
            del self.embedding_model
            self.embedding_model = None
        if MANAGE_VRAM:
            clear_device_memory()

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode and L2-normalize texts using SentenceTransformer (with TF-IDF fallback)."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        try:
            import torch

            self._load_embedding_model()
            emb_batch_size = resolve_dynamic_batch_size(
                device=get_optimal_device(DEVICE),
                requested_batch_size=BATCH_SIZE,
                items=[{"prompt": t} for t in texts],
            )
            with torch.inference_mode():
                embs = self.embedding_model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=emb_batch_size,
                    convert_to_numpy=True,
                )
            return np.asarray(embs, dtype=np.float32)
        except Exception as e:
            print(f"   -> Warning: SentenceTransformer fallback triggered ({e}); using normalized TF-IDF vectors.")
            vocab: Dict[str, int] = {}
            tokenized: List[List[str]] = []
            for t in texts:
                words = re.findall(r"[a-z0-9]{2,}", t.lower())
                tokenized.append(words)
                for w in set(words):
                    vocab[w] = vocab.get(w, 0) + 1
            terms = sorted(vocab.keys())
            if not terms:
                return np.eye(max(1, len(texts)), dtype=np.float32)
            term_idx = {w: i for i, w in enumerate(terms)}
            n_docs = len(texts)
            mat = np.zeros((n_docs, len(terms)), dtype=np.float32)
            for d_i, words in enumerate(tokenized):
                for w in words:
                    if w in term_idx:
                        mat[d_i, term_idx[w]] += 1.0
            idf = np.log((1.0 + n_docs) / (1.0 + np.array([vocab[w] for w in terms], dtype=np.float32))) + 1.0
            mat = mat * idf
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return mat / norms

    def get_or_compute_embeddings(self, terms: List[str], component: str) -> np.ndarray:
        """
        Model-aware vector cache with metadata validation.
        Prevents dimension mismatch collisions when switching embedding backbones.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        model_slug = re.sub(r"[^a-zA-Z0-9_]", "_", self.embedding_model_name)
        comp_key = component.lower()
        npy_path = self.cache_dir / f"{comp_key}_{model_slug}_embeddings.npy"
        meta_path = self.cache_dir / f"{comp_key}_{model_slug}_meta.json"

        if npy_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if meta.get("model_name") == self.embedding_model_name and meta.get("vocab") == terms:
                    embs = np.load(npy_path)
                    if embs.shape[0] == len(terms):
                        print(f"   -> [{component}] Loaded {len(terms)} cached vectors ({embs.shape[1]}d) from {npy_path.name}")
                        return embs
            except Exception as e:
                print(f"   -> [{component}] Cache read warning ({e}); recomputing embeddings...")

        embs = self._encode_texts(terms)
        try:
            np.save(npy_path, embs)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_name": self.embedding_model_name,
                        "dimensions": int(embs.shape[1]),
                        "vocab": terms,
                    },
                    f,
                    indent=2,
                )
            print(f"   -> [{component}] Computed and cached {len(terms)} vectors ({embs.shape[1]}d) to {npy_path.name}")
        except Exception as e:
            print(f"   -> [{component}] Warning: Failed to write vector cache: {e}")

        return embs

    # =========================================================================
    # STEP 2: TWO-TIER DIMENSION-AWARE SVO VECTOR CLUSTERING
    # =========================================================================

    @staticmethod
    def resolve_clustering_params(embedding_dim: int, component: str = "subject") -> Dict[str, Any]:
        """
        Dynamically resolve clustering hyperparameters based on embedding dimensionality D
        and component role ('entity' vs 'predicate').
        """
        is_pred = component.lower() == "predicate"
        if embedding_dim > 768:
            # High-dimensional embeddings (D > 768, e.g., BAAI/bge-large-en-v1.5 1024d)
            return {
                "umap_n_neighbors": 15,
                "umap_min_dist": 0.05,
                "hdbscan_min_samples": 2,
                "hdbscan_selection": "leaf",
                "macro_linkage": "complete",
                "rescue_tau": 0.15 if is_pred else 0.18,
                "tau_macro": 0.18 if is_pred else 0.22,
            }
        elif embedding_dim > 512:
            # Mid-dimensional embeddings (512 < D <= 768, e.g., all-mpnet-base-v2 768d)
            return {
                "umap_n_neighbors": 15,
                "umap_min_dist": 0.03,
                "hdbscan_min_samples": 2,
                "hdbscan_selection": "eom",
                "macro_linkage": "average",
                "rescue_tau": 0.22 if is_pred else 0.26,
                "tau_macro": 0.30 if is_pred else 0.36,
            }
        else:
            # Low-dimensional embeddings (D <= 512, e.g., all-MiniLM-L6-v2 384d)
            return {
                "umap_n_neighbors": 20,
                "umap_min_dist": 0.00,
                "hdbscan_min_samples": 1,
                "hdbscan_selection": "eom",
                "macro_linkage": "average",
                "rescue_tau": 0.28 if is_pred else 0.32,
                "tau_macro": 0.38 if is_pred else 0.48,
            }

    @staticmethod
    def compute_spherical_centroid(
        member_indices: List[int],
        embeddings: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the frequency-weighted Spherical Fréchet Mean projected onto the unit hypersphere:
        c = sum(w_i * v_i) / ||sum(w_i * v_i)||_2
        """
        vecs = embeddings[member_indices]
        w = weights[member_indices]
        w_sum = np.sum(vecs * w[:, None], axis=0)
        norm = np.linalg.norm(w_sum)
        return w_sum / norm if norm > 0 else w_sum

    @staticmethod
    def _compute_loo_distances(
        vecs: np.ndarray,
        weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the frequency-weighted unit centroid and Leave-One-Out (LOO) cosine distances
        for each member vector to eliminate self-bias in small or skewed-frequency clusters.
        """
        n = len(vecs)
        w = np.asarray(weights, dtype=np.float64)
        weighted_vecs = vecs * w[:, None]
        total_vec = np.sum(weighted_vecs, axis=0)
        c_norm = np.linalg.norm(total_vec)
        c_unit = (total_vec / c_norm if c_norm > 0 else total_vec).astype(np.float32)

        if n <= 1:
            return c_unit, np.zeros(n, dtype=np.float32)

        loo_vecs = total_vec[None, :] - weighted_vecs
        loo_norms = np.linalg.norm(loo_vecs, axis=1, keepdims=True)
        valid = loo_norms.squeeze(-1) > 1e-8
        loo_unit = np.zeros_like(vecs, dtype=np.float32)
        loo_unit[valid] = (loo_vecs[valid] / loo_norms[valid]).astype(np.float32)
        loo_unit[~valid] = c_unit

        loo_sims = np.sum(vecs * loo_unit, axis=1)
        loo_dists = np.clip(1.0 - loo_sims, 0.0, 2.0).astype(np.float32)
        return c_unit, loo_dists

    @staticmethod
    def _select_mmr_order(
        vecs: np.ndarray,
        loo_distances: np.ndarray,
        seed_indices: List[int],
        lambda_param: float = 0.50,
    ) -> List[int]:
        """
        Order cluster member indices via Maximal Marginal Relevance (MMR) starting from
        seed_indices (top centroid fallbacks) so LLM payloads span both the prototypical
        core and the diverse semantic boundary of the cluster.
        """
        n = len(vecs)
        if n <= len(seed_indices):
            return list(range(n))

        centrality = np.clip(1.0 - loo_distances, -1.0, 1.0)
        selected: List[int] = []
        remaining: List[int] = list(range(n))

        for s_idx in seed_indices:
            if 0 <= s_idx < n and s_idx not in selected:
                selected.append(s_idx)
                remaining.remove(s_idx)

        sim_matrix = np.clip(np.dot(vecs, vecs.T), -1.0, 1.0)
        while remaining:
            rem_arr = np.array(remaining, dtype=int)
            if not selected:
                best_pos = int(np.argmax(centrality[rem_arr]))
            else:
                sel_arr = np.array(selected, dtype=int)
                max_sim_to_sel = np.max(sim_matrix[np.ix_(rem_arr, sel_arr)], axis=1)
                mmr_scores = lambda_param * centrality[rem_arr] - (1.0 - lambda_param) * max_sim_to_sel
                best_pos = int(np.argmax(mmr_scores))
            chosen = remaining.pop(best_pos)
            selected.append(chosen)

        return selected

    @staticmethod
    def _calibrate_distance_thresholds(
        embeddings_l2: np.ndarray,
        prior_rescue: float,
        prior_macro: float,
    ) -> Tuple[float, float]:
        """
        Self-calibrate outlier rescue_tau and macro-agglomeration tau_macro against the
        empirical 1-NN and background pairwise cosine distance distribution of the active
        embedding model (preventing macro-collapse on narrow-cone anisotropic models).
        """
        n = int(embeddings_l2.shape[0])
        if n < 6:
            return prior_rescue, prior_macro

        sample_n = min(n, 512)
        if sample_n < n:
            sample_idx = np.linspace(0, n - 1, sample_n, dtype=int)
            sample_vecs = embeddings_l2[sample_idx]
        else:
            sample_vecs = embeddings_l2

        sim_mat = np.clip(np.dot(sample_vecs, sample_vecs.T), -1.0, 1.0)
        dist_mat = np.clip(1.0 - sim_mat, 0.0, 2.0)

        # 1-NN distance distribution
        np.fill_diagonal(dist_mat, np.inf)
        nn_dists = np.min(dist_mat, axis=1)
        p15_nn, p35_nn, p60_nn = np.percentile(nn_dists, [15, 35, 60])

        # Background off-diagonal distance distribution
        triu_idx = np.triu_indices(sample_n, k=1)
        bg_dists = dist_mat[triu_idx]
        mu_d = float(np.mean(bg_dists))
        sigma_d = float(np.std(bg_dists))

        emp_rescue = min(mu_d - 1.0 * sigma_d, float(p35_nn))
        min_rescue = max(0.04, 0.50 * prior_rescue)
        max_rescue = max(min_rescue, min(1.35 * prior_rescue, mu_d - 0.75 * sigma_d))
        eff_rescue = float(np.clip(emp_rescue, min_rescue, max_rescue))

        emp_macro = min(mu_d - 0.65 * sigma_d, float(p60_nn))
        min_macro = max(eff_rescue * 1.15, 0.50 * prior_macro)
        max_macro = max(min_macro, min(1.30 * prior_macro, mu_d - 0.50 * sigma_d))
        eff_macro = float(np.clip(emp_macro, min_macro, max_macro))

        return round(eff_rescue, 4), round(eff_macro, 4)

    def _cluster_field(
        self,
        terms_list: List[str],
        label_prefix: str,
        normalized_frequencies: Dict[str, int],
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray], Optional[List[str]]]:
        if not terms_list:
            return [], None, None

        num_terms = len(terms_list)
        term_to_idx = {t: i for i, t in enumerate(terms_list)}

        # 1. Obtain L2-normalized embeddings via model-aware vector cache
        term_embeddings_l2 = self.get_or_compute_embeddings(terms_list, label_prefix)
        dim = int(term_embeddings_l2.shape[1])

        # 2. Resolve dimension- and role-aware clustering hyperparameters with anisotropy self-calibration
        params = self.resolve_clustering_params(dim, label_prefix)
        eff_rescue, eff_macro = self._calibrate_distance_thresholds(
            term_embeddings_l2,
            params["rescue_tau"],
            params["tau_macro"],
        )
        params["rescue_tau"] = eff_rescue
        params["tau_macro"] = eff_macro
        weights = np.array([1.0 + math.log(normalized_frequencies.get(t, 1)) for t in terms_list])

        # Resolve UMAP and HDBSCAN backends
        umap_mod = None
        try:
            import umap as _umap
            umap_mod = _umap
        except Exception:
            umap_mod = None

        hdbscan_cls = None
        try:
            import hdbscan as _hdbscan
            hdbscan_cls = _hdbscan.HDBSCAN
        except Exception:
            try:
                from sklearn.cluster import HDBSCAN as _sk_hdbscan
                hdbscan_cls = _sk_hdbscan
            except Exception:
                hdbscan_cls = None

        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.manifold import trustworthiness
        except Exception:
            AgglomerativeClustering = None
            trustworthiness = None

        if num_terms < 4 or hdbscan_cls is None:
            cluster_records = []
            for idx, t in enumerate(terms_list):
                c_vec = term_embeddings_l2[idx]
                rec: Dict[str, Any] = {
                    "cluster_number": idx,
                    "centroid_term": t,
                    "members": [t],
                    "centroid_vector": c_vec.tolist(),
                }
                cluster_records.append(rec)
            return cluster_records, term_embeddings_l2, terms_list

        # 3. Vocabulary-scale-adaptive guardrails for UMAP reduction
        adaptive_k = int(2 + 2.5 * math.log(max(2, num_terms)))
        max_local_k = max(3, num_terms // 4) if num_terms >= 12 else max(2, num_terms - 1)
        n_neighbors = int(min(params["umap_n_neighbors"], max(2, min(adaptive_k, max_local_k, num_terms - 1))))
        adaptive_comp = int(max(2, min(8, round(math.log(max(4, num_terms))))))
        n_components = min(adaptive_comp, max(2, num_terms - 2))
        min_dist = params["umap_min_dist"]
        metric = "cosine"
        random_state = 42

        # Optional UMAP Trustworthiness evaluation plot (subsampled at <= 1,000 terms to avoid O(N^2) bottlenecks)
        if PLOT_TRUSTWORTHINESS and umap_mod is not None and trustworthiness is not None and num_terms >= 4:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                if num_terms > 1000:
                    eval_idx = np.linspace(0, num_terms - 1, 1000, dtype=int)
                    eval_embeddings = term_embeddings_l2[eval_idx]
                else:
                    eval_embeddings = term_embeddings_l2
                eval_n = int(eval_embeddings.shape[0])

                dim_candidates = [2, 3, 5, 8, 10, 15, 20, 30]
                valid_dims = sorted(list(set([d for d in dim_candidates if d < eval_n - 1] + [n_components])))
                trust_scores = []
                eval_neighbors = min(n_neighbors, max(1, (eval_n // 2) - 1))
                for d in valid_dims:
                    reducer_eval = umap_mod.UMAP(
                        n_components=d,
                        n_neighbors=min(n_neighbors, eval_n - 1),
                        min_dist=min_dist,
                        metric=metric,
                        random_state=random_state,
                        n_jobs=1,
                    )
                    embedded_eval = reducer_eval.fit_transform(eval_embeddings)
                    score = trustworthiness(eval_embeddings, embedded_eval, n_neighbors=eval_neighbors, metric=metric)
                    trust_scores.append(score)

                plt.figure(figsize=(8, 5))
                plt.plot(valid_dims, trust_scores, marker="o", linewidth=2, color="#2b5c8f", label="Trustworthiness Score")
                if n_components in valid_dims:
                    target_score = trust_scores[valid_dims.index(n_components)]
                    plt.plot(
                        n_components,
                        target_score,
                        marker="*",
                        markersize=14,
                        color="#d9534f",
                        label=f"Selected n_components={n_components}",
                    )
                plt.title(f"UMAP Trustworthiness Evaluation - {label_prefix} ({num_terms} terms)", fontsize=12, fontweight="bold")
                plt.xlabel("Dimensions (n_components)", fontsize=10)
                plt.ylabel("Trustworthiness Score [0 - 1]", fontsize=10)
                plt.ylim([0.0, 1.05])
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.legend(loc="lower right")
                plt.tight_layout()
                plot_path = self.out_dir / f"umap_trustworthiness_{label_prefix}.jpg"
                plt.savefig(plot_path, dpi=300)
                plt.close()
                print(f"   -> [{label_prefix}] Exported UMAP Trustworthiness plot to: {plot_path}")
            except Exception as pe:
                print(f"   -> [{label_prefix}] Warning: Failed to render trustworthiness plot: {pe}")

        # 4. TIER 1: Dense Micro-Clustering & Legitimate Singleton Protection
        if umap_mod is not None:
            reducer = umap_mod.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=random_state,
                n_jobs=1,
            )
            space_reduced = reducer.fit_transform(term_embeddings_l2)
        else:
            from sklearn.decomposition import TruncatedSVD

            svd = TruncatedSVD(n_components=min(n_components, max(1, dim - 1)), random_state=random_state)
            space_reduced = svd.fit_transform(term_embeddings_l2)

        hdb = hdbscan_cls(
            min_cluster_size=2,
            min_samples=params["hdbscan_min_samples"],
            metric="euclidean",
            cluster_selection_method=params["hdbscan_selection"],
        )
        labels_raw = hdb.fit_predict(space_reduced)

        raw_clusters: Dict[int, List[str]] = defaultdict(list)
        outliers: List[str] = []
        for t, lbl in zip(terms_list, labels_raw):
            if int(lbl) == -1:
                outliers.append(t)
            else:
                raw_clusters[int(lbl)].append(t)

        # Compute Spherical Fréchet Centroids for micro-clusters
        micro_centroids: Dict[int, np.ndarray] = {}
        for lbl, mems in raw_clusters.items():
            m_idx = [term_to_idx[t] for t in mems]
            micro_centroids[lbl] = self.compute_spherical_centroid(m_idx, term_embeddings_l2, weights)

        # Conservative Outlier Rescue (isolate legitimate singletons)
        singletons: List[str] = []
        if outliers and micro_centroids:
            c_lbls = list(micro_centroids.keys())
            c_matrix = np.array([micro_centroids[lbl] for lbl in c_lbls])
            out_idx = [term_to_idx[t] for t in outliers]
            out_mat = term_embeddings_l2[out_idx]

            dists = 1.0 - np.dot(out_mat, c_matrix.T)
            best_c_idx = np.argmin(dists, axis=1)
            min_dists = np.min(dists, axis=1)

            for i, out_t in enumerate(outliers):
                if min_dists[i] <= params["rescue_tau"]:
                    raw_clusters[c_lbls[int(best_c_idx[i])]].append(out_t)
                else:
                    singletons.append(out_t)
        else:
            singletons.extend(outliers)

        # Split into multi-member micro-clusters and true singletons
        multi_micro_clusters: Dict[int, List[str]] = {}
        for lbl, mems in raw_clusters.items():
            if len(mems) > 1:
                multi_micro_clusters[lbl] = mems
            else:
                singletons.extend(mems)

        # 5. TIER 2: Macro-Category Agglomeration
        final_clusters: Dict[int, List[str]] = defaultdict(list)
        if multi_micro_clusters:
            micro_lbls = list(multi_micro_clusters.keys())
            micro_cents = []
            for lbl in micro_lbls:
                m_idx = [term_to_idx[t] for t in multi_micro_clusters[lbl]]
                micro_cents.append(self.compute_spherical_centroid(m_idx, term_embeddings_l2, weights))
            micro_cents_mat = np.array(micro_cents)

            if len(micro_lbls) > 1 and AgglomerativeClustering is not None:
                agg = AgglomerativeClustering(
                    metric="cosine",
                    linkage=params["macro_linkage"],
                    distance_threshold=params["tau_macro"],
                    n_clusters=None,
                )
                macro_partitions = agg.fit_predict(micro_cents_mat)
                for micro_lbl, macro_cid in zip(micro_lbls, macro_partitions):
                    final_clusters[int(macro_cid)].extend(multi_micro_clusters[micro_lbl])
            else:
                for i, micro_lbl in enumerate(micro_lbls):
                    final_clusters[i].extend(multi_micro_clusters[micro_lbl])

        # Append protected singletons
        next_id = max(final_clusters.keys()) + 1 if final_clusters else 0
        for s_term in singletons:
            final_clusters[next_id] = [s_term]
            next_id += 1

        # 6. Determine Exemplar Terms (via LOO self-bias correction) & Build Output Records
        cluster_records: List[Dict[str, Any]] = []
        for cid, mems in final_clusters.items():
            m_sorted = sorted(list(set(mems)))
            if len(m_sorted) == 1:
                c_vec = term_embeddings_l2[term_to_idx[m_sorted[0]]]
                ex_term = m_sorted[0]
            else:
                m_idx = [term_to_idx[t] for t in m_sorted]
                c_vec, loo_dists = self._compute_loo_distances(term_embeddings_l2[m_idx], weights[m_idx])
                ex_term = m_sorted[int(np.argmin(loo_dists))]

            rec = {
                "cluster_number": int(cid),
                "centroid_term": ex_term,
                "members": m_sorted,
                "centroid_vector": c_vec.tolist(),
            }
            cluster_records.append(rec)

        cluster_records.sort(key=lambda c: len(c["members"]), reverse=True)
        for idx_rec, rec in enumerate(cluster_records):
            rec["cluster_number"] = idx_rec

        return cluster_records, term_embeddings_l2, terms_list

    # =========================================================================
    # STEP 3: CONCURRENT BATCH TAXONOMIC HYPERNYM LIFTING (PRISMRUNNER)
    # =========================================================================

    def _apply_context_window_cap(self) -> int:
        native_ctx = getattr(self.prism, "context_window", 4096) or 4096
        if isinstance(CONTEXT_WINDOW_CAP, int) and CONTEXT_WINDOW_CAP > 0:
            effective_ctx = min(native_ctx, CONTEXT_WINDOW_CAP)
        else:
            effective_ctx = native_ctx
        self.prism.context_window = effective_ctx
        if self.prism.engine is not None:
            self.prism.engine.context_window = effective_ctx
        return effective_ctx

    def _lift_component_clusters(
        self,
        clusters: List[Dict[str, Any]],
        term_embeddings_l2: Optional[np.ndarray],
        terms_list: Optional[List[str]],
        label_prefix: str,
        normalized_frequencies: Dict[str, int],
        dry_run: bool = DRY_RUN,
    ) -> Dict[str, str]:
        """
        Resolve canonical taxonomic hypernyms for all clusters of a component ('Subject', 'Predicate', 'Object').
        - Single-member clusters resolve locally without LLM calls.
        - Cached clusters in refinement_state.db resolve immediately when RESUME_MODE == 'skip'.
        - Remaining multi-member clusters are lifted concurrently in hardware-sized batches via PrismRunner.
        """
        target_map: Dict[str, str] = {}
        if not clusters or term_embeddings_l2 is None or not terms_list:
            return target_map

        term_to_idx = {t: i for i, t in enumerate(terms_list)}
        base_sys_prompt = (
            PREDICATE_LIFTING_SYSTEM_PROMPT
            if label_prefix.lower() == "predicate"
            else TAXONOMIC_LIFTING_SYSTEM_PROMPT
        )
        full_sys_prompt = f"{base_sys_prompt}\nDomain Context: {self.context.master_domain}"

        if not dry_run and self.prism.engine is None:
            self._log_checkpoint(f"Step 3: {label_prefix} Lift", f"Loading LLM '{MODEL_NAME}' onto device...")
            self.prism.init_engine(
                batch_size=BATCH_SIZE,
                max_new_tokens=LIFT_MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
            )
            eff_ctx = self._apply_context_window_cap()
            self._log_checkpoint(
                f"Step 3: {label_prefix} Lift",
                f"Model ready on device '{self.prism.device}' (effective context window: {eff_ctx} tokens)",
            )

        effective_ctx = self._apply_context_window_cap()
        try:
            budget = ContextBudget.from_tokenizer(
                tokenizer=self.prism.tokenizer,
                context_limit=effective_ctx,
                system_prompt=full_sys_prompt,
                template_text=TAXONOMIC_LIFTING_USER_PROMPT.format(payload_json="[]"),
                max_new_tokens=LIFT_MAX_NEW_TOKENS,
            )
            max_payload_tokens = budget.max_input_tokens
        except ValueError:
            max_payload_tokens = max(256, effective_ctx // 2)

        def _count_tokens(txt: str) -> int:
            if self.prism.tokenizer is not None:
                try:
                    ids = self.prism.tokenizer.encode(txt, add_special_tokens=False)
                    if isinstance(ids, list):
                        return len(ids)
                except Exception:
                    pass
            return max(1, int(len(txt.split()) * 1.35))

        pending_rows: List[Dict[str, Any]] = []
        pending_meta_by_uid: Dict[str, Dict[str, Any]] = {}

        for cluster in clusters:
            original_members = list(cluster["members"])
            cluster_terms = list(original_members)
            label = int(cluster["cluster_number"])

            if len(cluster_terms) == 1:
                target_map[cluster_terms[0]] = cluster_terms[0]
                continue

            legacy_key = ",".join(sorted(cluster_terms))
            cluster_key = f"{self.context.master_domain}::{label_prefix}::{legacy_key}"
            if self.resume:
                cached_mapped = self._db_get_cached_lift(cluster_key) or self._db_get_cached_lift(legacy_key)
                if cached_mapped:
                    print(f"      -> [{label_prefix}] Using cached taxonomic lift for cluster #{label} -> '{cached_mapped}'")
                    for t in original_members:
                        target_map[t] = cached_mapped
                    continue

            idx = [term_to_idx[t] for t in cluster_terms if t in term_to_idx]
            cluster_vecs = term_embeddings_l2[idx]
            weights = np.array([1.0 + math.log(normalized_frequencies.get(t, 1)) for t in cluster_terms])

            _, loo_distances = self._compute_loo_distances(cluster_vecs, weights)
            closest_indices = [int(i) for i in np.argsort(loo_distances)[:3]]
            fallback_candidates = [cluster_terms[i] for i in closest_indices]

            mmr_indices = self._select_mmr_order(
                cluster_vecs,
                loo_distances,
                seed_indices=closest_indices,
                lambda_param=0.50,
            )
            ordered_cluster_terms = [cluster_terms[i] for i in mmr_indices]

            payload = {
                "cluster_terms": ordered_cluster_terms,
                "top_3_centroid_fallbacks": fallback_candidates,
            }

            if _count_tokens(json.dumps(payload)) > max_payload_tokens:
                trimmed_terms = list(fallback_candidates)
                for term in ordered_cluster_terms:
                    if term not in trimmed_terms:
                        test_payload = {
                            "cluster_terms": trimmed_terms + [term],
                            "top_3_centroid_fallbacks": fallback_candidates,
                        }
                        if _count_tokens(json.dumps(test_payload)) <= max_payload_tokens:
                            trimmed_terms.append(term)
                        else:
                            break
                payload = {
                    "cluster_terms": trimmed_terms,
                    "top_3_centroid_fallbacks": fallback_candidates,
                }

            payload_json_str = json.dumps(payload)
            user_prompt = TAXONOMIC_LIFTING_USER_PROMPT.format(payload_json=payload_json_str)
            full_prompt = f"=== SYSTEM PROMPT ===\n{full_sys_prompt}\n\n=== USER PROMPT ===\n{user_prompt}"
            self._save_payload_text("stage_02_taxonomic_lifting.txt", full_prompt)

            uid = f"{label_prefix}__cluster_{label}"
            row_meta = {
                "uid": uid,
                "label": label,
                "cluster_key": cluster_key,
                "original_members": original_members,
                "fallback_candidates": fallback_candidates,
                "text": user_prompt,
            }
            pending_rows.append({"uid": uid, "label": label, "text": user_prompt})
            pending_meta_by_uid[uid] = row_meta

        if not pending_rows:
            return target_map

        total_pending = len(pending_rows)
        completed_count = 0
        map_lock = threading.Lock()
        resolved_uids: set = set()

        lift_template = {
            "version": "1.0",
            "task": "taxonomic_lifting",
            "system_prompt": full_sys_prompt,
            "prompt": "{{ text }}\n\n{schema_block}",
        }

        def _extract_hypernym(items: List[Any]) -> Optional[str]:
            for item in items:
                if isinstance(item, TaxonomicVerification) and item.formal_hypernym:
                    return item.formal_hypernym.strip()
                if isinstance(item, dict):
                    try:
                        parsed = TaxonomicVerification.model_validate(item)
                        if parsed.formal_hypernym:
                            return parsed.formal_hypernym.strip()
                    except Exception:
                        pass
            return None

        def _on_lift_batch_start(batch_idx: int, total_batches: int, batch_items: List[Dict[str, Any]]) -> None:
            cluster_labels = [
                f"#{pending_meta_by_uid[str(it.get('record_id') or '')]['label']}"
                for it in batch_items
                if str(it.get("record_id") or "") in pending_meta_by_uid
            ]
            self._log_checkpoint(
                f"Step 3: {label_prefix} Lift",
                f"Running concurrent batch {batch_idx}/{total_batches}: {len(batch_items)} cluster(s) "
                f"{cluster_labels[:6]}{'...' if len(cluster_labels) > 6 else ''}",
            )

        def _on_lift_chunk_processed(chunk_res: Dict[str, Any]) -> None:
            nonlocal completed_count
            uid = str(chunk_res.get("record_id") or chunk_res.get("id") or "")
            meta = pending_meta_by_uid.get(uid)
            if not meta:
                return
            items = chunk_res.get("parsed_items") or chunk_res.get("extracted_items") or []
            hypernym = _extract_hypernym(items)
            label = meta["label"]
            fallbacks = meta["fallback_candidates"]
            members = meta["original_members"]
            cluster_key = meta["cluster_key"]

            with map_lock:
                if uid in resolved_uids:
                    return
                resolved_uids.add(uid)
                completed_count += 1

                if hypernym:
                    mapped_term = hypernym
                    print(
                        f"      -> [{label_prefix}] ({completed_count}/{total_pending}) "
                        f"Resolved cluster #{label} ({len(members)} terms) -> Mapped to: '{mapped_term}'"
                    )
                else:
                    err_msg = chunk_res.get("parse_error") or "Empty or invalid TaxonomicVerification output"
                    self._log_error(
                        {
                            "phase": f"taxonomic_lifting_{label_prefix}",
                            "cluster_label": label,
                            "error": str(err_msg),
                        }
                    )
                    mapped_term = fallbacks[0] if fallbacks else members[0]
                    print(
                        f"      -> [{label_prefix}] ({completed_count}/{total_pending}) "
                        f"Fallback applied to cluster #{label} -> Mapped to: '{mapped_term}'"
                    )

                for t in members:
                    target_map[t] = mapped_term
                if self.resume:
                    self._db_save_lift(cluster_key, mapped_term)

        results = self.prism.run(
            task="extract",
            input_data=pending_rows,
            text_column="text",
            id_column="uid",
            schema_cls=TaxonomicVerification,
            prompt_template=lift_template,
            system_prompt=full_sys_prompt,
            chunk=False,
            as_array=False,
            batch_size=BATCH_SIZE,
            max_new_tokens=LIFT_MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            constrain_json=CONSTRAIN_JSON,
            dry_run=dry_run,
            on_batch_start=_on_lift_batch_start,
            on_chunk_processed=_on_lift_chunk_processed,
        )

        if dry_run:
            return target_map

        # Finalize any items not already handled by the streaming callback
        for row_meta, res in zip(pending_rows, results):
            uid = row_meta["uid"]
            if uid not in resolved_uids:
                _on_lift_chunk_processed(
                    {
                        "record_id": uid,
                        "parsed_items": res.get("extracted_items") or res.get("parsed_items") or [],
                        "parse_error": res.get("warning"),
                    }
                )

        return target_map

    # =========================================================================
    # STEP 4: CONTEXTUAL SENTENCE GROUNDING & EDGE FIDELITY (gamma_e)
    # =========================================================================

    def _compute_and_cache_sentence_context(
        self,
        refined_triples: List[Union[RawTriple, Dict[str, Any]]],
        out_dir: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embed unique source quotes and lifted SVO phrases, compute edge contextual fidelity (gamma_e),
        and export contextual_grounding_diagnostics.json.
        """
        target_out_dir = out_dir or self.out_dir
        print("[Refinement] Step 4: Computing Contextual Sentence Vectors & Edge Fidelity...")
        if MANAGE_VRAM:
            clear_device_memory()

        model_slug = re.sub(r"[^a-zA-Z0-9_]", "_", self.embedding_model_name)

        def get_field(item: Any, field_name: str) -> str:
            if hasattr(item, field_name):
                return str(getattr(item, field_name) or "").strip()
            if isinstance(item, dict):
                return str(item.get(field_name, "") or "").strip()
            return ""

        quotes_map: Dict[str, int] = {}
        for t in refined_triples:
            q = get_field(t, "source_quote")
            if q and q not in quotes_map:
                quotes_map[q] = len(quotes_map)

        unique_quotes = list(quotes_map.keys())
        print(f"   -> Extracted {len(unique_quotes)} unique source sentences across {len(refined_triples)} triplets.")

        sent_npy_path = self.cache_dir / f"sentence_{model_slug}_embeddings.npy"
        sent_meta_path = self.cache_dir / f"sentence_{model_slug}_meta.json"

        sent_embeddings = None
        if sent_npy_path.exists() and sent_meta_path.exists():
            try:
                meta = json.loads(sent_meta_path.read_text(encoding="utf-8"))
                if meta.get("model_name") == self.embedding_model_name and meta.get("vocab") == unique_quotes:
                    sent_embeddings = np.load(sent_npy_path)
                    print(f"   -> Loaded {len(unique_quotes)} cached sentence vectors ({sent_embeddings.shape[1]}d)")
            except Exception as e:
                print(f"   -> Sentence cache read warning: {e}; recomputing...")

        try:
            if sent_embeddings is None or len(sent_embeddings) != len(unique_quotes):
                sent_embeddings = self._encode_texts(unique_quotes)
                try:
                    np.save(sent_npy_path, sent_embeddings)
                    with open(sent_meta_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "model_name": self.embedding_model_name,
                                "dimensions": int(sent_embeddings.shape[1]),
                                "vocab": unique_quotes,
                            },
                            f,
                            indent=2,
                        )
                    print(f"   -> Cached {len(unique_quotes)} sentence vectors to {sent_npy_path.name}")
                except Exception as e:
                    print(f"   -> Warning: Failed to cache sentence vectors: {e}")

            phrase_list = [
                f"{get_field(t, 'subject')} {get_field(t, 'predicate')} {get_field(t, 'object')}".strip()
                for t in refined_triples
            ]
            unique_phrases = sorted(list(set(phrase_list)))
            phrase_to_idx = {p: i for i, p in enumerate(unique_phrases)}
            phrase_embeddings = self._encode_texts(unique_phrases)
        finally:
            self._unload_embedding_model()

        triplet_dicts: List[Dict[str, Any]] = []
        fidelity_scores: List[float] = []

        for i, t in enumerate(refined_triples):
            if hasattr(t, "model_dump"):
                td = t.model_dump()
            elif isinstance(t, dict):
                td = dict(t)
            else:
                td = dict(t.__dict__)

            q = get_field(t, "source_quote")
            phrase = phrase_list[i]

            if q in quotes_map and phrase in phrase_to_idx and len(sent_embeddings) > 0 and len(phrase_embeddings) > 0:
                q_idx = quotes_map[q]
                p_idx = phrase_to_idx[phrase]
                gamma_e = float(np.dot(phrase_embeddings[p_idx], sent_embeddings[q_idx]))
                gamma_e = max(0.0, min(1.0, gamma_e))
            else:
                gamma_e = 0.5

            td["contextual_fidelity"] = round(gamma_e, 4)
            td["sentence_id"] = quotes_map.get(q, -1)
            triplet_dicts.append(td)
            fidelity_scores.append(gamma_e)

        if fidelity_scores:
            f_arr = np.array(fidelity_scores)
            q25, q50, q75 = np.percentile(f_arr, [25, 50, 75])
        else:
            f_arr = np.array([0.5])
            q25 = q50 = q75 = 0.5

        indexed_scores = sorted(enumerate(fidelity_scores), key=lambda x: x[1], reverse=True)
        top_samples = [
            {
                "phrase": phrase_list[idx],
                "quote": get_field(refined_triples[idx], "source_quote")[:120],
                "fidelity": round(score, 4),
            }
            for idx, score in indexed_scores[:5]
        ]
        bottom_samples = [
            {
                "phrase": phrase_list[idx],
                "quote": get_field(refined_triples[idx], "source_quote")[:120],
                "fidelity": round(score, 4),
            }
            for idx, score in indexed_scores[-5:]
        ]

        diag_report = {
            "summary": {
                "total_triplets": len(refined_triples),
                "total_unique_sentences": len(unique_quotes),
                "fidelity_stats": {
                    "min": round(float(np.min(f_arr)), 4),
                    "max": round(float(np.max(f_arr)), 4),
                    "mean": round(float(np.mean(f_arr)), 4),
                    "median": round(float(q50), 4),
                    "std_dev": round(float(np.std(f_arr)), 4),
                    "q25": round(float(q25), 4),
                    "q75": round(float(q75), 4),
                },
            },
            "sample_high_fidelity_triplets": top_samples,
            "sample_low_fidelity_triplets": bottom_samples,
        }

        diag_path = target_out_dir / "contextual_grounding_diagnostics.json"
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diag_report, f, indent=2)

        print(f"   -> Fidelity: mean={np.mean(f_arr):.3f}, median={q50:.3f}, std={np.std(f_arr):.3f}")
        print(f"   -> Exported grounding diagnostics to: {diag_path}")
        return triplet_dicts

    # =========================================================================
    # FULL STAGE 2 EXECUTION
    # =========================================================================

    def execute(
        self,
        raw_triples: List[RawTriple],
        subject_map: Optional[Dict[str, str]] = None,
        predicate_map: Optional[Dict[str, str]] = None,
        object_map: Optional[Dict[str, str]] = None,
        dry_run: bool = DRY_RUN,
    ) -> List[RawTriple]:
        print("[Refinement] Starting Stage 2 Pipeline (Vector Clustering & Taxonomic Lifting)...")
        if subject_map is None:
            subject_map = {t.subject: self._nlp_preprocess(t.subject) for t in raw_triples}
        if predicate_map is None:
            predicate_map = {t.predicate: self._nlp_preprocess(t.predicate) for t in raw_triples}
        if object_map is None:
            object_map = {t.object: self._nlp_preprocess(t.object) for t in raw_triples}

        lift_subj = bool(LIFT_SUBJECTS)
        lift_pred = bool(LIFT_PREDICATES)
        lift_obj = bool(LIFT_OBJECTS)
        enable_lift = lift_subj or lift_pred or lift_obj

        subj_clusters: List[Dict[str, Any]] = []
        pred_clusters: List[Dict[str, Any]] = []
        obj_clusters: List[Dict[str, Any]] = []
        subj_embeddings_l2: Optional[np.ndarray] = None
        pred_embeddings_l2: Optional[np.ndarray] = None
        obj_embeddings_l2: Optional[np.ndarray] = None
        subj_terms: Optional[List[str]] = []
        pred_terms: Optional[List[str]] = []
        obj_terms: Optional[List[str]] = []

        # Calculate normalized frequencies for weighting Spherical Fréchet centroids
        normalized_frequencies: Dict[str, int] = defaultdict(int)
        for t in raw_triples:
            norm_subj = subject_map.get(t.subject, t.subject)
            normalized_frequencies[norm_subj] += 1

            norm_pred = predicate_map.get(t.predicate, t.predicate)
            normalized_frequencies[norm_pred] += 1

            norm_obj = object_map.get(t.object, t.object)
            normalized_frequencies[norm_obj] += 1

        self.context.term_frequencies = normalized_frequencies

        if enable_lift and (lift_subj or lift_pred or lift_obj):
            if MANAGE_VRAM:
                clear_device_memory()

            print("[Refinement] Step 2: SVO Vector Clustering")
            try:
                if lift_subj:
                    unique_subjects_norm = sorted(list(set(subject_map.values())))
                    self._log_checkpoint("Step 2: Clustering", f"Clustering {len(unique_subjects_norm)} unique Subjects...")
                    subj_clusters, subj_embeddings_l2, subj_terms = self._cluster_field(
                        unique_subjects_norm, "Subject", normalized_frequencies
                    )
                    with open(self.out_dir / "subject_clusters.json", "w", encoding="utf-8") as f:
                        json.dump(subj_clusters, f, indent=2)
                else:
                    print("   -> Clustering Subjects SKIPPED (toggled off in config).")

                if lift_pred:
                    unique_predicates_norm = sorted(list(set(predicate_map.values())))
                    self._log_checkpoint("Step 2: Clustering", f"Clustering {len(unique_predicates_norm)} unique Predicates...")
                    pred_clusters, pred_embeddings_l2, pred_terms = self._cluster_field(
                        unique_predicates_norm, "Predicate", normalized_frequencies
                    )
                    with open(self.out_dir / "predicate_clusters.json", "w", encoding="utf-8") as f:
                        json.dump(pred_clusters, f, indent=2)
                else:
                    print("   -> Clustering Predicates SKIPPED (toggled off in config).")

                if lift_obj:
                    unique_objects_norm = sorted(list(set(object_map.values())))
                    self._log_checkpoint("Step 2: Clustering", f"Clustering {len(unique_objects_norm)} unique Objects...")
                    obj_clusters, obj_embeddings_l2, obj_terms = self._cluster_field(
                        unique_objects_norm, "Object", normalized_frequencies
                    )
                    with open(self.out_dir / "object_clusters.json", "w", encoding="utf-8") as f:
                        json.dump(obj_clusters, f, indent=2)
                else:
                    print("   -> Clustering Objects SKIPPED (toggled off in config).")
            finally:
                # Offload SentenceTransformer before loading LLM for Step 3
                self._unload_embedding_model()
        else:
            print("[Refinement] Step 2 & 3: Taxonomic Lifting SKIPPED (toggled off in config).")

        # 3. Taxonomic Lifting via PrismRunner
        subject_taxonomic_map: Dict[str, str] = {}
        predicate_taxonomic_map: Dict[str, str] = {}
        object_taxonomic_map: Dict[str, str] = {}

        if enable_lift and (lift_subj or lift_pred or lift_obj):
            print("[Refinement] Step 3: Taxonomic Lifting (PrismRunner Batch Inference)")
            subj_singletons = sum(1 for c in subj_clusters if len(c["members"]) == 1)
            subj_multi = len(subj_clusters) - subj_singletons
            pred_singletons = sum(1 for c in pred_clusters if len(c["members"]) == 1)
            pred_multi = len(pred_clusters) - pred_singletons
            obj_singletons = sum(1 for c in obj_clusters if len(c["members"]) == 1)
            obj_multi = len(obj_clusters) - obj_singletons

            if lift_subj:
                print(
                    f"   -> Subjects: {len(subj_clusters)} total clusters "
                    f"({subj_singletons} single-member resolved locally, {subj_multi} multi-member requiring lifting)"
                )
                subject_taxonomic_map = self._lift_component_clusters(
                    subj_clusters,
                    subj_embeddings_l2,
                    subj_terms,
                    "Subject",
                    normalized_frequencies,
                    dry_run=dry_run,
                )

            if lift_pred:
                print(
                    f"   -> Predicates: {len(pred_clusters)} total clusters "
                    f"({pred_singletons} single-member resolved locally, {pred_multi} multi-member requiring lifting)"
                )
                predicate_taxonomic_map = self._lift_component_clusters(
                    pred_clusters,
                    pred_embeddings_l2,
                    pred_terms,
                    "Predicate",
                    normalized_frequencies,
                    dry_run=dry_run,
                )

            if lift_obj:
                print(
                    f"   -> Objects: {len(obj_clusters)} total clusters "
                    f"({obj_singletons} single-member resolved locally, {obj_multi} multi-member requiring lifting)"
                )
                object_taxonomic_map = self._lift_component_clusters(
                    obj_clusters,
                    obj_embeddings_l2,
                    obj_terms,
                    "Object",
                    normalized_frequencies,
                    dry_run=dry_run,
                )

            del subj_embeddings_l2, pred_embeddings_l2, obj_embeddings_l2
            if MANAGE_VRAM:
                clear_device_memory()

        if dry_run:
            return raw_triples

        taxonomic_map: Dict[str, str] = {}
        if lift_subj:
            with open(self.out_dir / "subject_taxonomic_map.json", "w", encoding="utf-8") as f:
                json.dump(subject_taxonomic_map, f, indent=2)
            taxonomic_map.update(subject_taxonomic_map)

        if lift_pred:
            with open(self.out_dir / "predicate_taxonomic_map.json", "w", encoding="utf-8") as f:
                json.dump(predicate_taxonomic_map, f, indent=2)
            taxonomic_map.update(predicate_taxonomic_map)

        if lift_obj:
            with open(self.out_dir / "object_taxonomic_map.json", "w", encoding="utf-8") as f:
                json.dump(object_taxonomic_map, f, indent=2)
            taxonomic_map.update(object_taxonomic_map)

        if taxonomic_map:
            with open(self.out_dir / "taxonomic_map.json", "w", encoding="utf-8") as f:
                json.dump(taxonomic_map, f, indent=2)

        # Build case-insensitive maps for final triple re-mapping lookups
        lower_subj_norm = {k.lower(): v for k, v in subject_map.items()}
        lower_pred_norm = {k.lower(): v for k, v in predicate_map.items()}
        lower_obj_norm = {k.lower(): v for k, v in object_map.items()}

        lower_subj_tax = {k.lower(): v for k, v in subject_taxonomic_map.items()}
        lower_pred_tax = {k.lower(): v for k, v in predicate_taxonomic_map.items()}
        lower_obj_tax = {k.lower(): v for k, v in object_taxonomic_map.items()}

        # Apply Taxonomic Lifting maps to raw triples
        refined_triples = [t.model_copy(deep=True) for t in raw_triples]
        for t in refined_triples:
            norm_subj = subject_map.get(t.subject)
            if norm_subj is None:
                pre_subj = self._nlp_preprocess(t.subject)
                norm_subj = lower_subj_norm.get(pre_subj.lower(), pre_subj)

            norm_pred = predicate_map.get(t.predicate)
            if norm_pred is None:
                pre_pred = self._nlp_preprocess(t.predicate)
                norm_pred = lower_pred_norm.get(pre_pred.lower(), pre_pred)

            norm_obj = object_map.get(t.object)
            if norm_obj is None:
                pre_obj = self._nlp_preprocess(t.object)
                norm_obj = lower_obj_norm.get(pre_obj.lower(), pre_obj)

            final_subj = subject_taxonomic_map.get(norm_subj)
            if final_subj is None:
                final_subj = lower_subj_tax.get(norm_subj.lower(), norm_subj)

            final_pred = predicate_taxonomic_map.get(norm_pred)
            if final_pred is None:
                final_pred = lower_pred_tax.get(norm_pred.lower(), norm_pred)

            final_obj = object_taxonomic_map.get(norm_obj)
            if final_obj is None:
                final_obj = lower_obj_tax.get(norm_obj.lower(), norm_obj)

            t.subject = final_subj
            t.predicate = final_pred
            t.object = final_obj

        # 4. Contextual Sentence Grounding & Fidelity Computation
        triplet_dicts = self._compute_and_cache_sentence_context(refined_triples, self.out_dir)

        refined_path = self.out_dir / "refined_triplets.json"
        with open(refined_path, "w", encoding="utf-8") as f:
            json.dump(triplet_dicts, f, indent=2)

        print(f"[Refinement] Exported refined triplets to: {refined_path}")
        print("[Refinement] Pipeline complete. Purging VRAM.")
        if MANAGE_VRAM:
            clear_device_memory()
        return refined_triples

    # Alias for backward compatibility with original SemanticPrism
    execute_part_2 = execute


# =============================================================================
# 4. CLI & ENTRYPOINT
# =============================================================================

def run_fidelity_only(pipeline: RefinementPipeline, out_dir: Path) -> None:
    """Fast computation of sentence vectors and contextual fidelity on existing refined triplets (< 10s)."""
    refined_path = out_dir / "refined_triplets.json"
    if not refined_path.exists():
        print(f"CRITICAL ERROR: Required file '{refined_path}' not found.")
        print("Please run Stage 2 fully first before running --fidelity-only.")
        sys.exit(1)

    print(f"=== SemanticPrism Stage 2: Fast Contextual Fidelity Update ({refined_path}) ===")
    with open(refined_path, "r", encoding="utf-8") as f:
        refined_data = json.load(f)

    triplet_dicts = pipeline._compute_and_cache_sentence_context(refined_data, out_dir)
    with open(refined_path, "w", encoding="utf-8") as f:
        json.dump(triplet_dicts, f, indent=2)

    print(f"Successfully updated {len(triplet_dicts)} triplets with contextual fidelity in '{refined_path}'.")
    print(f"Grounding diagnostics saved in '{out_dir / 'contextual_grounding_diagnostics.json'}'.")


def main() -> None:
    global MODEL_NAME, DEVICE, BATCH_SIZE, RESUME_MODE, DRY_RUN

    parser = argparse.ArgumentParser(
        description="SemanticPrism Stage 2: Refinement Pipeline (Vector Clustering & Taxonomic Lifting via PrismRunner)"
    )
    parser.add_argument(
        "--fidelity-only",
        action="store_true",
        help="Fast computation of sentence vectors and contextual fidelity on existing refined triplets (< 10s)",
    )
    parser.add_argument("--model", type=str, default=None, help="Override MODEL_NAME")
    parser.add_argument("--device", type=str, default=None, help="Override DEVICE ('auto', 'cuda', 'mps', 'cpu')")
    parser.add_argument("--batch-size", type=str, default=None, help="Override BATCH_SIZE ('auto' or integer)")
    parser.add_argument("--overwrite", action="store_true", help="Set RESUME_MODE='overwrite' to clear SQLite cache")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode without loading LLM weights")
    parser.add_argument("--live", action="store_true", help="Force live LLM execution (DRY_RUN = False)")
    args = parser.parse_args()

    if args.model:
        MODEL_NAME = args.model
    if args.device:
        DEVICE = args.device
    if args.batch_size:
        BATCH_SIZE = int(args.batch_size) if args.batch_size.isdigit() else args.batch_size
    if args.overwrite:
        RESUME_MODE = "overwrite"
    if args.dry_run:
        DRY_RUN = True
    elif args.live:
        DRY_RUN = False

    pipeline = RefinementPipeline()
    out_dir = pipeline.out_dir

    if args.fidelity_only:
        run_fidelity_only(pipeline, out_dir)
        return

    print(f"=== SemanticPrism Stage 2: Refinement Pipeline (Model: {MODEL_NAME}) ===")
    stage_start = time.time()

    input_dir = Path(INPUT_DIR)
    if not input_dir.exists() and (REPO_ROOT / input_dir).exists():
        input_dir = REPO_ROOT / input_dir

    triplets_path = input_dir / "original_triplets.json"
    if not triplets_path.exists():
        print(f"CRITICAL ERROR: Required file '{triplets_path}' not found.")
        print("Stage 2 requires Stage 1 triple extraction outputs ('outputs/01_extraction/original_triplets.json').")
        print("Please run Stage 1 first ('python3 run_stage_1_extraction.py').")
        sys.exit(1)

    try:
        with open(triplets_path, "r", encoding="utf-8") as f:
            triplets_data = json.load(f)
            raw_triples = [RawTriple(**t) for t in triplets_data]
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load original triplets: {e}")
        sys.exit(1)

    try:
        pipeline.execute(raw_triples=raw_triples, dry_run=DRY_RUN)

        total_duration = time.time() - stage_start
        print("\n" + "=" * 60)
        print("STAGE 2 REFINEMENT PIPELINE COMPLETE!")
        print(f"Total Stage 2 Time: {total_duration:.2f} seconds")
        print(f"Outputs saved in '{out_dir}':")
        print(" - subject_clusters.json")
        print(" - predicate_clusters.json")
        print(" - object_clusters.json")
        print(" - subject_taxonomic_map.json")
        print(" - predicate_taxonomic_map.json")
        print(" - object_taxonomic_map.json")
        print(" - taxonomic_map.json")
        print(" - refined_triplets.json")
        print(" - contextual_grounding_diagnostics.json")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"Error during Stage 2 Refinement execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

