"""
SemanticPrism Stage 2: Refinement & Taxonomic Hypernym Lifting Pipeline (Powered by PrismRunner)
"""

from __future__ import annotations

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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", message=r".*n_jobs value.*overridden to 1.*", category=UserWarning)

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_SIBLING_PRISMRUNNER = REPO_ROOT.parent / "PrismRunner"
if _SIBLING_PRISMRUNNER.exists() and str(_SIBLING_PRISMRUNNER) not in sys.path:
    sys.path.insert(0, str(_SIBLING_PRISMRUNNER))

from prismrunner import (
    ContextBudget,
    PrismPipeline,
    clear_device_memory,
    get_optimal_device,
    resolve_dynamic_batch_size,
)

from src.refinement.prompts import (
    PREDICATE_LIFTING_SYSTEM_PROMPT,
    TAXONOMIC_LIFTING_SYSTEM_PROMPT,
    TAXONOMIC_LIFTING_USER_PROMPT,
)
from src.refinement.schemas import RawTriple, TaxonomicVerification


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
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        embedding_model: str = "all-MiniLM-L6-v2",
        device: str = "auto",
        torch_dtype: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        attn_implementation: str = "auto",
        batch_size: Union[int, str] = "auto",
        temperature: float = 0.0,
        constrain_json: bool = True,
        manage_vram: bool = True,
        dry_run: bool = False,
        input_dir: str = "outputs/01_extraction",
        outputs_dir: str = "outputs",
        save_llm_payloads: bool = True,
        resume_mode: str = "skip",
        print_checkpoints: bool = True,
        plot_trustworthiness: bool = True,
        context_window_cap: Union[int, str] = "auto",
        lift_max_new_tokens: int = 256,
        lift_subjects: bool = True,
        lift_predicates: bool = False,
        lift_objects: bool = True,
    ):
        self.model_name = model_name
        self.embedding_model = None
        self.embedding_model_name = embedding_model
        self.device = device
        self.torch_dtype = torch_dtype
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.attn_implementation = attn_implementation
        self.batch_size = batch_size
        self.temperature = temperature
        self.constrain_json = constrain_json
        self.manage_vram = manage_vram
        self.dry_run = dry_run
        self.save_llm_payloads = save_llm_payloads
        self.resume_mode = resume_mode
        self.print_checkpoints = print_checkpoints
        self.plot_trustworthiness = plot_trustworthiness
        self.context_window_cap = context_window_cap
        self.lift_max_new_tokens = lift_max_new_tokens
        self.lift_subjects = lift_subjects
        self.lift_predicates = lift_predicates
        self.lift_objects = lift_objects

        base_out = Path(outputs_dir)
        self.in_dir = Path(input_dir)
        self.out_dir = base_out / "02_refinement"
        self.cache_dir = self.out_dir / "cache"
        self.log_dir = base_out / "logs"
        self.payload_dir = base_out / "payload_text"

        for d in (self.out_dir, self.cache_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)
        if self.save_llm_payloads:
            self.payload_dir.mkdir(parents=True, exist_ok=True)

        resolved_domain = self._resolve_master_domain()
        self.context = context or PipelineRunContext(master_domain=resolved_domain)
        if not self.context.master_domain or self.context.master_domain == "auto":
            self.context.master_domain = resolved_domain

        _lower_llm = str(self.model_name).lower()
        if any(pat in _lower_llm for pat in ("bge-", "minilm", "mpnet", "sentence-transformers/", "e5-", "nomic-embed")):
            raise ValueError(
                f"MODEL_NAME is set to '{self.model_name}', which is an encoder-only embedding model (not a generative LLM).\n"
                f"In run_stage_2_refinement.py:\n"
                f"  - Set EMBEDDING_MODEL = \"{self.model_name}\"  (for Step 2 SVO clustering & Step 4 sentence vectors)\n"
                f"  - Set MODEL_NAME      = \"Qwen/Qwen3.6-27B\" (or another generative LLM for Step 3 Taxonomic Lifting)"
            )

        log_file = self.log_dir / "stage_02_refinement_errors.json"
        if log_file.exists() and self.resume_mode != "skip":
            log_file.unlink()

        self.resume = str(self.resume_mode).lower().strip() == "skip"
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
                        "temperature": self.temperature,
                        "constrain_json": self.constrain_json,
                    },
                },
            )

        try:
            with open(self.out_dir / "model_execution_metadata.json", "w", encoding="utf-8") as mf:
                json.dump(
                    {
                        "stage": "stage_2_refinement",
                        "configured_model": self.model_name,
                        "active_model": self.model_name,
                        "embedding_model": self.embedding_model,
                        "quantization_or_dtype": (
                            "4-bit NF4"
                            if self.load_in_4bit
                            else ("8-bit" if self.load_in_8bit else f"native ({self.torch_dtype})")
                        ),
                        "fallback_used": False,
                        "fallback_reason": None,
                    },
                    mf,
                    indent=2,
                )
        except Exception:
            pass

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

    def _log_checkpoint(self, phase: str, message: str) -> None:
        if self.print_checkpoints:
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
        try:
            log_path = self.log_dir / "stage_02_refinement_errors.json"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[Refinement] Failed to write error log: {e}")

    def _save_payload_text(self, filename: str, content: str) -> None:
        if not self.save_llm_payloads:
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

        models_dir = REPO_ROOT / "models" / "embeddings"
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
        if self.embedding_model is not None:
            try:
                self.embedding_model.to("cpu")
            except Exception:
                pass
            del self.embedding_model
            self.embedding_model = None
        if self.manage_vram:
            clear_device_memory()

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        try:
            import torch

            self._load_embedding_model()
            emb_batch_size = resolve_dynamic_batch_size(
                device=get_optimal_device(self.device),
                requested_batch_size=self.batch_size,
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
        is_pred = component.lower() == "predicate"
        if embedding_dim > 768:
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

        np.fill_diagonal(dist_mat, np.inf)
        nn_dists = np.min(dist_mat, axis=1)
        p15_nn, p35_nn, p60_nn = np.percentile(nn_dists, [15, 35, 60])

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

        term_embeddings_l2 = self.get_or_compute_embeddings(terms_list, label_prefix)
        dim = int(term_embeddings_l2.shape[1])

        params = self.resolve_clustering_params(dim, label_prefix)
        eff_rescue, eff_macro = self._calibrate_distance_thresholds(
            term_embeddings_l2,
            params["rescue_tau"],
            params["tau_macro"],
        )
        params["rescue_tau"] = eff_rescue
        params["tau_macro"] = eff_macro
        weights = np.array([1.0 + math.log(normalized_frequencies.get(t, 1)) for t in terms_list])

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

        adaptive_k = int(2 + 2.5 * math.log(max(2, num_terms)))
        max_local_k = max(3, num_terms // 4) if num_terms >= 12 else max(2, num_terms - 1)
        n_neighbors = int(min(params["umap_n_neighbors"], max(2, min(adaptive_k, max_local_k, num_terms - 1))))
        adaptive_comp = int(max(2, min(8, round(math.log(max(4, num_terms))))))
        n_components = min(adaptive_comp, max(2, num_terms - 2))
        min_dist = params["umap_min_dist"]
        metric = "cosine"
        random_state = 42

        if self.plot_trustworthiness and umap_mod is not None and trustworthiness is not None and num_terms >= 4:
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

        micro_centroids: Dict[int, np.ndarray] = {}
        for lbl, mems in raw_clusters.items():
            m_idx = [term_to_idx[t] for t in mems]
            micro_centroids[lbl] = self.compute_spherical_centroid(m_idx, term_embeddings_l2, weights)

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

        multi_micro_clusters: Dict[int, List[str]] = {}
        for lbl, mems in raw_clusters.items():
            if len(mems) > 1:
                multi_micro_clusters[lbl] = mems
            else:
                singletons.extend(mems)

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

        next_id = max(final_clusters.keys()) + 1 if final_clusters else 0
        for s_term in singletons:
            final_clusters[next_id] = [s_term]
            next_id += 1

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
        if isinstance(self.context_window_cap, int) and self.context_window_cap > 0:
            effective_ctx = min(native_ctx, self.context_window_cap)
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
        dry_run: Optional[bool] = None,
    ) -> Dict[str, str]:
        is_dry = self.dry_run if dry_run is None else dry_run
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

        if not is_dry and self.prism.engine is None:
            self._log_checkpoint(f"Step 3: {label_prefix} Lift", f"Loading LLM '{self.model_name}' onto device...")
            self.prism.init_engine(
                batch_size=self.batch_size,
                max_new_tokens=self.lift_max_new_tokens,
                temperature=self.temperature,
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
                max_new_tokens=self.lift_max_new_tokens,
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
            batch_size=self.batch_size,
            max_new_tokens=self.lift_max_new_tokens,
            temperature=self.temperature,
            constrain_json=self.constrain_json,
            dry_run=is_dry,
            on_batch_start=_on_lift_batch_start,
            on_chunk_processed=_on_lift_chunk_processed,
        )

        if is_dry:
            return target_map

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
        target_out_dir = out_dir or self.out_dir
        print("[Refinement] Step 4: Computing Contextual Sentence Vectors & Edge Fidelity...")
        if self.manage_vram:
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
        dry_run: Optional[bool] = None,
    ) -> List[RawTriple]:
        is_dry = self.dry_run if dry_run is None else dry_run
        print("[Refinement] Starting Stage 2 Pipeline (Vector Clustering & Taxonomic Lifting)...")
        if subject_map is None:
            subject_map = {t.subject: self._nlp_preprocess(t.subject) for t in raw_triples}
        if predicate_map is None:
            predicate_map = {t.predicate: self._nlp_preprocess(t.predicate) for t in raw_triples}
        if object_map is None:
            object_map = {t.object: self._nlp_preprocess(t.object) for t in raw_triples}

        lift_subj = bool(self.lift_subjects)
        lift_pred = bool(self.lift_predicates)
        lift_obj = bool(self.lift_objects)
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
            if self.manage_vram:
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
                self._unload_embedding_model()
        else:
            print("[Refinement] Step 2 & 3: Taxonomic Lifting SKIPPED (toggled off in config).")

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
                    dry_run=is_dry,
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
                    dry_run=is_dry,
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
                    dry_run=is_dry,
                )

            del subj_embeddings_l2, pred_embeddings_l2, obj_embeddings_l2
            if self.manage_vram:
                clear_device_memory()

        if is_dry:
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

        lower_subj_norm = {k.lower(): v for k, v in subject_map.items()}
        lower_pred_norm = {k.lower(): v for k, v in predicate_map.items()}
        lower_obj_norm = {k.lower(): v for k, v in object_map.items()}

        lower_subj_tax = {k.lower(): v for k, v in subject_taxonomic_map.items()}
        lower_pred_tax = {k.lower(): v for k, v in predicate_taxonomic_map.items()}
        lower_obj_tax = {k.lower(): v for k, v in object_taxonomic_map.items()}

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

            if norm_subj and final_subj and norm_subj.strip().lower() != final_subj.strip().lower():
                t.surface_subject = norm_subj
            if norm_obj and final_obj and norm_obj.strip().lower() != final_obj.strip().lower():
                t.surface_object = norm_obj

            t.subject = final_subj
            t.predicate = final_pred
            t.object = final_obj

        triplet_dicts = self._compute_and_cache_sentence_context(refined_triples, self.out_dir)

        refined_path = self.out_dir / "refined_triplets.json"
        with open(refined_path, "w", encoding="utf-8") as f:
            json.dump(triplet_dicts, f, indent=2)

        print(f"[Refinement] Exported refined triplets to: {refined_path}")
        print("[Refinement] Pipeline complete. Purging VRAM.")
        if self.manage_vram:
            clear_device_memory()
        return refined_triples

    execute_part_2 = execute


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
