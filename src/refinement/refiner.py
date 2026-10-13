"""
SemanticPrism Stage 2: Refinement Pipeline
Handles embedding mapping, HDBSCAN clustering, and taxonomic lifting.
"""

import json
import re
import asyncio
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np

warnings.filterwarnings("ignore", message=r".*n_jobs value.*overridden to 1.*", category=UserWarning)

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from sklearn.preprocessing import normalize
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
    from sklearn.manifold import trustworthiness
except Exception:
    normalize = None
    AgglomerativeClustering = None
    cosine_distances = None
    trustworthiness = None

try:
    import umap
except Exception:
    umap = None

try:
    import hdbscan
except Exception:
    try:
        from sklearn.cluster import HDBSCAN as hdbscan_module
        hdbscan = hdbscan_module
    except Exception:
        hdbscan = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from src.agents.refinement_agents import lift_agent
from src.agents.vram_manager import purge_vram
from src.extraction.schemas import RawTriple  # Stage 1 schema
from src.refinement import prompts

class PipelineRunContext:
    """Stores global state safely for the pipeline."""
    def __init__(self, master_domain: str):
        self.master_domain = master_domain
        self.term_frequencies: Dict[str, int] = defaultdict(int)


class RefinementPipeline:
    def __init__(self, config: Dict[str, Any], context: PipelineRunContext):
        self.config = config
        self.context = context
        self.embedding_model = None
        self.embedding_model_name = self.config.get('refinement', {}).get('embedding_model', 'all-MiniLM-L6-v2')
        
        # Setup Output and Log Directories
        base_out_dir = self.config.get('directories', {}).get('outputs', 'outputs')
        self.use_async = self.config.get('pipeline', {}).get('use_async', False)
        self.log_dir = Path(base_out_dir) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear previous run logs to prevent continuous expansion
        log_file = self.log_dir / "stage_02_refinement_errors.json"
        if log_file.exists():
            log_file.unlink()

        # Setup SQLite Database and Lock for WAL-safe async operations
        pipeline_resume_mode = self.config.get('pipeline', {}).get('resume_mode', 'skip')
        self.resume = (str(pipeline_resume_mode).lower() == 'skip')

        mode_str = "skip (resume existing cache)" if self.resume else "overwrite (clear cache & restart)"
        print(f"[Refinement] Pipeline Resume Mode: {mode_str}\n")

        db_dir = Path(base_out_dir) / "02_refinement"
        db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_dir / "refinement_state.db"
        
        # Erase existing database file if resume is set to False (clean run requested)
        if not self.resume and self.db_path.exists():
            print(f"[Refinement] Starting fresh run: deleting existing cache database at {self.db_path}")
            try:
                self.db_path.unlink()
                for ext in [".db-wal", ".db-shm"]:
                    extra_file = self.db_path.with_suffix(ext)
                    if extra_file.exists():
                        extra_file.unlink()
            except Exception as e:
                print(f"[Refinement] Warning: Could not delete cache database: {e}")
                
        self.db_lock = asyncio.Lock()
        self._init_db()

    def _init_db(self):
        import sqlite3
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS taxonomic_lifting (
            cluster_key TEXT PRIMARY KEY,
            mapped_term TEXT
        );
        """)
        conn.commit()
        conn.close()

    async def _run_db_query(self, query: str, params: tuple = (), is_write: bool = False):
        import sqlite3
        def _execute():
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            cursor = conn.cursor()
            try:
                cursor.execute(query, params)
                if is_write:
                    conn.commit()
                    return None
                return cursor.fetchall()
            finally:
                conn.close()
                
        if is_write:
            async with self.db_lock:
                return await asyncio.to_thread(_execute)
        else:
            return await asyncio.to_thread(_execute)

    async def _run_db_executemany(self, query: str, param_list: list):
        """Helper to execute bulk SQLite insertions safely."""
        if not param_list:
            return
        import sqlite3

        def _execute():
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            cursor = conn.cursor()
            try:
                cursor.executemany(query, param_list)
                conn.commit()
            finally:
                conn.close()

        async with self.db_lock:
            await asyncio.to_thread(_execute)


    def _log_error(self, record: Dict[str, Any]):
        """Helper to append error records to stage_02_refinement_errors.json."""
        try:
            log_path = self.log_dir / "stage_02_refinement_errors.json"
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[Refinement] Failed to write error log: {e}")

    def _save_payload_text(self, filename: str, content: str):
        """Helper to export initial LLM prompt payloads to outputs/payload_text directory."""
        payload_dir = Path(self.config.get('directories', {}).get('outputs', 'outputs')) / "payload_text"
        payload_dir.mkdir(parents=True, exist_ok=True)
        target_path = payload_dir / filename
        if target_path.exists():
            return
        try:
            with open(target_path, "w", encoding="utf-8") as pf:
                pf.write(content)
        except Exception as e:
            print(f"Warning: Failed to save payload text file {filename}: {e}")

    def _nlp_preprocess(self, text: str) -> str:
        """Native cleaning."""
        if not text:
            return text
        original = text
        text = text.replace('_', ' ')
        text = re.sub(r'[<>/\\|\[\]{}]', '', text)
        text = text.lower()
        cleaned = " ".join(text.split())
        return cleaned if cleaned else original

    def _load_embedding_model(self):
        if not self.embedding_model:
            global SentenceTransformer
            if SentenceTransformer is None:
                from sentence_transformers import SentenceTransformer
            model_name = self.config.get('refinement', {}).get('embedding_model', 'all-MiniLM-L6-v2')
            self.embedding_model_name = model_name
            
            # Ensure model is downloaded and stored locally in the project directory
            models_dir = Path("models/embeddings")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                self.embedding_model = SentenceTransformer(
                    model_name, 
                    cache_folder=str(models_dir),
                    local_files_only=True  # Guarantees 100% offline execution if cached
                )
            except Exception as e:
                print(f"[Refinement] Local model '{model_name}' not found or failed to load offline ({e}).")
                print("[Refinement] Attempting online download from HuggingFace...")
                self.embedding_model = SentenceTransformer(
                    model_name, 
                    cache_folder=str(models_dir),
                    local_files_only=False
                )

    def get_or_compute_embeddings(self, terms: List[str], component: str) -> np.ndarray:
        """
        Model-aware vector cache with metadata validation.
        Prevents dimension mismatch collisions when switching backbones.
        """
        ref_cfg = self.config.get('refinement', {})
        cache_dir = Path(ref_cfg.get('vector_cache_dir', 'outputs/02_refinement/cache'))
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_name = getattr(self, 'embedding_model_name', ref_cfg.get('embedding_model', 'all-MiniLM-L6-v2'))
        model_slug = re.sub(r'[^a-zA-Z0-9_]', '_', model_name)
        comp_key = component.lower()
        npy_path = cache_dir / f"{comp_key}_{model_slug}_embeddings.npy"
        meta_path = cache_dir / f"{comp_key}_{model_slug}_meta.json"

        if npy_path.exists() and meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("model_name") == model_name and meta.get("vocab") == terms:
                    embs = np.load(npy_path)
                    if embs.shape[0] == len(terms):
                        print(f"   -> [{component}] Loaded {len(terms)} cached vectors ({embs.shape[1]}d) from {npy_path.name}")
                        return embs
            except Exception as e:
                print(f"   -> [{component}] Cache read warning ({e}); recomputing embeddings...")

        # Encode and L2-normalize
        embs = self.embedding_model.encode(
            terms,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=ref_cfg.get('batch_size', 128)
        )
        try:
            np.save(npy_path, embs)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "model_name": model_name,
                    "dimensions": int(embs.shape[1]),
                    "vocab": terms
                }, f, indent=2)
            print(f"   -> [{component}] Computed and cached {len(terms)} vectors ({embs.shape[1]}d) to {npy_path.name}")
        except Exception as e:
            print(f"   -> [{component}] Warning: Failed to write vector cache: {e}")

        return embs

    def resolve_clustering_params(self, embedding_dim: int, component: str = 'subject') -> dict:
        """
        Dynamically resolves clustering hyperparameters based on embedding dimension
        and component role directly from configs/refinement.yaml.
        """
        ref_cfg = self.config.get('refinement', {})
        calib_cfg = ref_cfg.get('calibration', {})
        is_pred = (component.lower() == 'predicate')

        if embedding_dim > 768:
            tier = calib_cfg.get('high_dim', {})
            def_neighbors = 15
            def_dist = 0.05
            def_samples = 2
            def_selection = "leaf"
            def_linkage = "complete"
            def_rescue = 0.15 if is_pred else 0.18
            def_macro = 0.18 if is_pred else 0.22
        elif embedding_dim > 512:
            tier = calib_cfg.get('mid_dim', {})
            def_neighbors = 15
            def_dist = 0.03
            def_samples = 2
            def_selection = "eom"
            def_linkage = "average"
            def_rescue = 0.22 if is_pred else 0.26
            def_macro = 0.30 if is_pred else 0.36
        else:
            tier = calib_cfg.get('low_dim', {})
            def_neighbors = 20
            def_dist = 0.00
            def_samples = 1
            def_selection = "eom"
            def_linkage = "average"
            def_rescue = 0.28 if is_pred else 0.32
            def_macro = 0.38 if is_pred else 0.48

        role_key = 'predicate' if is_pred else 'entity'
        role_cfg = tier.get(role_key, {})

        return {
            "umap_n_neighbors": tier.get('umap_n_neighbors', def_neighbors),
            "umap_min_dist": tier.get('umap_min_dist', def_dist),
            "hdbscan_min_samples": tier.get('hdbscan_min_samples', def_samples),
            "hdbscan_selection": tier.get('hdbscan_selection', def_selection),
            "macro_linkage": tier.get('macro_linkage', def_linkage),
            "rescue_tau": role_cfg.get('rescue_tau', def_rescue),
            "tau_macro": role_cfg.get('tau_macro', def_macro),
        }

    def compute_spherical_centroid(
        self,
        member_indices: List[int],
        embeddings: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Computes the frequency-weighted Spherical Fréchet Mean projected onto the unit hypersphere.
        Centroids are L2-normalized: c = sum(w_i * v_i) / ||sum(w_i * v_i)||_2.
        """
        vecs = embeddings[member_indices]
        w = weights[member_indices]
        w_sum = np.sum(vecs * w[:, None], axis=0)
        norm = np.linalg.norm(w_sum)
        return w_sum / norm if norm > 0 else w_sum

    def execute(
        self,
        raw_triples: List[RawTriple],
        subject_map: Optional[Dict[str, str]] = None,
        predicate_map: Optional[Dict[str, str]] = None,
        object_map: Optional[Dict[str, str]] = None,
    ) -> List[RawTriple]:
        print("[Refinement] Starting Stage 2 Pipeline (Vector Clustering & Taxonomic Lifting)...")
        if subject_map is None:
            subject_map = {t.subject: self._nlp_preprocess(t.subject) for t in raw_triples}
        if predicate_map is None:
            predicate_map = {t.predicate: self._nlp_preprocess(t.predicate) for t in raw_triples}
        if object_map is None:
            object_map = {t.object: self._nlp_preprocess(t.object) for t in raw_triples}
        ref_cfg = self.config.get('refinement', {})
        refinement_cap = ref_cfg.get('context_window_cap', 2048)
        max_async = ref_cfg.get('max_async_calls', 1)
        
        enable_lift = ref_cfg.get('enable_taxonomic_lifting', True)
        lift_subj = ref_cfg.get('lift_subjects', True) if enable_lift else False
        lift_pred = ref_cfg.get('lift_predicates', False) if enable_lift else False
        lift_obj = ref_cfg.get('lift_objects', True) if enable_lift else False

        out_dir = Path("outputs/02_refinement")
        out_dir.mkdir(parents=True, exist_ok=True)

        subj_clusters, subj_embeddings_l2, subj_terms = [], None, []
        pred_clusters, pred_embeddings_l2, pred_terms = [], None, []
        obj_clusters, obj_embeddings_l2, obj_terms = [], None, []

        # Calculate normalized frequencies for weighting centroids
        normalized_frequencies = defaultdict(int)
        for t in raw_triples:
            norm_subj = subject_map.get(t.subject, t.subject)
            normalized_frequencies[norm_subj] += 1
            
            norm_pred = predicate_map.get(t.predicate, t.predicate)
            normalized_frequencies[norm_pred] += 1
            
            norm_obj = object_map.get(t.object, t.object)
            normalized_frequencies[norm_obj] += 1

        def cluster_field(terms_list: List[str], label_prefix: str):
            if not terms_list:
                return [], None, None

            num_terms = len(terms_list)
            term_to_idx = {t: i for i, t in enumerate(terms_list)}

            # 1. Obtain L2-normalized embeddings via model-aware vector cache
            term_embeddings_l2 = self.get_or_compute_embeddings(terms_list, label_prefix)
            dim = term_embeddings_l2.shape[1]

            # 2. Resolve dimension- and role-aware clustering hyperparameters from config
            params = self.resolve_clustering_params(dim, label_prefix)

            weights = np.array([1.0 + np.log(normalized_frequencies.get(t, 1)) for t in terms_list])

            if num_terms < 4 or umap is None or hdbscan is None:
                # Direct single-cluster or singleton fallback for minimal vocabularies
                cluster_records = []
                for idx, t in enumerate(terms_list):
                    c_vec = term_embeddings_l2[idx]
                    rec = {
                        "cluster_number": idx,
                        "centroid_term": t,
                        "members": [t]
                    }
                    if ref_cfg.get('print_centroid_vector', True):
                        rec["centroid_vector"] = c_vec.tolist()
                    cluster_records.append(rec)
                return cluster_records, term_embeddings_l2, terms_list

            # 3. Dynamic guardrails for UMAP reduction
            n_neighbors = min(params["umap_n_neighbors"], max(2, num_terms - 1))
            n_components = min(5, max(2, num_terms - 2))
            min_dist = params["umap_min_dist"]
            metric = "cosine"
            random_state = 42

            # Plot UMAP Trustworthiness vs n_components if requested
            plot_trust = ref_cfg.get('plot_trustworthiness', False)
            if plot_trust and trustworthiness is not None and plt is not None and num_terms >= 4:
                try:
                    dim_candidates = [2, 3, 5, 8, 10, 15, 20, 30]
                    valid_dims = sorted(list(set([d for d in dim_candidates if d < num_terms - 1] + [n_components])))
                    trust_scores = []
                    eval_neighbors = min(n_neighbors, max(1, (num_terms // 2) - 1))
                    for d in valid_dims:
                        reducer_eval = umap.UMAP(
                            n_components=d,
                            n_neighbors=n_neighbors,
                            min_dist=min_dist,
                            metric=metric,
                            random_state=random_state,
                            n_jobs=1
                        )
                        embedded_eval = reducer_eval.fit_transform(term_embeddings_l2)
                        score = trustworthiness(term_embeddings_l2, embedded_eval, n_neighbors=eval_neighbors, metric=metric)
                        trust_scores.append(score)

                    plt.figure(figsize=(8, 5))
                    plt.plot(valid_dims, trust_scores, marker='o', linewidth=2, color='#2b5c8f', label='Trustworthiness Score')
                    if n_components in valid_dims:
                        target_score = trust_scores[valid_dims.index(n_components)]
                        plt.plot(n_components, target_score, marker='*', markersize=14, color='#d9534f', label=f'Selected n_components={n_components}')
                    plt.title(f"UMAP Trustworthiness Evaluation - {label_prefix} ({num_terms} terms)", fontsize=12, fontweight='bold')
                    plt.xlabel("Dimensions (n_components)", fontsize=10)
                    plt.ylabel("Trustworthiness Score [0 - 1]", fontsize=10)
                    plt.ylim([0.0, 1.05])
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.legend(loc='lower right')
                    plt.tight_layout()
                    plot_path = out_dir / f"umap_trustworthiness_{label_prefix}.jpg"
                    plt.savefig(plot_path, dpi=300)
                    plt.close()
                    print(f"   -> [{label_prefix}] Exported UMAP Trustworthiness plot to: {plot_path}")
                except Exception as pe:
                    print(f"   -> [{label_prefix}] Warning: Failed to render trustworthiness plot: {pe}")

            # 4. TIER 1: Dense Micro-Clustering & Legitimate Singleton Protection
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=random_state,
                n_jobs=1
            )
            space_reduced = reducer.fit_transform(term_embeddings_l2)

            hdb = hdbscan.HDBSCAN(
                min_cluster_size=2,
                min_samples=params["hdbscan_min_samples"],
                metric="euclidean",
                cluster_selection_method=params["hdbscan_selection"]
            )
            labels_raw = hdb.fit_predict(space_reduced)

            raw_clusters = defaultdict(list)
            outliers = []
            for t, lbl in zip(terms_list, labels_raw):
                if lbl == -1:
                    outliers.append(t)
                else:
                    raw_clusters[lbl].append(t)

            # Compute Spherical Fréchet Centroids for micro-clusters
            micro_centroids = {}
            for lbl, mems in raw_clusters.items():
                m_idx = [term_to_idx[t] for t in mems]
                micro_centroids[lbl] = self.compute_spherical_centroid(m_idx, term_embeddings_l2, weights)

            # Conservative Outlier Rescue (isolate legitimate singletons)
            singletons = []
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
                        raw_clusters[c_lbls[best_c_idx[i]]].append(out_t)
                    else:
                        singletons.append(out_t)
            else:
                singletons.extend(outliers)

            # Split into multi-member micro-clusters and true singletons
            multi_micro_clusters = {}
            for lbl, mems in raw_clusters.items():
                if len(mems) > 1:
                    multi_micro_clusters[lbl] = mems
                else:
                    singletons.extend(mems)

            # 5. TIER 2: Macro-Category Agglomeration
            final_clusters = defaultdict(list)

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
                        n_clusters=None
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

            # 6. Determine Exemplar Terms & Build Output Records
            print_centroid_vec = ref_cfg.get('print_centroid_vector', True)
            cluster_records = []

            for cid, mems in final_clusters.items():
                m_sorted = sorted(list(set(mems)))
                if len(m_sorted) == 1:
                    c_vec = term_embeddings_l2[term_to_idx[m_sorted[0]]]
                    ex_term = m_sorted[0]
                else:
                    m_idx = [term_to_idx[t] for t in m_sorted]
                    c_vec = self.compute_spherical_centroid(m_idx, term_embeddings_l2, weights)
                    dists = 1.0 - np.dot(term_embeddings_l2[m_idx], c_vec)
                    ex_term = m_sorted[int(np.argmin(dists))]

                rec = {
                    "cluster_number": int(cid),
                    "centroid_term": ex_term,
                    "members": m_sorted
                }
                if print_centroid_vec:
                    rec["centroid_vector"] = c_vec.tolist()
                cluster_records.append(rec)

            cluster_records.sort(key=lambda c: len(c["members"]), reverse=True)
            for idx_rec, rec in enumerate(cluster_records):
                rec["cluster_number"] = idx_rec

            return cluster_records, term_embeddings_l2, terms_list

        if enable_lift and (lift_subj or lift_pred or lift_obj):
            # Before loading embeddings, explicitly purge VRAM
            print("[Refinement] Purging VRAM before embedding initialization...")
            purge_vram()

            # 2. SVO Vector Clustering
            print("[Refinement] Step 2: SVO Vector Clustering")
            self._load_embedding_model()

            if lift_subj:
                unique_subjects_norm = sorted(list(set(subject_map.values())))
                print("   -> Clustering Subjects...")
                subj_clusters, subj_embeddings_l2, subj_terms = cluster_field(unique_subjects_norm, "Subject")
                with open(out_dir / "subject_clusters.json", "w") as f:
                    json.dump(subj_clusters, f, indent=2)
            else:
                print("   -> Clustering Subjects SKIPPED (toggled off in config).")

            if lift_pred:
                unique_predicates_norm = sorted(list(set(predicate_map.values())))
                print("   -> Clustering Predicates...")
                pred_clusters, pred_embeddings_l2, pred_terms = cluster_field(unique_predicates_norm, "Predicate")
                with open(out_dir / "predicate_clusters.json", "w") as f:
                    json.dump(pred_clusters, f, indent=2)
            else:
                print("   -> Clustering Predicates SKIPPED (toggled off in config).")

            if lift_obj:
                unique_objects_norm = sorted(list(set(object_map.values())))
                print("   -> Clustering Objects...")
                obj_clusters, obj_embeddings_l2, obj_terms = cluster_field(unique_objects_norm, "Object")
                with open(out_dir / "object_clusters.json", "w") as f:
                    json.dump(obj_clusters, f, indent=2)
            else:
                print("   -> Clustering Objects SKIPPED (toggled off in config).")
        else:
            print("[Refinement] Step 2 & 3: Taxonomic Lifting SKIPPED (toggled off in config).")

        # 3. Taxonomic Lifting
        subject_taxonomic_map = {}
        predicate_taxonomic_map = {}
        object_taxonomic_map = {}

        if enable_lift and (lift_subj or lift_pred or lift_obj):
            print("[Refinement] Step 3: Taxonomic Lifting")

            async def lift_cluster_async(cluster, sem, term_embeddings_l2, terms_list, label_prefix, target_map):
                original_cluster_members = list(cluster["members"])
                cluster_terms = list(original_cluster_members)
                label = cluster["cluster_number"]
                if len(cluster_terms) == 1:
                    target_map[cluster_terms[0]] = cluster_terms[0]
                    return
                
                cluster_key = ",".join(sorted(cluster_terms))
                if self.resume:
                    rows = await self._run_db_query(
                        "SELECT mapped_term FROM taxonomic_lifting WHERE cluster_key = ?",
                        (cluster_key,)
                    )
                    if rows and rows[0][0]:
                        cached_mapped = rows[0][0]
                        print(f"      -> [{label_prefix}] Using cached taxonomic lift for cluster #{label} -> '{cached_mapped}'")
                        for t in original_cluster_members:
                            target_map[t] = cached_mapped
                        return

                async with sem:
                    print(f"      -> [{label_prefix}] Lifting cluster #{label} ({len(cluster_terms)} terms: e.g. {cluster_terms[:3]})...")
                    term_to_idx = {t: i for i, t in enumerate(terms_list)}
                    idx = [term_to_idx[t] for t in cluster_terms]
                    cluster_vecs = term_embeddings_l2[idx]
                    weights = np.array([1.0 + np.log(normalized_frequencies.get(t, 1)) for t in cluster_terms])
                    
                    centroid = np.average(cluster_vecs, axis=0, weights=weights)
                    distances = cosine_distances([centroid], cluster_vecs)[0]
                    closest_indices = np.argsort(distances)[:3]
                    fallback_candidates = [cluster_terms[i] for i in closest_indices]
                    
                    payload = {
                        "cluster_terms": cluster_terms,
                        "top_3_centroid_fallbacks": fallback_candidates
                    }
                    
                    from src.utils.token_helper import estimate_tokens
                    from src.refinement import prompts as ref_prompts
                    
                    base_tokens = estimate_tokens(prompts.TAXONOMIC_LIFTING_USER_PROMPT.format(payload_json="[]"))
                    sys_tokens = estimate_tokens(ref_prompts.TAXONOMIC_LIFTING_SYSTEM_PROMPT) + estimate_tokens(f"\nDomain Context: {self.context.master_domain}")
                    max_payload_tokens = refinement_cap - base_tokens - sys_tokens - 500
                    
                    if estimate_tokens(json.dumps(payload)) > max_payload_tokens:
                        trimmed_terms = list(fallback_candidates)
                        for term in cluster_terms:
                            if term not in trimmed_terms:
                                test_payload = {
                                    "cluster_terms": trimmed_terms + [term],
                                    "top_3_centroid_fallbacks": fallback_candidates
                                }
                                if estimate_tokens(json.dumps(test_payload)) <= max_payload_tokens:
                                    trimmed_terms.append(term)
                                else:
                                    break
                        payload = {
                            "cluster_terms": trimmed_terms,
                            "top_3_centroid_fallbacks": fallback_candidates
                        }
                    
                    user_prompt = prompts.TAXONOMIC_LIFTING_USER_PROMPT.format(
                        payload_json=json.dumps(payload)
                    )
                    full_prompt = f"=== SYSTEM PROMPT ===\n{prompts.TAXONOMIC_LIFTING_SYSTEM_PROMPT}\nDomain Context: {self.context.master_domain}\n\n=== USER PROMPT ===\n{user_prompt}"
                    self._save_payload_text("stage_02_taxonomic_lifting.txt", full_prompt)
                    try:
                        result = await lift_agent.run(
                            user_prompt,
                            deps=self.context.master_domain
                        )
                        mapped_term = result.output.formal_hypernym if result.output.formal_hypernym else (fallback_candidates[0] if fallback_candidates else original_cluster_members[0])
                        print(f"      -> [{label_prefix}] Resolved cluster #{label} -> Mapped to: '{mapped_term}'")
                        for t in original_cluster_members:
                            target_map[t] = mapped_term
                        if self.resume:
                            await self._run_db_query(
                                "INSERT OR REPLACE INTO taxonomic_lifting (cluster_key, mapped_term) VALUES (?, ?)",
                                (cluster_key, mapped_term),
                                is_write=True
                            )
                    except Exception as e:
                        print(f"      -> [{label_prefix}] Error lifting cluster #{label}: {e}")
                        error_record = {
                            "phase": f"taxonomic_lifting_{label_prefix}",
                            "cluster_label": label,
                            "error": str(e)
                        }
                        self._log_error(error_record)
                        mapped_term = fallback_candidates[0] if fallback_candidates else original_cluster_members[0]
                        print(f"      -> [{label_prefix}] Applying fallback mapping to cluster #{label} -> Mapped to: '{mapped_term}'")
                        for t in original_cluster_members:
                            target_map[t] = mapped_term
                        if self.resume:
                            await self._run_db_query(
                                "INSERT OR REPLACE INTO taxonomic_lifting (cluster_key, mapped_term) VALUES (?, ?)",
                                (cluster_key, mapped_term),
                                is_write=True
                            )

            def lift_cluster_sync(cluster, term_embeddings_l2, terms_list, label_prefix, target_map):
                import sqlite3
                original_cluster_members = list(cluster["members"])
                cluster_terms = list(original_cluster_members)
                label = cluster["cluster_number"]
                if len(cluster_terms) == 1:
                    target_map[cluster_terms[0]] = cluster_terms[0]
                    return
                
                cluster_key = ",".join(sorted(cluster_terms))
                if self.resume:
                    conn = sqlite3.connect(self.db_path, timeout=30.0)
                    cursor = conn.cursor()
                    cursor.execute("SELECT mapped_term FROM taxonomic_lifting WHERE cluster_key = ?", (cluster_key,))
                    row = cursor.fetchone()
                    conn.close()
                    if row and row[0]:
                        cached_mapped = row[0]
                        print(f"      -> [{label_prefix}] Using cached taxonomic lift for cluster #{label} -> '{cached_mapped}'")
                        for t in original_cluster_members:
                            target_map[t] = cached_mapped
                        return
                
                print(f"      -> [{label_prefix}] Lifting cluster #{label} ({len(cluster_terms)} terms: e.g. {cluster_terms[:3]})...")
                term_to_idx = {t: i for i, t in enumerate(terms_list)}
                idx = [term_to_idx[t] for t in cluster_terms]
                cluster_vecs = term_embeddings_l2[idx]
                weights = np.array([1.0 + np.log(normalized_frequencies.get(t, 1)) for t in cluster_terms])
                
                centroid = np.average(cluster_vecs, axis=0, weights=weights)
                distances = cosine_distances([centroid], cluster_vecs)[0]
                closest_indices = np.argsort(distances)[:3]
                fallback_candidates = [cluster_terms[i] for i in closest_indices]
                
                payload = {
                    "cluster_terms": cluster_terms,
                    "top_3_centroid_fallbacks": fallback_candidates
                }
                
                from src.utils.token_helper import estimate_tokens
                from src.refinement import prompts as ref_prompts
                
                base_tokens = estimate_tokens(prompts.TAXONOMIC_LIFTING_USER_PROMPT.format(payload_json="[]"))
                sys_tokens = estimate_tokens(ref_prompts.TAXONOMIC_LIFTING_SYSTEM_PROMPT) + estimate_tokens(f"\nDomain Context: {self.context.master_domain}")
                max_payload_tokens = refinement_cap - base_tokens - sys_tokens - 500
                
                if estimate_tokens(json.dumps(payload)) > max_payload_tokens:
                    trimmed_terms = list(fallback_candidates)
                    for term in cluster_terms:
                        if term not in trimmed_terms:
                            test_payload = {
                                "cluster_terms": trimmed_terms + [term],
                                "top_3_centroid_fallbacks": fallback_candidates
                            }
                            if estimate_tokens(json.dumps(test_payload)) <= max_payload_tokens:
                                trimmed_terms.append(term)
                            else:
                                break
                    payload = {
                        "cluster_terms": trimmed_terms,
                        "top_3_centroid_fallbacks": fallback_candidates
                    }
                
                user_prompt = prompts.TAXONOMIC_LIFTING_USER_PROMPT.format(
                    payload_json=json.dumps(payload)
                )
                full_prompt = f"=== SYSTEM PROMPT ===\n{prompts.TAXONOMIC_LIFTING_SYSTEM_PROMPT}\nDomain Context: {self.context.master_domain}\n\n=== USER PROMPT ===\n{user_prompt}"
                self._save_payload_text("stage_02_taxonomic_lifting.txt", full_prompt)
                try:
                    result = lift_agent.run_sync(
                        user_prompt,
                        deps=self.context.master_domain
                    )
                    mapped_term = result.output.formal_hypernym if result.output.formal_hypernym else (fallback_candidates[0] if fallback_candidates else original_cluster_members[0])
                    print(f"      -> [{label_prefix}] Resolved cluster #{label} -> Mapped to: '{mapped_term}'")
                    for t in original_cluster_members:
                        target_map[t] = mapped_term
                    if self.resume:
                        conn = sqlite3.connect(self.db_path, timeout=30.0)
                        cursor = conn.cursor()
                        cursor.execute("INSERT OR REPLACE INTO taxonomic_lifting (cluster_key, mapped_term) VALUES (?, ?)", (cluster_key, mapped_term))
                        conn.commit()
                        conn.close()
                except Exception as e:
                    print(f"      -> [{label_prefix}] Error lifting cluster #{label}: {e}")
                    error_record = {
                        "phase": f"taxonomic_lifting_{label_prefix}",
                        "cluster_label": label,
                        "error": str(e)
                    }
                    self._log_error(error_record)
                    mapped_term = fallback_candidates[0] if fallback_candidates else original_cluster_members[0]
                    print(f"      -> [{label_prefix}] Applying fallback mapping to cluster #{label} -> Mapped to: '{mapped_term}'")
                    for t in original_cluster_members:
                        target_map[t] = mapped_term
                    if self.resume:
                        conn = sqlite3.connect(self.db_path, timeout=30.0)
                        cursor = conn.cursor()
                        cursor.execute("INSERT OR REPLACE INTO taxonomic_lifting (cluster_key, mapped_term) VALUES (?, ?)", (cluster_key, mapped_term))
                        conn.commit()
                        conn.close()

            subj_singletons = sum(1 for c in subj_clusters if len(c["members"]) == 1)
            subj_multi = len(subj_clusters) - subj_singletons
            pred_singletons = sum(1 for c in pred_clusters if len(c["members"]) == 1)
            pred_multi = len(pred_clusters) - pred_singletons
            obj_singletons = sum(1 for c in obj_clusters if len(c["members"]) == 1)
            obj_multi = len(obj_clusters) - obj_singletons

            if lift_subj:
                print(f"   -> Subjects: {len(subj_clusters)} total clusters ({subj_singletons} single-member resolved locally, {subj_multi} multi-member requiring lifting)")
            if lift_pred:
                print(f"   -> Predicates: {len(pred_clusters)} total clusters ({pred_singletons} single-member resolved locally, {pred_multi} multi-member requiring lifting)")
            if lift_obj:
                print(f"   -> Objects: {len(obj_clusters)} total clusters ({obj_singletons} single-member resolved locally, {obj_multi} multi-member requiring lifting)")

            if self.use_async:
                async def run_lifting():
                    sem = asyncio.Semaphore(max_async)
                    tasks = []
                    if lift_subj:
                        for cluster in subj_clusters:
                            tasks.append(lift_cluster_async(cluster, sem, subj_embeddings_l2, subj_terms, "Subject", subject_taxonomic_map))
                    if lift_pred:
                        for cluster in pred_clusters:
                            tasks.append(lift_cluster_async(cluster, sem, pred_embeddings_l2, pred_terms, "Predicate", predicate_taxonomic_map))
                    if lift_obj:
                        for cluster in obj_clusters:
                            tasks.append(lift_cluster_async(cluster, sem, obj_embeddings_l2, obj_terms, "Object", object_taxonomic_map))
                    if tasks:
                        await asyncio.gather(*tasks)
                asyncio.run(run_lifting())
            else:
                if lift_subj:
                    for cluster in subj_clusters:
                        lift_cluster_sync(cluster, subj_embeddings_l2, subj_terms, "Subject", subject_taxonomic_map)
                if lift_pred:
                    for cluster in pred_clusters:
                        lift_cluster_sync(cluster, pred_embeddings_l2, pred_terms, "Predicate", predicate_taxonomic_map)
                if lift_obj:
                    for cluster in obj_clusters:
                        lift_cluster_sync(cluster, obj_embeddings_l2, obj_terms, "Object", object_taxonomic_map)

        taxonomic_map = {}
        if lift_subj:
            with open(out_dir / "subject_taxonomic_map.json", "w") as f:
                json.dump(subject_taxonomic_map, f, indent=2)
            taxonomic_map.update(subject_taxonomic_map)

        if lift_pred:
            with open(out_dir / "predicate_taxonomic_map.json", "w") as f:
                json.dump(predicate_taxonomic_map, f, indent=2)
            taxonomic_map.update(predicate_taxonomic_map)

        if lift_obj:
            with open(out_dir / "object_taxonomic_map.json", "w") as f:
                json.dump(object_taxonomic_map, f, indent=2)
            taxonomic_map.update(object_taxonomic_map)

        if taxonomic_map:
            with open(out_dir / "taxonomic_map.json", "w") as f:
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
            # 1. Normalization lookup: case-sensitive direct with case-insensitive fallback
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
            
            # 2. Taxonomic lookup: case-sensitive direct with case-insensitive fallback
            final_subj = subject_taxonomic_map.get(norm_subj)
            if final_subj is None:
                final_subj = lower_subj_tax.get(norm_subj.lower(), norm_subj)
                
            final_pred = predicate_taxonomic_map.get(norm_pred)
            if final_pred is None:
                final_pred = lower_pred_tax.get(norm_pred.lower(), norm_pred)
                
            final_obj = object_taxonomic_map.get(norm_obj)
            if final_obj is None:
                final_obj = lower_obj_tax.get(norm_obj.lower(), norm_obj)
                
            # Assign final resolved values
            t.subject = final_subj
            t.predicate = final_pred
            t.object = final_obj
            
        # 4. Contextual Sentence Grounding & Fidelity Computation
        triplet_dicts = self._compute_and_cache_sentence_context(refined_triples, out_dir)

        # Save refined triplets to outputs/02_refinement/refined_triplets.json
        refined_path = out_dir / "refined_triplets.json"
        with open(refined_path, "w", encoding="utf-8") as f:
            json.dump(triplet_dicts, f, indent=2)

        print(f"[Refinement] Exported refined triplets to: {refined_path}")
        print("[Refinement] Pipeline complete. Purging VRAM.")
        purge_vram()
        return refined_triples

    def _compute_and_cache_sentence_context(
        self,
        refined_triples: List[RawTriple],
        out_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Embeds unique source quotes, computes edge contextual fidelity (gamma_e),
        and exports a non-breaking diagnostic report.
        """
        print("[Refinement] Step 4: Computing Contextual Sentence Vectors & Edge Fidelity...")
        self._load_embedding_model()

        ref_cfg = self.config.get('refinement', {})
        cache_dir = Path(ref_cfg.get('vector_cache_dir', 'outputs/02_refinement/cache'))
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_name = getattr(self, 'embedding_model_name', ref_cfg.get('embedding_model', 'all-MiniLM-L6-v2'))
        model_slug = re.sub(r'[^a-zA-Z0-9_]', '_', model_name)

        # 1. Collect unique source quotes
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

        # Cache paths for sentence vectors
        sent_npy_path = cache_dir / f"sentence_{model_slug}_embeddings.npy"
        sent_meta_path = cache_dir / f"sentence_{model_slug}_meta.json"

        sent_embeddings = None
        if sent_npy_path.exists() and sent_meta_path.exists():
            try:
                with open(sent_meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("model_name") == model_name and meta.get("vocab") == unique_quotes:
                    sent_embeddings = np.load(sent_npy_path)
                    print(f"   -> Loaded {len(unique_quotes)} cached sentence vectors ({sent_embeddings.shape[1]}d)")
            except Exception as e:
                print(f"   -> Sentence cache read warning: {e}; recomputing...")

        if sent_embeddings is None or len(sent_embeddings) != len(unique_quotes):
            sent_embeddings = self.embedding_model.encode(
                unique_quotes,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=ref_cfg.get('batch_size', 128)
            )
            try:
                np.save(sent_npy_path, sent_embeddings)
                with open(sent_meta_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "model_name": model_name,
                        "dimensions": int(sent_embeddings.shape[1]),
                        "vocab": unique_quotes
                    }, f, indent=2)
                print(f"   -> Cached {len(unique_quotes)} sentence vectors to {sent_npy_path.name}")
            except Exception as e:
                print(f"   -> Warning: Failed to cache sentence vectors: {e}")

        # 2. Encode unique relational phrases ("subject predicate object")
        phrase_list = [f"{get_field(t, 'subject')} {get_field(t, 'predicate')} {get_field(t, 'object')}".strip() for t in refined_triples]
        unique_phrases = sorted(list(set(phrase_list)))
        phrase_to_idx = {p: i for i, p in enumerate(unique_phrases)}

        phrase_embeddings = self.embedding_model.encode(
            unique_phrases,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=ref_cfg.get('batch_size', 128)
        )

        # 3. Compute Contextual Fidelity (gamma_e) for each triplet
        triplet_dicts = []
        fidelity_scores = []

        for i, t in enumerate(refined_triples):
            if hasattr(t, "model_dump"):
                td = t.model_dump()
            elif isinstance(t, dict):
                td = dict(t)
            else:
                td = dict(t.__dict__)

            q = get_field(t, "source_quote")
            phrase = phrase_list[i]

            if q in quotes_map and phrase in phrase_to_idx:
                q_idx = quotes_map[q]
                p_idx = phrase_to_idx[phrase]
                gamma_e = float(np.dot(phrase_embeddings[p_idx], sent_embeddings[q_idx]))
                gamma_e = max(0.0, min(1.0, gamma_e))  # Clip to [0.0, 1.0]
            else:
                gamma_e = 0.5  # Neutral fallback

            td["contextual_fidelity"] = round(gamma_e, 4)
            td["sentence_id"] = quotes_map.get(q, -1)
            triplet_dicts.append(td)
            fidelity_scores.append(gamma_e)

        # 4. Generate Diagnostic Grounding Report
        f_arr = np.array(fidelity_scores)
        q25, q50, q75 = np.percentile(f_arr, [25, 50, 75])

        indexed_scores = sorted(enumerate(fidelity_scores), key=lambda x: x[1], reverse=True)
        top_samples = [
            {
                "phrase": phrase_list[idx],
                "quote": get_field(refined_triples[idx], "source_quote")[:120],
                "fidelity": round(score, 4)
            }
            for idx, score in indexed_scores[:5]
        ]
        bottom_samples = [
            {
                "phrase": phrase_list[idx],
                "quote": get_field(refined_triples[idx], "source_quote")[:120],
                "fidelity": round(score, 4)
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
                    "q75": round(float(q75), 4)
                }
            },
            "sample_high_fidelity_triplets": top_samples,
            "sample_low_fidelity_triplets": bottom_samples
        }

        diag_path = out_dir / "contextual_grounding_diagnostics.json"
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diag_report, f, indent=2)

        print(f"   -> Fidelity: mean={np.mean(f_arr):.3f}, median={q50:.3f}, std={np.std(f_arr):.3f}")
        print(f"   -> Exported grounding diagnostics to: {diag_path}")

        return triplet_dicts

    # Alias for backward compatibility
    execute_part_2 = execute
