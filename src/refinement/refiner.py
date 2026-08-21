"""
SemanticPrism Stage 2: Refinement Pipeline
Handles lexical normalization, embedding mapping, clustering, and taxonomic lifting.
"""

import json
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from sklearn.preprocessing import normalize
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
except ImportError:
    normalize = None
    AgglomerativeClustering = None
    cosine_distances = None

from src.agents.refinement_agents import subject_norm_agent, predicate_norm_agent, object_norm_agent, lift_agent
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
        CREATE TABLE IF NOT EXISTS lexical_normalization (
            original_term TEXT PRIMARY KEY,
            normalized_term TEXT,
            term_type TEXT
        );
        """)
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

    def _log_error(self, record: Dict[str, Any]):
        """Helper to append error records to stage_02_refinement_errors.json."""
        try:
            log_path = self.log_dir / "stage_02_refinement_errors.json"
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[Refinement] Failed to write error log: {e}")

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
            model_name = self.config.get('refinement', {}).get('embedding_model', 'all-MiniLM-L6-v2')
            
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
                print(f"[Refinement] Attempting online download from HuggingFace...")
                self.embedding_model = SentenceTransformer(
                    model_name, 
                    cache_folder=str(models_dir),
                    local_files_only=False
                )

    def execute(self, raw_triples: List[RawTriple], original_themes: List[Any], master_themes: List[str]):
        print("[Refinement] Starting Stage 2 Pipeline...")
        subject_map, predicate_map, object_map = self.execute_part_1(raw_triples, master_themes)
        return self.execute_part_2(raw_triples, original_themes, master_themes, subject_map, predicate_map, object_map)

    def execute_part_1(self, raw_triples: List[RawTriple], master_themes: List[str]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        print("[Refinement] Starting Stage 2 Pipeline Part 1 (Lexical Normalization)...")
        # 1. Lexical Normalization
        print("[Refinement] Step 1: Lexical Normalization")
        normalized_triples = [t.model_copy(deep=True) for t in raw_triples]
        
        unique_subjects = set()
        unique_predicates = set()
        unique_objects = set()
        
        for t in normalized_triples:
            t.subject = self._nlp_preprocess(t.subject)
            t.predicate = self._nlp_preprocess(t.predicate)
            t.object = self._nlp_preprocess(t.object)
            unique_subjects.add(t.subject)
            unique_predicates.add(t.predicate)
            unique_objects.add(t.object)
            
            # Count frequencies
            self.context.term_frequencies[t.subject] += 1
            self.context.term_frequencies[t.predicate] += 1
            self.context.term_frequencies[t.object] += 1

        batch_size = self.config.get('refinement', {}).get('batch_size', 15)
        max_async = self.config.get('refinement', {}).get('max_async_calls', 1)
        refinement_cap = self.config.get('refinement', {}).get('context_window_cap', 2048)
        
        out_dir = Path("outputs/02_refinement")
        out_dir.mkdir(parents=True, exist_ok=True)

        async def process_batch(batch: list, sem: asyncio.Semaphore, agent, batch_idx: int, total_batches: int, target_map: dict, term_type: str):
            async with sem:
                print(f"      -> Running API call for batch {batch_idx}/{total_batches}...")
                user_prompt = prompts.LLM_PREPROCESSING_USER_PROMPT.format(
                    master_themes=', '.join(master_themes),
                    batch_json=json.dumps(batch)
                )
                try:
                    # Read timeout from config (default to 300.0s). Set to 0 or null to disable timeouts entirely.
                    timeout_val = self.config.get('refinement', {}).get('timeout', 300.0)
                    if timeout_val and timeout_val > 0:
                        result = await asyncio.wait_for(
                            agent.run(
                                user_prompt,
                                deps=self.context.master_domain
                            ),
                            timeout=timeout_val
                        )
                    else:
                        result = await agent.run(
                            user_prompt,
                            deps=self.context.master_domain
                        )
                    for pair in result.output.tokens:
                        target_map[pair.original] = pair.normalized
                        await self._run_db_query(
                            "INSERT OR REPLACE INTO lexical_normalization (original_term, normalized_term, term_type) VALUES (?, ?, ?)",
                            (pair.original, pair.normalized, term_type),
                            is_write=True
                        )
                except asyncio.TimeoutError:
                    print(f"      -> Timeout error on API call for batch {batch_idx}/{total_batches}")
                    error_record = {
                        "phase": "lexical_normalization",
                        "batch_index": batch_idx,
                        "total_batches": total_batches,
                        "error": "TimeoutError: API call exceeded configured limit"
                    }
                    self._log_error(error_record)
                    for term in batch:
                        target_map[term] = term
                        await self._run_db_query(
                            "INSERT OR REPLACE INTO lexical_normalization (original_term, normalized_term, term_type) VALUES (?, ?, ?)",
                            (term, term, term_type),
                            is_write=True
                        )
                except Exception as e:
                    print(f"      -> Error normalizing batch {batch_idx}/{total_batches}: {e}")
                    error_record = {
                        "phase": "lexical_normalization",
                        "batch_index": batch_idx,
                        "total_batches": total_batches,
                        "error": str(e)
                    }
                    self._log_error(error_record)
                    for term in batch:
                        target_map[term] = term
                        await self._run_db_query(
                            "INSERT OR REPLACE INTO lexical_normalization (original_term, normalized_term, term_type) VALUES (?, ?, ?)",
                            (term, term, term_type),
                            is_write=True
                        )

        async def run_all_batches(term_set: set, agent, target_map: dict, term_type: str):
            term_list = sorted(list(term_set))
            sem = asyncio.Semaphore(max_async)
            tasks = []
            
            # Estimate maximum allowed tokens for prompt JSON payload
            from src.utils.token_helper import estimate_tokens
            from src.refinement import prompts as ref_prompts
            
            base_prompt_tokens = estimate_tokens(
                prompts.LLM_PREPROCESSING_USER_PROMPT.format(
                    master_themes=', '.join(master_themes),
                    batch_json="[]"
                )
            )
            # Normalize with sys prompt size and output buffer (500 tokens)
            sys_prompt_tokens = estimate_tokens(ref_prompts.SUBJECT_NORMALIZATION_SYSTEM_PROMPT) + estimate_tokens(f"\nDomain Context: {self.context.master_domain}")
            max_json_tokens = refinement_cap - base_prompt_tokens - sys_prompt_tokens - 500
            if max_json_tokens < 100:
                max_json_tokens = 100
                
            # Dynamic batch grouping
            batches = []
            current_batch = []
            for term in term_list:
                hypothetical_batch = current_batch + [term]
                hypothetical_tokens = estimate_tokens(json.dumps(hypothetical_batch))
                if hypothetical_tokens > max_json_tokens or len(current_batch) >= batch_size:
                    if current_batch:
                        batches.append(current_batch)
                        current_batch = [term]
                    else:
                        batches.append([term])
                        current_batch = []
                else:
                    current_batch.append(term)
            if current_batch:
                batches.append(current_batch)
                
            total_batches = len(batches)
            
            async def wrapped_process(batch, sem, agent, batch_idx):
                print(f"      -> Enqueued batch {batch_idx}/{total_batches} ({len(batch)} terms)...")
                try:
                    await process_batch(batch, sem, agent, batch_idx, total_batches, target_map, term_type)
                    print(f"      -> Completed batch {batch_idx}/{total_batches}")
                except Exception as e:
                    print(f"      -> Unexpected error on batch {batch_idx}/{total_batches}: {e}")

            for idx, batch in enumerate(batches, start=1):
                tasks.append(wrapped_process(batch, sem, agent, idx))
                
            if tasks:
                await asyncio.gather(*tasks)

        pre_subject_map = {}
        pre_predicate_map = {}
        pre_object_map = {}

        async def process_all_terms():
            if self.resume:
                print("   -> Loading cached lexical normalization mappings from database...")
                cached_rows = await self._run_db_query(
                    "SELECT original_term, normalized_term, term_type FROM lexical_normalization"
                )
                for row in cached_rows:
                    orig, norm, term_type = row
                    if term_type == "subject":
                        pre_subject_map[orig] = norm
                    elif term_type == "predicate":
                        pre_predicate_map[orig] = norm
                    elif term_type == "object":
                        pre_object_map[orig] = norm
            else:
                print("   -> Starting fresh run (skipping cached database loading)...")

            subjects_to_process = unique_subjects - set(pre_subject_map.keys())
            predicates_to_process = unique_predicates - set(pre_predicate_map.keys())
            objects_to_process = unique_objects - set(pre_object_map.keys())

            print(f"   -> Normalizing Subjects ({len(subjects_to_process)} remaining)...")
            await run_all_batches(subjects_to_process, subject_norm_agent, pre_subject_map, "subject")
                
            print(f"   -> Normalizing Predicates ({len(predicates_to_process)} remaining)...")
            await run_all_batches(predicates_to_process, predicate_norm_agent, pre_predicate_map, "predicate")
                
            print(f"   -> Normalizing Objects ({len(objects_to_process)} remaining)...")
            await run_all_batches(objects_to_process, object_norm_agent, pre_object_map, "object")
            
        # Execute all normalization passes inside a single shared event loop
        asyncio.run(process_all_terms())

        # Construct final case-sensitive maps using the original raw string keys
        subject_map = {}
        predicate_map = {}
        object_map = {}

        for t in raw_triples:
            pre_subj = self._nlp_preprocess(t.subject)
            pre_pred = self._nlp_preprocess(t.predicate)
            pre_obj = self._nlp_preprocess(t.object)
            
            subject_map[t.subject] = pre_subject_map.get(pre_subj, pre_subj)
            predicate_map[t.predicate] = pre_predicate_map.get(pre_pred, pre_pred)
            object_map[t.object] = pre_object_map.get(pre_obj, pre_obj)

        with open(out_dir / "subject_normalization_map.json", "w") as f:
            json.dump(subject_map, f, indent=2)

        with open(out_dir / "predicate_normalization_map.json", "w") as f:
            json.dump(predicate_map, f, indent=2)

        with open(out_dir / "object_normalization_map.json", "w") as f:
            json.dump(object_map, f, indent=2)

        # Consolidate global normalization map for backwards compatibility
        normalization_map = {}
        normalization_map.update(subject_map)
        normalization_map.update(predicate_map)
        normalization_map.update(object_map)

        with open(out_dir / "normalization_map.json", "w") as f:
            json.dump(normalization_map, f, indent=2)

        print("[Refinement] Stage 2 Pipeline Part 1 Completed.")
        return subject_map, predicate_map, object_map

    def execute_part_2(self, raw_triples: List[RawTriple], original_themes: List[Any], master_themes: List[str], subject_map: Dict[str, str], predicate_map: Dict[str, str], object_map: Dict[str, str]) -> List[RawTriple]:
        print("[Refinement] Starting Stage 2 Pipeline Part 2 (Clustering, Lifting, Theme Mapping)...")
        refinement_cap = self.config.get('refinement', {}).get('context_window_cap', 2048)
        max_async = self.config.get('refinement', {}).get('max_async_calls', 1)
        out_dir = Path("outputs/02_refinement")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Before loading embeddings, explicitly purge VRAM
        print("[Refinement] Purging VRAM before embedding initialization...")
        purge_vram()

        # 2. SVO Vector Clustering
        print("[Refinement] Step 2: SVO Vector Clustering")
        self._load_embedding_model()

        # Calculate normalized frequencies for weighting centroids
        normalized_frequencies = defaultdict(int)
        for t in raw_triples:
            norm_subj = subject_map.get(t.subject, t.subject)
            normalized_frequencies[norm_subj] += 1
            
            norm_pred = predicate_map.get(t.predicate, t.predicate)
            normalized_frequencies[norm_pred] += 1
            
            norm_obj = object_map.get(t.object, t.object)
            normalized_frequencies[norm_obj] += 1

        threshold = self.config.get('refinement', {}).get('clustering_threshold', 0.4)

        def cluster_field(terms_list: List[str], label_prefix: str):
            if not terms_list:
                return [], None, None
                
            term_embeddings = self.embedding_model.encode(terms_list)
            term_embeddings_l2 = normalize(term_embeddings, norm='l2')
            
            clustering = AgglomerativeClustering(
                metric='cosine', 
                linkage='average', 
                distance_threshold=threshold, 
                n_clusters=None
            )
            
            if len(terms_list) > 1:
                cluster_labels = clustering.fit_predict(term_embeddings_l2)
            else:
                cluster_labels = [0]
                
            clusters_dict = defaultdict(list)
            for term, label in zip(terms_list, cluster_labels):
                clusters_dict[label].append(term)
                
            cluster_records = []
            for label, cluster_members in clusters_dict.items():
                if len(cluster_members) == 1:
                    centroid = term_embeddings_l2[terms_list.index(cluster_members[0])]
                    centroid_term = cluster_members[0]
                else:
                    idx = [terms_list.index(t) for t in cluster_members]
                    cluster_vecs = term_embeddings_l2[idx]
                    weights = np.array([normalized_frequencies.get(t, 1) for t in cluster_members])
                    
                    centroid = np.average(cluster_vecs, axis=0, weights=weights)
                    distances = cosine_distances([centroid], cluster_vecs)[0]
                    closest_idx = np.argmin(distances)
                    centroid_term = cluster_members[closest_idx]
                    
                cluster_records.append({
                    "cluster_number": int(label),
                    "centroid_term": centroid_term,
                    "centroid_vector": centroid.tolist(),
                    "members": sorted(cluster_members)
                })
                
            cluster_records.sort(key=lambda c: c["cluster_number"])
            return cluster_records, term_embeddings_l2, terms_list

        unique_subjects_norm = sorted(list(set(subject_map.values())))
        unique_predicates_norm = sorted(list(set(predicate_map.values())))
        unique_objects_norm = sorted(list(set(object_map.values())))

        print("   -> Clustering Subjects...")
        subj_clusters, subj_embeddings_l2, subj_terms = cluster_field(unique_subjects_norm, "Subject")
        with open(out_dir / "subject_clusters.json", "w") as f:
            json.dump(subj_clusters, f, indent=2)

        print("   -> Clustering Predicates...")
        pred_clusters, pred_embeddings_l2, pred_terms = cluster_field(unique_predicates_norm, "Predicate")
        with open(out_dir / "predicate_clusters.json", "w") as f:
            json.dump(pred_clusters, f, indent=2)

        print("   -> Clustering Objects...")
        obj_clusters, obj_embeddings_l2, obj_terms = cluster_field(unique_objects_norm, "Object")
        with open(out_dir / "object_clusters.json", "w") as f:
            json.dump(obj_clusters, f, indent=2)

        # 3. Taxonomic Lifting
        print("[Refinement] Step 3: Taxonomic Lifting")

        subject_taxonomic_map = {}
        predicate_taxonomic_map = {p: p for p in unique_predicates_norm}
        object_taxonomic_map = {}

        async def lift_cluster_async(cluster, sem, term_embeddings_l2, terms_list, label_prefix, target_map):
            cluster_terms = cluster["members"]
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
                    for t in cluster_terms:
                        target_map[t] = cached_mapped
                    return

            async with sem:
                print(f"      -> [{label_prefix}] Lifting cluster #{label} ({len(cluster_terms)} terms: e.g. {cluster_terms[:3]})...")
                idx = [terms_list.index(t) for t in cluster_terms]
                cluster_vecs = term_embeddings_l2[idx]
                weights = np.array([normalized_frequencies.get(t, 1) for t in cluster_terms])
                
                centroid = np.average(cluster_vecs, axis=0, weights=weights)
                distances = cosine_distances([centroid], cluster_vecs)[0]
                closest_indices = np.argsort(distances)[:3]
                fallback_candidates = [cluster_terms[i] for i in closest_indices]
                
                payload = {
                    "cluster_terms": cluster_terms,
                    "top_3_centroid_fallbacks": fallback_candidates
                }
                
                # Validate cluster payload size
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
                    cluster_terms = trimmed_terms
                    payload = {
                        "cluster_terms": cluster_terms,
                        "top_3_centroid_fallbacks": fallback_candidates
                    }
                
                user_prompt = prompts.TAXONOMIC_LIFTING_USER_PROMPT.format(
                    payload_json=json.dumps(payload)
                )
                try:
                    result = await lift_agent.run(
                        user_prompt,
                        deps=self.context.master_domain
                    )
                    mapped_term = result.output.formal_hypernym if result.output.formal_hypernym else (fallback_candidates[0] if fallback_candidates else cluster_terms[0])
                    print(f"      -> [{label_prefix}] Resolved cluster #{label} -> Mapped to: '{mapped_term}'")
                    for t in cluster_terms:
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
                    mapped_term = fallback_candidates[0] if fallback_candidates else cluster_terms[0]
                    print(f"      -> [{label_prefix}] Applying fallback mapping to cluster #{label} -> Mapped to: '{mapped_term}'")
                    for t in cluster_terms:
                        target_map[t] = mapped_term
                    if self.resume:
                        await self._run_db_query(
                            "INSERT OR REPLACE INTO taxonomic_lifting (cluster_key, mapped_term) VALUES (?, ?)",
                            (cluster_key, mapped_term),
                            is_write=True
                        )

        def lift_cluster_sync(cluster, term_embeddings_l2, terms_list, label_prefix, target_map):
            import sqlite3
            cluster_terms = cluster["members"]
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
                    for t in cluster_terms:
                        target_map[t] = cached_mapped
                    return
            
            print(f"      -> [{label_prefix}] Lifting cluster #{label} ({len(cluster_terms)} terms: e.g. {cluster_terms[:3]})...")
            idx = [terms_list.index(t) for t in cluster_terms]
            cluster_vecs = term_embeddings_l2[idx]
            weights = np.array([normalized_frequencies.get(t, 1) for t in cluster_terms])
            
            centroid = np.average(cluster_vecs, axis=0, weights=weights)
            distances = cosine_distances([centroid], cluster_vecs)[0]
            closest_indices = np.argsort(distances)[:3]
            fallback_candidates = [cluster_terms[i] for i in closest_indices]
            
            payload = {
                "cluster_terms": cluster_terms,
                "top_3_centroid_fallbacks": fallback_candidates
            }
            
            # Validate cluster payload size
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
                cluster_terms = trimmed_terms
                payload = {
                    "cluster_terms": cluster_terms,
                    "top_3_centroid_fallbacks": fallback_candidates
                }
            
            user_prompt = prompts.TAXONOMIC_LIFTING_USER_PROMPT.format(
                payload_json=json.dumps(payload)
            )
            try:
                result = lift_agent.run_sync(
                    user_prompt,
                    deps=self.context.master_domain
                )
                mapped_term = result.output.formal_hypernym if result.output.formal_hypernym else (fallback_candidates[0] if fallback_candidates else cluster_terms[0])
                print(f"      -> [{label_prefix}] Resolved cluster #{label} -> Mapped to: '{mapped_term}'")
                for t in cluster_terms:
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
                mapped_term = fallback_candidates[0] if fallback_candidates else cluster_terms[0]
                print(f"      -> [{label_prefix}] Applying fallback mapping to cluster #{label} -> Mapped to: '{mapped_term}'")
                for t in cluster_terms:
                    target_map[t] = mapped_term
                if self.resume:
                    conn = sqlite3.connect(self.db_path, timeout=30.0)
                    cursor = conn.cursor()
                    cursor.execute("INSERT OR REPLACE INTO taxonomic_lifting (cluster_key, mapped_term) VALUES (?, ?)", (cluster_key, mapped_term))
                    conn.commit()
                    conn.close()

        # Count clusters that need LLM resolution vs those resolved locally (singletons)
        subj_singletons = sum(1 for c in subj_clusters if len(c["members"]) == 1)
        subj_multi = len(subj_clusters) - subj_singletons
        obj_singletons = sum(1 for c in obj_clusters if len(c["members"]) == 1)
        obj_multi = len(obj_clusters) - obj_singletons

        print(f"   -> Subjects: {len(subj_clusters)} total clusters ({subj_singletons} single-member resolved locally, {subj_multi} multi-member requiring lifting)")
        print(f"   -> Objects: {len(obj_clusters)} total clusters ({obj_singletons} single-member resolved locally, {obj_multi} multi-member requiring lifting)")

        if self.use_async:
            async def run_lifting():
                sem = asyncio.Semaphore(max_async)
                tasks = []
                for cluster in subj_clusters:
                    tasks.append(lift_cluster_async(cluster, sem, subj_embeddings_l2, subj_terms, "Subject", subject_taxonomic_map))
                for cluster in obj_clusters:
                    tasks.append(lift_cluster_async(cluster, sem, obj_embeddings_l2, obj_terms, "Object", object_taxonomic_map))
                await asyncio.gather(*tasks)
            asyncio.run(run_lifting())
        else:
            for cluster in subj_clusters:
                lift_cluster_sync(cluster, subj_embeddings_l2, subj_terms, "Subject", subject_taxonomic_map)
            for cluster in obj_clusters:
                lift_cluster_sync(cluster, obj_embeddings_l2, obj_terms, "Object", object_taxonomic_map)

        # Write individual taxonomic maps to disk
        with open(out_dir / "subject_taxonomic_map.json", "w") as f:
            json.dump(subject_taxonomic_map, f, indent=2)

        # Re-verify and clean up other maps
        with open(out_dir / "predicate_taxonomic_map.json", "w") as f:
            json.dump(predicate_taxonomic_map, f, indent=2)

        with open(out_dir / "object_taxonomic_map.json", "w") as f:
            json.dump(object_taxonomic_map, f, indent=2)

        # Consolidate global taxonomic map for backwards compatibility
        taxonomic_map = {}
        taxonomic_map.update(subject_taxonomic_map)
        taxonomic_map.update(predicate_taxonomic_map)
        taxonomic_map.update(object_taxonomic_map)

        with open(out_dir / "taxonomic_map.json", "w") as f:
            json.dump(taxonomic_map, f, indent=2)

        # Build case-insensitive maps for final triple re-mapping lookups
        lower_subj_norm = {k.lower(): v for k, v in subject_map.items()}
        lower_pred_norm = {k.lower(): v for k, v in predicate_map.items()}
        lower_obj_norm = {k.lower(): v for k, v in object_map.items()}
        
        lower_subj_tax = {k.lower(): v for k, v in subject_taxonomic_map.items()}
        lower_pred_tax = {k.lower(): v for k, v in predicate_taxonomic_map.items()}
        lower_obj_tax = {k.lower(): v for k, v in object_taxonomic_map.items()}

        # Apply both Lexical Normalization and Taxonomic Lifting maps to raw triples
        normalized_triples = [t.model_copy(deep=True) for t in raw_triples]
        for t in normalized_triples:
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
            
        with open(out_dir / "refined_triplets.json", "w") as f:
            json.dump([t.model_dump() for t in normalized_triples], f, indent=2)

        # 4. Theme-Based Embedding Mapping
        print("[Refinement] Step 4: Theme-Based Embedding Mapping")
        
        orig_theme_texts = []
        for theme in original_themes:
            title = theme.title if hasattr(theme, 'title') else theme.get('title', '')
            desc = theme.description if hasattr(theme, 'description') else theme.get('description', '')
            reasoning = theme.reasoning if hasattr(theme, 'reasoning') else theme.get('reasoning', '')
            orig_theme_texts.append(f"{title} {desc} {reasoning}")
            
        master_theme_texts = [f"{self.context.master_domain} {mt}" for mt in master_themes]
        
        orig_embeddings = self.embedding_model.encode(orig_theme_texts, normalize_embeddings=True)
        master_embeddings = self.embedding_model.encode(master_theme_texts, normalize_embeddings=True)
        
        theme_mapping = defaultdict(list)
        similarity_matrix = cosine_similarity(orig_embeddings, master_embeddings)
        
        for i in range(len(orig_theme_texts)):
            best_idx = np.argmax(similarity_matrix[i])
            orig_title = original_themes[i].title if hasattr(original_themes[i], 'title') else original_themes[i].get('title', f"theme_{i}")
            theme_mapping[master_themes[best_idx]].append(orig_title)

        with open(out_dir / "theme_mapping_clusters.json", "w") as f:
            json.dump(theme_mapping, f, indent=2)

        print("[Refinement] Pipeline complete. Purging VRAM.")
        purge_vram()
        return normalized_triples
