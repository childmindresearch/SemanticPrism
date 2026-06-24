"""
SemanticPrism Stage 2: Refinement Pipeline
Handles lexical normalization, embedding mapping, clustering, and taxonomic lifting.
"""

import json
import re
import asyncio
from pathlib import Path
from typing import List, Dict, Any
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

        async def process_batch(batch: list, sem: asyncio.Semaphore, agent, batch_idx: int, total_batches: int, target_map: dict):
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

        async def run_all_batches(term_set: set, agent, target_map: dict):
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
                    await process_batch(batch, sem, agent, batch_idx, total_batches, target_map)
                    print(f"      -> Completed batch {batch_idx}/{total_batches}")
                except Exception as e:
                    print(f"      -> Unexpected error on batch {batch_idx}/{total_batches}: {e}")

            for idx, batch in enumerate(batches, start=1):
                tasks.append(wrapped_process(batch, sem, agent, idx))
                
            if tasks:
                await asyncio.gather(*tasks)

        subject_map = {}
        predicate_map = {}
        object_map = {}

        async def process_all_terms():
            print("   -> Normalizing Subjects...")
            await run_all_batches(unique_subjects, subject_norm_agent, subject_map)
            with open(out_dir / "subject_normalization_map.json", "w") as f:
                json.dump(subject_map, f, indent=2)
                
            print("   -> Normalizing Predicates...")
            await run_all_batches(unique_predicates, predicate_norm_agent, predicate_map)
            with open(out_dir / "predicate_normalization_map.json", "w") as f:
                json.dump(predicate_map, f, indent=2)
                
            print("   -> Normalizing Objects...")
            await run_all_batches(unique_objects, object_norm_agent, object_map)
            with open(out_dir / "object_normalization_map.json", "w") as f:
                json.dump(object_map, f, indent=2)
            
        # Execute all normalization passes inside a single shared event loop
        asyncio.run(process_all_terms())

        # Consolidate global normalization map for backwards compatibility
        normalization_map = {}
        normalization_map.update(subject_map)
        normalization_map.update(predicate_map)
        normalization_map.update(object_map)

        with open(out_dir / "normalization_map.json", "w") as f:
            json.dump(normalization_map, f, indent=2)

        # Before loading embeddings, explicitly purge VRAM
        print("[Refinement] Purging VRAM before embedding initialization...")
        purge_vram()

        # 2. SVO Vector Clustering
        print("[Refinement] Step 2: SVO Vector Clustering")
        self._load_embedding_model()

        # Calculate normalized frequencies for weighting centroids
        normalized_frequencies = defaultdict(int)
        for t in raw_triples:
            pre_subj = self._nlp_preprocess(t.subject)
            norm_subj = subject_map.get(pre_subj, pre_subj)
            normalized_frequencies[norm_subj] += 1
            
            pre_pred = self._nlp_preprocess(t.predicate)
            norm_pred = predicate_map.get(pre_pred, pre_pred)
            normalized_frequencies[norm_pred] += 1
            
            pre_obj = self._nlp_preprocess(t.object)
            norm_obj = object_map.get(pre_obj, pre_obj)
            normalized_frequencies[norm_obj] += 1

        taxonomic_map = {}
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

        async def lift_cluster_async(cluster, sem, term_embeddings_l2, terms_list, label_prefix):
            async with sem:
                cluster_terms = cluster["members"]
                label = cluster["cluster_number"]
                if len(cluster_terms) == 1:
                    taxonomic_map[cluster_terms[0]] = cluster_terms[0]
                    return
                    
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
                    for t in cluster_terms:
                        taxonomic_map[t] = mapped_term
                except Exception as e:
                    print(f"[Refinement] Error lifting {label_prefix} cluster {label}: {e}")
                    error_record = {
                        "phase": f"taxonomic_lifting_{label_prefix}",
                        "cluster_label": label,
                        "error": str(e)
                    }
                    self._log_error(error_record)
                    for t in cluster_terms:
                        taxonomic_map[t] = fallback_candidates[0] if fallback_candidates else t

        def lift_cluster_sync(cluster, term_embeddings_l2, terms_list, label_prefix):
            cluster_terms = cluster["members"]
            label = cluster["cluster_number"]
            if len(cluster_terms) == 1:
                taxonomic_map[cluster_terms[0]] = cluster_terms[0]
                return
                
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
                for t in cluster_terms:
                    taxonomic_map[t] = mapped_term
            except Exception as e:
                print(f"[Refinement] Error lifting {label_prefix} cluster {label}: {e}")
                error_record = {
                    "phase": f"taxonomic_lifting_{label_prefix}",
                    "cluster_label": label,
                    "error": str(e)
                }
                self._log_error(error_record)
                for t in cluster_terms:
                    taxonomic_map[t] = fallback_candidates[0] if fallback_candidates else t

        if self.use_async:
            async def run_lifting():
                sem = asyncio.Semaphore(max_async)
                tasks = []
                for cluster in subj_clusters:
                    tasks.append(lift_cluster_async(cluster, sem, subj_embeddings_l2, subj_terms, "Subject"))
                for cluster in obj_clusters:
                    tasks.append(lift_cluster_async(cluster, sem, obj_embeddings_l2, obj_terms, "Object"))
                await asyncio.gather(*tasks)
            asyncio.run(run_lifting())
        else:
            for cluster in subj_clusters:
                lift_cluster_sync(cluster, subj_embeddings_l2, subj_terms, "Subject")
            for cluster in obj_clusters:
                lift_cluster_sync(cluster, obj_embeddings_l2, obj_terms, "Object")

        # Apply both Lexical Normalization and Taxonomic Lifting maps to raw triples
        normalized_triples = [t.model_copy(deep=True) for t in raw_triples]
        for t in normalized_triples:
            pre_subj = self._nlp_preprocess(t.subject)
            pre_pred = self._nlp_preprocess(t.predicate)
            pre_obj = self._nlp_preprocess(t.object)
            
            norm_subj = subject_map.get(pre_subj, pre_subj)
            norm_pred = predicate_map.get(pre_pred, pre_pred)
            norm_obj = object_map.get(pre_obj, pre_obj)
            
            t.subject = taxonomic_map.get(norm_subj, norm_subj)
            t.predicate = norm_pred
            t.object = taxonomic_map.get(norm_obj, norm_obj)
            
        with open(out_dir / "refined_triplets.json", "w") as f:
            json.dump([t.model_dump() for t in normalized_triples], f, indent=2)
            
        with open(out_dir / "taxonomic_map.json", "w") as f:
            json.dump(taxonomic_map, f, indent=2)

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
