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
            
            self.embedding_model = SentenceTransformer(
                model_name, 
                cache_folder=str(models_dir),
                local_files_only=True  # Guarantees 100% offline execution
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

        normalization_map = {}
        batch_size = 15
        max_async = self.config.get('refinement', {}).get('max_async_calls', 1)
        
        async def process_batch(batch: list, sem: asyncio.Semaphore, agent):
            async with sem:
                user_prompt = prompts.LLM_PREPROCESSING_USER_PROMPT.format(
                    master_themes=', '.join(master_themes),
                    batch_json=json.dumps(batch)
                )
                try:
                    result = await agent.run(
                        user_prompt,
                        deps=self.context.master_domain
                    )
                    for pair in result.output.tokens:
                        normalization_map[pair.original] = pair.normalized
                except Exception as e:
                    print(f"[Refinement] Error normalizing batch: {e}")
                    for term in batch:
                        normalization_map[term] = term

        async def run_all_batches(term_set: set, agent):
            term_list = sorted(list(term_set))
            sem = asyncio.Semaphore(max_async)
            tasks = []
            total_batches = (len(term_list) + batch_size - 1) // batch_size
            
            async def wrapped_process(batch, sem, agent, batch_idx):
                print(f"      -> Starting batch {batch_idx}/{total_batches} ({len(batch)} terms)...")
                try:
                    # Apply a timeout to prevent infinite hangs
                    await asyncio.wait_for(process_batch(batch, sem, agent), timeout=60.0)
                    print(f"      -> Completed batch {batch_idx}/{total_batches}")
                except asyncio.TimeoutError:
                    print(f"      -> Timeout error on batch {batch_idx}/{total_batches}")
                    for term in batch:
                        normalization_map[term] = term
                except Exception as e:
                    print(f"      -> Error on batch {batch_idx}/{total_batches}: {e}")

            for i in range(0, len(term_list), batch_size):
                batch = term_list[i:i+batch_size]
                batch_idx = (i // batch_size) + 1
                tasks.append(wrapped_process(batch, sem, agent, batch_idx))
                
            if tasks:
                await asyncio.gather(*tasks)

        async def process_all_terms():
            print("   -> Normalizing Subjects...")
            await run_all_batches(unique_subjects, subject_norm_agent)
            print("   -> Normalizing Predicates...")
            await run_all_batches(unique_predicates, predicate_norm_agent)
            print("   -> Normalizing Objects...")
            await run_all_batches(unique_objects, object_norm_agent)
            
        # Execute all normalization passes inside a single shared event loop
        asyncio.run(process_all_terms())

        # Re-map triples
        for t in normalized_triples:
            mapped_subj = normalization_map.get(t.subject, t.subject)
            t.subject = mapped_subj if mapped_subj and str(mapped_subj).strip() else t.subject
            
            mapped_pred = normalization_map.get(t.predicate, t.predicate)
            t.predicate = mapped_pred if mapped_pred and str(mapped_pred).strip() else t.predicate
            
            mapped_obj = normalization_map.get(t.object, t.object)
            t.object = mapped_obj if mapped_obj and str(mapped_obj).strip() else t.object

        out_dir = Path("outputs/02_refinement")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        with open(out_dir / "normalized_triplets.json", "w") as f:
            json.dump([t.model_dump() for t in normalized_triples], f, indent=2)

        with open(out_dir / "normalization_map.json", "w") as f:
            json.dump(normalization_map, f, indent=2)

        # Before loading embeddings, explicitly purge VRAM
        print("[Refinement] Purging VRAM before embedding initialization...")
        purge_vram()

        # 2. Theme-Based Embedding Mapping
        print("[Refinement] Step 2: Theme-Based Embedding Mapping")
        self._load_embedding_model()
        
        # Prepare original themes texts
        orig_theme_texts = []
        for theme in original_themes:
            # Assume ThemeDiscoveryResult structure or dict
            title = theme.title if hasattr(theme, 'title') else theme.get('title', '')
            desc = theme.description if hasattr(theme, 'description') else theme.get('description', '')
            reasoning = theme.reasoning if hasattr(theme, 'reasoning') else theme.get('reasoning', '')
            orig_theme_texts.append(f"{title} {desc} {reasoning}")
            
        master_theme_texts = [f"{self.context.master_domain} {mt}" for mt in master_themes]
        
        # L2 Normalize embeddings directly during encoding for rigorous cosine math
        orig_embeddings = self.embedding_model.encode(orig_theme_texts, normalize_embeddings=True)
        master_embeddings = self.embedding_model.encode(master_theme_texts, normalize_embeddings=True)
        
        theme_mapping = defaultdict(list)
        
        # Calculate full similarity matrix at once
        similarity_matrix = cosine_similarity(orig_embeddings, master_embeddings)
        
        for i in range(len(orig_theme_texts)):
            # Find index of the master theme with the highest similarity score
            best_idx = np.argmax(similarity_matrix[i])
            
            # theme title
            orig_title = original_themes[i].title if hasattr(original_themes[i], 'title') else original_themes[i].get('title', f"theme_{i}")
            theme_mapping[master_themes[best_idx]].append(orig_title)

        with open(out_dir / "theme_mapping_clusters.json", "w") as f:
            json.dump(theme_mapping, f, indent=2)

        # 3 & 4. Triple Vector Clustering and Taxonomic Lifting (Isolated by SVO)
        print("[Refinement] Step 3 & 4: SVO Vector Clustering and Taxonomic Lifting")
        
        taxonomic_map = {}
        threshold = self.config.get('refinement', {}).get('clustering_threshold', 0.4)

        def cluster_and_lift(terms_list: List[str], label_prefix: str):
            if not terms_list:
                return
            
            term_embeddings = self.embedding_model.encode(terms_list)
            term_embeddings_l2 = normalize(term_embeddings, norm='l2')
            
            clustering = AgglomerativeClustering(metric='cosine', linkage='average', distance_threshold=threshold, n_clusters=None)
            
            if len(terms_list) > 1:
                cluster_labels = clustering.fit_predict(term_embeddings_l2)
            else:
                cluster_labels = [0]
                
            clusters = defaultdict(list)
            for term, label in zip(terms_list, cluster_labels):
                clusters[label].append(term)
                
            for label, cluster_terms in clusters.items():
                if len(cluster_terms) == 1:
                    taxonomic_map[cluster_terms[0]] = cluster_terms[0]
                    continue
                    
                idx = [terms_list.index(t) for t in cluster_terms]
                cluster_vecs = term_embeddings_l2[idx]
                weights = np.array([self.context.term_frequencies.get(t, 1) for t in cluster_terms])
                
                centroid = np.average(cluster_vecs, axis=0, weights=weights)
                distances = cosine_distances([centroid], cluster_vecs)[0]
                closest_indices = np.argsort(distances)[:3]
                fallback_candidates = [cluster_terms[i] for i in closest_indices]
                
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
                    for t in cluster_terms:
                        taxonomic_map[t] = fallback_candidates[0] if fallback_candidates else t

        unique_subjects_norm = list(set([t.subject for t in normalized_triples]))
        unique_objects_norm = list(set([t.object for t in normalized_triples]))

        print("   -> Clustering and Lifting Subjects...")
        cluster_and_lift(unique_subjects_norm, "Subject")
        
        print("   -> Clustering and Lifting Objects...")
        cluster_and_lift(unique_objects_norm, "Object")

        # Re-Map Triples
        for t in normalized_triples:
            t.subject = taxonomic_map.get(t.subject, t.subject)
            t.object = taxonomic_map.get(t.object, t.object)
            # Note: t.predicate is NOT mapped here because predicates are verbs/edges and should not be taxonomically lifted into nouns.
            
        with open(out_dir / "refined_triplets.json", "w") as f:
            json.dump([t.model_dump() for t in normalized_triples], f, indent=2)
            
        with open(out_dir / "taxonomic_map.json", "w") as f:
            json.dump(taxonomic_map, f, indent=2)
            
        print("[Refinement] Pipeline complete. Purging VRAM.")
        purge_vram()
        return normalized_triples
