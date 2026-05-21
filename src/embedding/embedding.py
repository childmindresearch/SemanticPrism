"""
SemanticPrism: Offline Embedding Pipeline
The  via vector algebra.
"""

import yaml
import os
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from collections import Counter

from src.extraction.schemas import RawTriple, ThemeDiscoveryResult, MasterThemeSynthesisResult
from src.core.logger import get_logger

logger = get_logger("EmbeddingPipeline")


class EmbeddingPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initializes the EmbeddingPipeline by loading configuration settings and instantiating the sentence transformer model either locally or via download.
        """
        # Load explicit configurations safely
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.critical(f"Missing config natively statically securely: {e}")
            raise e
            
        ref_cfg = config.get('refinement', {})
        self.model_name = ref_cfg.get('embedding_model', 'all-MiniLM-L6-v2')
        self.similarity_threshold = float(ref_cfg.get('similarity_threshold', 0.15))
        self.variance_retention = float(ref_cfg.get('spectral_variance_retention', 0.95))
        self.compress_fields = ref_cfg.get('compress_fields', ["subject", "object", "predicate"])

        logger.info(f"Initializing Offline Embedding Pipeline (Model: {self.model_name})")
        local_model_path = os.path.join("models", "embeddings", self.model_name.replace("/", "_"))
        
        if os.path.exists(local_model_path):
            logger.info(f"Loading embedding model locally from: {local_model_path}")
            self.encoder = SentenceTransformer(local_model_path)
        else:
            logger.info(f"Downloading model {self.model_name} from HuggingFace.")
            self.encoder = SentenceTransformer(self.model_name)
            os.makedirs(local_model_path, exist_ok=True)
            self.encoder.save(local_model_path)
            logger.info(f"Model saved locally to: {local_model_path}")

    def theme_based_embedding(
        self, 
        original_themes: List[ThemeDiscoveryResult], 
        master_context: MasterThemeSynthesisResult,
        output_dir: str = "outputs/02_embedding"
    ) -> Dict[str, List[str]]:
        """
        Embeds original and master themes to calculate similarity and map original themes to their most similar master theme.
        """
        import json
        logger.info("Executing theme-based embedding and master theme mapping...")

        # 1. Group original themes by title (case-insensitive for key, retaining original casing from first occurrence)
        theme_groups: Dict[str, List[Tuple[str, str]]] = {}  # normalized_title -> list of (description, reasoning)
        first_casing_map: Dict[str, str] = {}     # normalized_title -> original_casing
        theme_counts: Dict[str, int] = {}          # normalized_title -> count

        for tr in original_themes:
            if not tr.themes:
                continue
            for t in tr.themes:
                title = t.title.strip() if t.title else ""
                if not title:
                    continue
                norm_title = title.lower()
                
                if norm_title not in theme_groups:
                    theme_groups[norm_title] = []
                    first_casing_map[norm_title] = title
                    theme_counts[norm_title] = 0
                
                theme_groups[norm_title].append((t.description or "", t.reasoning or ""))
                theme_counts[norm_title] += 1

        if not theme_groups:
            logger.warning("No original themes found for theme-based embedding mapping.")
            return {}

        # Construct original theme concatenated texts and map them back to original casings
        unique_original_titles = []
        original_texts = []
        original_title_counts = []

        for norm_title, items in theme_groups.items():
            orig_title = first_casing_map[norm_title]
            unique_original_titles.append(orig_title)
            original_title_counts.append(theme_counts[norm_title])

            # Concatenate all descriptions and reasonings
            descriptions = [desc.strip() for desc, _ in items if desc.strip()]
            reasonings = [reason.strip() for _, reason in items if reason.strip()]

            desc_concat = " ".join(descriptions)
            reasoning_concat = " ".join(reasonings)
            
            combined_text = f"Title: {orig_title}. Description: {desc_concat}. Reasoning: {reasoning_concat}."
            original_texts.append(combined_text)

        # 2. Embed the text for each value field of concatenated text
        logger.info(f"Embedding {len(original_texts)} unique consolidated original theme texts...")
        original_embeddings = self.encoder.encode(
            original_texts, 
            batch_size=2048,
            convert_to_numpy=True
        )

        # 3. Combine each individual master theme from distilled themes + the master domain text into a single string.
        master_domain = master_context.master_domain or ""
        master_themes = master_context.master_themes or []

        if not master_themes:
            logger.warning("No master themes found in master context.")
            return {}

        master_strings = [
            f"Theme: {mt}. Domain: {master_domain}."
            for mt in master_themes
        ]

        logger.info(f"Embedding {len(master_strings)} combined master themes...")
        master_embeddings = self.encoder.encode(
            master_strings,
            batch_size=2048,
            convert_to_numpy=True
        )

        # 4. Perform a similarity measurement to identify clusters of the most similar original themes that map to a master theme.
        # Initialize the mapping structure
        mapping: Dict[str, List[str]] = {mt: [] for mt in master_themes}

        # Helper to compute cosine similarity
        def get_cosine_similarity(v1, v2):
            dot_val = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(dot_val / (norm1 * norm2))

        for orig_idx, orig_title in enumerate(unique_original_titles):
            orig_emb = original_embeddings[orig_idx]
            best_score = -1.0
            best_master = master_themes[0]  # Fallback

            for mast_idx, master_theme in enumerate(master_themes):
                mast_emb = master_embeddings[mast_idx]
                sim = get_cosine_similarity(orig_emb, mast_emb)
                if sim > best_score:
                    best_score = sim
                    best_master = master_theme

            # 5. Return this mapping and store the key: value pair of master theme : [list of original themes].
            # For instances where an original title was duplicated, return that title x number of times present into the list.
            count = original_title_counts[orig_idx]
            mapping[best_master].extend([orig_title] * count)

        # Sort each list of original themes alphabetically
        for mt in mapping:
            mapping[mt] = sorted(mapping[mt])

        # 6. Save this file to a .json object in the embeddings output folder
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, "theme_mapping_clusters.json")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(mapping, f, indent=4, ensure_ascii=False)
            logger.info(f"Theme-based embedding clusters saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save theme mapping clusters to JSON: {e}")

        return mapping

    def extract_and_group(self, triples: List[RawTriple]) -> Dict[str, List[str]]:
        """
        Separates a list of raw triples into distinct lists of subjects, predicates, and objects based on configured compression fields.
        """
        groups = {"subject": [], "predicate": [], "object": []}
        
        for t in triples:
            if "subject" in self.compress_fields and t.subject:
                groups["subject"].append(t.subject)
            if "predicate" in self.compress_fields and t.predicate:
                groups["predicate"].append(t.predicate)
            if "object" in self.compress_fields and t.object:
                groups["object"].append(t.object)
                
        return groups

    def _process_isolated_group(self, item_strings: List[str]) -> List[List[str]]:
        """
        Processes a group of strings by generating embeddings, applying PCA for dimensionality reduction, and clustering them using Agglomerative Clustering. Returns lists of clustered strings.
        """
        logger.info(f"Processing vector mappings rigidly explicitly cleanly.")
        
        if not item_strings:
            return []

        item_counts = Counter(item_strings)
        unique_items = list(item_counts.keys())
        counts = [item_counts[x] for x in unique_items]
        
        if len(unique_items) <= 1:
            return [[unique_items[0]]] if unique_items else []
            
        logger.info(f"Generating vectors. Unique items: {len(unique_items)}")
        
        # 1. Math Encoding completely
        embeddings_matrix = self.encoder.encode(
            unique_items, 
            batch_size=2048,
            device='cuda',
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # 2. PCA Weighted statically perfectly. 
        # Duplicating rows physically to reflect absolute frequencies strictly perfectly.
        expanded_embeddings = []
        for emb, count in zip(embeddings_matrix, counts):
            expanded_embeddings.extend([emb] * count)
        
        expanded_np = np.array(expanded_embeddings)
        
        max_components = min(len(expanded_np), len(expanded_np[0]) if expanded_np.ndim > 1 else 1)
        if max_components <= 1:
            logger.info("Insufficient variance automatically. Mapping purely identical cleanly.")
        else:
            pca_full = PCA()
            pca_full.fit(expanded_np)
            evr = pca_full.explained_variance_ratio_
            
            if len(evr) > 2:
                eigenvalues = pca_full.explained_variance_
                gaps = eigenvalues[:-1] - eigenvalues[1:]
                optimal_components = np.argmax(gaps) + 1
                
                retention = np.sum(evr[:optimal_components])
                if retention < 0.5:
                    cumulative_var = np.cumsum(evr)
                    optimal_components = np.argmax(cumulative_var >= 0.85) + 1
            else:
                optimal_components = len(evr)
                
            logger.info(f"Dynamic Eigengap Analysis identified optimal components: {optimal_components}/{len(evr)}")
            pca = PCA(n_components=optimal_components)
            pca.fit(expanded_np)
            embeddings_matrix = pca.transform(embeddings_matrix)        
        logger.info(f"PCA reduced logically gracefully. Dimensions elegantly: {embeddings_matrix.shape}")
        
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric='cosine',
            linkage='average',
            distance_threshold=self.similarity_threshold
        )
        
        try:
            labels = clusterer.fit_predict(embeddings_matrix)
        except Exception as e:
            logger.warning(f"Agglomerative grouping dynamically intrinsically safely rigorously failed: {e}. Fallback cleanly.")
            labels = np.zeros(len(unique_items), dtype=int)
            
        grouped: Dict[int, List[str]] = {}
        for idx, label in enumerate(labels):
            if label not in grouped:
                grouped[label] = []
            grouped[label].append(unique_items[idx])
            
        proposals = list(grouped.values())
        return proposals
        
    def process_triples(self, triples: List[RawTriple]) -> Dict[str, List[List[str]]]:
        """
        Orchestrates the grouping and clustering of raw triples by isolating them into fields and processing each field to generate clustered proposals.
        """
        logger.info("Executing Offline Embedding Matrix completely accurately intelligently.")
        isolated = self.extract_and_group(triples)
        
        final_proposals = {}
        
        for k, arr in isolated.items():
            if not arr:
                final_proposals[k] = []
                continue
            logger.info(f"Grouping '{k}' logic array accurately cleanly dependably gracefully.")
            proposals = self._process_isolated_group(arr)
            final_proposals[k] = proposals
            
        return final_proposals
