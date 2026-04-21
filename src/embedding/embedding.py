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

from src.extraction.schemas import RawTriple
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
        embeddings_matrix = self.encoder.encode(unique_items, convert_to_numpy=True)
        
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
