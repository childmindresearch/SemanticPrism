"""
SemanticPrism: Hypernym Lifting Pipeline
The hybrid entity normalization mapping logic securely merging mathematically 
defined centroids with LLM validated taxonomy explicitly natively.
"""

import yaml
import asyncio
import json
import os
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

from src.nlp.schemas import ClusterContextualValidation, TaxonomicVerification
from src.core.logger import get_logger
import src.nlp.prompts as prompts
from src.llm.llm_client import SemanticLLMClient

logger = get_logger("HypernymPipeline")

class HypernymPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initializes the HypernymPipeline by loading configuration, setting up the LLM client, and instantiating the sentence transformer model for embeddings.
        """
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logger.critical(f"Config load failed inherently: {e}")
            raise e
            
        self.use_async = self.config.get('pipeline', {}).get('use_async', False)
        self.max_concurrent = self.config.get('pipeline', {}).get('max_concurrent_llm_calls', 3)
        self.embedding_model = self.config.get('refinement', {}).get('embedding_model', 'all-MiniLM-L6-v2')
        self.fallback_domain = self.config.get('extraction', {}).get('domain', 'General')
        
        logger.info("Initializing Master LLM Factory for context lifting.")
        self.llm = SemanticLLMClient(config_path)
        logger.info("Initializing Dedicated Embedding Encoder for centroid projections.")
        local_model_path = os.path.join("models", "embeddings", self.embedding_model.replace("/", "_"))
        
        if os.path.exists(local_model_path):
            logger.info(f"Loading embedding model locally from: {local_model_path}")
            self.encoder = SentenceTransformer(local_model_path)
        else:
            logger.info(f"Downloading model {self.embedding_model} from HuggingFace.")
            self.encoder = SentenceTransformer(self.embedding_model)
            os.makedirs(local_model_path, exist_ok=True)
            self.encoder.save(local_model_path)
            logger.info(f"Model saved locally to: {local_model_path}")

    async def validate_context_vectors(self, proposals: Dict[str, List[List[str]]], master_domain: str = "") -> Dict[str, List[List[str]]]:
        """
        Uses the LLM to evaluate proposed clusters of strings; if rejected, splits the cluster back into isolated elements. Returns verified proposals.
        """
        logger.info("Phase 3.1: Validating mathematical proposals...")
        domain_string = master_domain if master_domain else self.fallback_domain
        sem = asyncio.Semaphore(self.max_concurrent)
        
        verified_proposals = {}
        
        async def evaluate_cluster(cluster: List[str]) -> List[List[str]]:
            # Ignore single item natively efficiently
            if len(cluster) <= 1:
                return [cluster]
                
            user_msg = prompts.CONTEXT_VECTOR_VALIDATION_USER_PROMPT.format(
                domain_context=f"Domain: {domain_string}\n",
                proposed_cluster_json=json.dumps(cluster)
            )
            
            if self.use_async:
                async with sem:
                    res = await self.llm.safe_api_call_async(
                        prompts.CONTEXT_VECTOR_VALIDATION_SYSTEM_PROMPT,
                        user_msg,
                        ClusterContextualValidation
                    )
            else:
                res = self.llm.safe_api_call_sync(
                    prompts.CONTEXT_VECTOR_VALIDATION_SYSTEM_PROMPT,
                    user_msg,
                    ClusterContextualValidation
                )
                
            if res and res.accuracy_destroyed:
                logger.debug(f"Convergence rejected structurally: {res.condition_detected}. Splitting strictly.")
                return [[item] for item in cluster] # Revert into pure isolated elements natively safely
            
            # Verified! Keep natively merged functionally realistically.
            return [cluster]

        for k, clusters in proposals.items():
            logger.info(f"Evaluating {len(clusters)} mathematical clusters for [{k}]")
            verified_proposals[k] = []
            
            if self.use_async:
                tasks = [evaluate_cluster(c) for c in clusters]
                results = await asyncio.gather(*tasks)
            else:
                results = []
                for c in clusters:
                    results.append(await evaluate_cluster(c))
            
            for mapped_split in results:
                verified_proposals[k].extend(mapped_split)
                
        return verified_proposals

    def _find_semantic_center(self, cluster: List[str]) -> str:
        """
        Calculates the geometric centroid of a cluster of strings using embeddings and returns the string closest to the mean vector based on cosine distance.
        """
        if len(cluster) == 1:
            return cluster[0]
            
        embeddings = self.encoder.encode(cluster, convert_to_numpy=True)
        # Identify central Euclidean coordinate properly cleanly optimally natively
        mean_vector = np.mean(embeddings, axis=0, keepdims=True)
        # Map Cosine distance formally
        distances = cosine_distances(mean_vector, embeddings)[0]
        # Locate item index.
        idx = np.argmin(distances)
        return cluster[idx]

    async def taxonomic_lift(self, verified_clusters: Dict[str, List[List[str]]], master_domain: str = "") -> Dict[str, Dict[str, str]]:
        """
        Uses the LLM to determine a formal taxonomic hypernym for each validated cluster, falling back to the geometric centroid if validation fails. Returns a mapping from raw strings to their elevated standard.
        """
        logger.info("Phase 3.2: Native Taxonomic Lifting.")
        domain_string = master_domain if master_domain else self.fallback_domain
        sem = asyncio.Semaphore(self.max_concurrent)
        
        final_mapping = {"subject": {}, "predicate": {}, "object": {}}
        
        async def evaluate_lifting(cluster: List[str]) -> tuple[List[str], str]:
            centroid = self._find_semantic_center(cluster)
            
            if len(cluster) <= 1:
                return (cluster, centroid)
                
            data_map = {
                "mathematical_centroid": centroid,
                "geometric_members": cluster
            }
            user_msg = prompts.TAXONOMIC_LIFTING_USER_PROMPT.format(
                domain_context=f"Domain: {domain_string}\n",
                taxonomic_map_json=json.dumps(data_map)
            )
            
            if self.use_async:
                async with sem:
                    res = await self.llm.safe_api_call_async(
                        prompts.TAXONOMIC_LIFTING_SYSTEM_PROMPT,
                        user_msg,
                        TaxonomicVerification
                    )
            else:
                res = self.llm.safe_api_call_sync(
                    prompts.TAXONOMIC_LIFTING_SYSTEM_PROMPT,
                    user_msg,
                    TaxonomicVerification
                )
                
            if res and res.members_verified:
                return (cluster, res.formal_hypernym)
            
            logger.debug(f"Taxonomy natively rejected. Utilizing geometric centroid fallback.")
            return (cluster, centroid)

        for k, clusters in verified_clusters.items():
            if k not in final_mapping:
                final_mapping[k] = {}
                
            if self.use_async:
                tasks = [evaluate_lifting(c) for c in clusters]
                results = await asyncio.gather(*tasks)
            else:
                results = []
                for c in clusters:
                    results.append(await evaluate_lifting(c))
            
            for cluster_members, final_label in results:
                for member in cluster_members:
                    final_mapping[k][member] = final_label
                    
        return final_mapping
