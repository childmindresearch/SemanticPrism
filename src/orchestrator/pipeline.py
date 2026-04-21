"""
SemanticPrism: Master Pipeline Orchestrator

The absolute global entrypoint governing the sequential execution
of all mathematical extraction, verification, topology, and synthesis logic.
"""

import asyncio
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List
import yaml

from src.core.logger import get_logger, save_execution_log
from src.extraction.extractor import ExtractionPipeline
from src.extraction.normalize_text import execute_normalization_phase
from src.embedding.embedding import EmbeddingPipeline
from src.nlp.hypernyms import HypernymPipeline
from src.nlp.nlp_mapping import NamingResolutionPipeline
from src.topology.graph_builder import TopologyEngine
from src.synthesis.synthesizer import SynthesisEngine
from src.helpers.visualizer import SemanticVisualizer

logger = get_logger("SemanticPrismOrchestrator")

class SemanticPrismOrchestrator:
    def __init__(self, config_path: str = "config.yaml"):
        logger.info("Initializing Master Pipeline Orchestrator.")
        self.extractor = ExtractionPipeline(config_path)
        self.embedder = EmbeddingPipeline(config_path)
        self.hypernyms = HypernymPipeline(config_path)
        self.mapper = NamingResolutionPipeline()
        self.topology = TopologyEngine()
        self.synthesizer = SynthesisEngine(config_path)
        self.visualizer = SemanticVisualizer()

    def _save_state(self, data: Any, filepath: str):
        """Helper to silently safely dump phase state natively."""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                default_encoder = lambda x: list(x) if isinstance(x, set) else str(x)
                if isinstance(data, list) and len(data) > 0 and hasattr(data[0], 'model_dump'):
                    json.dump([item.model_dump(mode='json') for item in data], f, indent=4, default=default_encoder)
                elif hasattr(data, 'model_dump'):
                    json.dump(data.model_dump(mode='json'), f, indent=4, default=default_encoder)
                else:
                    json.dump(data, f, indent=4, default=default_encoder)
        except Exception as e:
            logger.warning(f"Failed parsing natively to write state to {filepath}: {e}")

    async def execute_knowledge_pipeline(self, documents: List[str]) -> str:
        """
        Executes the explicit linear logic sequence parsing multiple text matrices.
        Returns the absolute filepath to the finalized semantic master map JSON.
        """
        start_time = time.time()
        start_datetime = datetime.now()
        pipeline_errors = []
        doc_lengths = [len(doc) for doc in documents]
        
        original_subjs = set()
        original_preds = set()
        original_objs = set()
        norm_subjs = set()
        norm_preds = set()
        norm_objs = set()
        
        all_themes = []
        master_context = None
        raw_triples = []
        file_path = ""
        
        def _dump_current_log():
            all_errors = pipeline_errors.copy()
            if hasattr(self.extractor, 'llm'): all_errors.extend(self.extractor.llm.error_history)
            if hasattr(self.hypernyms, 'llm'): all_errors.extend(self.hypernyms.llm.error_history)
            if hasattr(self.synthesizer, 'llm'): all_errors.extend(self.synthesizer.llm.error_history)
            
            all_ctxs = []
            if hasattr(self.extractor, 'llm'): all_ctxs.extend(self.extractor.llm.context_history)
            if hasattr(self.hypernyms, 'llm'): all_ctxs.extend(self.hypernyms.llm.context_history)
            if hasattr(self.synthesizer, 'llm'): all_ctxs.extend(self.synthesizer.llm.context_history)

            distilled_t_count = len(master_context.master_themes) if master_context and hasattr(master_context, 'master_themes') else 0
            
            metrics = {
                "start_datetime": start_datetime,
                "duration": time.time() - start_time,
                "use_async": getattr(self.extractor, 'use_async', False),
                "model_name": self.extractor.config.get('llm', {}).get('model_name', 'Unknown'),
                "connection_protocol": self.extractor.config.get('llm', {}).get('connection_protocol', 'Unknown'),
                "doc_count": len(documents),
                "doc_lengths": doc_lengths,
                "all_ctxs": all_ctxs,
                "all_themes_count": len(all_themes),
                "distilled_t_count": distilled_t_count,
                "raw_triples_count": len(raw_triples),
                "orig_subjs": len(original_subjs),
                "orig_preds": len(original_preds),
                "orig_objs": len(original_objs),
                "norm_subjs": len(norm_subjs),
                "norm_preds": len(norm_preds),
                "norm_objs": len(norm_objs),
                "all_errors": all_errors
            }
            save_execution_log(metrics, logger)
        
        try:
            logger.info("==================================================")
            logger.info("STAGE 1: LLM EXTRACTION & THEME CONSOLIDATION")
            logger.info("==================================================")
            
            for idx, text in enumerate(documents):
                logger.info(f"Processing themes for document {idx + 1}/{len(documents)}")
                themes = await self.extractor.discover_themes(text)
                all_themes.extend(themes)
            
            self._save_state(all_themes, "outputs/01_extraction/original_themes.json")
        
            weighted_string = self.extractor.weight_themes(all_themes)
            master_context = await self.extractor.consolidate_themes(weighted_string)
            self._save_state(master_context, "outputs/01_extraction/distilled_themes.json")
            _dump_current_log()
        
            master_domain = master_context.master_domain if master_context else "General"
        
            raw_triples = []
            for idx, text in enumerate(documents):
                logger.info(f"Processing triples for document {idx + 1}/{len(documents)}")
                triples = await self.extractor.extract_triples(text, master_context)
                raw_triples.extend(triples)
        
            if not raw_triples:
                logger.warning("Pipeline terminated early. No logical triples discovered.")
                return ""
            
            self._save_state(raw_triples, "outputs/01_extraction/original_triplets.json")
            self.visualizer.visualize_triples(raw_triples, "outputs/01_extraction/01_raw_triples_graph.html", "Phase 1: Raw Extractions")
        
            original_subjs = {t.subject for t in raw_triples}
            original_preds = {t.predicate for t in raw_triples}
            original_objs = {t.object for t in raw_triples}
        
            # New Phase 2.5 block
            normalized_triples, norm_subjs, norm_preds, norm_objs = await execute_normalization_phase(
                self.extractor,
                raw_triples,
                master_domain,
                self._save_state
            )
            _dump_current_log()
            
            logger.info("==================================================")
            logger.info("STAGE 2: OFFLINE EMBEDDING & MODULARITY PROPOSALS")
            logger.info("==================================================")
            proposed_clusters = self.embedder.process_triples(normalized_triples)
            self._save_state(proposed_clusters, "outputs/02_embedding/clusters_identified.json")
            _dump_current_log()
        
            logger.info("==================================================")
            logger.info("STAGE 3: HYBRID HYPERNYM TAXONOMIC LIFTING")
            logger.info("==================================================")
            verified_clusters = await self.hypernyms.validate_context_vectors(proposed_clusters, master_domain)
            self._save_state(verified_clusters, "outputs/03_hypernym_lifting/verified_clusters.json")
        
            hypernym_mapping = await self.hypernyms.taxonomic_lift(verified_clusters, master_domain)
            self._save_state(hypernym_mapping, "outputs/03_hypernym_lifting/hypernym_mapping.json")
            _dump_current_log()
        
            logger.info("==================================================")
            logger.info("STAGE 4: TAXONOMIC RESOLUTION MAPPING")
            logger.info("==================================================")
            mapped_triples = self.mapper.resolve_names(normalized_triples, hypernym_mapping)
            self._save_state(mapped_triples, "outputs/04_mapping/mapped_triplets.json")
            self.visualizer.visualize_triples(mapped_triples, "outputs/04_mapping/02_resolved_triples_graph.html", "Phase 4: Abstracted Topology")
            _dump_current_log()
        
            logger.info("==================================================")
            logger.info("STAGE 5: TOPOLOGICAL GRAPH MATRICES")
            logger.info("==================================================")
            graph = self.topology.build_graph(mapped_triples)
            partition = self.topology.detect_communities(graph)
            hierarchy = self.topology.extract_hierarchy(graph, partition)
            self._save_state(partition, "outputs/05_topology/modularity_partition.json")
            self._save_state(hierarchy, "outputs/05_topology/extracted_hierarchy.json")
            self.visualizer.visualize_topology(graph, partition, "outputs/05_topology/03_topology_communities_graph.html", "Phase 5: Global Modularity Map")
            _dump_current_log()
        
            logger.info("==================================================")
            logger.info("STAGE 6: GENERATIVE SCHEMA SYNTHESIS")
            logger.info("==================================================")
            resolved_schemas = await self.synthesizer.generate_schemas(hierarchy, master_domain)
            file_path = self.synthesizer.build_global_context(resolved_schemas)
            _dump_current_log()
        
            logger.info("==================================================")
            logger.info("PIPELINE EXECUTION FINALIZED.")
            logger.info("==================================================")
        
        except Exception as e:
            pipeline_errors.append(f"Pipeline crashed abruptly: {str(e)}")
            logger.error(f"Pipeline crashed: {e}")
            raise e
            
        finally:
            _dump_current_log()
                
        return file_path
