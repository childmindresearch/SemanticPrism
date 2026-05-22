"""
SemanticPrism: Master Pipeline Orchestrator

The absolute global entrypoint governing the sequential execution
of all mathematical extraction, verification, topology, and synthesis logic.
"""

import asyncio
import os
import json
import time
import csv
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
        self.config_path = config_path
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

    async def execute_knowledge_pipeline(self, documents: List[Dict[str, str]]) -> str:
        """
        Executes the explicit linear logic sequence parsing multiple text matrices in parallel.
        Accepts documents in structured format: [{'filename': ..., 'text': ...}]
        Returns the absolute filepath to the finalized semantic master models file.
        """
        start_time = time.time()
        start_datetime = datetime.now()
        pipeline_errors = []
        doc_lengths = [len(doc["text"]) for doc in documents]
        
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
        
        # Initialize telemetry tracking
        extraction_telemetry = {
            doc["filename"]: {
                "theme_extraction": "Failed/Null",
                "triple_extraction": "Failed/Null"
            }
            for doc in documents
        }
        
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
            
            # 1.A: Parallel Theme Discovery
            logger.info(f"Discovering theme nodes concurrently for {len(documents)} documents...")
            async def safe_discover_themes(doc):
                try:
                    theme_results = await self.extractor.discover_themes(doc["text"])
                    extraction_telemetry[doc["filename"]]["theme_extraction"] = "Success"
                    return theme_results
                except Exception as e:
                    logger.error(f"Theme extraction failed for {doc['filename']}: {e}")
                    return []

            theme_tasks = [safe_discover_themes(doc) for doc in documents]
            theme_results = await asyncio.gather(*theme_tasks)
            
            for themes in theme_results:
                all_themes.extend(themes)
            
            self._save_state(all_themes, "outputs/01_extraction/original_themes.json")
        
            # 1.B: Master Theme Synthesis
            logger.info("Consolidating theme lists to identify master global domain...")
            weighted_string = self.extractor.weight_themes(all_themes)
            master_context = await self.extractor.consolidate_themes(weighted_string)
            master_domain = master_context.master_domain if master_context else "General Complex Logic"
            self._save_state(master_context, "outputs/01_extraction/distilled_themes.json")
            logger.info(f"🎯 Master Domain Distilled: {master_domain}")
            _dump_current_log()
        
            # 1.C: Parallel Logical Triple Extraction (SVO)
            logger.info("Extracting logical triples concurrently...")
            async def safe_extract_triples(doc, context):
                try:
                    triples = await self.extractor.extract_triples(doc["text"], context)
                    for trip in triples:
                        trip.source_document = doc["filename"]
                    extraction_telemetry[doc["filename"]]["triple_extraction"] = "Success"
                    return triples
                except Exception as e:
                    logger.error(f"Triple extraction failed for {doc['filename']}: {e}")
                    return []

            triple_tasks = [safe_extract_triples(doc, master_context) for doc in documents]
            triple_results = await asyncio.gather(*triple_tasks)
            
            for trips in triple_results:
                raw_triples.extend(trips)
        
            if not raw_triples:
                logger.warning("Pipeline terminated early. No logical triples discovered.")
                return ""
            
            self._save_state(raw_triples, "outputs/01_extraction/original_triplets.json")
            self.visualizer.visualize_triples(raw_triples, "outputs/01_extraction/01_raw_triples_graph.html", "Phase 1: Raw Extractions")
            
            # Export extraction telemetry report to CSV
            csv_path = "outputs/01_extraction/extraction_telemetry.csv"
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            try:
                with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["document_name", "theme_extraction_status", "triple_extraction_status"])
                    for doc_name, status in extraction_telemetry.items():
                        writer.writerow([
                            os.path.basename(doc_name),
                            status["theme_extraction"],
                            status["triple_extraction"]
                        ])
                logger.info(f"Extraction telemetry CSV report saved to: {csv_path}")
            except Exception as e:
                logger.error(f"Failed to export extraction telemetry CSV: {e}")
        
            original_subjs = {t.subject for t in raw_triples}
            original_preds = {t.predicate for t in raw_triples}
            original_objs = {t.object for t in raw_triples}
        
            # 1.D: Text Normalization
            logger.info("Executing NLP Lexical Normalization phase...")
            try:
                normalized_triples, norm_subjs, norm_preds, norm_objs = await execute_normalization_phase(
                    self.extractor,
                    raw_triples,
                    master_domain,
                    self._save_state
                )
            except Exception as e:
                logger.error(f"Normalization phase failed, bypassing: {e}")
                normalized_triples = raw_triples
                norm_subjs, norm_preds, norm_objs = original_subjs, original_preds, original_objs
            _dump_current_log()
            
            logger.info("==================================================")
            logger.info("STAGE 2: OFFLINE EMBEDDING & MODULARITY PROPOSALS")
            logger.info("==================================================")
            
            # Theme-based Embedding Process
            if all_themes and master_context:
                logger.info("Executing theme-based embedding mapping...")
                try:
                    self.embedder.theme_based_embedding(all_themes, master_context)
                except Exception as e:
                    logger.error(f"Theme-based embedding failed: {e}")
            else:
                logger.warning("Skipping theme-based embedding: original themes or master context is missing.")
                
            # Triple Vector Clustering offloaded to a background thread
            logger.info("Offloading heavy offline matrix computation to background thread...")
            try:
                proposed_clusters = await asyncio.to_thread(self.embedder.process_triples, normalized_triples)
                self._save_state(proposed_clusters, "outputs/02_embedding/clusters_identified.json")
            except Exception as e:
                logger.error(f"Embedding processing failed: {e}")
                proposed_clusters = []
            _dump_current_log()
        
            logger.info("==================================================")
            logger.info("STAGE 3: HYBRID HYPERNYM TAXONOMIC LIFTING")
            logger.info("==================================================")
            try:
                verified_clusters = await self.hypernyms.validate_context_vectors(proposed_clusters, master_domain)
                self._save_state(verified_clusters, "outputs/03_hypernym_lifting/verified_clusters.json")
            
                hypernym_mapping = await self.hypernyms.taxonomic_lift(verified_clusters, master_domain)
                self._save_state(hypernym_mapping, "outputs/03_hypernym_lifting/hypernym_mapping.json")
            except Exception as e:
                logger.error(f"Taxonomic hypernym lifting failed: {e}")
                hypernym_mapping = {}
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
            try:
                # Construct directed network graph
                graph = self.topology.build_graph(mapped_triples)
                
                # Fetch Topology hyperparameters dynamically from config
                overlap_threshold = 0.80
                leiden_resolution = 1.0
                min_community_size = 4
                synthesis_strategy = "standard"
                if os.path.exists(self.config_path):
                    try:
                        with open(self.config_path, "r") as f:
                            cfg = yaml.safe_load(f)
                            overlap_threshold = cfg.get("topology", {}).get("inheritance_overlap_threshold", 0.80)
                            leiden_resolution = cfg.get("topology", {}).get("leiden_resolution", 1.0)
                            min_community_size = cfg.get("topology", {}).get("min_community_size", 4)
                            synthesis_strategy = cfg.get("synthesis", {}).get("strategy", "standard")
                    except Exception as ecf:
                        logger.warning(f"Could not load custom topology parameters from config: {ecf}")

                if synthesis_strategy == "hub_and_spoke":
                    hierarchy_payload = self.topology.build_hub_and_spoke_hierarchy(graph, min_size=min_community_size, resolution=leiden_resolution)
                    partition = self.topology.detect_communities(graph, resolution=leiden_resolution)
                else:
                    partition = self.topology.detect_communities(graph, resolution=leiden_resolution)
                    hierarchy = self.topology.extract_hierarchy(graph, partition, min_size=min_community_size)
                    hierarchy_payload = {"strategy": "standard", "communities": hierarchy}
                
                self._save_state(partition, "outputs/05_topology/modularity_partition.json")
                self._save_state(hierarchy_payload, "outputs/05_topology/extracted_hierarchy.json")
                self.visualizer.visualize_topology(graph, partition, "outputs/05_topology/03_topology_communities_graph.html", "Phase 5: Global Modularity Map")
                
                # Bipartite Hypergraph Topology expansion and spectral matrices
                logger.info("Building N-ary Hypergraph Topology matrices")
                hypergraph_res = self.topology.build_hypergraph_topology(mapped_triples, overlap_threshold=overlap_threshold)
                self.topology.visualize_hypergraph(hypergraph_res["B"], "outputs/05_topology")
                
                hierarchy_export = {
                    "entities_count": hypergraph_res["entities"],
                    "themes_count": hypergraph_res["themes"],
                    "theme_inheritance_map": hypergraph_res.get("theme_inheritance_map", {})
                }
                self._save_state(hierarchy_export, "outputs/05_topology/theme_inheritance_hierarchy.json")

                spectral_export = {
                    "H_matrix": hypergraph_res["H"].tolist(),
                    "L_matrix": hypergraph_res["L"].tolist()
                }
                self._save_state(spectral_export, "outputs/05_topology/hypergraph_spectral_matrices.json")
            except Exception as e:
                logger.error(f"Topological graph analysis failed: {e}")
                hypergraph_res = {}
                hierarchy = {}
            _dump_current_log()
        
            logger.info("==================================================")
            logger.info("STAGE 6: GENERATIVE SCHEMA SYNTHESIS")
            logger.info("==================================================")
            try:
                theme_inheritance_map = hypergraph_res.get("theme_inheritance_map", {})
                resolved_schemas = await self.synthesizer.generate_schemas(hierarchy_payload, master_domain, theme_inheritance_map, strategy=synthesis_strategy)
                file_path = self.synthesizer.build_global_context(resolved_schemas)
            except Exception as e:
                logger.error(f"Generative schema synthesis failed: {e}")
                file_path = ""
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
