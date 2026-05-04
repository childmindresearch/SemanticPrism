#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SemanticPrism: Master Workflow Pipeline
This script serves as the definitive orchestrator for the entire SemanticPrism extraction architecture.
It heavily documents every phase, handles deep generative synthesis via Pydantic AI, and natively 
exports intermediate logical matrices into structured output directories for full diagnostic transparency.
"""

import os
import json
import glob
import time
import asyncio
import yaml
from typing import Any

# =====================================================================
# CORE ENGINE IMPORTS
# =====================================================================
from src.core.logger import get_logger
from src.extraction.extractor import ExtractionPipeline
from src.extraction.normalize_text import execute_normalization_phase
from src.embedding.embedding import EmbeddingPipeline
from src.nlp.hypernyms import HypernymPipeline
from src.nlp.nlp_mapping import NamingResolutionPipeline
from src.topology.graph_builder import TopologyEngine
from src.synthesis.synthesizer import SynthesisEngine
from src.helpers.visualizer import SemanticVisualizer

logger = get_logger("MasterWorkflow")

def _save_state(data: Any, filepath: str):
    """
    Safely serializes Pydantic schemas, Python Sets, and Base Dictionaries natively to disk.
    Ensures intermediate logic steps are physically stored for diagnostic audits.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        # Default encoder collapses native Python sets into list arrays
        default_encoder = lambda x: list(x) if isinstance(x, set) else str(x)
        
        # Branch 1: The data is a raw list containing Pydantic AI validated schemas
        if isinstance(data, list) and len(data) > 0 and hasattr(data[0], 'model_dump'):
            json.dump([item.model_dump(mode='json') for item in data], f, indent=4, default=default_encoder)
            
        # Branch 2: The data is a singular Pydantic AI validated schema
        elif hasattr(data, 'model_dump'):
            json.dump(data.model_dump(mode='json'), f, indent=4, default=default_encoder)
            
        # Branch 3: Standard generic Python array/dictionary structure
        else:
            json.dump(data, f, indent=4, default=default_encoder)

async def execute_master_pipeline():
    """
    Executes the comprehensive 6-stage semantic pipeline natively.
    """
    start_time = time.time()
    logger.info("Initializing Master Orchestrator Pipeline natively.")
    
    # -----------------------------------------------------------------
    # STEP 0: INGESTION
    # -----------------------------------------------------------------
    # Identify and load the raw unstructured texts intended for logical mapping
    target_dir = "inputs/testdocs"
    files = glob.glob(os.path.join(target_dir, "*.txt")) + glob.glob(os.path.join(target_dir, "*.md"))
    
    raw_texts = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            raw_texts.append(f.read())
            
    logger.info(f"Ingested {len(raw_texts)} logic documents natively.")
    if not raw_texts:
        logger.critical("No documents found in inputs/testdocs. Terminating securely.")
        return

    # Instantiate all underlying mathematical and Generative engines
    extractor = ExtractionPipeline("config.yaml")
    embedder = EmbeddingPipeline("config.yaml")
    hypernyms = HypernymPipeline("config.yaml")
    mapper = NamingResolutionPipeline()
    topology = TopologyEngine()
    synthesizer = SynthesisEngine("config.yaml")
    visualizer = SemanticVisualizer()

    # -----------------------------------------------------------------
    # STAGE 1: DISCOVERY & THEMATIC EXTRACTION
    # -----------------------------------------------------------------
    logger.info("========================================")
    logger.info("STAGE 1: LLM EXTRACTION & DISCOVERY")
    logger.info("========================================")
    
    # Process 1.A: Parallel Sub-Thematic Discovery
    # We iterate over the documents to find isolated domain concepts natively
    all_themes = []
    for idx, text in enumerate(raw_texts):
        logger.info(f"Discovering thematic nodes inside document {idx + 1}/{len(raw_texts)}")
        themes = await extractor.discover_themes(text)
        all_themes.extend(themes)
    
    # Save mathematical state
    _save_state(all_themes, "outputs/01_extraction/original_themes.json")
    
    # Process 1.B: Master Domain Synthesis
    # Weights high-frequency themes mathematically and collapses them into a master taxonomy rulebook
    weighted_string = extractor.weight_themes(all_themes)
    master_context = await extractor.consolidate_themes(weighted_string)
    
    # Provide safe fallback string if the LLM failed validation
    master_domain = master_context.master_domain if master_context else "General Complex Logic"
    _save_state(master_context, "outputs/01_extraction/distilled_themes.json")
    
    # Process 1.C: Logical Triple Extraction (SVO)
    # Extracts concrete logic facts mapped explicitly against the Master Domain context
    raw_triples = []
    for idx, text in enumerate(raw_texts):
        logger.info(f"Extracting factual logic triples from document {idx + 1}/{len(raw_texts)}")
        trips = await extractor.extract_triples(text, master_context)
        raw_triples.extend(trips)
        
    _save_state(raw_triples, "outputs/01_extraction/original_triplets.json")
    
    # Generate PyVis HTML network visual
    visualizer.visualize_triples(raw_triples, "outputs/01_extraction/01_raw_triples_graph.html", "Phase 1: Raw Extractions")
    
    if not raw_triples:
        logger.warning("No logic facts securely extracted. Terminating pipeline gracefully.")
        return

    # Process 1.D: Text Normalization Phase (Cleaning logic arrays)
    logger.info("Executing NLP Lexical Normalization arrays...")
    normalized_triples, _, _, _ = await execute_normalization_phase(
        extractor, raw_triples, master_domain, _save_state
    )

    # -----------------------------------------------------------------
    # STAGE 2: OFFLINE EMBEDDING COMPRESSION
    # -----------------------------------------------------------------
    logger.info("========================================")
    logger.info("STAGE 2: EMBEDDING & CLUSTERING")
    logger.info("========================================")
    
    # Maps text into High-Dimensional Euclidean Space (SentenceTransformers).
    # Then mathematically compresses dimensions using PCA (Eigen-gap analysis).
    # Finally clusters the tight logic nodes cleanly using Agglomerative Cosine metrics.
    proposed_clusters = embedder.process_triples(normalized_triples)
    _save_state(proposed_clusters, "outputs/02_embedding/clusters_identified.json")
    
    logger.info(f"Mathematical embedding compression generated {len(proposed_clusters)} distinct physical clusters.")

    # -----------------------------------------------------------------
    # STAGE 3: HYBRID TAXONOMIC LIFTING
    # -----------------------------------------------------------------
    logger.info("========================================")
    logger.info("STAGE 3: TAXONOMIC LIFTING")
    logger.info("========================================")
    
    # Process 3.A: Contextual Validation
    # Gating mechanism. The LLM validates if the mathematical cluster logically makes sense.
    verified_clusters = await hypernyms.validate_context_vectors(proposed_clusters, master_domain)
    _save_state(verified_clusters, "outputs/03_hypernym_lifting/verified_clusters.json")
    
    # Process 3.B: Taxonomic Lift (Hypernym creation)
    # Replaces the verified cluster arrays with a single overarching conceptual "Hypernym" natively.
    hypernym_mapping = await hypernyms.taxonomic_lift(verified_clusters, master_domain)
    _save_state(hypernym_mapping, "outputs/03_hypernym_lifting/hypernym_mapping.json")

    # -----------------------------------------------------------------
    # STAGE 4: NAMING RESOLUTION
    # -----------------------------------------------------------------
    logger.info("========================================")
    logger.info("STAGE 4: TAXONOMIC NAMING RESOLUTION")
    logger.info("========================================")
    
    # Substitutes the raw, messy logic nodes mathematically with their clean hypernym counterparts
    mapped_triples = mapper.resolve_names(normalized_triples, hypernym_mapping)
    _save_state(mapped_triples, "outputs/04_mapping/mapped_triplets.json")
    
    # Visualize the newly simplified taxonomy graph
    visualizer.visualize_triples(mapped_triples, "outputs/04_mapping/02_resolved_triples_graph.html", "Phase 4: Abstracted Taxonomy")

    # -----------------------------------------------------------------
    # STAGE 5: NETWORK TOPOLOGY MATRICES
    # -----------------------------------------------------------------
    logger.info("========================================")
    logger.info("STAGE 5: TOPOLOGICAL GRAPH BUILDER")
    logger.info("========================================")
    
    # 5.A: Construct standard DiGraph NetworkX mathematically collapsing identical edges
    graph = topology.build_graph(mapped_triples)
    
    # Load Topology configurations safely mathematically
    overlap_threshold = 0.80
    leiden_resolution = 1.0
    min_community_size = 4
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
            overlap_threshold = cfg.get("topology", {}).get("inheritance_overlap_threshold", 0.80)
            leiden_resolution = cfg.get("topology", {}).get("leiden_resolution", 1.0)
            min_community_size = cfg.get("topology", {}).get("min_community_size", 4)
            
    # 5.B: Community Detection
    # Utilizes the Leiden mathematical Modularity algorithm via C-backed igraph
    partition = topology.detect_communities(graph, resolution=leiden_resolution)
    hierarchy = topology.extract_hierarchy(graph, partition, min_size=min_community_size)
    
    _save_state(partition, "outputs/05_topology/modularity_partition.json")
    _save_state(hierarchy, "outputs/05_topology/extracted_hierarchy.json")
    
    visualizer.visualize_topology(graph, partition, "outputs/05_topology/03_topology_communities_graph.html", "Phase 5: Modularity Topology")
    
    # 5.C: N-ary Hypergraph Calculation
    # Computes Incidence (H) and Laplacian (L) matrices natively
    hyper_data = topology.build_hypergraph_topology(mapped_triples, overlap_threshold=overlap_threshold)
    topology.visualize_hypergraph(hyper_data["B"], "outputs/05_topology")
    
    hyper_export = {
        "entities_count": hyper_data["entities"],
        "themes_count": hyper_data["themes"],
        "H_matrix": hyper_data["H"].tolist(),
        "L_matrix": hyper_data["L"].tolist(),
        "theme_inheritance_map": hyper_data.get("theme_inheritance_map", {})
    }
    _save_state(hyper_export, "outputs/05_topology/hypergraph_matrices.json")

    # -----------------------------------------------------------------
    # STAGE 6: GENERATIVE SYNTHESIS
    # -----------------------------------------------------------------
    logger.info("========================================")
    logger.info("STAGE 6: SEMANTIC SYNTHESIS GENERATION")
    logger.info("========================================")
    
    # Parses the discrete Leiden communities into structural Pydantic models cleanly
    theme_inheritance_map = hyper_data.get("theme_inheritance_map", {})
    resolved_schemas = await synthesizer.generate_schemas(hierarchy, master_domain, theme_inheritance_map)
    
    # Formats the final structural Python class code blocks and writes them logically
    file_path = synthesizer.build_global_context(resolved_schemas)
    logger.info(f"Final logical schema outputs smoothly synthesized to: {file_path}")

    # -----------------------------------------------------------------
    # WRAP UP
    # -----------------------------------------------------------------
    logger.info(f"Master Pipeline securely completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    asyncio.run(execute_master_pipeline())
