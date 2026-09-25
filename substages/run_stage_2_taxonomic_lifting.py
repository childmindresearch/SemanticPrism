"""
Isolated Execution Script for SemanticPrism Stage 2: Refinement Pipeline (Clustering, Taxonomic Lifting, Theme Mapping)
"""

import sys
import os
import json
from pathlib import Path

# Append project root to sys.path to ensure modules can be imported directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings
from src.extraction.schemas import RawTriple
from src.refinement.refiner import RefinementPipeline, PipelineRunContext

def main():
    print("=== SemanticPrism Stage 2: Refinement Pipeline (Clustering, Taxonomic Lifting, Theme Mapping) ===")
    
    # Determine resume mode settings
    resume_mode = settings.get('pipeline', {}).get('resume_mode', 'skip')
    print(f"Pipeline Resume Mode: {resume_mode}\n")
    
    # Define paths
    input_dir = Path("outputs/01_extraction")
    refinement_dir = Path("outputs/02_refinement")
    
    if not input_dir.exists():
        print(f"CRITICAL ERROR: Required input directory '{input_dir}' not found.")
        print("Please run Stage 1 first ('python3 run_stage_1.py' or 'python3 run_pipeline.py').")
        sys.exit(1)

    # Check if Normalization outputs exist
    sub_map_path = refinement_dir / "subject_normalization_map.json"
    pred_map_path = refinement_dir / "predicate_normalization_map.json"
    obj_map_path = refinement_dir / "object_normalization_map.json"
    
    # Load original triplets
    triplets_path = input_dir / "original_triplets.json"
    if not triplets_path.exists():
        print(f"CRITICAL ERROR: Required file '{triplets_path}' not found.")
        print("Stage 2 requires Stage 1 triple extraction outputs. Please run Stage 1 first ('python3 run_stage_1.py' or 'python3 run_pipeline.py').")
        sys.exit(1)

    try:
        with open(triplets_path, "r") as f:
            triplets_data = json.load(f)
            raw_triples = [RawTriple(**t) for t in triplets_data]
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load original triplets: {e}")
        sys.exit(1)

    # Initialize Pipeline Context and Pipeline
    master_domain = settings.get('extraction', {}).get('domain', 'General')
    context = PipelineRunContext(master_domain=master_domain)
    pipeline = RefinementPipeline(config=settings, context=context)

    # Load normalization maps or fallback to preprocessed identity mapping
    if sub_map_path.exists():
        with open(sub_map_path, "r") as f:
            subject_map = json.load(f)
    else:
        print("   -> subject_normalization_map.json not found; using preprocessed identity map.")
        subject_map = {t.subject: pipeline._nlp_preprocess(t.subject) for t in raw_triples}

    if pred_map_path.exists():
        with open(pred_map_path, "r") as f:
            predicate_map = json.load(f)
    else:
        print("   -> predicate_normalization_map.json not found; using preprocessed identity map.")
        predicate_map = {t.predicate: pipeline._nlp_preprocess(t.predicate) for t in raw_triples}

    if obj_map_path.exists():
        with open(obj_map_path, "r") as f:
            object_map = json.load(f)
    else:
        print("   -> object_normalization_map.json not found; using preprocessed identity map.")
        object_map = {t.object: pipeline._nlp_preprocess(t.object) for t in raw_triples}

    try:
        pipeline.execute_part_2(
            raw_triples=raw_triples,
            subject_map=subject_map,
            predicate_map=predicate_map,
            object_map=object_map
        )
        print("=== Stage 2 Taxonomic Lifting Completed Successfully ===")
    except Exception as e:
        print(f"Error during Taxonomic Lifting execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
