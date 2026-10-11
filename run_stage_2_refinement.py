"""
SemanticPrism Stage 2: Refinement Pipeline
Executes two-tier vector clustering, taxonomic hypernym lifting, and theme association.
"""

import sys
import json
import time
from pathlib import Path

from src.config import settings
from src.extraction.schemas import RawTriple
from src.refinement.refiner import RefinementPipeline, PipelineRunContext

def main():
    print("=== SemanticPrism Stage 2: Refinement Pipeline (Vector Clustering & Taxonomic Lifting) ===")
    
    resume_mode = settings.get('pipeline', {}).get('resume_mode', 'skip')
    print(f"Pipeline Resume Mode: {resume_mode}\n")
    
    stage_start = time.time()
    
    # Define paths
    input_dir = Path("outputs/01_extraction")
    if not input_dir.exists():
        print(f"CRITICAL ERROR: Required input directory '{input_dir}' not found.")
        print("Please run Stage 1 first ('python3 run_stage_1_extraction.py' or 'python3 run_pipeline.py').")
        sys.exit(1)

    # Load original triplets
    triplets_path = input_dir / "original_triplets.json"
    if not triplets_path.exists():
        print(f"CRITICAL ERROR: Required file '{triplets_path}' not found.")
        print("Stage 2 requires Stage 1 triple extraction outputs. Please run Stage 1 first ('python3 run_stage_1_extraction.py' or 'python3 run_pipeline.py').")
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

    try:
        pipeline.execute(raw_triples=raw_triples)
        
        total_duration = time.time() - stage_start
        print("\n" + "="*60)
        print("🎉 STAGE 2 REFINEMENT PIPELINE COMPLETE!")
        print(f"Total Stage 2 Time: {total_duration:.2f} seconds")
        print("Outputs saved in 'outputs/02_refinement':")
        print(" - subject_clusters.json")
        print(" - predicate_clusters.json")
        print(" - object_clusters.json")
        print(" - subject_taxonomic_map.json")
        print(" - predicate_taxonomic_map.json")
        print(" - object_taxonomic_map.json")
        print(" - refined_triplets.json")
        print("="*60 + "\n")
    except Exception as e:
        print(f"Error during Stage 2 Refinement execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
