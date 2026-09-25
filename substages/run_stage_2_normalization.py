"""
Isolated Execution Script for SemanticPrism Stage 2: Refinement Pipeline (Lexical Normalization)
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
    print("=== SemanticPrism Stage 2: Refinement Pipeline (Lexical Normalization) ===")
    
    # Determine resume mode settings
    resume_mode = settings.get('pipeline', {}).get('resume_mode', 'skip')
    print(f"Pipeline Resume Mode: {resume_mode}\n")
    
    # Define paths
    input_dir = Path("outputs/01_extraction")
    if not input_dir.exists():
        print(f"CRITICAL ERROR: Required input directory '{input_dir}' not found.")
        print("Please run Stage 1 first ('python3 run_stage_1.py' or 'python3 run_pipeline.py').")
        sys.exit(1)

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

    # Initialize Pipeline Context
    master_domain = settings.get('extraction', {}).get('domain', 'General')
    context = PipelineRunContext(master_domain=master_domain)

    # Execute Refinement Pipeline (Lexical Normalization)
    pipeline = RefinementPipeline(config=settings, context=context)
    
    try:
        pipeline.execute_part_1(raw_triples=raw_triples)
        print("=== Stage 2 Lexical Normalization Completed Successfully ===")
    except Exception as e:
        print(f"Error during Lexical Normalization execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
