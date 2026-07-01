"""
Isolated Execution Script for SemanticPrism Stage 2: Refinement Pipeline (Part 1 - Lexical Normalization)
"""

import json
from pathlib import Path

from src.config import settings
from src.extraction.schemas import RawTriple
from src.refinement.refiner import RefinementPipeline, PipelineRunContext

def main():
    print("=== SemanticPrism Stage 2: Refinement Pipeline (Part 1 - Lexical Normalization) ===")
    
    # Define paths
    input_dir = Path("outputs/01_extraction")
    if not input_dir.exists():
        print(f"Error: Required input directory '{input_dir}' not found.")
        print("Please run Stage 1 first.")
        return

    # Load master themes
    try:
        with open(input_dir / "master_themes.json", "r") as f:
            master_data = json.load(f)
            master_domain = master_data.get("master_domain", settings.get('extraction', {}).get('domain', 'Unknown'))
            master_themes = master_data.get("master_themes", [])
    except Exception as e:
        print(f"Failed to load master themes: {e}")
        return

    # Load original triplets
    try:
        with open(input_dir / "original_triplets.json", "r") as f:
            triplets_data = json.load(f)
            # Parse into Pydantic models
            raw_triples = [RawTriple(**t) for t in triplets_data]
    except Exception as e:
        print(f"Failed to load original triplets: {e}")
        return

    # Initialize Pipeline Context
    context = PipelineRunContext(master_domain=master_domain)

    # Execute Refinement Pipeline Part 1
    pipeline = RefinementPipeline(config=settings, context=context)
    
    try:
        pipeline.execute_part_1(
            raw_triples=raw_triples,
            master_themes=master_themes
        )
        print("=== Stage 2 Refinement Part 1 Completed Successfully ===")
    except Exception as e:
        print(f"Error during Refinement execution (Part 1): {e}")

if __name__ == "__main__":
    main()
