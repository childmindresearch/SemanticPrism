"""
Isolated Execution Script for SemanticPrism Stage 2: Refinement Pipeline (Part 2 - Clustering, Taxonomic Lifting, Theme Mapping)
"""

import json
from pathlib import Path

from src.config import settings
from src.extraction.schemas import RawTriple
from src.refinement.refiner import RefinementPipeline, PipelineRunContext

def main():
    print("=== SemanticPrism Stage 2: Refinement Pipeline (Part 2 - Clustering, Taxonomic Lifting, Theme Mapping) ===")
    
    # Define paths
    input_dir = Path("outputs/01_extraction")
    refinement_dir = Path("outputs/02_refinement")
    
    if not input_dir.exists():
        print(f"Error: Required input directory '{input_dir}' not found.")
        print("Please run Stage 1 first.")
        return

    # Check if Part 1 outputs exist
    sub_map_path = refinement_dir / "subject_normalization_map.json"
    pred_map_path = refinement_dir / "predicate_normalization_map.json"
    obj_map_path = refinement_dir / "object_normalization_map.json"
    
    if not (sub_map_path.exists() and pred_map_path.exists() and obj_map_path.exists()):
        print("Error: Normalization maps from Stage 2 Part 1 not found in outputs/02_refinement/.")
        print("Please run Stage 2 Part 1 first using: python run_stage_2_part_1.py")
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

    # Load original themes
    try:
        with open(input_dir / "all_themes.json", "r") as f:
            original_themes = json.load(f)
    except Exception as e:
        print(f"Failed to load original themes: {e}")
        return

    # Load original triplets
    try:
        with open(input_dir / "original_triplets.json", "r") as f:
            triplets_data = json.load(f)
            raw_triples = [RawTriple(**t) for t in triplets_data]
    except Exception as e:
        print(f"Failed to load original triplets: {e}")
        return

    # Load normalization maps
    try:
        with open(sub_map_path, "r") as f:
            subject_map = json.load(f)
        with open(pred_map_path, "r") as f:
            predicate_map = json.load(f)
        with open(obj_map_path, "r") as f:
            object_map = json.load(f)
    except Exception as e:
        print(f"Failed to load normalization maps: {e}")
        return

    # Initialize Pipeline Context
    context = PipelineRunContext(master_domain=master_domain)

    # Execute Refinement Pipeline Part 2
    pipeline = RefinementPipeline(config=settings, context=context)
    
    try:
        pipeline.execute_part_2(
            raw_triples=raw_triples,
            original_themes=original_themes,
            master_themes=master_themes,
            subject_map=subject_map,
            predicate_map=predicate_map,
            object_map=object_map
        )
        print("=== Stage 2 Refinement Part 2 Completed Successfully ===")
    except Exception as e:
        print(f"Error during Refinement execution (Part 2): {e}")

if __name__ == "__main__":
    main()
