"""
SemanticPrism Stage 2: Refinement Pipeline
Executes two-tier vector clustering, taxonomic hypernym lifting, and theme association.
"""

import sys
import json
import time
import argparse
from pathlib import Path

from src.config import settings
from src.extraction.schemas import RawTriple
from src.refinement.refiner import RefinementPipeline, PipelineRunContext


def run_fidelity_only(pipeline: RefinementPipeline, out_dir: Path):
    """Fast computation of sentence vectors and contextual fidelity on existing refined triplets (< 10s)."""
    refined_path = out_dir / "refined_triplets.json"
    if not refined_path.exists():
        print(f"CRITICAL ERROR: Required file '{refined_path}' not found.")
        print("Please run Stage 2 fully first before running --fidelity-only.")
        sys.exit(1)

    print(f"=== SemanticPrism Stage 2: Fast Contextual Fidelity Update ({refined_path}) ===")
    with open(refined_path, "r", encoding="utf-8") as f:
        refined_data = json.load(f)

    triplet_dicts = pipeline._compute_and_cache_sentence_context(refined_data, out_dir)
    with open(refined_path, "w", encoding="utf-8") as f:
        json.dump(triplet_dicts, f, indent=2)

    print(f"✅ Successfully updated {len(triplet_dicts)} triplets with contextual fidelity in '{refined_path}'.")
    print(f"✅ Grounding diagnostics saved in '{out_dir / 'contextual_grounding_diagnostics.json'}'.")


def main():
    parser = argparse.ArgumentParser(
        description="SemanticPrism Stage 2: Refinement Pipeline (Vector Clustering & Taxonomic Lifting)"
    )
    parser.add_argument(
        "--fidelity-only",
        action="store_true",
        help="Fast computation of sentence vectors and contextual fidelity on existing refined triplets (< 10s)",
    )
    args = parser.parse_args()

    master_domain = settings.get('extraction', {}).get('domain', 'General')
    context = PipelineRunContext(master_domain=master_domain)
    pipeline = RefinementPipeline(config=settings, context=context)
    out_dir = Path("outputs/02_refinement")

    if args.fidelity_only:
        run_fidelity_only(pipeline, out_dir)
        return

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
        print(" - contextual_grounding_diagnostics.json")
        print("="*60 + "\n")
    except Exception as e:
        print(f"Error during Stage 2 Refinement execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
