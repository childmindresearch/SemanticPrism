"""
SemanticPrism Stage 2: Full Refinement Pipeline Orchestrator
This script sequentially executes Stage 2 Lexical Normalization and Stage 2 Taxonomic Lifting
in isolated subprocesses to ensure memory and VRAM are cleanly garbage collected.
"""

import subprocess
import sys
import time

def run_substage(script_name: str, substage_desc: str):
    print("\n" + "="*60)
    print(f"🚀 INITIATING {substage_desc.upper()}")
    print("="*60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"\n❌ CRITICAL ERROR in {script_name}!")
        print(f"Stage 2 halted at {substage_desc}.")
        sys.exit(1)
        
    duration = time.time() - start_time
    print(f"\n✅ {substage_desc} completed in {duration:.2f} seconds.")

def main():
    print("=== SemanticPrism Stage 2: Full Refinement Pipeline ===")
    
    stage_start = time.time()
    
    run_substage("run_stage_2_normalization.py", "Stage 2 Part 1: Lexical Normalization")
    run_substage("run_stage_2_taxonomic_lifting.py", "Stage 2 Part 2: Taxonomic Lifting & Theme Mapping")
    
    total_duration = time.time() - stage_start
    print("\n" + "="*60)
    print(f"🎉 STAGE 2 FULL REFINEMENT PIPELINE COMPLETE!")
    print(f"Total Stage 2 Time: {total_duration:.2f} seconds")
    print("Outputs saved in 'outputs/02_refinement':")
    print(" - subject_normalization_map.json")
    print(" - predicate_normalization_map.json")
    print(" - object_normalization_map.json")
    print(" - refined_triplets.json")
    print(" - entity_clusters.json")
    print(" - theme_taxonomy_mapping.json")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
