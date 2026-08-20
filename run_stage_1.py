"""
SemanticPrism Stage 1: Full Orchestrator Script
This script sequentially executes Stage 1 Theme Extraction and Stage 1 Triple Extraction
in isolated subprocesses to ensure VRAM and memory are cleanly garbage collected.
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
        print(f"Stage 1 halted at {substage_desc}.")
        sys.exit(1)
        
    duration = time.time() - start_time
    print(f"\n✅ {substage_desc} completed in {duration:.2f} seconds.")

def main():
    print("=== SemanticPrism Stage 1: Full Extraction Pipeline ===")
    
    stage_start = time.time()
    
    run_substage("run_stage_1_themes.py", "Stage 1 Part 1: Theme Discovery & Synthesis")
    run_substage("run_stage_1_triples.py", "Stage 1 Part 2: Triple Extraction & Aggregation")
    
    total_duration = time.time() - stage_start
    print("\n" + "="*60)
    print(f"🎉 STAGE 1 FULL EXTRACTION PIPELINE COMPLETE!")
    print(f"Total Stage 1 Time: {total_duration:.2f} seconds")
    print("Outputs saved in 'outputs/01_extraction':")
    print(" - all_themes.json")
    print(" - master_themes.json")
    print(" - original_triplets.json")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
