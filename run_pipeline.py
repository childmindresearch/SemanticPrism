"""
SemanticPrism Master Pipeline Execution Script
This script sequentially executes Stages 1 through 4 of the SemanticPrism pipeline.
It utilizes subprocesses to ensure memory and VRAM are cleanly garbage collected
between each intensive LLM and mathematical operation.
"""

import subprocess
import sys
import time

def run_stage(script_name: str, stage_desc: str):
    print("\n" + "="*60)
    print(f"🚀 INITIATING {stage_desc.upper()}")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Run the script as a subprocess, piping output directly to the main terminal
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"\n❌ CRITICAL ERROR in {script_name}!")
        print(f"Pipeline halted at {stage_desc}.")
        sys.exit(1)
        
    duration = time.time() - start_time
    print(f"\n✅ {stage_desc} completed in {duration:.2f} seconds.")

def main():
    print("Welcome to SemanticPrism.")
    print("Starting full extraction, refinement, topology, and synthesis pipeline...\n")
    
    global_start = time.time()
    
    stages = [
        ("run_stage_1.py", "Stage 1: Triplet Extraction & Theme Discovery"),
        ("run_stage_2.py", "Stage 2: Lexical Refinement & Taxonomic Lifting"),
        ("run_stage_3.py", "Stage 3: Topology Graphing & Structural Clustering"),
        ("run_stage_4.py", "Stage 4: Pydantic Schema Synthesis")
    ]
    
    for script, desc in stages:
        run_stage(script, desc)
        
    global_duration = time.time() - global_start
    print("\n" + "="*60)
    print(f"🎉 FULL SEMANTIC PRISM PIPELINE COMPLETE!")
    print(f"Total Execution Time: {global_duration:.2f} seconds")
    print("Check the 'outputs/' directory for your final topology graphs and Pydantic schemas.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
