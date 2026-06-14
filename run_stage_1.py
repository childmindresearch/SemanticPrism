"""
SemanticPrism Stage 1: Execution Script
This standalone script initializes the Stage 1 pipeline, loads the central config,
reads documents, and fires off the extraction process. It ensures the stage can run in isolation.
"""

import sys
import os
from pathlib import Path

# Append project root to sys.path to ensure modules can be imported directly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import settings
from src.extraction.extractor import ExtractionPipeline, PipelineRunContext
from src.agents.vram_manager import purge_vram

def main():
    print("Initializing Stage 1: Extraction Pipeline...")
    
    # Ensure the designated input directory exists
    input_dir = Path(settings.get('directories', {}).get('inputs', 'inputs/testdocs'))
    input_dir.mkdir(parents=True, exist_ok=True) 
    
    # Initialize the core pipeline state tracker
    context = PipelineRunContext()
    pipeline = ExtractionPipeline(context)
    
    # Locate all raw text files to process
    txt_files = list(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in '{input_dir}'. Please place some test documents and run again.")
        sys.exit(0)
        
    print(f"Found {len(txt_files)} document(s) in {input_dir}. Beginning extraction...\n")
    
    # Phase 1: Global Theme Discovery
    print("=== Phase 1: Global Theme Discovery ===")
    for txt_file in txt_files:
        print(f"-> Discovering themes in: {txt_file.name}")
        text = txt_file.read_text(encoding='utf-8')
        pipeline.discover_themes(text, source_doc=txt_file.name)
        
    # Phase 2: Master Theme Synthesis
    print("\n=== Phase 2: Master Theme Synthesis ===")
    print("-> Synthesizing master themes from all documents...")
    pipeline.synthesize_master_themes()
    
    print("\nPurging VRAM...")
    purge_vram()
    
    # Phase 3: Global Triple Extraction
    print("\n=== Phase 3: Global Triple Extraction ===")
    for txt_file in txt_files:
        print(f"-> Extracting triplets from: {txt_file.name}")
        text = txt_file.read_text(encoding='utf-8')
        pipeline.extract_triples(text, source_doc=txt_file.name)
        
    print("\nStage 1 Complete!")
    
    # Output final summary metrics
    all_themes_count = len(context.all_discovered_themes)
    master_themes_count = len(context.master_themes.master_themes) if context.master_themes else 0
    print(f"Total Raw Themes Discovered: {all_themes_count}")
    print(f"Total Master Themes Synthesized: {master_themes_count}")
    print(f"Total Triples Extracted: {len(context.raw_triples)}")
    
    out_dir = Path(settings.get('directories', {}).get('outputs', 'outputs')) / "01_extraction"
    print(f"Outputs saved to '{out_dir}':")
    print(" - all_themes.json")
    print(" - master_themes.json")
    print(" - original_triplets.json")
    
    print("\nPurging VRAM...")
    purge_vram()

if __name__ == "__main__":
    main()
