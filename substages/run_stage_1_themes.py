"""
SemanticPrism Stage 1: Theme Discovery & Synthesis Script
This standalone script initializes the Stage 1 pipeline for themes, loads the central config,
reads input documents, discovers raw themes across documents, aggregates them, and synthesizes master themes.
"""

import asyncio
import sys
import os
from pathlib import Path

# Append project root to sys.path to ensure modules can be imported directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings
from src.extraction.extractor import ExtractionPipeline, PipelineRunContext, load_input_documents
from src.agents.vram_manager import purge_vram
from src.agents.extraction_agents import theme_model

async def main_async():
    print("Initializing Stage 1: Theme Discovery & Synthesis...")
    
    # Initialize the core pipeline state tracker
    context = PipelineRunContext()
    pipeline = ExtractionPipeline(context)
    pipeline.clear_stage_logs("themes")
    pipeline.clear_stage_logs("master")
    
    # Load documents based on ingestion settings
    try:
        documents = load_input_documents()
    except Exception as e:
        print(f"Warning: Loading input documents encountered issue: {e}")
        documents = []
        
    if not documents:
        existing_theme_files = list(pipeline.themes_dir.glob("*_themes.json"))
        all_themes_file = pipeline.out_dir / "all_themes.json"
        master_themes_file = pipeline.out_dir / "master_themes.json"
        
        if not existing_theme_files and not all_themes_file.exists() and not master_themes_file.exists():
            print("❌ ERROR: No input documents found AND no existing theme files found on disk.")
            print("Please check your ingestion settings or input file paths.")
            sys.exit(1)
        else:
            print("-> No input documents loaded, but existing theme files were found on disk.")
            print("-> Proceeding to Theme Aggregation & Master Synthesis/Mapping from disk state...\n")
    else:
        print(f"Found {len(documents)} document(s) to process. Beginning theme extraction...\n")
    
    # Determine resume mode settings
    resume_mode = settings.get('pipeline', {}).get('resume_mode', 'skip')
    print(f"Pipeline Resume Mode: {resume_mode}\n")
    
    # Phase 1: Global Theme Discovery
    print(f"=== Phase 1: Global Theme Discovery (Model: {theme_model}) ===")
    for doc in documents:
        safe_name = pipeline.sanitize_filename(doc['id'])
        theme_file_path = pipeline.themes_dir / f"{safe_name}_themes.json"
        
        if resume_mode == 'skip' and theme_file_path.exists():
            print(f"-> Skipping theme discovery for: {doc['id']} (already exists on disk)")
        else:
            print(f"-> Discovering themes in: {doc['id']}")
            if pipeline.use_async:
                await pipeline.discover_themes_async(doc['text'], source_doc=doc['id'])
            else:
                pipeline.discover_themes(doc['text'], source_doc=doc['id'])
            
    # Phase 1.5: Theme Aggregation
    print("\n=== Phase 1.5: Theme Aggregation ===")
    print("-> Consolidating individual themes files...")
    pipeline.aggregate_themes()
        
    # Phase 2: Master Theme Synthesis
    print(f"\n=== Phase 2: Master Theme Synthesis (Model: {theme_model}) ===")
    print("-> Synthesizing master themes from all documents...")
    await pipeline.synthesize_master_themes_async()
    
    print("\nPurging VRAM...")
    purge_vram()
    
    print("\nStage 1 Theme Extraction Complete!")
    
    # Output final summary metrics
    all_themes_count = len(context.all_discovered_themes)
    master_themes_count = len(context.master_themes.master_themes) if context.master_themes else 0
    print(f"Total Raw Themes Discovered: {all_themes_count}")
    print(f"Total Master Themes Synthesized: {master_themes_count}")
    
    out_dir = Path(settings.get('directories', {}).get('outputs', 'outputs')) / "01_extraction"
    print(f"Outputs saved to '{out_dir}':")
    print(" - all_themes.json")
    print(" - clustering_summary.json")
    print(" - theme_cluster_galaxy_3d.html")
    print(" - cluster_candidates.json")
    print(" - master_themes.json")
    print(" - theme_mapping_clusters.json")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
