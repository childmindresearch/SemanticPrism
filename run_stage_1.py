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

def load_input_documents() -> list[dict]:
    """
    Loads input documents based on configuration settings.
    Supports directory of txt files or a single parquet file (via polars).
    Returns a list of dicts: [{'id': str, 'text': str}]
    """
    ingestion_cfg = settings.get('ingestion', {})
    source_type = ingestion_cfg.get('source_type', 'directory')
    
    docs = []
    if source_type == 'parquet':
        parquet_cfg = ingestion_cfg.get('parquet', {})
        filename = parquet_cfg.get('filename')
        id_field = parquet_cfg.get('id_field')
        text_field = parquet_cfg.get('text_field')
        
        if not filename or not id_field or not text_field:
            raise ValueError(
                f"Parquet ingestion is selected but filename ({filename}), "
                f"id_field ({id_field}), or text_field ({text_field}) is missing from config."
            )
            
        print(f"Loading input documents from Parquet file: {filename}")
        parquet_path = Path(filename)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found at: {filename}")
            
        import polars as pl
        df = pl.read_parquet(filename)
        
        # Validate columns
        if id_field not in df.columns:
            raise ValueError(f"id_field '{id_field}' not found in Parquet columns: {df.columns}")
        if text_field not in df.columns:
            raise ValueError(f"text_field '{text_field}' not found in Parquet columns: {df.columns}")
            
        # Extract documents
        for row in df.iter_rows(named=True):
            doc_id = str(row[id_field])
            text = str(row[text_field])
            docs.append({'id': doc_id, 'text': text})
            
    else:
        # Default: Directory of txt files
        input_dir = Path(settings.get('directories', {}).get('inputs', 'inputs/testdocs'))
        print(f"Loading input documents from Directory: {input_dir}")
        input_dir.mkdir(parents=True, exist_ok=True)
        txt_files = sorted(list(input_dir.glob("*.txt")))
        
        for txt_file in txt_files:
            text = txt_file.read_text(encoding='utf-8')
            docs.append({'id': txt_file.name, 'text': text})
            
    return docs

def main():
    print("Initializing Stage 1: Extraction Pipeline...")
    
    # Initialize the core pipeline state tracker
    context = PipelineRunContext()
    pipeline = ExtractionPipeline(context)
    
    # Load documents based on ingestion settings
    try:
        documents = load_input_documents()
    except Exception as e:
        print(f"Error loading input documents: {e}")
        sys.exit(1)
        
    if not documents:
        print("No documents found to process. Please check inputs and run again.")
        sys.exit(0)
        
    print(f"Found {len(documents)} document(s) to process. Beginning extraction...\n")
    
    # Determine resume mode settings
    resume_mode = settings.get('pipeline', {}).get('resume_mode', 'skip')
    print(f"Pipeline Resume Mode: {resume_mode}\n")
    
    # Phase 1: Global Theme Discovery
    print("=== Phase 1: Global Theme Discovery ===")
    for doc in documents:
        safe_name = pipeline.sanitize_filename(doc['id'])
        theme_file_path = pipeline.themes_dir / f"{safe_name}_themes.json"
        
        if resume_mode == 'skip' and theme_file_path.exists():
            print(f"-> Skipping theme discovery for: {doc['id']} (already exists on disk)")
        else:
            print(f"-> Discovering themes in: {doc['id']}")
            pipeline.discover_themes(doc['text'], source_doc=doc['id'])
            
    # Theme Aggregation Step
    print("\n=== Phase 1.5: Theme Aggregation ===")
    print("-> Consolidating individual themes files...")
    pipeline.aggregate_themes()
        
    # Phase 2: Master Theme Synthesis
    print("\n=== Phase 2: Master Theme Synthesis ===")
    print("-> Synthesizing master themes from all documents...")
    pipeline.synthesize_master_themes()
    
    print("\nPurging VRAM...")
    purge_vram()
    
    # Phase 3: Global Triple Extraction
    print("\n=== Phase 3: Global Triple Extraction ===")
    for doc in documents:
        safe_name = pipeline.sanitize_filename(doc['id'])
        triple_file_path = pipeline.triples_dir / f"{safe_name}_triplets.json"
        
        if resume_mode == 'skip' and triple_file_path.exists():
            print(f"-> Skipping triple extraction for: {doc['id']} (already exists on disk)")
        else:
            print(f"-> Extracting triplets from: {doc['id']}")
            pipeline.extract_triples(doc['text'], source_doc=doc['id'])
            
    # Triple Aggregation Step
    print("\n=== Phase 3.5: Triple Aggregation ===")
    print("-> Consolidating individual triplets files...")
    pipeline.aggregate_triplets()
        
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
