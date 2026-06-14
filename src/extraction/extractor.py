"""
SemanticPrism Stage 1: Extractor Pipeline
This module orchestrates the Pydantic AI agents to process raw text, extract themes, and mine SVO triplets.
"""

import json
import os

from typing import List, Set, Optional, Tuple
from pathlib import Path

from . import schemas
from . import prompts
from src.agents.extraction_agents import theme_agent, master_theme_agent, triple_agent, TripleContext
from src.config import settings

class PipelineRunContext:
    """
    State tracking object passed throughout the pipeline execution to persist data
    across chunk iterations and multi-document passes without using global variables.
    """
    def __init__(self):
        self.all_discovered_themes: List[schemas.Theme] = []
        self.master_themes: Optional[schemas.MasterThemeSynthesisResult] = None
        self.raw_triples: List[dict] = []

class ExtractionPipeline:
    """
    Master pipeline class that chunks text and coordinates the agents to perform extraction.
    """
    def __init__(self, context: PipelineRunContext):
        self.context = context
        
        # Load chunk size constraints from config
        self.theme_chunk_size = settings['extraction']['theme_chunk_max_words']
        self.triple_chunk_size = settings['extraction']['triple_chunk_max_words']
        
        # Setup Output Directories
        base_out_dir = settings.get('directories', {}).get('outputs', 'outputs')
        self.out_dir = Path(base_out_dir) / "01_extraction"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(base_out_dir) / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear previous run logs to prevent continuous expansion across isolated runs
        for log_file in [
            "stage_01_theme_discovery_errors.json", 
            "stage_01_master_theme_errors.json", 
            "stage_01_triple_extraction_errors.json"
        ]:
            log_path = self.log_dir / log_file
            if log_path.exists():
                log_path.unlink()

    def chunk_text(self, text: str, max_words: int) -> List[Tuple[str, int, int]]:
        """
        Splits raw text into manageable chunks.
        Returns a list of tuples: (chunk_text, start_word_index, end_word_index)
        """
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            
            # If we aren't at the end of the text, look forward for a sentence boundary
            if end < len(words):
                while end < len(words) and not words[end - 1].endswith(('.', '!', '?', '."', '!"', '?"')):
                    end += 1
                    
            chunk = " ".join(words[start:end])
            chunks.append((chunk, start, end))
            
            if end >= len(words):
                break
                
            # Start the next chunk exactly where the last sentence ended
            start = end
            
        return chunks

    def discover_themes(self, text: str, source_doc: str):
        """
        Phase 1: Extracts themes from text chunks and aggregates them globally.
        Checkpoints to disk as 'all_themes.json'.
        """
        theme_chunks = self.chunk_text(text, self.theme_chunk_size)
        
        for chunk, start_idx, end_idx in theme_chunks:
            user_prompt = prompts.THEME_DISCOVERY_USER_PROMPT.format(text_content=chunk)
            
            try:
                result = theme_agent.run_sync(user_prompt)
            except Exception as e:
                error_record = {
                    "source_document": source_doc,
                    "start_word": start_idx,
                    "end_word": end_idx,
                    "error": str(e)
                }
                with open(self.log_dir / "stage_01_theme_discovery_errors.json", "a") as f:
                    f.write(json.dumps(error_record) + "\n")
                continue
            
            # Record themes and append to global context
            for theme in result.output.themes:
                self.context.all_discovered_themes.append(theme)
                
        # Checkpoint: Save all accumulated themes
        with open(self.out_dir / "all_themes.json", "w") as f:
            json.dump([t.model_dump() for t in self.context.all_discovered_themes], f, indent=2)

    def synthesize_master_themes(self):
        """
        Phase 2: Synthesizes the global aggregated themes into a consolidated Master Ontology.
        Checkpoints to disk as 'master_themes.json'.
        """
        if not self.context.all_discovered_themes:
            return
            
        # Consolidate ALL chunk-level themes across the corpus into a master list
        themes_str = "\n".join([f"- {t.title}: {t.description}" for t in self.context.all_discovered_themes])
        master_user_prompt = prompts.MASTER_THEME_USER_PROMPT.format(all_extracted_themes=themes_str)
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                master_result = master_theme_agent.run_sync(master_user_prompt)
                
                # Persist master themes in state
                self.context.master_themes = master_result.output
                
                # Checkpoint: Save master themes to disk immediately
                with open(self.out_dir / "master_themes.json", "w") as f:
                    f.write(master_result.output.model_dump_json(indent=2))
                
                break # Success, exit the retry loop
                
            except Exception as e:
                if attempt == max_retries - 1:
                    error_record = {
                        "phase": "master_theme_synthesis",
                        "error": str(e)
                    }
                    with open(self.log_dir / "stage_01_master_theme_errors.json", "a") as f:
                        f.write(json.dumps(error_record) + "\n")
                    print(f"Warning: Master theme synthesis failed after {max_retries} attempts.")
                    # Gracefully allow to exit without crashing
                else:
                    print(f"Master theme synthesis attempt {attempt + 1} failed, retrying...")

    def extract_triples(self, text: str, source_doc: str):
        """
        Phase 3: Extracts SVO triplets using global Master Themes and coreference context.
        Checkpoints to disk as 'original_triplets.json'.
        """
        triple_chunks = self.chunk_text(text, self.triple_chunk_size)
        
        for chunk, start_idx, end_idx in triple_chunks:
            # Package state context for the agent
            deps = TripleContext(
                master_themes=self.context.master_themes
            )
            
            # Prepare textual context for the user prompt
            themes_context_str = f"Discovered Themes:\n{self.context.master_themes.model_dump_json()}\n\n" if self.context.master_themes else ""
            
            user_prompt = prompts.TRIPLE_EXTRACTION_USER_PROMPT.format(
                themes_context=themes_context_str,
                text_content=chunk
            )
            
            try:
                # Execute extraction
                triple_result = triple_agent.run_sync(user_prompt, deps=deps)
            except Exception as e:
                error_record = {
                    "source_document": source_doc,
                    "start_word": start_idx,
                    "end_word": end_idx,
                    "error": str(e)
                }
                with open(self.log_dir / "stage_01_triple_extraction_errors.json", "a") as f:
                    f.write(json.dumps(error_record) + "\n")
                continue
            
            # Update state with new entities and save triplets
            for t in triple_result.output.triples:
                t_dict = t.model_dump()
                t_dict["source_document"] = source_doc # Guarantee source mapping natively
                self.context.raw_triples.append(t_dict)
                
            # Checkpoint: Save pristine, unmodified triples to disk incrementally per chunk
            self.save_triplets()

    def save_triplets(self):
        """Helper to save the current state of raw_triples."""
        with open(self.out_dir / "original_triplets.json", "w") as f:
            json.dump(self.context.raw_triples, f, indent=2)
