"""
SemanticPrism Stage 1: Extractor Pipeline
This module orchestrates the Pydantic AI agents to process raw text, extract themes, and mine SVO triplets.
"""

import json
import os
import asyncio

from typing import List, Set, Optional, Tuple
from pathlib import Path

from . import schemas
from . import prompts
from src.agents.extraction_agents import theme_agent, master_theme_agent, triple_agent, TripleContext, triple_reformat_agent, triple_model
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
        self.processed_documents = set()

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
        
        self.themes_dir = self.out_dir / "themes"
        self.themes_dir.mkdir(parents=True, exist_ok=True)
        
        self.triples_dir = self.out_dir / "triples"
        self.triples_dir.mkdir(parents=True, exist_ok=True)
        
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

        # Concurrency settings
        self.use_async = settings.get('pipeline', {}).get('use_async', False)
        self.max_async = settings.get('extraction', {}).get('max_async_calls', 1)
        self.context_cap = settings.get('extraction', {}).get('context_window_cap', 8192)

        # Automatically load master themes if available on disk and not already set
        if self.context.master_themes is None:
            self.load_master_themes()

    def load_master_themes(self) -> Optional[schemas.MasterThemeSynthesisResult]:
        """Loads master_themes.json from disk into the context if it exists."""
        master_path = self.out_dir / "master_themes.json"
        if master_path.exists():
            try:
                with open(master_path, "r", encoding="utf-8") as f:
                    master_data = json.load(f)
                    self.context.master_themes = schemas.MasterThemeSynthesisResult(**master_data)
                    return self.context.master_themes
            except Exception as e:
                print(f"Warning: Failed to load master themes from {master_path}: {e}")
        return None

    def sanitize_filename(self, name: str) -> str:
        """Sanitizes document identifiers to be filesystem-safe."""
        import re
        return re.sub(r'[\\/:*?"<>|]', '_', name)

    def _ensure_fit(self, chunk: str, template: str, system_prompt: str) -> str:
        """Estimates token footprint and slices/truncates chunk to fit within the extraction context window cap."""
        from src.utils.token_helper import validate_and_trim_prompt
        full_sys_prompt = system_prompt + "\n" + template.replace("{text_content}", "")
        return validate_and_trim_prompt(chunk, full_sys_prompt, self.context_cap, output_buffer=1000)

    def _extract_malformed_text(self, messages, exception: Exception) -> str:
        """Extracts the raw malformed output string from the agent's message history or exception."""
        from pydantic_ai.messages import ModelResponse, ToolCallPart, TextPart
        for msg in reversed(messages):
            if isinstance(msg, ModelResponse):
                tool_calls = [p for p in msg.parts if isinstance(p, ToolCallPart)]
                if tool_calls:
                    args = tool_calls[0].args
                    return json.dumps(args, indent=2) if isinstance(args, dict) else str(args)
                text_parts = [p for p in msg.parts if isinstance(p, TextPart)]
                if text_parts:
                    content = text_parts[0].content
                    if "```" in content:
                        lines = content.splitlines()
                        cleaned = [l for l in lines if not l.strip().startswith("```")]
                        content = "\n".join(cleaned)
                    return content.strip()
                    
        # Fallback to exception string if message history is empty or uninformative
        if exception:
            return str(exception)
        return ""

    def _log_extraction_error(self, phase: str, source_doc: str, start_idx: int, end_idx: int, error: Exception, malformed: str):
        """Helper to log extraction failures and validation errors systematically."""
        error_record = {
            "phase": phase,
            "source_document": source_doc,
            "start_word": start_idx,
            "end_word": end_idx,
            "error_message": str(error),
            "malformed_output": malformed
        }
        with open(self.log_dir / "stage_01_triple_extraction_errors.json", "a") as f:
            f.write(json.dumps(error_record) + "\n")

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

    async def _discover_themes_async(self, text: str, source_doc: str):
        """Async implementation of theme discovery using asyncio.gather and Semaphore."""
        theme_chunks = self.chunk_text(text, self.theme_chunk_size)
        sem = asyncio.Semaphore(self.max_async)
        
        async def process_chunk(chunk, start_idx, end_idx):
            async with sem:
                chunk = self._ensure_fit(chunk, prompts.THEME_DISCOVERY_USER_PROMPT, prompts.THEME_DISCOVERY_SYSTEM_PROMPT)
                user_prompt = prompts.THEME_DISCOVERY_USER_PROMPT.format(text_content=chunk)
                try:
                    result = await theme_agent.run(user_prompt)
                    return result.output.themes
                except Exception as e:
                    error_record = {
                        "source_document": source_doc,
                        "start_word": start_idx,
                        "end_word": end_idx,
                        "error": str(e)
                    }
                    with open(self.log_dir / "stage_01_theme_discovery_errors.json", "a") as f:
                        f.write(json.dumps(error_record) + "\n")
                    return []

        tasks = [process_chunk(chunk, start, end) for chunk, start, end in theme_chunks]
        results = await asyncio.gather(*tasks)
        
        doc_themes = []
        for theme_list in results:
            for theme in theme_list:
                doc_themes.append(theme)
                self.context.all_discovered_themes.append(theme)
                
        # Save individual themes file for this document
        safe_name = self.sanitize_filename(source_doc)
        with open(self.themes_dir / f"{safe_name}_themes.json", "w") as f:
            json.dump([t.model_dump() for t in doc_themes], f, indent=2)

    def discover_themes(self, text: str, source_doc: str):
        """
        Phase 1: Extracts themes from text chunks and aggregates them globally.
        Persists them per-document to themes directory.
        """
        if self.use_async:
            asyncio.run(self._discover_themes_async(text, source_doc))
            return

        theme_chunks = self.chunk_text(text, self.theme_chunk_size)
        doc_themes = []
        
        for chunk, start_idx, end_idx in theme_chunks:
            chunk = self._ensure_fit(chunk, prompts.THEME_DISCOVERY_USER_PROMPT, prompts.THEME_DISCOVERY_SYSTEM_PROMPT)
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
                doc_themes.append(theme)
                self.context.all_discovered_themes.append(theme)
                
        # Save individual themes file for this document
        safe_name = self.sanitize_filename(source_doc)
        with open(self.themes_dir / f"{safe_name}_themes.json", "w") as f:
            json.dump([t.model_dump() for t in doc_themes], f, indent=2)

    def aggregate_themes(self) -> List[schemas.Theme]:
        """Loads and consolidates all individual themes from the themes directory."""
        all_themes = []
        for fpath in sorted(self.themes_dir.glob("*_themes.json")):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    themes_data = json.load(f)
                    all_themes.extend(themes_data)
            except Exception as e:
                print(f"Warning: Failed to load theme file {fpath.name}: {e}")
                
        # Update pipeline context
        self.context.all_discovered_themes = [schemas.Theme(**t) for t in all_themes]
        
        # Write unified master themes list
        with open(self.out_dir / "all_themes.json", "w") as f:
            json.dump(all_themes, f, indent=2)
            
        return self.context.all_discovered_themes

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

    async def _extract_triples_async(self, text: str, source_doc: str):
        """Async implementation of triple extraction using asyncio.gather and Semaphore."""
        self.context.processed_documents.add(source_doc)
        triple_chunks = self.chunk_text(text, self.triple_chunk_size)
        sem = asyncio.Semaphore(self.max_async)
        
        async def process_chunk(chunk, start_idx, end_idx):
            async with sem:
                # Package state context for the agent
                deps = TripleContext(
                    master_themes=self.context.master_themes
                )
                chunk = self._ensure_fit(chunk, prompts.TRIPLE_EXTRACTION_USER_PROMPT, prompts.TRIPLE_EXTRACTION_SYSTEM_PROMPT)
                user_prompt = prompts.TRIPLE_EXTRACTION_USER_PROMPT.format(
                    text_content=chunk
                )
                from pydantic_ai import capture_run_messages
                
                with capture_run_messages() as messages:
                    try:
                        result = await triple_agent.run(user_prompt, deps=deps)
                        print(f"      -> Initial extraction successful for {source_doc} [{start_idx}-{end_idx}] (extracted {len(result.output.triples)} triples)")
                        return result.output.triples
                    except Exception as e:
                        malformed_text = self._extract_malformed_text(messages, e)
                        self._log_extraction_error("initial_triple_extraction_failed", source_doc, start_idx, end_idx, e, malformed_text)

                        from pydantic_ai.messages import ModelResponse
                        from pydantic_ai.exceptions import UnexpectedModelBehavior
                        from pydantic import ValidationError

                        has_response = any(isinstance(msg, ModelResponse) for msg in messages)
                        is_traceback = any(term in malformed_text.lower() for term in ["validation error", "field required", "input_value="])
                        is_model_error = isinstance(e, (UnexpectedModelBehavior, ValidationError, ValueError))

                        if not is_model_error and (not has_response or not malformed_text or is_traceback):
                            print(f"      -> Extraction failed due to a connection/system/traceback error. Raising immediately.")
                            raise e

                        if is_model_error and (not has_response or not malformed_text or is_traceback):
                            print(f"      -> Extraction failed due to a validation/model behavior error. Logging and skipping this chunk.")
                            return []

                        print(f"      -> Initial extraction failed after standard retry. Retrying with custom JSON reformatter (Model: {triple_model})...")

                        themes_str = self.context.master_themes.model_dump_json() if self.context.master_themes else "None"
                        reformat_prompt = (
                            f"Discovered Themes Context:\n{themes_str}\n\n"
                            f"Malformed input payload that failed schema validation:\n{malformed_text}\n\n"
                            f"Please reformat the text to strictly match the schemas.TripleExtractionResult schema, "
                            f"aligning any theme_association values to the themes provided above."
                        )
                        try:
                            ref_result = await triple_reformat_agent.run(reformat_prompt)
                            print(f"      -> Custom reformatting successful for {source_doc} [{start_idx}-{end_idx}] (recovered {len(ref_result.output.triples)} triples)!")
                            return ref_result.output.triples
                        except Exception as reformat_err:
                            self._log_extraction_error("triple_reformat_failed", source_doc, start_idx, end_idx, reformat_err, malformed_text)
                            print(f"      -> Custom reformatting failed for {source_doc} [{start_idx}-{end_idx}]: {reformat_err}")
                            return []

        tasks = [process_chunk(chunk, start, end) for chunk, start, end in triple_chunks]
        results = await asyncio.gather(*tasks)
        
        doc_triples = []
        for triples in results:
            for t in triples:
                t_dict = t.model_dump()
                t_dict["source_document"] = source_doc
                doc_triples.append(t_dict)
                self.context.raw_triples.append(t_dict)
                
        # Save individual triples file for this document
        safe_name = self.sanitize_filename(source_doc)
        with open(self.triples_dir / f"{safe_name}_triplets.json", "w") as f:
            json.dump(doc_triples, f, indent=2)

    def extract_triples(self, text: str, source_doc: str):
        """
        Phase 3: Extracts SVO triplets using global Master Themes and coreference context.
        Persists them per-document to triples directory.
        """
        self.context.processed_documents.add(source_doc)
        if self.use_async:
            asyncio.run(self._extract_triples_async(text, source_doc))
            return

        triple_chunks = self.chunk_text(text, self.triple_chunk_size)
        doc_triples = []
        
        for chunk, start_idx, end_idx in triple_chunks:
            # Package state context for the agent
            deps = TripleContext(
                master_themes=self.context.master_themes
            )
            
            chunk = self._ensure_fit(chunk, prompts.TRIPLE_EXTRACTION_USER_PROMPT, prompts.TRIPLE_EXTRACTION_SYSTEM_PROMPT)
            
            user_prompt = prompts.TRIPLE_EXTRACTION_USER_PROMPT.format(
                text_content=chunk
            )
            
            from pydantic_ai import capture_run_messages
            
            triples_list = []
            with capture_run_messages() as messages:
                try:
                    # Execute extraction
                    triple_result = triple_agent.run_sync(user_prompt, deps=deps)
                    triples_list = triple_result.output.triples
                    print(f"-> Initial extraction successful for {source_doc} [{start_idx}-{end_idx}] (extracted {len(triples_list)} triples)")
                except Exception as e:
                    malformed_text = self._extract_malformed_text(messages, e)
                    self._log_extraction_error("initial_triple_extraction_failed", source_doc, start_idx, end_idx, e, malformed_text)

                    from pydantic_ai.messages import ModelResponse
                    from pydantic_ai.exceptions import UnexpectedModelBehavior
                    from pydantic import ValidationError

                    has_response = any(isinstance(msg, ModelResponse) for msg in messages)
                    is_traceback = any(term in malformed_text.lower() for term in ["validation error", "field required", "input_value="])
                    is_model_error = isinstance(e, (UnexpectedModelBehavior, ValidationError, ValueError))

                    if not is_model_error and (not has_response or not malformed_text or is_traceback):
                        print(f"-> Extraction failed due to a connection/system/traceback error. Raising immediately.")
                        raise e

                    if is_model_error and (not has_response or not malformed_text or is_traceback):
                        print(f"-> Extraction failed due to a validation/model behavior error. Logging and skipping this chunk.")
                        continue

                    print(f"-> Initial extraction failed after standard retry. Retrying with custom JSON reformatter (Model: {triple_model})...")

                    themes_str = self.context.master_themes.model_dump_json() if self.context.master_themes else "None"
                    reformat_prompt = (
                        f"Discovered Themes Context:\n{themes_str}\n\n"
                        f"Malformed input payload that failed schema validation:\n{malformed_text}\n\n"
                        f"Please reformat the text to strictly match the schemas.TripleExtractionResult schema, "
                        f"aligning any theme_association values to the themes provided above."
                    )
                    try:
                        ref_result = triple_reformat_agent.run_sync(reformat_prompt)
                        print(f"-> Custom reformatting successful for {source_doc} [{start_idx}-{end_idx}] (recovered {len(ref_result.output.triples)} triples)!")
                        triples_list = ref_result.output.triples
                    except Exception as reformat_err:
                        self._log_extraction_error("triple_reformat_failed", source_doc, start_idx, end_idx, reformat_err, malformed_text)
                        print(f"-> Custom reformatting failed for {source_doc} [{start_idx}-{end_idx}]: {reformat_err}")
            
            # Update state with new entities and save triplets
            for t in triples_list:
                t_dict = t.model_dump()
                t_dict["source_document"] = source_doc # Guarantee source mapping natively
                doc_triples.append(t_dict)
                self.context.raw_triples.append(t_dict)
                
        # Save individual triples file for this document
        safe_name = self.sanitize_filename(source_doc)
        with open(self.triples_dir / f"{safe_name}_triplets.json", "w") as f:
            json.dump(doc_triples, f, indent=2)

    def aggregate_triplets(self) -> List[dict]:
        """Loads and consolidates all individual triplets from the triples directory."""
        all_triples = []
        processed_docs = set()
        
        for fpath in sorted(self.triples_dir.glob("*_triplets.json")):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    triples_data = json.load(f)
                    all_triples.extend(triples_data)
                    for t in triples_data:
                        doc = t.get("source_document")
                        if doc:
                            processed_docs.add(doc)
            except Exception as e:
                print(f"Warning: Failed to load triplet file {fpath.name}: {e}")
                
        # Update pipeline context
        self.context.raw_triples = all_triples
        self.context.processed_documents = processed_docs
        
        # Save aggregated master JSON
        with open(self.out_dir / "original_triplets.json", "w") as f:
            json.dump(all_triples, f, indent=2)
            
        # Count per document and save CSV
        counts = {doc: 0 for doc in processed_docs}
        for t in all_triples:
            doc = t.get("source_document", "unknown")
            counts[doc] = counts.get(doc, 0) + 1
            
        csv_path = self.out_dir / "triplet_counts.csv"
        try:
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("document_name,triplet_count\n")
                for doc, count in sorted(counts.items()):
                    f.write(f"{doc},{count}\n")
        except Exception as e:
            print(f"Warning: Failed to save triplet counts CSV: {e}")
            
        return all_triples
