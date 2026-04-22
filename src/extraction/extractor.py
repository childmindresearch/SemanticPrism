"""
SemanticPrism: Extraction Pipeline
Handles strictly LLM-based mappings of strings to Pydantic objects natively.
"""
import yaml
import asyncio
import json
from typing import List, Optional
from pydantic import BaseModel

from src.extraction.schemas import (
    Theme, ThemeDiscoveryResult, MasterThemeSynthesisResult, 
    RawTriple, TripleExtractionResult, NormalizedStrings
)
from src.core.chunking import chunk_text
from src.core.logger import get_logger
from src.helpers.context_manager import ContextManager
import src.extraction.prompts as prompts
from src.llm.llm_client import SemanticLLMClient

logger = get_logger("ExtractionPipeline")

class ExtractionPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initializes the ExtractionPipeline by loading configuration from a YAML file, setting up the LLM client, and configuring extraction parameters.
        """
        # Explicit configuration load without hardcoding
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logger.critical(f"Failed to parse explicitly required configuration natively: {e}")
            raise e
            
        self.use_async = self.config.get('pipeline', {}).get('use_async', False)
        self.max_concurrent = self.config.get('pipeline', {}).get('max_concurrent_llm_calls', 3)
        self.fallback_domain = self.config.get('extraction', {}).get('domain', 'General')
        
        # Centralized LLM Backend Mapping 
        self.llm = SemanticLLMClient(config_path)
        self.model_name = self.llm.model_name
        
        # Hardware context bounds explicitly governed independently if overridden
        self.theme_max_words = self.config.get('extraction', {}).get('theme_chunk_max_words', 8000)
        self.triple_max_words = self.config.get('extraction', {}).get('triple_chunk_max_words', 2500)
        
        logger.info(f"Initialized ExtractionPipeline (Model: {self.model_name}, Use Async: {self.use_async})")

    async def discover_themes(self, text: str) -> List[ThemeDiscoveryResult]:
        """
        Splits input text into chunks and extracts themes using the LLM. Supports both synchronous and asynchronous execution.
        """
        logger.info("Phase 1: Executing Theme Discovery...")
        chunks = chunk_text(text, self.theme_max_words)
        
        if self.use_async:
            sem = asyncio.Semaphore(self.max_concurrent)
            async def _process_chunk(chunk: str):
                async with sem:
                    user_msg = prompts.THEME_DISCOVERY_USER_PROMPT.format(text_content=chunk)
                    return await self.llm.safe_api_call_async(
                        prompts.THEME_DISCOVERY_SYSTEM_PROMPT, 
                        user_msg, 
                        ThemeDiscoveryResult
                    )
                    
            tasks = [_process_chunk(c) for c in chunks]
            results = await asyncio.gather(*tasks)
            return [r for r in results if r is not None]
        else:
            results = []
            for c in chunks:
                user_msg = prompts.THEME_DISCOVERY_USER_PROMPT.format(text_content=c)
                res = self.llm.safe_api_call_sync(
                    prompts.THEME_DISCOVERY_SYSTEM_PROMPT, 
                    user_msg, 
                    ThemeDiscoveryResult
                )
                if res:
                    results.append(res)
            return results

    def weight_themes(self, themes_list: List[ThemeDiscoveryResult]) -> str:
        """
        Consolidates and counts the frequency of extracted themes from multiple chunks, returning a formatted string ordered by frequency.
        """
        theme_counts = {}
        theme_mappings = {}
        
        for tr in themes_list:
            for t in tr.themes:
                norm_title = t.title.strip().lower()
                if norm_title not in theme_counts:
                    theme_counts[norm_title] = 0
                    theme_mappings[norm_title] = t
                theme_counts[norm_title] += 1
                
        formatted_blocks = []
        for norm_title, count in sorted(theme_counts.items(), key=lambda item: item[1], reverse=True):
            t_obj = theme_mappings[norm_title]
            formatted_blocks.append(
                f"Title: {t_obj.title} (Frequency: {count})\n"
                f"Description: {t_obj.description}\n"
                f"Reasoning: {t_obj.reasoning}"
            )
            
        return "\n\n".join(formatted_blocks)

    async def consolidate_themes(self, formatted_themes: str) -> Optional[MasterThemeSynthesisResult]:
        """
        Takes the formatted list of discovered themes and uses the LLM to synthesize an overarching master domain and combined themes.
        """
        logger.info("Phase 1.5: Consolidating overarching Master Domain.")
        if not formatted_themes:
            logger.warning("No logic themes securely discovered to consolidate.")
            return None
            
        user_msg = prompts.MASTER_THEME_USER_PROMPT.format(all_extracted_themes=formatted_themes)
        
        if self.use_async:
            return await self.llm.safe_api_call_async(
                prompts.MASTER_THEME_SYSTEM_PROMPT,
                user_msg,
                MasterThemeSynthesisResult
            )
        else:
            return self.llm.safe_api_call_sync(
                prompts.MASTER_THEME_SYSTEM_PROMPT,
                user_msg,
                MasterThemeSynthesisResult
            )

    async def extract_triples(self, text: str, master_theme_context: Optional[MasterThemeSynthesisResult]) -> List[RawTriple]:
        """
        Splits input text into chunks and uses the LLM to extract logical subject-predicate-object triples, utilizing the consolidated themes and a running entity registry for context.
        """
        logger.info("Phase 2: Executing Logical Triple Extraction Explicitly...")
        chunks = chunk_text(text, self.triple_max_words)
        
        themes_context = ""
        if master_theme_context:
            themes_context = f"Discovered Themes Context: {master_theme_context.model_dump_json()}\n\n"
            
        all_triples = []
        entity_registry = set()
        
        for idx, chunk in enumerate(chunks):
            logger.info(f"Extracting triples from logic chunk {idx + 1}/{len(chunks)}...")
            
            previous_entities_context = ""
            if entity_registry:
                recent_entities = list(entity_registry)[-100:]
                previous_entities_context = f"Previously Discovered Entities (for Coreference Resolution): {json.dumps(recent_entities)}\n\n"
                
            user_msg = prompts.TRIPLE_EXTRACTION_USER_PROMPT.format(
                themes_context=themes_context,
                previous_entities_context=previous_entities_context,
                text_content=chunk
            )
            
            if self.use_async:
                res = await self.llm.safe_api_call_async(
                    prompts.TRIPLE_EXTRACTION_SYSTEM_PROMPT,
                    user_msg,
                    TripleExtractionResult,
                    num_ctx=8192
                )
            else:
                res = self.llm.safe_api_call_sync(
                    prompts.TRIPLE_EXTRACTION_SYSTEM_PROMPT,
                    user_msg,
                    TripleExtractionResult,
                    num_ctx=8192
                )
                
            if res is not None and res.triples:
                all_triples.extend(res.triples)
                for t in res.triples:
                    if t.subject: entity_registry.add(t.subject)
                    if t.object: entity_registry.add(t.object)
                    
        return all_triples

    async def normalize_triples_strings(self, raw_tokens: List[str], domain_context: str = "") -> dict[str, str]:
        """
        Uses the LLM to perform syntactic normalization on a list of extracted string tokens to simplify graph topology.
        Returns a dictionary mapping the original string to its normalized form.
        """
        logger.info("Phase 2.5: Executing Strict Lexical Normalization")
        
        context_string = ""
        if domain_context:
            context_string = f"Domain Context: {domain_context}\n\n"
        elif self.fallback_domain:
            context_string = f"Domain Context: {self.fallback_domain}\n\n"
            
        user_msg = prompts.LLM_PREPROCESSING_USER_PROMPT.format(
            domain_context=context_string,
            raw_tokens_json=json.dumps(raw_tokens)
        )
        

        if self.use_async:
            sem = asyncio.Semaphore(self.max_concurrent)
            async with sem:
                res = await self.llm.safe_api_call_async(
                    prompts.LLM_PREPROCESSING_SYSTEM_PROMPT,
                    user_msg,
                    NormalizedStrings,
                    num_ctx=4096
                )
        else:
            res = self.llm.safe_api_call_sync(
                prompts.LLM_PREPROCESSING_SYSTEM_PROMPT,
                user_msg,
                NormalizedStrings,
                num_ctx=4096
            )
            
        if res and hasattr(res, "tokens"):
            return {t.original: t.normalized for t in res.tokens}
        return {k: k for k in raw_tokens}
