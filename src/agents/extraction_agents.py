"""
SemanticPrism Stage 1: Extraction Agents
This module centralizes the initialization and configuration of Pydantic AI agents for the extraction phase.
"""

from typing import Set, Optional
import os
from pydantic_ai import Agent, RunContext, ModelSettings
from dataclasses import dataclass

from src.extraction import schemas
from src.extraction import prompts
from src.config import settings

# Determine LLM configuration for Theme and Triplet extraction separately with fallbacks
def resolve_llm_config(sub_key: str) -> tuple[str, ModelSettings]:
    stage_cfg = settings.get('extraction', {})
    llm_config = stage_cfg.get(sub_key)
    
    # Fallback to general 'llm' under extraction
    if not llm_config:
        llm_config = stage_cfg.get('llm')
    # Fallback to global 'llm'
    if not llm_config:
        llm_config = settings.get('llm', {})
        
    provider = llm_config.get('provider')
    model_name = llm_config.get('model_name')
    base_url = llm_config.get('base_url')
    
    # Configure model backend
    if provider == 'ollama':
        if base_url:
            os.environ['OLLAMA_BASE_URL'] = base_url
        pydantic_model = f"ollama:{model_name}"
    else:
        api_key = llm_config.get('api_key', '')
        if api_key:
            if provider == 'google':
                os.environ['GOOGLE_API_KEY'] = api_key
        pydantic_model = f"{provider}:{model_name}"
        
    extraction_cap = stage_cfg.get('context_window_cap', 8192)
    model_settings = ModelSettings(
        max_tokens=extraction_cap,
        extra_body={"options": {"num_ctx": extraction_cap}} if provider == 'ollama' else {}
    )
    return pydantic_model, model_settings

# Resolve separate configurations
theme_model, theme_settings = resolve_llm_config('llm_theme')
triple_model, triple_settings = resolve_llm_config('llm_triple')

# Agent 1: Extracts localized themes from individual text chunks.
theme_agent = Agent(
    model=theme_model,
    output_type=schemas.ThemeDiscoveryResult,
    system_prompt=prompts.THEME_DISCOVERY_SYSTEM_PROMPT,
    model_settings=theme_settings,
    retries=3
)

# Agent 2: Synthesizes all localized themes into a global master list.
master_theme_agent = Agent(
    model=theme_model,
    output_type=schemas.MasterThemeSynthesisResult,
    system_prompt=prompts.MASTER_THEME_SYSTEM_PROMPT,
    model_settings=theme_settings,
    retries=3
)

@dataclass
class TripleContext:
    """Dependency object passed to the Triple Extraction Agent to provide global context."""
    master_themes: Optional[schemas.MasterThemeSynthesisResult]

# Agent 3: Extracts Subject-Predicate-Object triplets and assigns them to the master themes.
triple_agent = Agent(
    model=triple_model,
    deps_type=TripleContext,
    output_type=schemas.TripleExtractionResult,
    system_prompt=prompts.TRIPLE_EXTRACTION_SYSTEM_PROMPT,
    model_settings=triple_settings,
    retries=1
)

# Agent 3b: Reformats failed/malformed JSON outputs to fit the desired schema.
triple_reformat_agent = Agent(
    model=triple_model,
    output_type=schemas.TripleExtractionResult,
    system_prompt=prompts.TRIPLE_REFORMAT_SYSTEM_PROMPT,
    model_settings=triple_settings,
    retries=1
)

@triple_agent.system_prompt
def add_triple_context(ctx: RunContext[TripleContext]) -> str:
    """
    Dynamically injects context into the Triple Agent's system prompt before each run.
    Provides the master themes.
    """
    themes = ctx.deps.master_themes.model_dump_json() if ctx.deps.master_themes else "None"
    return f"\nDiscovered Themes Context: {themes}"
