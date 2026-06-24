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

# Determine Stage-Specific or Global LLM configuration
llm_config = settings.get('extraction', {}).get('llm')
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
        # Add other providers here if necessary in the future
    pydantic_model = f"{provider}:{model_name}"

# Setup stage-specific context limit and model settings
extraction_cap = settings.get('extraction', {}).get('context_window_cap', 8192)
model_settings = ModelSettings(
    max_tokens=extraction_cap,
    extra_body={"options": {"num_ctx": extraction_cap}} if provider == 'ollama' else {}
)

# Agent 1: Extracts localized themes from individual text chunks.
theme_agent = Agent(
    model=pydantic_model,
    output_type=schemas.ThemeDiscoveryResult,
    system_prompt=prompts.THEME_DISCOVERY_SYSTEM_PROMPT,
    model_settings=model_settings,
    retries=3
)

# Agent 2: Synthesizes all localized themes into a global master list.
master_theme_agent = Agent(
    model=pydantic_model,
    output_type=schemas.MasterThemeSynthesisResult,
    system_prompt=prompts.MASTER_THEME_SYSTEM_PROMPT,
    model_settings=model_settings,
    retries=3
)

@dataclass
class TripleContext:
    """Dependency object passed to the Triple Extraction Agent to provide global context."""
    master_themes: Optional[schemas.MasterThemeSynthesisResult]

# Agent 3: Extracts Subject-Predicate-Object triplets and assigns them to the master themes.
triple_agent = Agent(
    model=pydantic_model,
    deps_type=TripleContext,
    output_type=schemas.TripleExtractionResult,
    system_prompt=prompts.TRIPLE_EXTRACTION_SYSTEM_PROMPT,
    model_settings=model_settings,
    retries=1
)

# Agent 3b: Reformats failed/malformed JSON outputs to fit the desired schema.
triple_reformat_agent = Agent(
    model=pydantic_model,
    output_type=schemas.TripleExtractionResult,
    system_prompt=prompts.TRIPLE_REFORMAT_SYSTEM_PROMPT,
    model_settings=model_settings,
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
