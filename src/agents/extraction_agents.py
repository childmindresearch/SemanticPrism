"""
SemanticPrism Stage 1: Extraction Agents
This module centralizes the initialization and configuration of Pydantic AI agents for the extraction phase.
"""

from typing import Set, Optional
import os
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from pydantic_ai.models.ollama import OllamaModel

from src.extraction import schemas
from src.extraction import prompts
from src.config import settings

# Determine AI Model Provider Details from global settings
provider = settings['llm']['provider']
model_name = settings['llm']['model_name']
base_url = settings['llm'].get('base_url')

# Configure model backend
if provider == 'ollama':
    from pydantic_ai.providers.ollama import OllamaProvider
    custom_provider = OllamaProvider(base_url=base_url)
    pydantic_model = OllamaModel(model_name, provider=custom_provider)
else:
    api_key = settings['llm'].get('api_key', '')
    if api_key:
        if provider == 'google':
            os.environ['GOOGLE_API_KEY'] = api_key
        # Add other providers here if necessary in the future
    pydantic_model = f"{provider}:{model_name}"

# Agent 1: Extracts localized themes from individual text chunks.
theme_agent = Agent(
    model=pydantic_model,
    output_type=schemas.ThemeDiscoveryResult,
    system_prompt=prompts.THEME_DISCOVERY_SYSTEM_PROMPT,
    retries=3
)

# Agent 2: Synthesizes all localized themes into a global master list.
master_theme_agent = Agent(
    model=pydantic_model,
    output_type=schemas.MasterThemeSynthesisResult,
    system_prompt=prompts.MASTER_THEME_SYSTEM_PROMPT,
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
    retries=3
)

@triple_agent.system_prompt
def add_triple_context(ctx: RunContext[TripleContext]) -> str:
    """
    Dynamically injects context into the Triple Agent's system prompt before each run.
    Provides the master themes.
    """
    themes = ctx.deps.master_themes.model_dump_json() if ctx.deps.master_themes else "None"
    return f"\nDiscovered Themes Context: {themes}"
