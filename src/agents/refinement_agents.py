"""
SemanticPrism Stage 2: Refinement Agents
This module centralizes the initialization and configuration of Pydantic AI agents for the refinement phase.
"""

import os
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel

from src.refinement import schemas
from src.refinement import prompts
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

# Agent 1a: Subject Normalization
subject_norm_agent = Agent(
    model=pydantic_model,
    deps_type=str,
    output_type=schemas.NormalizedStrings,
    system_prompt=prompts.SUBJECT_NORMALIZATION_SYSTEM_PROMPT,
    retries=3
)

@subject_norm_agent.system_prompt
def add_subj_context(ctx: RunContext[str]) -> str:
    return f"\nDomain Context: {ctx.deps}"

# Agent 1b: Predicate Normalization
predicate_norm_agent = Agent(
    model=pydantic_model,
    deps_type=str,
    output_type=schemas.NormalizedStrings,
    system_prompt=prompts.PREDICATE_NORMALIZATION_SYSTEM_PROMPT,
    retries=3
)

@predicate_norm_agent.system_prompt
def add_pred_context(ctx: RunContext[str]) -> str:
    return f"\nDomain Context: {ctx.deps}"

# Agent 1c: Object Normalization
object_norm_agent = Agent(
    model=pydantic_model,
    deps_type=str,
    output_type=schemas.NormalizedStrings,
    system_prompt=prompts.OBJECT_NORMALIZATION_SYSTEM_PROMPT,
    retries=3
)

@object_norm_agent.system_prompt
def add_obj_context(ctx: RunContext[str]) -> str:
    return f"\nDomain Context: {ctx.deps}"

# Agent 2: Taxonomic Lift
lift_agent = Agent(
    model=pydantic_model,
    deps_type=str,
    output_type=schemas.TaxonomicVerification,
    system_prompt=prompts.TAXONOMIC_LIFTING_SYSTEM_PROMPT,
    retries=3
)

@lift_agent.system_prompt
def add_lift_context(ctx: RunContext[str]) -> str:
    """Injects master domain context."""
    return f"\nDomain Context: {ctx.deps}"
