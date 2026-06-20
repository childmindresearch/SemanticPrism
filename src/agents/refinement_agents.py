"""
SemanticPrism Stage 2: Refinement Agents
This module centralizes the initialization and configuration of Pydantic AI agents for the refinement phase.
"""

import os
from pydantic_ai import Agent, RunContext, ModelSettings

from src.refinement import schemas
from src.refinement import prompts
from src.config import settings

# Determine Stage-Specific or Global LLM configuration
llm_config = settings.get('refinement', {}).get('llm')
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
refinement_cap = settings.get('refinement', {}).get('context_window_cap', 2048)
model_settings = ModelSettings(
    max_tokens=refinement_cap,
    extra_body={"options": {"num_ctx": refinement_cap}} if provider == 'ollama' else {}
)

# Agent 1a: Subject Normalization
subject_norm_agent = Agent(
    model=pydantic_model,
    deps_type=str,
    output_type=schemas.NormalizedStrings,
    system_prompt=prompts.SUBJECT_NORMALIZATION_SYSTEM_PROMPT,
    model_settings=model_settings,
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
    model_settings=model_settings,
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
    model_settings=model_settings,
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
    model_settings=model_settings,
    retries=3
)

@lift_agent.system_prompt
def add_lift_context(ctx: RunContext[str]) -> str:
    """Injects master domain context."""
    return f"\nDomain Context: {ctx.deps}"
