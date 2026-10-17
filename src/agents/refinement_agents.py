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

model_name = llm_config.get('model_name')
base_url = llm_config.get('base_url')

# Configure local Ollama backend
if base_url:
    os.environ['OLLAMA_BASE_URL'] = base_url
pydantic_model = f"ollama:{model_name}"

# Setup stage-specific context limit and model settings
refinement_cap = settings.get('refinement', {}).get('context_window_cap', 2048)
model_settings = ModelSettings(
    max_tokens=refinement_cap,
    temperature=llm_config.get('temperature', 0.0),
    extra_body={"options": {"num_ctx": refinement_cap}}
)

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
