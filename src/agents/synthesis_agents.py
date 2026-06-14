"""
SemanticPrism Stage 4: Synthesis Agents
This module centralizes the initialization and configuration of Pydantic AI agents for the synthesis phase.
"""

from typing import List
import os
import json
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel

from src.synthesis import schemas
from src.synthesis import prompts
from src.config import settings

# Determine AI Model Provider Details
provider = settings['llm']['provider']
model_name = settings['llm']['model_name']
base_url = settings['llm'].get('base_url')

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

@dataclass
class OrphanContext:
    master_themes: List[str]

@dataclass
class SynthesisContext:
    global_enums: str
    theme_inheritance: str

# Agent 1: Orphan Enum Agent
orphan_agent = Agent(
    model=pydantic_model,
    deps_type=OrphanContext,
    output_type=schemas.GeneratedModule,
    system_prompt=prompts.ORPHAN_SYNTHESIS_SYSTEM_PROMPT,
    retries=3
)

@orphan_agent.system_prompt
def add_orphan_context(ctx: RunContext[OrphanContext]) -> str:
    return f"\nMaster Themes Context: {json.dumps(ctx.deps.master_themes)}"

# Agent 2a: Leiden Ontology Schema Agent
leiden_schema_agent = Agent(
    model=pydantic_model,
    deps_type=SynthesisContext,
    output_type=schemas.GeneratedModule,
    system_prompt=prompts.LEIDEN_SCHEMA_SYNTHESIS_PROMPT,
    retries=3
)

@leiden_schema_agent.system_prompt
def add_leiden_schema_context(ctx: RunContext[SynthesisContext]) -> str:
    return f"\nTheme Inheritance Mapping:\n{ctx.deps.theme_inheritance}\n\nGlobal Enums Available:\n{ctx.deps.global_enums}"

# Agent 2b: Node2Vec Ontology Schema Agent
node2vec_schema_agent = Agent(
    model=pydantic_model,
    deps_type=SynthesisContext,
    output_type=schemas.GeneratedModule,
    system_prompt=prompts.NODE2VEC_SCHEMA_SYNTHESIS_PROMPT,
    retries=3
)

@node2vec_schema_agent.system_prompt
def add_node2vec_schema_context(ctx: RunContext[SynthesisContext]) -> str:
    return f"\nTheme Inheritance Mapping:\n{ctx.deps.theme_inheritance}\n\nGlobal Enums Available:\n{ctx.deps.global_enums}"

# Agent 3: Consolidation Agent
consolidation_agent = Agent(
    model=pydantic_model,
    output_type=schemas.GeneratedModule,
    system_prompt=prompts.CONSOLIDATION_SYSTEM_PROMPT,
    retries=3
)

# Agent 4: Comprehensive Ontology Agent
comprehensive_ontology_agent = Agent(
    model=pydantic_model,
    output_type=schemas.GeneratedModule,
    system_prompt=prompts.FINAL_ONTOLOGY_SYSTEM_PROMPT,
    retries=3
)
