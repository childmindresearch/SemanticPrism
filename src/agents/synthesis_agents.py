"""
SemanticPrism Stage 4: Synthesis Agents
This module centralizes the initialization and configuration of Pydantic AI agents for the synthesis phase.
"""

from typing import List, Any
import os
import json
from dataclasses import dataclass
from contextvars import ContextVar
from pydantic_ai import Agent, RunContext, ModelSettings
from pydantic_ai.models import Model, infer_model
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart

from src.synthesis import schemas
from src.synthesis import prompts
from src.config import settings

# Determine Stage-Specific or Global LLM configuration
llm_config = settings.get('synthesis', {}).get('llm')
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

# Setup context-local history variable and logger wrapper
last_llm_responses: ContextVar[List[Any]] = ContextVar("last_llm_responses", default=[])

class LoggingModel(Model):
    def __init__(self, wrapped: Model):
        self.wrapped = wrapped

    @property
    def model_name(self) -> str:
        return self.wrapped.model_name

    @property
    def system(self) -> str:
        return self.wrapped.system

    @property
    def base_url(self) -> str | None:
        return self.wrapped.base_url

    async def request(self, messages: list[ModelMessage], model_settings: Any, model_request_parameters: Any) -> ModelResponse:
        response = await self.wrapped.request(messages, model_settings, model_request_parameters)
        
        extracted = []
        for part in response.parts:
            if isinstance(part, TextPart):
                extracted.append(part.content)
            elif isinstance(part, ToolCallPart):
                extracted.append(part.args)
            else:
                extracted.append(str(part))
                
        current_history = last_llm_responses.get()
        current_history.append(extracted)
        
        return response

    async def request_stream(self, messages: list[ModelMessage], model_settings: Any, model_request_parameters: Any, run_context: Any = None) -> Any:
        return self.wrapped.request_stream(messages, model_settings, model_request_parameters, run_context)

resolved_model = infer_model(pydantic_model)
pydantic_model = LoggingModel(resolved_model)

# Setup stage-specific context limit and model settings
synthesis_cap = settings.get('synthesis', {}).get('context_window_cap', 16384)
model_settings = ModelSettings(
    # Avoid setting max_tokens for Ollama to prevent pydantic-ai from sending max_completion_tokens, which Ollama rejects
    **({"max_tokens": 4096} if provider != 'ollama' else {}),
    extra_body={"options": {"num_ctx": synthesis_cap}} if provider == 'ollama' else {}
)

@dataclass
class OrphanContext:
    master_themes: List[str]

@dataclass
class SynthesisContext:
    global_enums: str

# Agent 1: Orphan Enum Agent
orphan_agent = Agent(
    model=pydantic_model,
    deps_type=OrphanContext,
    output_type=schemas.GeneratedModule,
    system_prompt=prompts.ORPHAN_SYNTHESIS_SYSTEM_PROMPT,
    model_settings=model_settings,
    retries=3
)

@orphan_agent.system_prompt
def add_orphan_context(ctx: RunContext[OrphanContext]) -> str:
    return f"\nMaster Themes Context: {json.dumps(ctx.deps.master_themes)}"

# Agent 2: Schema Synthesis Agent
schema_synthesis_agent = Agent(
    model=pydantic_model,
    deps_type=SynthesisContext,
    output_type=schemas.GeneratedModule,
    system_prompt=prompts.SCHEMA_SYNTHESIS_PROMPT,
    model_settings=model_settings,
    retries=1
)

@schema_synthesis_agent.system_prompt
def add_schema_context(ctx: RunContext[SynthesisContext]) -> str:
    return f"\nGlobal Enums Available:\n{ctx.deps.global_enums}"

leiden_schema_agent = schema_synthesis_agent
node2vec_schema_agent = schema_synthesis_agent

# Agent 3: Consolidation Agent
consolidation_agent = Agent(
    model=pydantic_model,
    output_type=schemas.GeneratedModule,
    system_prompt=prompts.CONSOLIDATION_SYSTEM_PROMPT,
    model_settings=model_settings,
    retries=3
)

# Agent 4: Comprehensive Ontology Agent
comprehensive_ontology_agent = Agent(
    model=pydantic_model,
    output_type=schemas.GeneratedModule,
    system_prompt=prompts.FINAL_ONTOLOGY_SYSTEM_PROMPT,
    model_settings=model_settings,
    retries=3
)

# Agent 5: Schema Reformat Agent
schema_reformat_agent = Agent(
    model=pydantic_model,
    output_type=schemas.GeneratedModule,
    system_prompt=prompts.SCHEMA_REFORMAT_SYSTEM_PROMPT,
    model_settings=model_settings,
    retries=1
)
