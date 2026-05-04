"""
SemanticPrism: Pydantic AI Default Workflow

This module introduces a minimalist, focused approach to knowledge extraction 
by leveraging the `pydantic-ai` framework. It replaces the previous `instructor` 
wrapper with native, type-safe agents built by the Pydantic team.

This workflow demonstrates how to use `RunContext` for dependency injection 
and strictly validated output schemas for deterministic LLM interactions.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import List, Optional


# Import native Pydantic AI components
from pydantic_ai import Agent, RunContext, NativeOutput
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

# Import our existing schemas and prompts
from src.extraction.schemas import (
    ThemeDiscoveryResult,
    MasterThemeSynthesisResult,
    TripleExtractionResult,
    RawTriple
)
import src.extraction.prompts as prompts

# ============================================================================
# Dependency Injection Definition
# ============================================================================
@dataclass
class ExtractionDeps:
    """
    Dependencies injected into the Pydantic AI agents.
    This replaces global state and passes the current execution context 
    safely down to the agents and their dynamic instructions.
    """
    chunk_text: str
    domain_context: str = "General"
    master_theme_context: Optional[MasterThemeSynthesisResult] = None
    previous_entities: Optional[List[str]] = None


# ============================================================================
# Agent Definitions
# ============================================================================
# In Pydantic AI, we define explicit Agents tailored for specific tasks.
# We configure the model (here we point it to the Ollama endpoint defined in config)
# and strictly define the deps_type and output_type for full type safety.

# For this default workflow, we will use an Ollama-compatible interface 
# pointing to a local Ollama instance (as defined in config.yaml).
model = OllamaModel(
    'mistral-nemo:12b-instruct-2407-q4_K_M',
    provider=OllamaProvider(base_url='http://localhost:11434/v1')
)

# 1. Theme Discovery Agent
# This agent handles Phase 1: extracting overarching themes from a text chunk.
theme_agent = Agent(
    model,
    deps_type=ExtractionDeps,
    output_type=NativeOutput(ThemeDiscoveryResult),
    system_prompt=prompts.THEME_DISCOVERY_SYSTEM_PROMPT,
)

@theme_agent.system_prompt
def inject_theme_user_prompt(ctx: RunContext[ExtractionDeps]) -> str:
    """
    Dynamically injects the user prompt based on the chunk text in the dependencies.
    Pydantic AI allows dynamic system/user instructions based on the runtime context.
    """
    return prompts.THEME_DISCOVERY_USER_PROMPT.format(text_content=ctx.deps.chunk_text)

# 2. Triple Extraction Agent
# This agent handles Phase 2: extracting strictly formatted (Subject, Predicate, Object) triples.
triple_agent = Agent(
    model,
    deps_type=ExtractionDeps,
    output_type=NativeOutput(TripleExtractionResult),
    system_prompt=prompts.TRIPLE_EXTRACTION_SYSTEM_PROMPT,
)

@triple_agent.system_prompt
def inject_triple_user_prompt(ctx: RunContext[ExtractionDeps]) -> str:
    """
    Dynamically injects themes and previous entities into the triple extraction prompt.
    """
    themes_context = ""
    if ctx.deps.master_theme_context:
        # Pydantic models need to be serialized carefully, model_dump_json() ensures valid format
        themes_context = f"Discovered Themes Context: {ctx.deps.master_theme_context.model_dump_json()}\n\n"
        
    previous_entities_context = ""
    if ctx.deps.previous_entities:
        previous_entities_context = f"Previously Discovered Entities (for Coreference Resolution): {json.dumps(ctx.deps.previous_entities)}\n\n"
        
    return prompts.TRIPLE_EXTRACTION_USER_PROMPT.format(
        themes_context=themes_context,
        previous_entities_context=previous_entities_context,
        text_content=ctx.deps.chunk_text
    )


# ============================================================================
# Workflow Orchestration
# ============================================================================
async def run_pydantic_ai_workflow(text: str) -> List[RawTriple]:
    """
    Executes the minimalist Pydantic AI workflow.
    
    1. Extracts themes from the provided text.
    2. Extracts triples using the discovered themes as context.
    """
    print("==================================================")
    print("Starting Pydantic AI Minimalist Workflow")
    print("==================================================")

    # ---------------------------------------------------------
    # Phase 1: Theme Discovery
    # ---------------------------------------------------------
    print("\n[Phase 1] Discovering Themes...")
    theme_deps = ExtractionDeps(chunk_text=text)
    
    # Run the agent asynchronously. It strictly returns a ThemeDiscoveryResult.
    # The string passed to `run` is the user message.
    theme_result = await theme_agent.run(
        "Please extract the themes.",
        deps=theme_deps
    )
    
    # We can access the strictly validated output directly via .output
    discovered_themes = theme_result.output.themes
    print(f"Discovered {len(discovered_themes)} themes.")
    for theme in discovered_themes:
        print(f"  - {theme.title}: {theme.description}")

    # Mocking Master Synthesis for the sake of this default focused workflow
    master_context = MasterThemeSynthesisResult(
        master_domain="Semantic Default Domain",
        master_themes=[t.title for t in discovered_themes]
    )

    # ---------------------------------------------------------
    # Phase 2: Triple Extraction
    # ---------------------------------------------------------
    print("\n[Phase 2] Extracting Triples...")
    triple_deps = ExtractionDeps(
        chunk_text=text,
        master_theme_context=master_context,
        previous_entities=[]
    )
    
    # Run the agent asynchronously. It strictly returns a TripleExtractionResult.
    # Pydantic AI handles the JSON schema constraints and self-correction reflection loops natively.
    triple_result = await triple_agent.run(
        "Please extract the triples based on the text and context.",
        deps=triple_deps
    )
    
    extracted_triples = triple_result.output.triples
    print(f"Extracted {len(extracted_triples)} triples.")
    for triple in extracted_triples:
        print(f"  [{triple.subject}] --({triple.predicate})--> [{triple.object}] (Theme: {triple.theme_association})")

    print("\n==================================================")
    print("Pydantic AI Workflow Completed Successfully")
    print("==================================================")
    
    return extracted_triples

if __name__ == "__main__":
    # A simple test to verify the script runs standalone
    sample_text = (
        "SemanticPrism is an ontology extractor. It focuses on accuracy and philosophy. "
        "Plato taught Aristotle. Aristotle wrote Metaphysics."
    )
    asyncio.run(run_pydantic_ai_workflow(sample_text))
