"""
Stage 4: Ontology & Pydantic Schema Synthesis Package.
"""

from src.synthesis.schemas import GeneratedModule
from src.synthesis.prompts import (
    get_schema_synthesis_prompt,
    get_consolidation_system_prompt,
    get_schema_reformat_system_prompt,
    SCHEMA_SYNTHESIS_PROMPT,
    CONSOLIDATION_SYSTEM_PROMPT,
    SCHEMA_REFORMAT_SYSTEM_PROMPT,
)
from src.synthesis.sanitizers import (
    estimate_tokens,
    configure_sanitizer_defaults,
    clean_python_code,
    deduplicate_consolidated_classes,
    sanitize_module_name,
    validate_python_syntax,
)
from src.synthesis.pipeline import (
    TargetResolver,
    SynthesisPipeline,
    configure_synthesis_defaults,
)

__all__ = [
    "GeneratedModule",
    "get_schema_synthesis_prompt",
    "get_consolidation_system_prompt",
    "get_schema_reformat_system_prompt",
    "SCHEMA_SYNTHESIS_PROMPT",
    "CONSOLIDATION_SYSTEM_PROMPT",
    "SCHEMA_REFORMAT_SYSTEM_PROMPT",
    "estimate_tokens",
    "configure_sanitizer_defaults",
    "clean_python_code",
    "deduplicate_consolidated_classes",
    "sanitize_module_name",
    "validate_python_syntax",
    "TargetResolver",
    "SynthesisPipeline",
    "configure_synthesis_defaults",
]
