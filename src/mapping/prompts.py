"""
Stage 5 Phase 3a & 3b: Schema-Driven Triage & Extraction Prompt Builders.
"""

from __future__ import annotations

from enum import Enum
import inspect
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union, get_args, get_origin

from pydantic import BaseModel

from src.mapping.schemas import SchemaMeta, extract_referenced_models

EXTRACTION_RULES = """
CRITICAL EXTRACTION RULES:
1. **Verbatim Fidelity & Granular Specificity**: Extract mentions, phrases, and values EXACTLY as they appear in the source text. Do NOT paraphrase, summarize, synthesize, or generalize into high-level categories.
2. **Strict Null Rule (Zero Hallucination)**: If a specific field is NOT explicitly mentioned or supported by the text, you MUST leave it as `null` (`None`) or an empty list `[]`. NEVER guess, infer, extrapolate, or fabricate default placeholder values.
3. **Context Grounding**: Every extracted attribute must be directly grounded in the provided document excerpt. If the excerpt contains no relevant information for the entire schema, return `null` (`None`) or empty lists `[]` for all fields.
""".strip()

TRIAGE_RULES = """
CRITICAL TRIAGE RULES:
1. **Evidence-Based Selection**: Select a schema ONLY if the text excerpt contains explicit factual information, entities, or measurements that belong to that schema's domain and fields.
2. **Conservative Filtering**: Do NOT select a schema based on vague tangential associations. If no schemas are relevant to the excerpt, return an empty list `[]`.
3. **Exact Class Names**: Return ONLY exact schema class names from the provided candidate list in `relevant_schemas`.
""".strip()


def _format_type_annotation(annotation: Any) -> str:
    """Formats a Python/Pydantic type annotation into a clean, readable string for prompts."""
    if annotation is None or annotation is type(None):
        return "None"

    origin = get_origin(annotation)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return f"Optional[{_format_type_annotation(args[0])}]"
        return "Union[" + ", ".join(_format_type_annotation(a) for a in args) + "]"

    if origin in (list, List, set, Set, tuple, Tuple):
        args = get_args(annotation)
        inner = _format_type_annotation(args[0]) if args else "Any"
        return f"List[{inner}]"

    if origin in (dict, Dict):
        args = get_args(annotation)
        if len(args) == 2:
            return f"Dict[{_format_type_annotation(args[0])}, {_format_type_annotation(args[1])}]"
        return "Dict[str, Any]"

    if inspect.isclass(annotation):
        if issubclass(annotation, Enum):
            choices = [repr(e.value) for e in annotation]
            return f"Enum({annotation.__name__}: {', '.join(choices)})"
        return annotation.__name__

    return str(annotation).replace("typing.", "")


def format_schema_field_instructions(
    schema_cls: Type[BaseModel],
    indent: int = 0,
    visited: Optional[Set[Type[BaseModel]]] = None,
) -> str:
    """
    Recursively inspects a Pydantic BaseModel and formats its fields, types, descriptions,
    Enum choices, and nested child BaseModel definitions into structured prompt instructions.
    """
    if visited is None:
        visited = set()

    visited.add(schema_cls)
    lines: List[str] = []
    prefix = "  " * indent

    for field_name, field_info in schema_cls.model_fields.items():
        type_str = _format_type_annotation(field_info.annotation)
        desc = field_info.description or "Extract exact mentions from text if present; otherwise null."
        lines.append(f"{prefix}- `{field_name}` ({type_str}): {desc}")

        referenced = extract_referenced_models(field_info.annotation)
        for child_cls in referenced:
            if child_cls not in visited:
                child_doc = (child_cls.__doc__ or "").strip().splitlines()
                child_summary = f" — {child_doc[0].strip()}" if child_doc else ""
                lines.append(f"{prefix}  * Nested `{child_cls.__name__}` structure{child_summary}:")
                lines.append(
                    format_schema_field_instructions(
                        child_cls,
                        indent=indent + 2,
                        visited=visited.copy(),
                    )
                )

    return "\n".join(lines)


def build_triage_prompt(candidate_schemas: List[SchemaMeta]) -> str:
    """
    Constructs the system prompt for Pass 3a (Fast Multi-Schema Relevance Triage),
    listing each candidate schema's class name, docstring summary, and key fields.
    """
    schema_descriptions: List[str] = []
    for meta in candidate_schemas:
        doc = (meta.schema_cls.__doc__ or "").strip()
        first_line = doc.splitlines()[0].strip() if doc else ""
        field_names = list(meta.schema_cls.model_fields.keys())
        fields_preview = ", ".join(field_names[:10])
        if len(field_names) > 10:
            fields_preview += ", ..."
        if first_line:
            desc_str = f"{first_line} (Fields: {fields_preview})"
        else:
            desc_str = f"Extracts fields: {fields_preview}"
        schema_descriptions.append(f"- **{meta.name}**: {desc_str}")

    schemas_block = "\n".join(schema_descriptions)

    return (
        "You are an expert semantic triage router.\n"
        "Review the document excerpt and identify which of the following data schemas "
        "contain explicit factual evidence in the text.\n\n"
        "### CANDIDATE SCHEMAS\n"
        f"{schemas_block}\n\n"
        f"{TRIAGE_RULES}"
    )


def build_extraction_prompt(schema_cls: Type[BaseModel]) -> str:
    """
    Constructs the schema-specific system prompt for Pass 3b (Deep Structured Extraction).
    """
    doc = (schema_cls.__doc__ or "").strip()
    doc_section = f"\nSchema Purpose: {doc}\n" if doc else "\n"
    fields_section = format_schema_field_instructions(schema_cls)

    return (
        f"You are an expert precision data extraction agent for the `{schema_cls.__name__}` schema."
        f"{doc_section}\n"
        f"### TARGET SCHEMA FIELDS (`{schema_cls.__name__}`)\n"
        f"{fields_section}\n\n"
        f"{EXTRACTION_RULES}"
    )
