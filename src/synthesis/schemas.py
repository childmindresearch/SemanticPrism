"""
Pydantic v2 Schemas for Stage 4: Ontological Schema Synthesis
=============================================================
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, model_validator


class GeneratedModule(BaseModel):
    module_name: str = Field(
        description=(
            "A specific 2-to-4 word snake_case module name derived directly from this cluster's highest-ranked "
            "top_domain_terms and primary BaseModel class (e.g., 'resource_allocation_workflow' or "
            "'incident_escalation_protocol'). NEVER use broad macro theme titles or generic suffixes like "
            "'_schema', '_module', or '_ontology'."
        )
    )
    source_code: str = Field(
        description=(
            "The raw Python source code containing the generated Pydantic BaseModels and Enums, including all necessary "
            "import statements. Every BaseModel class MUST include a descriptive triple-quoted docstring (\"\"\"...\"\"\") "
            "and have at most 15 fields (decompose sub-entities into separate "
            "models instead of prefix-flattened fields, and use List/Enum fields instead of boolean flag lists). "
            "Every Enum class MUST have at most 15 concise atomic members (<= 4 words each)."
        )
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_module_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            mod_name = (
                data.get("module_name")
                or data.get("name")
                or data.get("filename")
                or data.get("module")
                or "domain_schema"
            )
            src_code = (
                data.get("source_code")
                or data.get("code")
                or data.get("python_code")
                or data.get("content")
                or ""
            )
            out = dict(data)
            out["module_name"] = str(mod_name).strip()
            out["source_code"] = str(src_code).strip()
            return out
        return data
