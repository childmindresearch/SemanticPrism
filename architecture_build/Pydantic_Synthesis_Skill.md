# Pydantic Synthesis Skill Guidelines

**Purpose:** This document provides strict rules and best practices for the Stage 4 `Ontology Schema Agent`. When translating semantic clusters into Pydantic code, the resulting `BaseModel` classes must be optimized for *future LLM-driven data extraction*. 

If a schema is too rigid, subsequent LLM extraction passes will crash with validation errors. To prevent this, the generated schemas must be extremely flexible, highly descriptive, and utilize modern Pydantic V2 standards.

## 1. Optional by Default (Resilience)
LLMs frequently miss or skip fields during raw text extraction. If a field is strictly required, the entire Pydantic validation will fail, causing data loss.
*   **Rule:** Almost every field must be typed as `Optional` and default to `None`.
*   **Exception:** Only make a field strictly required if the ontological concept fundamentally cannot exist without it (e.g., a primary name or unique identifier).
*   **Syntax:** Use standard typing: `Optional[str] = Field(default=None)`.

## 2. Descriptive Fields (LLM Tool Optimization)
Because a future LLM will use these classes as "Structured Outputs", the field names and descriptions are the *only* context the LLM has to understand what data it should pull from the text.
*   **Rule:** Every single field MUST include a `Field(description="...")` parameter.
*   **Rule:** The description must be semantically rich. Do not just repeat the variable name; explicitly state what the field represents in the context of the corpus.

## 3. Graceful Defaults for Collections
If a concept involves a list of items (e.g., related properties, aliases, tags), do not let it default to `None`.
*   **Rule:** Use `default_factory=list` for all collection types to ensure the extraction engine always has an iterable object to append to, preventing `Nonetype object is not iterable` errors.
*   **Syntax:** `aliases: List[str] = Field(default_factory=list, description="...")`.

## 4. Modern Pydantic V2 Syntax
Ensure the generated code is compliant with modern Pydantic V2 standards.
*   **Rule:** Inherit strictly from `pydantic.BaseModel`.
*   **Rule:** Do not use legacy `__root__` validators. Keep the classes as pure data schemas; avoid complex `@model_validator` functions unless absolutely necessary, as they can cause unexpected rejection of LLM outputs.

## 5. Leverage Global Enums
In Stage 4, Phase 1 generates an `enums.py` file containing isolated low-degree nodes (Orphans).
*   **Rule:** When a field represents a classification, state, or rigid category, the agent must import and type-hint the field using the relevant `Enum` from `enums.py` rather than allowing an open-ended `str`. 
*   **Why:** This severely limits LLM hallucinations during future extraction runs by forcing it to select from a predefined list of valid strings.

## 6. Flat Over Deeply Nested
Deeply nested JSON schemas degrade LLM extraction performance, confuse the model's spatial awareness, and rapidly consume token limits.
*   **Rule:** Prefer flatter class structures. If a community schema is getting excessively deep, break the nested components into separate top-level `BaseModel` classes within the same file and reference them.

## 7. The "Escape Hatch" Principle (Handling Unknowns)
Because rigid schemas cannot perfectly predict all future text variations, you must provide logical "escape hatches" so the extraction LLM doesn't crash or hallucinate when it encounters unexpected data.
*   **Enum Fallbacks:** The `Orphan Enum Agent` must always inject an `OTHER = "Other"` and `UNKNOWN = "Unknown"` value into every generated Enum class.
*   **Supplemental Raw Fields:** Whenever you type-hint a field with an Enum, always pair it with an optional string field to capture the raw text in case the LLM is forced to select `OTHER`. (e.g., pair `status: Optional[EntityStatus]` with `status_raw: Optional[str]`).
*   **Catch-All Buckets:** Every primary `BaseModel` should include a generic dictionary field (e.g., `unmapped_attributes: Dict[str, str] = Field(default_factory=dict)`) to serve as a semantic bucket for highly relevant data the LLM finds that simply doesn't fit into the predefined schema fields.

---

## Example of an Optimal Schema Output

```python
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from enums import EntityStatus

class SpecializedDatabase(ParentDatabaseConcept):
    """
    Ontological representation of a highly specialized database system.
    Inherits core fields from ParentDatabaseConcept.
    """
    
    system_architecture: Optional[str] = Field(
        default=None, 
        description="The underlying technical architecture of the database (e.g., distributed, relational)."
    )
    
    supported_protocols: List[str] = Field(
        default_factory=list,
        description="A list of networking or data protocols natively supported by this system."
    )
    
    operational_status: Optional[EntityStatus] = Field(
        default=None,
        description="The current deployment or operational status of the database in the target environment."
    )
    
    operational_status_raw: Optional[str] = Field(
        default=None,
        description="If operational_status is OTHER, provide the exact raw status text extracted from the source here."
    )
    
    unmapped_attributes: Dict[str, str] = Field(
        default_factory=dict,
        description="A catch-all dictionary for any highly relevant database properties found in the text that do not fit into the rigid fields above."
    )
```
