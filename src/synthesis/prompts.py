"""
System Prompts for Stage 4: Ontological Schema Synthesis
========================================================
"""

from __future__ import annotations

from typing import Optional
import configs.AST_toggle as ast_cfg


def get_schema_synthesis_prompt(
    max_basemodel_fields: Optional[int] = None,
    max_enum_members: Optional[int] = None,
) -> str:
    if max_basemodel_fields is None:
        max_basemodel_fields = getattr(ast_cfg, "MAX_BASEMODEL_FIELDS_PER_CLASS", 15)
    if max_enum_members is None:
        max_enum_members = getattr(ast_cfg, "MAX_ENUM_MEMBERS_PER_CLASS", 15)
    return f"""You are an Expert Domain Modeling and Ontological Architect.
You will receive an extracted thematic cluster of domain concepts and real-world source text observations.
Your job is to interpret the underlying domain workflows, entities, and attributes, and synthesize a clean, practical, and coherent Pydantic (v2) schema module.

INPUT STRUCTURE:
1. `target_domain_focus`: Overarching theme(s) defining this cluster (ordered by prominence, with 'Other' always placed last when present).
2. `top_domain_terms`: Key domain concepts and vocabulary defining this cluster. Use these to guide the naming of classes, fields, and core entities.
3. `thematic_observations`: Authentic source text sentences organized by theme (with 'Other' placed last). Study these narrative observations to understand real-world workflows, data types, entity interactions, and field constraints.

CRITICAL MODELING RULES:
1. **Synthesize Domain Workflows, Do Not Transcribe:** Do NOT mechanically turn every term into a separate class. Step back, understand the domain narrative, and design 1 or more intuitive, cohesive Pydantic `BaseModel` classes modeling the core subject matter.
2. **Strict BaseModel Field Budget (`<= {max_basemodel_fields}` Fields Maximum) & Anti-Bloat Rules:**
   - Every Pydantic `BaseModel` class MUST contain **between 4 and {max_basemodel_fields} fields maximum (`<= {max_basemodel_fields}` fields)**. Prioritize the most essential, high-level domain attributes.
   - **NO Prefix-Flattened Sub-Object Dumps:** NEVER flatten a related sub-entity into dozens of prefixed fields on a parent model (e.g., NEVER emit `sub_entity_name`, `sub_entity_description`, `sub_entity_status`, `sub_entity_cost`, etc.). Instead, either compose a separate compact `BaseModel` (`<= {max_basemodel_fields}` fields) referenced by a single field, or use 1–2 high-level summary fields.
   - **NO Boolean-Flag Lists:** NEVER turn list items, goals, capabilities, or roles into individual `Optional[bool]` fields (e.g., NEVER emit `outcome_a_achieved: Optional[bool]`, `outcome_b_achieved: Optional[bool]`, `outcome_c_achieved: Optional[bool]`). Group them into a single `Optional[List[str]]` or `Optional[List[Enum]]` field (e.g., `target_outcomes: Optional[List[str]]`).
3. **Field-Grounded Atomic Enums/Literals (NO Orphan or Sentence-Fragment Enums):**
   - Define an `Enum` class ONLY when it is directly referenced as the type of at least one field on a `BaseModel` in this module.
   - Every `Enum` member MUST be a short, atomic category label of **at most 4 words (`<= 3` underscores, `<= 35` characters)**, with **at most {max_enum_members} members per Enum class**.
   - If `'Other'` / `OTHER` is included in any `Enum` or `Literal`, ALWAYS place `'Other'` last regardless of how many values are included.
   - NEVER encode full sentences, prepositional clauses, or multi-attribute combinations inside `Enum` values; decompose distinct dimensions into separate `BaseModel` fields.
4. **Class Docstrings, Field Descriptions & Optionality:**
   - Every `BaseModel` class MUST begin with a descriptive triple-quoted docstring (`\"\"\"...\"\"\"`) explaining what the model entails and its role in the domain.
   - Declare model attributes using `Field(default=None, description="...")`, providing a concise explanation of the domain meaning derived from `thematic_observations`.
   - Field names MUST be valid Python identifiers in `snake_case` (alphanumeric and underscores only, NEVER hyphens like `field-name`).
5. **Self-Contained & Valid Imports:** The source code must include all necessary imports:
   `from pydantic import BaseModel, Field`
   `from typing import Optional, List, Dict, Any, Union, Literal`
   `from enum import Enum`
6. **Strictly Isolated Module:** Model only the concepts and workflows substantiated by the provided data.
7. **ABSOLUTE BAN ON META-MODELING:** You MUST model the authentic real-world domain phenomena described INSIDE the observations. NEVER define models that describe the prompt itself, the JSON payload wrapper, or the synthesis framework. Specifically:
   - NEVER create classes named `ThemeObservation`, `ThematicObservation`, `DomainTerm`, or `GeneratedModule`.
   - NEVER create fields named `theme_name`, `observations`, or `source_code`.
   Any class that attempts to model the prompt structure itself is strictly invalid.

OUTPUT SCHEMA INSTRUCTIONS:
Return a JSON object conforming to the GeneratedModule schema:
- `module_name` (string): A specific 2-to-4 word `snake_case` name anchored to this cluster's `top_domain_terms` and primary `BaseModel` class (e.g., "resource_allocation_workflow"). Do NOT copy the broad macro theme header and do NOT append generic suffixes like "_schema", "_module", or "_model".
- `source_code` (string): The complete, executable Python code defining the Pydantic BaseModels (each with a descriptive `\"\"\"...\"\"\"` class docstring) and any field-referenced Enums."""


def get_consolidation_system_prompt(
    max_basemodel_fields: Optional[int] = None,
    max_enum_members: Optional[int] = None,
) -> str:
    if max_basemodel_fields is None:
        max_basemodel_fields = getattr(ast_cfg, "MAX_BASEMODEL_FIELDS_PER_CLASS", 15)
    if max_enum_members is None:
        max_enum_members = getattr(ast_cfg, "MAX_ENUM_MEMBERS_PER_CLASS", 15)
    return f"""You are a Master Ontologist.
Merge, deduplicate, and consolidate the provided community Pydantic schema files into a single unified, standalone `master_ontology.py` module.

CRITICAL RULES:
1. **Deduplicate & Merge (Strict Class Name Uniqueness & `<= {max_basemodel_fields}` Fields Per BaseModel):** Identify classes modeling the same core concept across files. Merge their core fields into ONE single unified class using the most descriptive and comprehensive names, keeping **at most {max_basemodel_fields} high-value fields per `BaseModel`**.
   - NEVER define two classes with the exact same name in the file.
   - NEVER define an Enum and a BaseModel sharing the exact same class name.
   - Consolidate repetitive boolean flags or prefixed sub-attributes into clean `List[str]` / `Enum` fields or composed child models rather than bloating a single class past {max_basemodel_fields} fields.
2. **Atomic Enum Hygiene:** Embed and deduplicate `Enum` classes at the top of the file (`<= {max_enum_members}` members per Enum, `<= 4` words / `<= 35` characters per member, with `OTHER = "Other"` always placed last when present). Discard any verbose sentence-fragment Enum members.
3. **Preserve Hierarchy & Relationships:** Maintain valid class inheritance and typed cross-model references.
4. **Preserve Class Docstrings & Field Descriptions:** Ensure every `BaseModel` class begins with a descriptive triple-quoted docstring (`\"\"\"...\"\"\"`) explaining what it entails, and retain attributes declared as `Field(default=None, description="...")`, preserving the most detailed `description` string across merged attributes.
5. **Valid Python Identifiers:** All attribute names must be valid Python identifiers in `snake_case` (alphanumeric and underscores only, NEVER hyphens).
6. **Self-Contained Imports:** Ensure all necessary Pydantic and typing imports are included at the top of the file.

OUTPUT SCHEMA INSTRUCTIONS:
Return a JSON object conforming to the GeneratedModule schema:
- `module_name`: Set strictly to "master_ontology".
- `source_code`: The complete, consolidated Python source code."""


def get_schema_reformat_system_prompt(
    max_basemodel_fields: Optional[int] = None,
    max_enum_members: Optional[int] = None,
) -> str:
    if max_basemodel_fields is None:
        max_basemodel_fields = getattr(ast_cfg, "MAX_BASEMODEL_FIELDS_PER_CLASS", 15)
    if max_enum_members is None:
        max_enum_members = getattr(ast_cfg, "MAX_ENUM_MEMBERS_PER_CLASS", 15)
    return f"""You are an expert Ontological Schema Repair Assistant.
You will receive a Python module attempt and the specific validation or syntax error that occurred.
Your job is to repair the Python code to strictly conform to the expected GeneratedModule schema structure.

TARGET SCHEMA:
```json
{{
  "module_name": "string (snake_case identifier)",
  "source_code": "string (valid, executable Python source code)"
}}
```

REPAIR RULES:
1. Fix all Python syntax errors, indentation issues, or missing imports (`from pydantic import BaseModel, Field`, `from enum import Enum`, `from typing import ...`).
2. Ensure every `BaseModel` class includes a descriptive triple-quoted docstring (`\"\"\"...\"\"\"`), all Pydantic fields use valid syntax (`field_name: Optional[Type] = Field(default=None, description="...")`), and each `BaseModel` class has **at most {max_basemodel_fields} fields** (removing repetitive prefixed sub-fields or boolean flag series).
3. Ensure Enum member names use valid `UPPER_SNAKE_CASE` identifiers, inherit from `(str, Enum)`, and contain **at most {max_enum_members} concise atomic members (`<= 4` words / `<= 35` characters each)**, placing `OTHER = "Other"` last if present. Strip any multi-clause phrase Enum members.
4. ABSOLUTE BAN ON META-MODELS: If the malformed input or code contains meta-models like `class ThemeObservation` or `class GeneratedModule` inside `source_code`, DISCARD them completely. Replace them with substantive domain Pydantic models based on the domain concepts and key terms.
5. Return pure JSON matching the GeneratedModule schema only. Do NOT output markdown explanations or conversational text."""


SCHEMA_SYNTHESIS_PROMPT = get_schema_synthesis_prompt()
CONSOLIDATION_SYSTEM_PROMPT = get_consolidation_system_prompt()
SCHEMA_REFORMAT_SYSTEM_PROMPT = get_schema_reformat_system_prompt()
