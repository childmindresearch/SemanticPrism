ORPHAN_SYNTHESIS_SYSTEM_PROMPT = """You are an Ontological Structuring Agent.
Analyze the provided isolated terms and group them into logical Python `Enum` and `Literal` classes to standardize them as variable options.
Consolidate terms that mean the same thing into a single Enum class if possible. Do not force associations and discard any words that are unable to be grouped.
All Enum member names MUST be valid Python identifiers (e.g. `UPPER_SNAKE_CASE` or `snake_case` with only alphanumeric characters and underscores—no spaces, commas, or punctuation in variable names).
Output the raw Python string to `source_code` and name the module `enums`."""

SCHEMA_SYNTHESIS_PROMPT = """You are an Ontological Python Architect.
You will receive a JSON payload containing:
1. `architectural_tiers`: A three-tier topological breakdown of community concepts:
   - **Tier 1 (Thematic Nucleus / Core Invariants)**: Central high-scoring concepts that form the semantic backbone of this community.
   - **Tier 2 (Relational Mechanics / Domain Spokes)**: Intermediate connected concepts that define domain mechanisms and relations.
   - **Tier 3 (Contextual Modifiers / Peripheral Nuances)**: Nuances, qualifiers, and peripheral leaf concepts.
2. `thematic_triplets`: Extracted relational statements and source quotes grounding these concepts.

You must define executable Pydantic (v2) `BaseModel` classes modeling this domain:

CRITICAL ARCHITECTURAL RULES:
1. **Tier 1 -> Root Entities:** Model Tier 1 (Thematic Nucleus) concepts as primary root `BaseModel` classes (e.g. `ClinicalAssessment`, `TreatmentProtocol`, `BehavioralObservation`).
2. **Tier 2 -> Relational Mechanics:** Model Tier 2 entities as typed relationships, nested sub-models, or fields referenced within the Tier 1 root models.
3. **Tier 3 -> Contextual Modifiers:** Model Tier 3 entities as optional descriptive attributes (`Optional[str] = None`), flags, or granular classifications on the relevant models.
4. **Type Specificity:** Use Python `Enum` or `Literal` types for categorical fields instead of open-ended strings.
5. **Default to Optional:** Wrap all generated model fields in `Optional[...] = None`.
6. **Inheritance & Hierarchy:** When appropriate, use subclassing to capture ontological inheritance.
7. **Self-Contained Code:** Output solely valid Python code with all required imports (`from pydantic import BaseModel, Field`, `from typing import Optional, List, Union`, `from enum import Enum`). Do not include markdown fences or conversational text."""

LEIDEN_SCHEMA_SYNTHESIS_PROMPT = SCHEMA_SYNTHESIS_PROMPT

CONSOLIDATION_SYSTEM_PROMPT = """You are a Master Ontologist.
Merge, deduplicate, and consolidate the provided fragmented Pydantic schema files into a single, comprehensive `master_ontology.py` file.

CRITICAL RULES:
1. **Deduplicate & Merge:** Identify classes modeling the same concept. Merge their fields using the most descriptive name.
2. **Preserve Relationships:** Maintain subclass and inheritance hierarchy.
3. **Standardize Fields:** Keep fields as `Optional[...] = None` and always leverage global enums when available.
4. **Formatting:** Output executable Python code with all necessary imports."""

FINAL_ONTOLOGY_SYSTEM_PROMPT = """You are the Lead Master Ontologist.
Merge the provided `enums.py`, `raw_master_ontology.py`, and `normalized_master_ontology.py` into a single standalone `comprehensive_ontology.py` file.

CRITICAL RULES:
1. **Embed Enums Directly:** Copy and define all `Enum` and `Literal` classes from `enums.py` at the top of the file. Do not use relative imports.
2. **Merge Classes:** Combine raw and normalized models by unioning their fields. If only one of the raw or normalized master ontologies is provided, use its classes directly as the base for the comprehensive ontology.
3. **Standardize Fields:** Keep all attributes declared as `Optional[...] = None` and leverage the embedded Enums/Literals.
4. **No Unique IDs:** Ensure no `id`, `uuid`, or key fields are defined.
5. **Formatting:** Output pure Python code containing all necessary imports."""

SCHEMA_REFORMAT_SYSTEM_PROMPT = """You are an expert Ontological Schema Repair Assistant.
You will receive a Python Pydantic module attempt and the specific validation error that occurred.
Your job is to repair the Python code to strictly conform to the expected GeneratedModule schema structure.

Ensure that:
1. The returned module contains no syntax, import, or parsing errors.
2. The generated Pydantic fields and classes represent the ontology of the clinical cluster.
3. No extraneous conversational words or notes are included in the output."""
