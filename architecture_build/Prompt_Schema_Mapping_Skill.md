# SemanticPrism: Prompt & Schema Mapping Skill

This document provides the exact legacy prompt text and JSON schema designs that must be ported directly into the new **Pydantic AI** 4-stage architecture. Do not alter the core logic of these prompts or the structure of these schemas; they are the result of extensive prompt engineering and must be preserved.

## Stage 1: Extraction (`src/extraction/`)

### Process 1.1: Theme Discovery
**Goal:** Extract the most critical underlying abstractions or structural 'forms' of information.

**System Prompt (`THEME_DISCOVERY_SYSTEM_PROMPT`):**
```text
You are an Ontological Concept Engine tasked with Phase 1: Theme Discovery.
Your objective is to read the provided text and strictly discover the most critical underlying abstractions or structural 'forms' of information it represents, moving explicitly away from isolated specific subjects and maintaining distinction across themes.
Examples:
1. Rather than mapping specific instances (e.g., Plato, Aristotle, Metaphysics, Logic) as distinct themes, you MUST map the broader systemic abstractions that govern them (e.g., 'Historical Philosophical Figures', 'Epistemological Methodology', 'Ontological Theory').
2. Rather than mapping specific instances (e.g., Metformin, HbA1c testing, Type 2 Diabetes, Insulin resistance), you MUST map the broader systemic abstractions that govern them (e.g., 'Endocrine Metabolic Disorders', 'Pharmacotherapeutic Interventions', 'Biomarker Diagnostic Metrics').
3. Rather than mapping specific instances (e.g., Phishing email, CrowdStrike sensor, Ransomware, ISO 27001 compliance), you MUST map the broader systemic abstractions that govern them (e.g., 'Threat Vector Vectors', 'Endpoint Telemetry Mechanisms', 'Malicious Payload Typologies', 'Information Security Governance Frameworks').
Do not force finding themes if none exist, but accurately map as many relevant, abstract overarching categories as the depth of text naturally demands. Maintaining efficiency, the exact amount of categories should be dictated dynamically by the text's content.
These themes will act as the macro-level ontological categories for downstream factual extraction.
For each theme, provide its title, a brief description, and your reasoning as to why it is a critical class of information.
```

**User Prompt (`THEME_DISCOVERY_USER_PROMPT`):**
```text
Read this text and dynamically list the most critical overarching themes/classes of information based on the text:

<source_text>
{text_content}
</source_text>
```

**Schema Instructions (`ThemeDiscoveryResult`):**
*   Create a `Theme` BaseModel with `title`, `description`, and `reasoning` (all strings with descriptive `Field(...)` parameters).
*   Create a `ThemeDiscoveryResult` BaseModel containing `themes: List[Theme]`.

**Associated Schema (`ThemeDiscoveryResult` linked to `THEME_DISCOVERY_SYSTEM_PROMPT` & `THEME_DISCOVERY_USER_PROMPT`):**
```python
class Theme(BaseModel):
    """Represents an isolated, high-level structural theme."""
    title: str = Field(description="The formal title of the extracted theme.")
    description: str = Field(description="A brief description of the theme.")
    reasoning: str = Field(description="Reasoning for classifying this as a critical theme.")

class ThemeDiscoveryResult(BaseModel):
    """The aggregate output of the Theme Discovery phase."""
    themes: List[Theme] = Field(description="List of themes discovered in the text.")
```

---

### Process 1.2: Master Theme Synthesis
**Goal:** Deduplicate and formalize raw themes into a clean corpus-level list.

**System Prompt (`MASTER_THEME_SYSTEM_PROMPT`):**
```text
You are an Ontological Master Synthesizer. 
You will receive a massive aggregated list of document-level themes discovered individually across an entire corpus.
Your objective is to deduplicate, unify, and formalize this raw semantic noise into a single, clean, standardized 'Master Theme List' representing the entire corpus.
Your absolute priority is to dynamically identify the deep abstractions universally linking these themes. You must actively elevate overly-specific concepts into unified, systemic 'Forms' bridging entire datasets together. The final number of Master Themes MUST NOT be arbitrarily restricted; allow the text to dynamically scale the resulting volume of themes accurately.
Consolidate overlapping ideas into robust, formal generalized abstractions that maintain broad thematic reach while retaining just enough precision to be functionally discrete. Do not drop critical categories, but strictly merge them upward logically.

CRITICAL: You must return a single JSON data object containing your synthesized results. You must provide ONE single overarching `master_domain` string capturing the root logic, alongside the unified `master_themes` list containing the finalized theme names as strings. Do NOT output the JSON Schema definition itself.

The `master_themes` list should be a simple array of strings representing the final abstraction titles.
```

**User Prompt (`MASTER_THEME_USER_PROMPT`):**
```text
Consolidate the following document-level themes into a single Master Ontology for the corpus:

<extracted_themes>
{all_extracted_themes}
</extracted_themes>
```

**Schema Instructions (`MasterThemeSynthesisResult`):**
*   Create a BaseModel containing `master_domain` (string) and `master_themes` (List of strings).

**Associated Schema (`MasterThemeSynthesisResult` linked to `MASTER_THEME_SYSTEM_PROMPT` & `MASTER_THEME_USER_PROMPT`):**
```python
class MasterThemeSynthesisResult(BaseModel):
    """The normalized global output consolidating multiple document themes."""
    master_domain: str = Field(description="The singular, overarching domain describing the entire corpus.")
    master_themes: List[str] = Field(description="The consolidated, deduplicated list of formal theme names.")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "master_domain": "Ontology & Knowledge Systems",
                "master_themes": [
                    "System Architecture",
                    "Data Modeling",
                    "Semantic Graph Infrastructure"
                ]
            }
        }
    }
```

---

### Process 1.3: Triple Extraction
**Goal:** Extract node-edge-node facts with certainty and theme assignment.

**System Prompt (`TRIPLE_EXTRACTION_SYSTEM_PROMPT`):**
```text
You are an unconstrained Triple Extractor Agent running schema-mapped discovery.
Extract every single meaningful relationship found in the source text as a raw (Subject, Predicate, Object) triple.

If 'Discovered Themes' are provided to you, you MUST tentatively assign each extracted triple to its most logically associated theme title.
CRITICAL: Do not restrict extraction too early! If a triple possesses high semantic value but DOES NOT map cleanly into any supplied Discovered Theme, you MUST assign its `theme_association` to 'Other'. Do not discard critical isolated triples simply because they lack an explicit thematic category!

For EVERY entity you extract, you MUST:
1. Find the exact 'Source Quote' in the text that justifies its existence.
2. Assign a 'Certainty Score' (0.0 to 1.0).
ONLY where it exists, you MUST return the node-edge graph relationships:
1. Identify the core identity and connectedness of each entity (For example, "a young man walks his brown dog" = [man]-[walks]-[dog])
2. Do not infer information and only return what is explicitly stated in the text
3. An entity can have 0...N relationships from the text
        
Focus strictly on minimizing false positives. Do not hallucinate entities not strictly in the text.
```

**User Prompt (`TRIPLE_EXTRACTION_USER_PROMPT`):**
```text
{themes_context}{previous_entities_context}Extract the triplets from the following text and tentatively assign them to the themes above (if applicable):

<source_text>
{text_content}
</source_text>
```

**Schema Instructions (`TripleExtractionResult`):**
*   Create a `RawTriple` BaseModel with `subject`, `predicate`, `object`, `source_quote`, `certainty_score` (float), `theme_association` (Optional str), and `source_document`.
*   **Crucial Validation:** Add a `@field_validator` to `subject`, `predicate`, and `object` to reject empty strings.
*   Create a `TripleExtractionResult` BaseModel containing `triples: List[RawTriple]`.
*   **Crucial Validation:** Add a `mode='before'` validator to silently catch and discard hallucinated JSON dicts that fail the strict `RawTriple` validation without crashing the entire extraction loop.

**Associated Schema (`TripleExtractionResult` linked to `TRIPLE_EXTRACTION_SYSTEM_PROMPT` & `TRIPLE_EXTRACTION_USER_PROMPT`):**
```python
class RawTriple(BaseModel):
    """
    Extracts a high-fidelity subject-predicate-object relationship.
    Focuses on atomic facts to ensure graph density and accuracy.
    """
    subject: str = Field(..., description="The exact source entity exactly as it appears in the source text. Do not modify the casing or format.")
    predicate: str = Field(..., description="The exact relationship verb or linking phrase exactly as it appears in the source text. Do not use snake_case.")
    object: str = Field(..., description="The exact target entity or attribute value exactly as it appears in the source text. Do not modify the casing or format.")
    source_quote: str = Field(..., description="The exact snippet from the text that proves this relationship exists.")
    certainty_score: float = Field(ge=0, le=1, description="Confidence score: 1.0 for explicit facts, 0.5 for inferred.")
    theme_association: Optional[str] = Field(default="Other", description="The theme this triple most closely aligns with, if any.")
    source_document: str = Field(default="Unknown", description="The original document file this triple was extracted from.")

    @field_validator('subject', 'predicate', 'object')
    @classmethod
    def prevent_empty(cls, v):
        if not v or len(str(v).strip()) == 0:
            raise ValueError("Fields cannot be empty strings")
        return str(v).strip()

class TripleExtractionResult(BaseModel):
    """The aggregate result grouping multiple triples from a single text block."""
    triples: List[RawTriple] = Field(description="The complete list of extracted node-edge relationships.")

    @field_validator('triples', mode='before')
    @classmethod
    def drop_invalid_triples(cls, v):
        """Intercepts the raw JSON dictionaries before strict Pydantic crash enforcement."""
        if not isinstance(v, list):
            return v
            
        valid_chunk = []
        for item in v:
            try:
                if isinstance(item, dict):
                    # Test explicit validation against the strict RawTriple bounds natively
                    valid_chunk.append(RawTriple.model_validate(item))
                elif isinstance(item, RawTriple):
                    valid_chunk.append(item)
            except Exception:
                # Silently discard the corrupted/incomplete hallucination structurally 
                pass
        return valid_chunk
```

---

## Stage 2: Refinement (`src/refinement/`)

### Process 2.1: Lexical Normalization / Taxonomic Lifting
**Goal:** Transform raw entities into standardized hypernyms.

**System Prompt (`LLM_PREPROCESSING_SYSTEM_PROMPT`):**
```text
You are a Lexical Normalization Engine. Your goal is to transform raw NLP extracts into standardized, "atomized" strings to improve the accuracy of downstream vector embeddings.

### CORE DIRECTIVES:
1. **Lemmatization & Case:** - Convert all nouns to Singular form (e.g., 'data warehouses' -> 'data warehouse'). Maintain the lowercase format provided to you.
   - Convert all verbs to Third-Person Singular Present (e.g., 'running' -> 'runs').

2. **Noise Stripping:** - Remove determiners (the, a, an).
   - Remove corporate suffixes unless critical (e.g., 'Apple Inc.' -> 'Apple').
   - Remove "soft" adjectives that don't change the core entity (e.g., 'Large Database' -> 'Database').

3. **De-jargonizing:** - Expand common abbreviations ONLY if they are unambiguous in the provided context (e.g., 'K8s' -> 'Kubernetes').

4. **Structural Predicates:** - Standardize relationship strings. Convert 'is a part of', 'part of', 'component in' all to 'part of'.

### DOMAIN GUARDRAILS:
If a term is a specific technical product or a unique named entity, do NOT over-simplify it. 
- Keep: 'PostgreSQL' (Do not simplify to 'Database').
- Simplify: 'PostgreSQL Server Instance' -> 'PostgreSQL'.

CRITICAL: You must return a single JSON data object containing your normalized results. Do NOT output the JSON Schema definition itself. The output MUST match the provided schema exactly.
```

**User Prompt (`LLM_PREPROCESSING_USER_PROMPT`):**
```text
{domain_context}For each of the following raw strings, process them 1 by 1 and return the explicit mapping of the original string to its normalized form:
{raw_tokens_json}
```

**Schema Instructions (`NormalizedStrings`):**
*   Create a `NormalizedToken` BaseModel mapping `original` (str) to `normalized` (str).
*   Create a `NormalizedStrings` BaseModel containing `tokens: List[NormalizedToken]`.

**Associated Schema (`NormalizedStrings` linked to `LLM_PREPROCESSING_SYSTEM_PROMPT` & `LLM_PREPROCESSING_USER_PROMPT`):**
```python
class NormalizedToken(BaseModel):
    """A mapping from the original raw string to its normalized form."""
    original: str = Field(description="The exact original string provided.")
    normalized: str = Field(description="The finalized, normalized version of the string.")

class NormalizedStrings(BaseModel):
    """The normalized array mapping original strings to their transformed states."""
    tokens: List[NormalizedToken] = Field(description="The explicit mappings of original strings to normalized strings.")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "tokens": [
                    {
                        "original": "data warehouses",
                        "normalized": "data warehouse"
                    },
                    {
                        "original": "is a part of",
                        "normalized": "part of"
                    }
                ]
            }
        }
    }
```

---

## Stage 4: Synthesis (`src/synthesis/`)

### Process 4.1: Orphan Aggregation
**Goal:** Synthesize fragmented axioms into Python Enums.

**System Prompt (`ORPHAN_ENUMS_SYSTEM_PROMPT`):**
```text
You are an elite Python Architect.
You are receiving a raw list of isolated micro-components (disconnected facts with few nodes) from a topological graph.

Your TASK is to synthesize these fragmented axioms purely into Enums, Literal types, and Constants.

RULES:
1. 'summary': Provide a high-level summary of the concepts extracted.
2. 'enums_code': Output STRICTLY Python `Enum` classes, `Literal` types, and `Constants` only. NO BaseModels. NO Protocols.
3. GENERALIZATION & CONSOLIDATION: You MUST strictly organize and classify these enums under the provided `Master Themes` list. Do not invent new macro-categories; map the orphaned concepts logically into the existing thematic structure (e.g., `class {MasterThemeName}Options(Enum):`).
4. FALLBACK: If an orphaned concept does not logically fit into any of the Master Themes, you MUST classify it under an explicit `UncategorizedOptions(Enum)` or `OtherOptions(Enum)` to prevent forcing concepts into incorrect categories.
5. CRITICAL JSON FORMATTING: Do not wrap your response in markdown blocks (```json). You MUST properly escape all newlines (\n) and double quotes (\") inside the code strings.
```

**User Prompt (`ORPHAN_ENUMS_USER_PROMPT`):**
```text
<master_themes>
{master_themes_list}
</master_themes>

<orphaned_nodes>
{orphans_list}
</orphaned_nodes>
```

**Schema Instructions (`OrphanEnumSchema`):**
*   Create an `OrphanEnumSchema` (or migrate to the universal `GeneratedModule` schema) requiring `summary` and `enums_code` (or `source_code`). The code field MUST NOT be a dict; it must be a raw python string containing the literal code.

**Associated Schema (`OrphanEnumSchema` linked to `ORPHAN_ENUMS_SYSTEM_PROMPT` & `ORPHAN_ENUMS_USER_PROMPT`):**
```python
class OrphanEnumSchema(BaseModel):
    """Represents isolated constants and enums synthesized from micro-components."""
    summary: str = Field(description="A concise summary of the isolated concepts represented.")
    enums_code: str = Field(default="", description="Strictly typed Python Enums, Literal types, and Constants.")
```

---

### Process 4.2: Ontology Schema Generation
**Goal:** Synthesize the topological map into strictly typed Pydantic code using mathematical inheritance.

**System Prompt (`PYDANTIC_CODE_GEN_SYSTEM_PROMPT`):**
```text
You are an elite Python Architect and Ontologist. 
You are receiving a formalized Graph Topology map containing mathematically categorized nested structures (`root_classes`, `nested_classes`).

Your TASK is to synthesize this topology into perfectly structured JSON matching the strictly requested `GeneratedSchema` output!

RULES:
1. 'title', 'summary', & 'core_theme': Provide a concise but exhaustive title, summary, and 'core_theme' mathematically bounding the topology nodes physically natively.
2. 'key_learnings' & 'isolated_facts': Extract the logical axioms and unique properties driving this specific community.
3. 'protocols_code': Output `typing.Protocol` interfaces (Duck Typing) for cross-community linking. Do NOT use strict `class Child(Parent):` inheritance. Rely exclusively on Protocols and Composition for maximum flexibility.
4. 'concrete_models_code': Output the literal strictly typed Pydantic v2 Python classes here. Define Enums first, then Nested BaseModels, and finally Root BaseModels.
5. GENERALIZATION (CRITICAL): All schema field names MUST be abstracted. Do not use specific raw text (e.g., `wisc_v_score`). You must use broad Taxonomic Hypernyms (e.g., `assessment_score`) to ensure the schema is generalizable to similar documents.
6. INHERITANCE: If 'Inheritance Guidelines' are provided in the payload, you MUST implement them by generating dynamic `typing.Protocol` definitions to ensure your Pydantic schemas remain extremely decoupled and universally applicable.
7. CRITICAL JSON FORMATTING: Do not wrap your response in markdown blocks (```json). You MUST properly escape all newlines (\n) and double quotes (\") inside the code strings or the system will critically crash!
```

**User Prompt (`PYDANTIC_CODE_GEN_USER_PROMPT`):**
```text
Translate the following rigorously categorized topological structure into perfect nested strictly typed Pydantic Schema logic mapping exactly over Enums and List[...] subclass fields:

<inheritance_guidelines>
{inheritance_guidelines}
</inheritance_guidelines>

<hierarchy_payload>
{community_graph_json}
</hierarchy_payload>
```

**Schema Instructions (`GeneratedSchema`):**
*   Create a BaseModel (e.g., `GeneratedModule`) containing `title`, `summary`, `core_theme`, `key_learnings` (List[str]), `isolated_facts` (List[str]), `protocols_code` (str), and `concrete_models_code` (str).
*   *Note:* In the new Pydantic AI architecture, this can be merged into a simpler `module_name` and `source_code` string if preferred, as long as the generated code string accurately fulfills the prompt's structural demands.

**Associated Schema (`GeneratedSchema` linked to `PYDANTIC_CODE_GEN_SYSTEM_PROMPT` & `PYDANTIC_CODE_GEN_USER_PROMPT`):**
```python
class GeneratedSchema(BaseModel):
    """Represents the final Python schema synthesized from the topology map."""
    title: str = Field(description="The functional title abstracting the core semantic nodes.")
    summary: str = Field(description="A concise summary of the interlocked hierarchy map.")
    core_theme: str = Field(description="The foundational underlying concept binding the nodes.")
    key_learnings: List[str] = Field(description="The primary factual insights derived.")
    isolated_facts: List[str] = Field(description="Key unique properties abstracted from the topological array.")
    protocols_code: str = Field(default="", description="Strictly typed Protocol/Mixin interface blocks. Output empty string if none apply.")
    concrete_models_code: str = Field(default="", description="Strictly typed Pydantic code blocks for concrete representation. Output empty string if none apply.")
```
