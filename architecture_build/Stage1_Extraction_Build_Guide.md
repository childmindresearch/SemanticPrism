# SemanticPrism Stage 1: Extraction Pipeline Build Specification

This document provides explicit instructions for building the Stage 1 Extraction phase using **Pydantic AI**. It strictly adheres to the constraints defined in `Global_Architecture_Setup.md` (no external `core/` dependencies, mandatory config loading, and explicit state passing).

## 1. Architectural Layout
Construct the module using three cleanly separated files inside `src/extraction/`:
1.  `schemas.py`: Contains strictly Pydantic data models.
2.  `prompts.py`: Contains static text and system prompts as strings.
3.  `extractor.py`: Contains the text chunking logic, Pydantic AI Agent definitions, and execution pipeline class.

---

## 2. Schemas (`src/extraction/schemas.py`)
Implement the following Pydantic models. Ensure strict typing and documentation strings.

### Core Models:
1.  **`Theme`**: 
    *   Fields: `title` (str), `description` (str), `reasoning` (str).
2.  **`ThemeDiscoveryResult`**: 
    *   Fields: `themes` (List[Theme]).
3.  **`MasterThemeSynthesisResult`**: 
    *   Fields: `master_domain` (str), `master_themes` (List[str]).
4.  **`RawTriple`**: 
    *   Fields: `subject` (str), `predicate` (str), `object` (str), `source_quote` (str), `certainty_score` (float), `theme_association` (Optional[str]), `source_document` (str).
    *   Requirement: Include a `@field_validator` to prevent any field from being an empty string.
5.  **`TripleExtractionResult`**: 
    *   Fields: `triples` (List[RawTriple]).
    *   Requirement: Include a `@field_validator` with `mode='before'` to silently discard invalid dictionaries that fail strict `RawTriple` validation without crashing the process.

*(Note: Lexical Normalization models have been moved to Stage 2: Refinement).*

---

## 3. Prompts (`src/extraction/prompts.py`)
Define the following strings as constants. Do not include `.format()` variables directly in the system prompts.

### Required Constants:
*   `THEME_DISCOVERY_SYSTEM_PROMPT`
*   `MASTER_THEME_SYSTEM_PROMPT`
*   `TRIPLE_EXTRACTION_SYSTEM_PROMPT`

---

## 4. Agent Definitions (`src/extraction/extractor.py`)

Instantiate three distinct `pydantic_ai.Agent` objects at the module level. **Do not hardcode the model strings**; they must be dynamically supplied via `config.yaml` dynamically when the orchestrator initializes the module.

### Agent 1: Theme Discovery
*   **System Prompt:** `prompts.THEME_DISCOVERY_SYSTEM_PROMPT`.
*   **Result Type:** `schemas.ThemeDiscoveryResult`.

### Agent 2: Master Theme Synthesis
*   **System Prompt:** `prompts.MASTER_THEME_SYSTEM_PROMPT`.
*   **Result Type:** `schemas.MasterThemeSynthesisResult`.

### Agent 3: Triple Extraction
*   **System Prompt:** `prompts.TRIPLE_EXTRACTION_SYSTEM_PROMPT`.
*   **Result Type:** `schemas.TripleExtractionResult`.
*   **Dependency Injection:** 
    *   Define a `@dataclass` named `TripleContext` with fields `master_themes: schemas.MasterThemeSynthesisResult` and `previous_entities: set[str]`.
    *   Set the `deps_type` on the agent to `TripleContext`.
    *   Use the `@triple_agent.system_prompt` decorator to dynamically inject the context into the prompt:
        ```python
        @triple_agent.system_prompt
        def add_triple_context(ctx: RunContext[TripleContext]) -> str:
            themes = ctx.deps.master_themes.model_dump_json() if ctx.deps.master_themes else "None"
            entities = list(ctx.deps.previous_entities)[-100:] if ctx.deps.previous_entities else "None"
            return f"\\nDiscovered Themes Context: {themes}\\nPreviously Discovered Entities: {entities}"
        ```

---

## 5. Master Execution Pipeline

Create a master class `ExtractionPipeline` to execute the sequence logically. It must accept the global `config` mapping and a `PipelineRunContext` to properly persist state.

### Step 5.1: Native Utilities
1.  **Text Chunking (`chunk_text`):** Build a localized function to chunk text based on the limits defined in `config['extraction']['theme_chunk_max_words']`. It should split text by word boundaries and support a sliding overlap window to preserve context edges.

### Step 5.2: Theme Extraction Workflow
1.  **Text Chunking:** Chunk incoming text using the config-defined limits.
2.  **Discovery:** Loop over chunks. Call `theme_agent.run_sync(chunk, model=config_model)`. Collect outputs.
3.  **Consolidation:** Format all discovered themes into a single string. Call `master_theme_agent.run_sync(formatted_string, model=config_model)` to generate the `MasterThemeSynthesisResult`. Attach this result to the `PipelineRunContext`. **Critically, save the `master_themes` list to `outputs/01_extraction/master_themes.json` for downstream use.**

### Step 5.3: Triple Extraction Workflow
1.  **Extraction Loop:** Loop over text chunks using the config-defined limit `triple_chunk_max_words`.
2.  **Dependencies:** Instantiate `TripleContext(master_themes=PipelineRunContext.master_themes, previous_entities=PipelineRunContext.entity_registry)`.
3.  **Run Agent:** Call `triple_agent.run_sync(chunk, deps=TripleContext, model=config_model)`.
4.  **State Update:** Append subjects and objects from the resulting triples back to the `PipelineRunContext.entity_registry` for coreference tracking on the next chunk. 
5.  **Data Persistence:** Attach the final `raw_triples` to the `PipelineRunContext` for downstream modules. Save the pristine `raw_triples` to `outputs/01_extraction/original_triplets.json`.

---

## 6. Isolated Execution (`run_stage_1.py`)
To ensure strict modularity, this stage must be executable in complete isolation. Create a standalone `run_stage_1.py` execution script that initializes the pipeline. 

**Ingestion Requirements:** The script must read an `input_directory` path from `config.yaml` (e.g., `inputs/testdocs/`). It must use native Python logic (`pathlib` or `glob`) to find and read all individual `.txt` files inside that directory. The combined corpus text is then fed into the `ExtractionPipeline`.

This guarantees that the stage can be tested, debugged, and run entirely independently across varying batches of documents.
