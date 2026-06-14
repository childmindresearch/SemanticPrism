# SemanticPrism Stage 4: Synthesis Pipeline Build Specification

This document provides explicit instructions for building the Stage 4 Synthesis phase using **Pydantic AI**. This phase is responsible for translating the raw mathematical topology partitions into an accurate ontological schema using Pydantic `BaseModel`s. 

**Crucial Architecture Note:** "Hubs" and "Communities" are treated identically from an ontological perspective. A Hub is not a global "Root Schema"; it is simply a massive, high-gravity topic area that manages more content. Other communities will identify independent schemas. Inheritance between them is dictated *only* by the mathematical `ThemeInheritance` overlap mapping, not by hub status.

## 1. Architectural Layout
Construct the module using three cleanly separated files inside `src/synthesis/`:
1.  `schemas.py`: Contains strictly Pydantic data models to govern the LLM's code output.
2.  `prompts.py`: Contains static system prompts for Python synthesis.
3.  `synthesizer.py`: Contains the Pydantic AI Agent definitions and execution pipeline class.

---

## 2. Schemas (`src/synthesis/schemas.py`)
Implement the following Pydantic models.

### Core Models:
1.  **`GeneratedModule`**: 
    *   Fields: `module_name` (str), `source_code` (str).
    *   *Constraint:* Ensure `source_code` is formatted as raw Python text, explicitly containing necessary `import` statements. `module_name` must be a snake_case string defined as "The functional title abstracting the core semantic nodes".

---

## 3. Prompts (`src/synthesis/prompts.py`)
Define the following strings as constants. 

### Required Constants:
*   `ORPHAN_SYNTHESIS_SYSTEM_PROMPT`: Instructions to analyze a list of disconnected terms (orphans) and group them into logical Python `Enum` and `Literal` classes to standardize variable options.
*   `SCHEMA_SYNTHESIS_SYSTEM_PROMPT`: Instructions to construct granular Pydantic `BaseModel` classes. The LLM must determine the dominant theme of the provided triples, check the injected `ThemeInheritance` rules, and explicitly subclass the parent model if a parent-child relationship exists. If no inheritance relationship exists, it generates an independent root schema.

---

## 4. Agent Definitions (`src/agents/synthesis_agents.py`)

Instantiate the five specific `pydantic_ai.Agent` objects in a centralized agents file. Ensure the model string is passed dynamically from `config.yaml`.

### Agent 1: Orphan Enum Agent
*   **System Prompt:** `prompts.ORPHAN_SYNTHESIS_SYSTEM_PROMPT`.
*   **Result Type:** `schemas.GeneratedModule`.
*   **Dependency Injection:** Define a `@dataclass` named `OrphanContext` with a field `master_themes: list[str]`. Set `deps_type=OrphanContext`. Use `@orphan_agent.system_prompt` to dynamically inject the `master_themes` list directly into the prompt.

### Agent 2a: Leiden Ontology Schema Agent
*   **System Prompt:** `prompts.LEIDEN_SCHEMA_SYNTHESIS_PROMPT` (Instructs the LLM to model interconnected procedural narratives and workflows).
*   **Result Type:** `schemas.GeneratedModule`.
*   **Dependency Injection:** Define a `SynthesisContext` with fields `global_enums` and `theme_inheritance`.

### Agent 2b: Node2Vec Ontology Schema Agent
*   **System Prompt:** `prompts.NODE2VEC_SCHEMA_SYNTHESIS_PROMPT` (Instructs the LLM to model pure, decoupled semantic categories and completely ignore workflow logic).
*   **Result Type:** `schemas.GeneratedModule`.
*   **Dependency Injection:** Shares `SynthesisContext` with Agent 2a.

### Agent 3: Consolidation Agent
*   **System Prompt:** `prompts.CONSOLIDATION_SYSTEM_PROMPT`.
*   **Result Type:** `schemas.GeneratedModule`.

### Agent 4: Comprehensive Master Agent
*   **System Prompt:** `prompts.FINAL_ONTOLOGY_SYSTEM_PROMPT`.
*   **Result Type:** `schemas.GeneratedModule`.
*   **Dependency Injection:** 
    *   Define a `@dataclass` named `SynthesisContext` with fields `global_enums: str` and `theme_inheritance: str`.
    *   Set the `deps_type` on the agent to `SynthesisContext`.
    *   Use `@schema_agent.system_prompt` to inject these components into the prompt so the LLM correctly references Enums and knows when to apply true mathematical subclassing.

---

## 5. Master Execution Pipeline (`synthesizer.py`)

Create a master class `SynthesisPipeline` to execute the sequence logically. It must accept the global `config` and `PipelineRunContext`.

### Step 5.1: Setup and Ingestion
1.  **Load Topology:** Read the `topology_partitions.json` from Stage 3.
2.  **Load Logic:** Read `refined_triplets.json`, `original_triplets.json`, `taxonomic_map.json`, and `master_themes.json` from earlier stages.

### Step 5.2: Phase 1 - Orphan Aggregation (Enums)
1.  **Dependencies:** Instantiate `OrphanContext(master_themes=...)` from the loaded JSON.
2.  **Run Agent:** Pass the list of `orphans` to `orphan_agent.run_sync(orphans_payload, deps=OrphanContext)`.
2.  **Save Output:** Extract `result.data.source_code` and save it to both `outputs/schemas/normalized/enums.py` and `outputs/schemas/raw/enums.py`.
3.  **State Update:** Attach the generated source code to the `PipelineRunContext` as `global_enums`.

### Step 5.3: Phase 2 - Dual-Pass Schema Generation (Dynamic Routing)
*To allow for comparative analysis, the generation loop must be run twice: once using the smoothed hypernyms, and once using the original raw SVO text extracted in Stage 1.*

**Target Selection & Agent Routing:**
1. Read `clustering_strategy` from `config.yaml`.
2. If `"leiden"`, iterate over `communities` and use `leiden_schema_agent`.
3. If `"node2vec"`, iterate over `structural_clusters` and use `node2vec_schema_agent`.

**Pass A (Normalized):**
1.  **Dependencies:** Instantiate `SynthesisContext(global_enums=..., theme_inheritance=...)` from the `PipelineRunContext`.
2.  **Consolidate Targets:** Combine the `global_hubs` list and the configured cluster targets into a single iteration queue. **Sort this queue descending from largest cluster to smallest** (based on the number of nodes).
3.  **Loop Targets:** Iterate over the sorted targets, tracking the loop index `i` (starting at 1):
    *   Extract all `refined_triplets` where the subject or object exists in that specific partition.
    *   Serialize the subset of triples to JSON.
    *   Call `active_agent.run_sync(partition_json, deps=SynthesisContext, model=config_model)`.
4.  **Inheritance Wiring:** The agent evaluates the dominant `theme_association` of the triples within the current partition. It cross-references this dominant theme against the injected `ThemeInheritance` cheat sheet. If the dominant theme is listed as a `child_theme`, the agent physically alters the generated Python code to strictly subclass the parent schema.
5.  **Save Outputs:** Construct the filename using the zero-padded index `i` followed by the `result.data.module_name` returned from the LLM. Save the file (e.g., `outputs/schemas/normalized/01_core_database_architecture.py`).

**Pass B (Raw Unwound):**
1.  **Iterate Targets:** Same as above.
2.  **Filter Original Triples (Forward Map Lookup):** Iterate through the list of `original_triplets`. For each raw triple, dynamically translate its subject and object using the 1:1 `taxonomic_map` (e.g., `sub_mapped = taxonomic_map.get(raw_sub, raw_sub)`). If `sub_mapped` or `obj_mapped` exists in the current community's node list, include the **exact, un-mutated** raw triple in the subset. Serialize this raw payload to JSON.
3.  **Execute:** Call `active_agent.run_sync(raw_partition_json, deps=SynthesisContext, model=config_model)`.
4.  **Save:** Construct the filename identically. Save to `outputs/schemas/raw/01_core_database_architecture.py`.

### Step 5.4: Phase 3 - Global Consolidation
1. **Directory Consolidation:** Loop through the `raw/` and `normalized/` directories individually.
2. Read all generated `.py` files (except enums) and concatenate them into a single string.
3. Pass the string to the `Consolidation Agent`. It will deduplicate classes and resolve overlapping schemas.
4. Save the output to `master_ontology.py` in each respective directory.

### Step 5.5: Phase 4 - Final Comprehensive Ontology Generation
1. Read the text of `enums.py`, `normalized/master_ontology.py`, and `raw/master_ontology.py`.
2. Pass all three strings to the `Comprehensive Master Agent`.
3. **Equal Precedence Rule:** Instruct the agent to give equal weight to raw and normalized fields, merging them into a single class if there is a collision.
4. **Enum Direct Injection:** Instruct the agent to physically copy the Enum definitions into the top of the file rather than using relative imports.
5. Save the output to `outputs/schemas/comprehensive_ontology.py`. This is the final, standalone, deployable SDK.

### Step 5.6: Finalization
1.  **Create Init:** Generate a basic `__init__.py` in all schema output directories to formally expose the generated components as a usable python package.
2.  **Linting:** Run `ruff check --fix` and `ruff format` on the output directories to ensure syntax validity.

---

## 6. Isolated Execution (`run_stage_4.py`)
To ensure strict modularity, this stage must be executable in complete isolation. Create a standalone `run_stage_4.py` execution script that initializes the pipeline, loads the `topology_partitions.json`, `refined_triplets.json`, `original_triplets.json`, and `taxonomic_map.json` from the earlier stages, executes the `SynthesisPipeline`, and writes the final Pydantic schemas. This guarantees that the stage can be tested, debugged, and run entirely independently.
