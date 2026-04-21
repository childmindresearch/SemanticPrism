# SemanticPrism: Step-By-Step Reconstruction Summary

When reconstructing the SemanticPrism architecture entirely from scratch, it is imperative to follow a strict sequential flow to prevent circular dependency collisions.

### Step 1: Base Architecture & Dependencies
1. Create `requirements.txt` ensuring the presence of:
   - `instructor`, `ollama` (LLM communication)
   - `pydantic` (Entity structures)
   - `sentence-transformers`, `scikit-learn` (Math operations)
   - `networkx`, `pyvis` (Topology and Visualization)
   - `python-louvain` or `cdlib` (Community Modularity)
2. Create the core file system: `src/core`, `src/helpers`, `src/extraction`, `src/embedding`, `src/nlp`, `src/topology`, `src/synthesis`, `src/orchestrator`.
3. Create `config.yaml` matching the documented configuration features safely.
4. Build initial `orchestrator/pipeline.py` with outline of placeholder details. Wrap everything seamlessly inside sequential phase blockers.
5. Build prompt .py files for each module using exact text from `04_prompts_reference.md`

### Step 2: Core Primitives (`src/core/`)
1. Build `models.py` mapping out the Pydantic schemas explicitly (e.g., `Theme`, `ThemeDiscoveryResult`, `MasterThemeSynthesisResult`, `RawTriple`, `TripleExtractionResult`, `TaxonomicVerification`, `ClusterContextualValidation`, `GeneratedSchema`). 
2. Build `logger.py` for global standardized logging dynamically.
3. Build `chunking.py` utilizing the `SemanticChunker` overlap logic precisely.  Include dynamic reference to Step 3 context size identified.

### Step 3: Global Orchestration Sub-Modules
1. Build `context_manager.py` for evaluting the current GPU and available resources to dynamically identify appropriate context sizes. Maps hardware limitations natively. 
2. Integrate the precise semantic Prompts strings in separate utility files (e.g., `src/extraction/prompts.py`, `src/embedding/prompts.py`, `src/synthesis/prompts.py`, etc.) to keep the primary logic modules mathematically pure.

### Step 4: Logic Integration Nodes (The "Brain")
Sequentially build modules from `02_module_instructions.md`

### Step 5: The Primary Gateway
1. Finally, review all integrations to ensure `orchestrator/pipeline.py` wraps everything seamlessly inside sequential phase blockers.

By executing this framework sequentially and independently, you produce a highly cohesive, formally structured pipeline.
