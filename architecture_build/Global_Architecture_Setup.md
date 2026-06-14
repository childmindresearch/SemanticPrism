# SemanticPrism: Global Architecture & Project Setup

This document serves as the **root specification** for initializing the SemanticPrism project. Read this file *before* building any individual modules. It dictates the global directory structure, centralized configuration management, and the input/output lifecycle.

## 1. Directory Structure

The project strictly follows a domain-driven modular layout. All logic is housed inside `src/`, separated by pipeline phase. Do not cross-contaminate domain boundaries.

```text
SemanticPrism/
├── inputs/
│   └── testdocs/           # Place raw .txt and .md documents here for ingestion
├── outputs/
│   ├── 01_extraction/      # Raw and normalized triples, theme jsons, extraction logs
│   ├── 02_refinement/      # Vectors, matrices, and taxonomic hypernym mappings
│   ├── 03_topology/        # Modularity partitions and spectral metrics
│   ├── schemas/            # Generated Pydantic Python code outputs
│   │   ├── normalized/     # Schemas built from smoothed hypernyms
│   │   └── raw/            # Schemas built from unwound original source SVO text
│   └── visuals/            # PyVis HTML interactive graph visualizations
├── models/
│   └── embeddings/         # Locally cached HuggingFace sentence-transformers
├── src/
│   ├── extraction/         # Phase 1: Pydantic AI Agents, chunking, and text normalization
│   ├── refinement/         # Phase 2: Embedding matrices, clustering, and LLM taxonomic lifting
│   ├── topology/           # Phase 3: NetworkX, Leiden modularity, Hypergraph spectral math
│   └── synthesis/          # Phase 4: Pydantic AI Agents for Python code generation
├── config.yaml             # The single source of truth for global hyperparameters
├── run_pipeline.py         # Master script executing the phases sequentially
└── requirements.txt        # Python dependencies
```

## 2. Centralized Configuration Management (`config.yaml`)

To ensure the pipeline is heavily parameter-driven and easily adjustable, **all** configurable variables must be pulled dynamically from `config.yaml`. Hardcoding hyperparameters inside module files is strictly forbidden.

### Core Configuration Schema Template
The orchestrator must load `config.yaml` on boot and pass the relevant settings to the modules.

```yaml
# SemanticPrism Central Configuration

llm:
  # Connection details for Pydantic AI Agents
  provider: "openai" # e.g., 'openai', 'ollama', 'vertex'
  model_name: "gpt-4o"
  base_url: "http://localhost:11434/v1" # Relevant if using local proxies
  api_key_env_var: "OPENAI_API_KEY"
  temperature: 0.0

extraction:
  domain: "General Complex Logic" # Fallback explicit domain if agent discovery fails
  theme_chunk_max_words: 6000
  triple_chunk_max_words: 4000
  normalize_text: true

refinement:
  embedding_model: "BAAI/bge-m3" # HuggingFace model string
  similarity_threshold: 0.25     # Distance tolerance for agglomerative clustering
  spectral_variance_retention: 0.95

topology:
  inheritance_overlap_threshold: 0.75 # Minimum entity overlap % to subclass a theme
  leiden_resolution: 1.0              # Granularity control for Leiden clustering
  min_community_size: 3               # Pruning floor for disconnected orphans

synthesis:
  strategy: "hub_and_spoke"           # "standard" or "hub_and_spoke"
```

## 3. Data Lifecycle & State Management

To prevent data loss and ensure debuggability, the pipeline utilizes a strict "save state" mechanism. 

### Rule 1: Immutable Raw Data
When raw data is extracted from the LLM (e.g., `raw_triples`), it must be persisted immediately to disk (e.g., `outputs/01_extraction/original_triplets.json`). If transformations are applied (like lexical normalization), they must occur on a **deep copy** of the data, and the transformed result must be saved to a separate file (e.g., `outputs/01_extraction/normalized_triplets.json`).

### Rule 2: Pass Context via Objects, Not Globals
Global singletons (like importing `FREQUENCY_REGISTRY` from a logger) are forbidden. 
*   Create a `PipelineRunContext` or `StateTracker` class at the start of `run_pipeline.py`.
*   Pass this object sequentially down through the phases (Extraction $\rightarrow$ Embedding $\rightarrow$ NLP $\rightarrow$ Topology $\rightarrow$ Synthesis).
*   Any global tracking (like term frequency) should be attached to this context object.

### Rule 3: Pydantic AI Integration
All interactions with Large Language Models must be driven exclusively through **Pydantic AI Agents**. 
*   Avoid building custom raw HTTP clients or complex `aiohttp` wrappers.
*   Define your `Agent` instances in the respective modules.
*   Enforce structured JSON output natively by setting the `result_type` parameter on the Agent to a highly structured `pydantic.BaseModel`.
*   Inject runtime dependencies (like the dynamically calculated master domain or previous entity states) using Pydantic AI's `@agent.system_prompt` decorator and `RunContext[Deps]`.
