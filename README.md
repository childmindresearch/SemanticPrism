# SemanticPrism

SemanticPrism is an advanced, autonomous agentic pipeline designed to process unstructured, highly complex domain knowledge (such as clinical diagnostic texts) and mathematically synthesize it into structured, deployable Python Ontologies (Pydantic models).

The purpose of SemanticPrism is to solve the "hallucination and overlap" problem inherent in standard Large Language Model (LLM) schema generation. Rather than asking an LLM to guess the structure of a document in one shot, SemanticPrism breaks the text down into mathematical graph networks, clusters those networks using advanced graph theory and representation learning algorithms, and uses those mathematical boundaries to generate perfectly decoupled, high-fidelity data models.

---

## Overall Pipeline Execution Flow

The entire pipeline is orchestratable via the master script [run_pipeline.py](file:///Users/david.lobue/Desktop/Agents/Refactor/SemanticPrism/run_pipeline.py).
To run all stages in sequence, execute:
```bash
python3 run_pipeline.py
```
This execution uses separate subprocesses for each stage to ensure that GPU VRAM and system memory are cleanly garbage collected between intensive LLM and mathematical clustering processes.

---

## Detailed Pipeline Architecture & Stage Breakdowns

The pipeline runs sequentially across four stages, utilizing outputs from preceding runs.

### Stage 1: Extraction ([run_stage_1.py](file:///Users/david.lobue/Desktop/Agents/Refactor/SemanticPrism/run_stage_1.py))
**Purpose:** To ingest unstructured source texts and extract raw semantic relationships and global themes.

*   **Global Theme Discovery:** The pipeline scans the input documents in sliding text chunks to identify overarching "Master Themes" (e.g., *Symptomology*, *Interventions*, *Accommodations*).
    *   *Underlying Detail (Context Anchoring):* The extracted themes provide semantic boundaries for the LLM during triple extraction. By anchoring the LLM to predefined themes, it prevents "hallucinated drift" when parsing massive, dense documents.
*   **Triplet Extraction:** Employs Pydantic-AI agents to extract Subject-Predicate-Object (S-P-O) triplets alongside their associated theme.
    *   *Underlying Detail (Graph Construction Foundation):* This process translates dense, natural prose grammar into discrete entities (Subject, Object) and relationships (Predicate), transforming the text into nodes and directed edges for the graph representation in Stage 3.
*   **Output Files:** Saves individual run files under `outputs/01_extraction/` and aggregates them into:
    *   `all_themes.json`: Raw discovered themes.
    *   `master_themes.json`: Synthesized master domain and theme mappings.
    *   `original_triplets.json`: All raw extracted S-P-O triplets.

---

### Stage 2: Refinement ([run_stage_2.py](file:///Users/david.lobue/Desktop/Agents/Refactor/SemanticPrism/run_stage_2.py))
**Purpose:** To clean, deduplicate, and normalize the raw, noisy triplets into a standardized, clean taxonomy.

*   **Lexical Normalization:** Cleans text syntax (stripping underscores, lowercase formatting) and uses batch LLM agents to correct spelling mistakes, expand acronyms, and normalize phrasing.
    *   *Underlying Detail (Preventing Graph Fragmentation):* Ensures that variations such as *"WISC-5"*, *"wisc-v"*, and *"wisc 5"* map to the exact same string. Otherwise, they would form disjoint nodes, fracturing the graph topology.
*   **Sentence Embeddings:** Converts normalized terms into dense, high-dimensional vector representations using `SentenceTransformers` (defaulting to `all-MiniLM-L6-v2`).
    *   *Underlying Detail (Semantic Coordinate Mapping):* Embeddings translate text words into coordinates in a semantic vector space. The distance between points corresponds to semantic meaning; terms representing similar concepts (e.g., *"severe anxiety"* and *"extreme worry"*) are mapped close to one another in space.
*   **SVO Vector Clustering (Agglomerative Clustering):** Uses Scikit-learn's `AgglomerativeClustering` using average linkage and cosine distance.
    *   *Underlying Detail (Mathematical Deduplication):* Groups terms whose cosine distance is below the `clustering_threshold`. It calculates a centroid for each cluster using frequency-weighted vector averages. The term closest to the centroid vector is selected as the representative term.
*   **Taxonomic Lifting:** Passes cluster terms to an LLM to resolve them into a formal hypernym (parent concept).
    *   *Underlying Detail (Standardizing Vocabulary):* Translates synonym groups into formal class names (e.g., mapping *"wisc-v"*, *"wisc 5"*, and *"cognitive test"* to *"WISC psychometric assessment"*).
*   **Triple Remapping:** Re-maps the subjects, predicates, and objects of all raw triplets to their taxonomic hypernyms, outputting clean, normalized triplets.
*   **Theme-Based Embedding Mapping:** Groups lower-level themes under master themes by computing the cosine similarity between their respective sentence embeddings.
*   **Output Files (saved under `outputs/02_refinement/`):**
    *   `subject_normalization_map.json` / `predicate_normalization_map.json` / `object_normalization_map.json`
    *   `taxonomic_map.json`: Combined normalization and taxonomic map.
    *   `refined_triplets.json`: Fully cleaned and remapped triplets.
    *   `theme_mapping_clusters.json`: Hierarchical mappings of themes.

---

### Stage 3: Dual-Path Graph Topology ([run_stage_3.py](file:///Users/david.lobue/Desktop/Agents/Refactor/SemanticPrism/run_stage_3.py))
**Purpose:** To map refined triplets into a network graph and execute **Dual-Path Topological Partitioning** (`topology.execution_mode: "community"` | `"embedding"` | `"both"`) to discover workflow sequences and structural category taxonomies.

#### Path 1: Community Workflow Topology (Approach A)
*   **Objective:** Isolate narrative workflows, event sequences, and clinical processes.
*   **Hub Identification Mechanism:**
    *   **Participation Coefficient ($P_i \ge 0.65$):** Identifies nodes whose incident edges span $\ge 65\%$ across multiple distinct domain themes.
    *   **Betweenness Centrality Chokepoint (Top 5%):** Identifies shortest-path bottleneck nodes (`betweenness_centrality > 0.02`) that bridge narrative workflow steps.
*   **Clustering Engine:** **Leiden Modularity Algorithm** on the pruned sub-graph (with hubs and singletons temporarily isolated).
*   **Output Location:** `outputs/03_topology/community/topology_partitions.json`
*   **Visualizations:** `outputs/visuals/community/`

#### Path 2: Embedding Categorical Topology (Approach B)
*   **Objective:** Isolate pure, decoupled ontological categories and structural roles.
*   **Hub Identification Mechanism:**
    *   **Participation Coefficient ($P_i \ge 0.45$):** Identifies cross-category structural connectors.
    *   **Modularity Vitality Pruning ($\Delta Q < -0.005$):** Prunes boundary-blurring nodes whose removal *increases* global modularity ($Q$) and have degree $\ge 3$.
*   **Clustering Engine:** **Node2Vec 64D/128D Random-Walk Embeddings** + **Dynamic K-Means Silhouette Optimization** ($K=2 \dots 12$).
*   **Output Location:** `outputs/03_topology/embedding/topology_partitions.json`
*   **Visualizations:** `outputs/visuals/embedding/`

#### Diagnostic HTML Visualizations Engine
Stage 3 exports 10 interactive PyVis HTML diagnostic files under `outputs/visuals/community/` and `outputs/visuals/embedding/`:
1. `interactive_topology_graph.html`: Full network topology with hub/orphan node styling.
2. `interactive_hubs_ego_network.html`: Global hubs and their connected spoke nodes.
3. `interactive_global_hubs.html`: **Global Hubs Only Topology** showing inter-hub directed edges with gold highlights.
4. `interactive_communities_only.html` / `interactive_structural_clusters_only.html`: Intra-cluster member sub-graphs.
5. `interactive_workflow_narratives.html`: Shortest-path narrative chokepoints and betweenness hotspots.
6. `interactive_participation_dispersion.html`: Participation coefficient ($P_i$) heatmap dispersion.
7. `interactive_modularity_vitality_landscape.html`: Modularity vitality ($\Delta Q$) category separation.
8. `interactive_node2vec_embeddings.html`: 2D PCA projection of Node2Vec embedding space.
9. `interactive_llm_payload_gallery.html`: 100% raw cluster payload cards passed to Stage 4 LLMs.
10. `interactive_collapsed_modules.html`: High-level architecture block diagram mapping hub anchors to module nodes.

---

### Stage 4: Synthesis Engine ([run_stage_4.py](file:///Users/david.lobue/Desktop/Agents/Refactor/SemanticPrism/run_stage_4.py))
**Purpose:** To translate mathematical graph partitions back into deployable Python Pydantic ontologies using path-isolated, multi-pass LLM synthesis.

*   **Path Isolation & Schema Pass Control (`config.yaml`):**
    *   `synthesis.execution_mode`: `"community"` (Path 1), `"embedding"` (Path 2), or `"both"`. Outputs schemas into `outputs/schemas/community/` or `outputs/schemas/embedding/`.
    *   `synthesis.schema_pass_mode`: `"normalized"` (Pass A from `refined_triplets.json`), `"raw"` (Pass B from `original_triplets.json`), or `"both"`.
*   **Phase 1: Orphan & Low-Density Node Enum Synthesis (`enums.py`):**
    *   Bundles all isolated singleton nodes AND entity nodes from low-density communities below `min_cluster_size` ($<5$ entities).
    *   Synthesizes them into standardized Python `Enum` classes in `enums.py` in **1 single LLM call**, ensuring **zero data is lost**.
*   **Phase 2a: Global Hub Base Model Synthesis (`hubs.py`):**
    *   Synthesizes `global_hubs` **first** into foundational Pydantic `BaseModel` classes using Phase 1's `enums.py` context.
*   **Phase 2b: Qualified Community Schema Generation (`01_<module_name>.py`):**
    *   Processes qualified clusters ($\ge \text{min\_cluster\_size}$ nodes) using `leiden_schema_agent` or `node2vec_schema_agent`.
    *   Injects both `global_enums` AND `global_hubs` into system prompt context (`deps`).
    *   Community schemas synthesize Pydantic classes that **inherit from** or **reference** root models in `hubs.py` as typed fields.
*   **Phase 3: Global Consolidation:**
    *   Combines `enums.py`, `hubs.py`, and community module files into a unified `master_ontology.py` using `consolidation_agent` with token estimation and hierarchical sub-batching.
*   **Phase 4: Comprehensive Ontology & Code Formatting:**
    *   Synthesizes `comprehensive_ontology.py` merging raw and normalized master ontologies, and runs `ruff format` on all generated SDK files.

---

## Configuration Parameter Guide (`config.yaml`)

The entire execution of SemanticPrism is parameterized through [config.yaml](file:///Users/david.lobue/Desktop/Agents/Refactor/SemanticPrism/config.yaml):

| Config Section | Parameter | Type | Default | Description & Selection Guidance |
| :--- | :--- | :--- | :--- | :--- |
| **`pipeline`** | `use_async` | `boolean` | `true` | When enabled, API calls to the LLM (in Stages 2 and 4) are fired concurrently. |
| **`pipeline`** | `resume_mode` | `string` | `"skip"` | `"skip"` resumes from existing on-disk stage outputs; `"overwrite"` clears previous runs. |
| **`directories`**| `inputs` | `string` | `"inputs/testdocs"` | Directory path containing raw text source documents (`.txt` or `.md`). |
| **`directories`**| `outputs` | `string` | `"outputs"` | Base directory path where all stage outputs, logs, and schemas are saved. |
| **`llm`** | `provider` | `string` | `"ollama"` | Model provider host (e.g., `"ollama"`, `"google"`, `"openai"`). |
| **`llm`** | `model_name` | `string` | `"gemma4:26b"` | Model identifier on provider host. |
| **`llm`** | `temperature` | `float` | `0.0` | Controls LLM creativity. Keep at `0.0` for maximum determinism. |
| **`llm`** | `manage_vram` | `boolean` | `true` | Purges system VRAM between stages to prevent Out-Of-Memory (OOM) GPU errors. |
| **`extraction`** | `domain` | `string` | `"General Subjects"`| Context domain setting given to Stage 1 extraction agents. |
| **`refinement`** | `clustering_threshold`| `float` | `0.6` | Agglomerative Clustering cosine distance cutoff for term deduplication. |
| **`topology`** | `execution_mode` | `string` | `"both"` | Stage 3 partition execution mode (`"community"`, `"embedding"`, or `"both"`). |
| **`topology`** | `community_path.participation_threshold` | `float` | `0.65` | Participation cutoff ($P_i$) for Path 1 cross-community hubs. |
| **`topology`** | `community_path.betweenness_percentile` | `float` | `0.95` | Betweenness percentile cutoff for Path 1 narrative chokepoints. |
| **`topology`** | `embedding_path.participation_threshold` | `float` | `0.45` | Participation cutoff ($P_i$) for Path 2 cross-category hubs. |
| **`topology`** | `embedding_path.enable_modularity_vitality_pruning` | `boolean` | `true` | Prunes boundary-blurring nodes ($\Delta Q < -0.005$) to isolate pure category clusters. |
| **`synthesis`** | `execution_mode` | `string` | `"community"` | Stage 4 topology ingestion mode (`"community"`, `"embedding"`, or `"both"`). |
| **`synthesis`** | `schema_pass_mode` | `string` | `"both"` | Stage 4 pass mode (`"normalized"`, `"raw"`, or `"both"`). |
| **`synthesis`** | `min_cluster_size` | `integer` | `5` | Entity threshold for Stage 4 target qualification ($<5$ routed to `enums.py`). |
| **`synthesis`** | `context_window_cap`| `integer` | `16384` | Context window cap in tokens for schema generation and consolidation. |
