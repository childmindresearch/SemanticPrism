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

### Stage 3: Topology ([run_stage_3.py](file:///Users/david.lobue/Desktop/Agents/Refactor/SemanticPrism/run_stage_3.py))
**Purpose:** To map the refined triplets into a network graph and discover structural properties, communities, and hierarchies.

*   **Network Graph Construction:** Constructs directed and undirected graphs using NetworkX from the refined S-P-O triplets.
*   **Spectral Centrality (Hub & Orphan Identification):** Calculates PageRank and Degree Centrality scores.
    *   *Underlying Detail (Prestige & Influence):* PageRank simulates random walks across the graph to measure structural prestige. Nodes in the top $N\%$ of PageRank scores (where $N$ is configured by `spectral_variance_retention`, default `0.05` or 5%) are designated as **Global Hubs** (e.g., *"patient"* or *"diagnosis"*). Nodes with degree centrality $\le 1$ that are not hubs are isolated as **Orphans** (leaf values).
*   **Graph Pruning:** Temporarily removes hubs and orphans from the graph prior to clustering.
    *   *Underlying Detail (Reducing Structural Blur):* Hubs act as highly connected "hairballs" that smear boundaries, while orphans act as disconnected noise. Removing them temporarily allows clustering algorithms to identify clear topological borders.
*   **Leiden Modularity Community Detection:** Detects densely connected communities in the pruned subgraph.
    *   *Underlying Detail (Workflow & Event Clustering):* Modularity clustering groups nodes that have dense internal links but sparse external links. Since edges represent semantic associations, these communities discover **procedural workflows and event sequences** (e.g., *symptom* $\to$ *assessment* $\to$ *diagnoses*).
*   **Node2Vec Structural Equivalence:** Runs Node2Vec random walks on the pruned graph, trains embeddings using a skip-gram model, and clusters the vectors via K-Means.
    *   *Underlying Detail (Role & Category Identification):* Unlike modularity, Node2Vec clusters nodes that play similar structural roles in the network (e.g., grouping all *psychometric tests* or all *intervention types*), even if they are in completely different parts of the graph and do not connect. The optimal number of clusters ($K$) is determined dynamically using the Silhouette Score.
*   **Hypergraph Theme Inheritance:** Measures hypergraph containment (overlap scores) between the sets of nodes associated with different themes to determine child-parent inheritance.
*   **Output Files:**
    *   `outputs/03_topology/topology_partitions.json`: Full serialization of structural clusters, communities, hubs, orphans, node metrics, and theme inheritance.
    *   `outputs/visuals/`: Interactive HTML files for graph exploration (Standard, Hypergraph, Communities Only, Collapsed Communities, Dual Perspective, Node2Vec 2D PCA Space, Role-Based Network).

---

### Stage 4: Synthesis ([run_stage_4.py](file:///Users/david.lobue/Desktop/Agents/Refactor/SemanticPrism/run_stage_4.py))
**Purpose:** To translate mathematical graph partitions back into formatted, deployable Python Pydantic ontologies.

*   **Phase 1: Orphan Aggregation (Enums):** Bundles all isolated "Orphan" nodes (degree $\le 1$) into Python `Enum` and `Literal` structures.
    *   *Underlying Detail (Strict Data Typing):* Ensures highly specific, static values are mapped as strict types in the schema (e.g., specific test accommodations).
*   **Phase 2: Dual-Pass Schema Generation:** Processes the target clusters (Leiden or Node2Vec) using LLM agents to generate target schemas in two modes:
    *   *Pass A (Normalized):* Generates Python Pydantic classes representing clean, taxonomically lifted relationships.
    *   *Pass B (Raw):* Generates models mapping to the raw, detailed source triplet terms.
*   **Phase 3: Global Consolidation:** Combines individual cluster python files in both normalized and raw directories into a single unified `master_ontology.py` using a consolidation agent.
    *   *Underlying Detail (Hierarchical Batching):* If the combined code length exceeds the configured `context_window_cap`, the pipeline automatically strips comments/docstrings. If it is still too large, it performs hierarchical batching, merging smaller subsets into intermediate code blocks before a final merge.
*   **Phase 4: Comprehensive Ontology:** Integrates raw master schemas, normalized master schemas, and the generated enums into a final standalone package.
*   **Finalization:** Automatically runs the `ruff` formatter tool on all outputs to ensure PEP-8 compliance.
*   **Output Files:**
    *   `outputs/schemas/normalized/`: Individual schemas and `master_ontology.py`.
    *   `outputs/schemas/raw/`: Individual schemas and `master_ontology.py`.
    *   `outputs/schemas/comprehensive_ontology.py`: The final, complete, deployable SDK.

---

## Configuration Parameter Guide (`config.yaml`)

The entire execution of SemanticPrism is parameterized through [config.yaml](file:///Users/david.lobue/Desktop/Agents/Refactor/SemanticPrism/config.yaml). Below is an explanation of every key parameter:

| Config Section | Parameter | Type | Default | Description & Selection Guidance |
| :--- | :--- | :--- | :--- | :--- |
| **`pipeline`** | `use_async` | `boolean` | `true` | When enabled, API calls to the LLM (in Stages 2 and 4) are fired concurrently. **Select `true`** for high performance, but **select `false`** if you face rate limits or memory bottlenecks on your local LLM server. |
| **`pipeline`** | `resume_mode` | `string` | `"skip"` | Determines execution behavior on Stage 1. `"skip"` will check if a document's extracted themes and triplets exist on disk and skip processing it, allowing incremental resumption. `"overwrite"` will clear and re-process all documents. |
| **`directories`**| `inputs` | `string` | `"inputs/testdocs"` | Directory path containing raw text source documents (`.txt` or `.md`). |
| **`directories`**| `outputs` | `string` | `"outputs"` | Base directory path where all stage outputs, logs, and schemas are saved. |
| **`ingestion`** | `source_type` | `string` | `"directory"` | Ingestion source. Use `"directory"` to ingest all individual text documents in the inputs folder. Use `"parquet"` to load from a structured parquet dataset. |
| **`ingestion`** | `parquet.filename` | `string` | - | Filename of the parquet database file to load. |
| **`ingestion`** | `parquet.id_field` | `string` | - | Column name in the parquet file representing the document's unique ID. |
| **`ingestion`** | `parquet.text_field`| `string` | - | Column name in the parquet file containing the document's raw text content. |
| **`llm`** | `provider` | `string` | `"ollama"` | Under global LLM config. Set your model host provider (e.g., `"ollama"`, `"google"`, `"openai"`). |
| **`llm`** | `model_name` | `string` | - | Model identifier on the provider (e.g., `"gemma4:26b"`, `"gemini-1.5-pro"`). |
| **`llm`** | `temperature` | `float` | `0.0` | Controls LLM response creativity. Keep at `0.0` to maximize determinism and factual accuracy. |
| **`llm`** | `manage_vram` | `boolean` | `true` | If enabled, purges the system VRAM between stages. **Select `true`** when hosting local models on a consumer GPU to prevent Out-Of-Memory (OOM) errors. |
| **`extraction`** | `domain` | `string` | `"General Subjects"`| Context domain setting given to Stage 1 extraction agents (e.g., `"Clinical Diagnostics"`). |
| **`extraction`** | `max_async_calls` | `integer` | `2` | Number of concurrent API worker tasks in Stage 1 extraction. |
| **`extraction`** | `theme_chunk_max_words` | `integer` | `3500` | Word limit per chunk for Stage 1 theme discovery. |
| **`extraction`** | `triple_chunk_max_words` | `integer` | `1200` | Word limit per chunk for Stage 1 triple extraction. |
| **`extraction`** | `context_window_cap`| `integer` | `8192` | Input token limit to optimize KV cache in triple extraction. |
| **`refinement`** | `max_async_calls` | `integer` | `6` | Number of concurrent batch worker tasks in Stage 2 normalization. |
| **`refinement`** | `batch_size` | `integer` | `25` | Number of terms packaged in a single LLM normalization call. |
| **`refinement`** | `timeout` | `float` | `300.0` | Timeout limit in seconds for a single refinement LLM call. Set to `0` to disable timeouts. |
| **`refinement`** | `embedding_model` | `string` | `"all-MiniLM-L6-v2"`| SentenceTransformer model used to calculate dense vector embeddings. |
| **`refinement`** | `clustering_threshold`| `float` | `0.6` | The Agglomerative Clustering cosine distance cutoff. **Selection Guidance:** A **lower value (e.g., 0.4)** creates tighter, highly precise clusters, leading to granular schemas. A **higher value (e.g., 0.8)** merges synonyms aggressively, leading to broader, higher-level abstract schemas. |
| **`refinement`** | `context_window_cap`| `integer` | `2048` | Context window cap in tokens for term normalization. |
| **`topology`** | `max_structural_clusters`| `integer` | `10` | The maximum structural clusters bounds searched by K-Means. |
| **`topology`** | `spectral_variance_retention` | `float` | `0.05` | PageRank percentile cutoff defining what proportion of top nodes are isolated as **Global Hubs**. **Selection Guidance:** A **lower value (e.g., 0.02)** isolates only the absolute core concepts (top 2%), whereas a **higher value (e.g., 0.10)** marks more nodes (top 10%) as hubs. |
| **`synthesis`** | `max_async_calls` | `integer` | `3` | Number of concurrent worker tasks in Stage 4 schema generation. |
| **`synthesis`** | `clustering_strategy`| `string` | `"node2vec"`| The mathematical clustering method used to partition the graph for schema generation. Options: `"leiden"` or `"node2vec"`. **(See Detailed Selection Guidance Below)** |
| **`synthesis`** | `min_cluster_size` | `integer` | `3` | The minimum size (node count) a cluster/community must have to qualify for target schema generation. Small clusters are ignored or treated as minor leaf nodes. |
| **`synthesis`** | `context_window_cap`| `integer` | `16384` | Max input context tokens allowed for schema consolidation. Must be large enough to prevent code truncations during mergers. |

---

## Detailed Selection Guidance: `leiden` vs `node2vec`

The setting `synthesis.clustering_strategy` in `config.yaml` controls how the graph structure is partitioned before being converted into Python classes. Choosing the wrong strategy will yield schemas that do not fit your architecture.

### Option A: `"leiden"` (Community Modularity Detection)
*   **What it does:** Leiden modularity groups nodes based on direct local connectivity and link density. Nodes that are linked directly or share sequential edges with each other are grouped together.
*   **Resulting Schema Type:** Event-driven, procedural **Workflows and Sequences**.
*   **When to select:** 
    *   Choose `"leiden"` when your domain knowledge represents step-by-step processes, state transitions, or chronologies.
    *   *Example (Clinical Pathway):* A pipeline tracking `Referral` $\to$ `Clinical Interview` $\to$ `WISC-V Assessment` $\to$ `Diagnosed Dyscalculia` $\to$ `Extra Time Accommodations`. These nodes will form a single Leiden community because they occur in chronological sequence.
*   **Pros:** Captures procedural linkages, data flows, and end-to-end execution paths.
*   **Cons:** Can result in highly coupled, heterogeneous schemas (e.g., a single class containing symptoms, tests, and outcomes mixed together).

### Option B: `"node2vec"` (Structural Equivalence Clustering)
*   **What it does:** Node2Vec uses structural random walks to represent node neighborhoods. Nodes are clustered if they share similar network connection patterns (roles), **even if they are in completely separate parts of the graph and never connect directly**.
*   **Resulting Schema Type:** Decoupled, formal **Logical Categories (Taxonomies)**.
*   **When to select:** 
    *   Choose `"node2vec"` when you want to build a cleanly segregated, modular object hierarchy where components are categorized by their nature.
    *   *Example (Taxonomy):* Groups all *psychometric tests* (WISC-V, WAIS, NEPSY) into a `DiagnosticTools` cluster, and all *accommodations* (Extra Time, Visual Aids, Quiet Room) into an `Accommodations` cluster. Although "WISC-V" and "Extra Time" never connect to each other directly in the graph, they connect in identical patterns to other nodes, making them structurally equivalent.
*   **Pros:** Generates perfectly decoupled, reusable libraries that follow clean object-oriented design principles.
*   **Cons:** Loses sequential/procedural context (e.g., it will not tell you which diagnostic tool leads to which specific accommodation).
