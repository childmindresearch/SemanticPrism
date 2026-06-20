# SemanticPrism

SemanticPrism is an advanced, autonomous agentic pipeline designed to process unstructured, highly complex domain knowledge (such as clinical diagnostic texts) and mathematically synthesize it into structured, deployable Python Ontologies (Pydantic models).

The purpose of SemanticPrism is to solve the "hallucination and overlap" problem inherent in standard Large Language Model (LLM) schema generation. Rather than asking an LLM to guess the structure of a document in one shot, SemanticPrism breaks the text down into mathematical graph networks, clusters those networks using advanced algorithms, and uses those mathematical boundaries to generate perfectly decoupled, high-fidelity data models.

## Pipeline Architecture

The pipeline is driven by configuration settings in `config.yaml` and executes across four distinct, sequential stages.

---

### Stage 1: Extraction (`src/extraction/extractor.py`)
**Purpose:** To ingest unstructured source texts and convert them into granular, mathematical relationships.

- **Theme Discovery:** The pipeline first scans the entire corpus to identify "Master Themes" (e.g., Symptomology, Interventions).
  - *Why:* This anchors the LLM, providing a conceptual boundary so it doesn't drift when extracting data from massive documents.
- **Triplet Extraction:** Utilizes Pydantic-AI to comb through the text and extract Subject-Predicate-Object (S-V-O) triplets.
  - *Why:* This translates dense narrative prose into discrete mathematical edges that can be plotted on a graph.

---

### Stage 2: Refinement (`src/refinement/refiner.py`)
**Purpose:** To clean, deduplicate, and normalize the raw, noisy triplets generated in Stage 1 into a standardized taxonomy.

- **Lexical Normalization:** Uses an LLM to correct spelling, expand acronyms, and standardize phrasing.
  - *Why:* Ensures that variations like "WISC-5" and "wisc-v" are mapped to the exact same string to prevent graph fragmentation.
- **Taxonomic Lifting (Agglomerative Clustering):** Leverages `SentenceTransformers` (`all-MiniLM-L6-v2`) to generate vector embeddings of every entity, grouping them hierarchically based on a configurable `clustering_threshold`.
  - *Why:* This mathematically proves that "severe anxiety" and "extreme worry" mean the same thing, allowing us to map them to a single root concept.
- **Triple Remapping:** The original triplets are overwritten using the newly established taxonomic root concepts.

---

### Stage 3: Topology (`src/topology/graph_builder.py`)
**Purpose:** To map the normalized triplets into a `NetworkX` graph to discover hidden structural patterns, role groups, and communities.

#### Detailed Approach
1.  **Network Graph Construction**: Ingests the normalized S-V-O triplets from Stage 2 and constructs directed and undirected graphs using NetworkX.
2.  **Spectral Centrality (Hubs & Orphans)**: Computes PageRank and Degree Centrality scores. The top $N\%$ of nodes sorted by PageRank (where $N$ is determined by the `spectral_variance_retention` parameter in `config.yaml`, default `0.05` or 5%) are isolated as **Global Hubs**. Nodes with degree $\le 1$ that are not hubs are isolated as **Orphans**.
3.  **Graph Pruning**: Hubs and Orphans are temporarily removed from the clustering subgraph. This prevents highly central "hairballs" and isolated "leafs" from blurring cluster boundaries.
4.  **Leiden Modularity Community Detection**: Runs the Leiden modularity clustering algorithm on the pruned subgraph to group nodes based on local connectivity and proximity.
5.  **Node2Vec Structural Equivalence**: Embeds the pruned graph structure into a 64-dimensional vector space using random walks. The pipeline then dynamically searches for the optimal number of clusters ($K$) using the Silhouette Score and groups the nodes using K-Means.
6.  **Theme Inheritance**: Evaluates hypergraph overlap scores between the semantic theme node sets to determine child-parent relationships.

#### Output File (`outputs/03_topology/topology_partitions.json`)
The pipeline serializes the `TopologyResult` Pydantic model into this file. It contains:
*   **`global_hubs`** (List of Strings): Core central concepts.
*   **`orphans`** (List of Strings): Low-connectivity leaf concepts.
*   **`communities`** (List of CommunityPartition): Leiden modularity community partitions.
*   **`structural_clusters`** (List of StructuralCluster): Node2Vec K-Means cluster partitions.
*   **`node_metrics`** (Dict of Node ID to NodeMetrics): Centrality scores, hub/orphan status, and directed edge collections for each node.
*   **`theme_inheritance`** (List of ThemeInheritance): Theme overlap scores representing inheritance hierarchies.

#### What the Results Indicate
*   **Global Hubs**: Indicate core domain definitions or master entities (e.g., `patient` or `therapy`). They act as the primary structural routing nodes in the domain.
*   **Orphans**: Indicate highly specific details, individual values, or leaf nodes (e.g., a specific psychometric score or a localized behavior) that do not connect widely.
*   **Leiden Communities**: Discover procedural or event-driven **Clinical Workflows** (nodes grouped because they are connected sequentially, like clinical symptoms leading to a test, leading to a diagnosis, leading to an intervention).
*   **Node2Vec Clusters**: Discover functional **Ontological Categories** (nodes grouped because they play similar roles in the network structure, such as grouping all diagnostic instruments together, or all symptoms together, regardless of whether they connect directly).

---

---

### Stage 4: Synthesis (`src/synthesis/synthesizer.py`)
**Purpose:** To translate the mathematical graph partitions generated in Stage 3 back into fully formatted, deployable Python code. 

- **Phase 1: Orphan Aggregation (Enums):** Bundles all the isolated "Orphan" nodes into Python `Enum` and `Literal` classes.
  - *Why:* This creates standardized, highly specific data types for our eventual schemas (e.g., a dropdown list of specific accommodations).
- **Phase 2: Dual-Pass Schema Generation:** Loops through the graph targets (configurable to use either Leiden Communities or Node2Vec Clusters via `config.yaml`) and instructs highly-tuned Pydantic-AI agents to generate schemas for each cluster independently. It does this for both raw and normalized data.
  - *Why:* By generating schemas strictly within mathematical boundaries, we prevent the LLM from hallucinating relationships that don't exist in the data.
- **Phase 3: Global Consolidation:** Bundles the fragmented schemas and passes them to a Master Ontologist agent to merge overlapping classes and standardize inheritance structures.
  - *Why:* To mathematically deduplicate the data and ensure a clean schema inheritance tree.
- **Phase 4: Comprehensive Ontology:** Merges the raw master schemas, the normalized master schemas, and the Enums into one final file.
  - *Why:* To produce `comprehensive_ontology.py`—a 100% portable, standalone SDK that can be handed directly to an extraction LLM out of the box.

## Interactive Visualizations (Stage 3)

The Stage 3 Topology pipeline generates 7 interactive HTML visualization files under `outputs/visuals/` to explore community and structural patterns in the domain knowledge graph:

1.  **Standard Topology Visual** (`interactive_topology_graph.html`)
    *   *Description:* A complete rendering of all Subject-Predicate-Object relationships in the knowledge network.
    *   *Details Included:* Nodes are colored by their Leiden community partition. Global hubs are styled as **red stars**, orphans as **gray circles**, and standard entities as standard circles. Tooltips display node roles (Hub, Orphan, or Entity), Leiden community ID, and degree centrality.
2.  **Hypergraph Visual** (`interactive_hypergraph.html`)
    *   *Description:* A macro-representation showing high-level semantic theme associations and their inheritance patterns.
    *   *Details Included:* Blue star-shaped nodes represent discovered themes (hyperedges), connected by thin lines to their member entity nodes. Directional arrows between theme nodes represent parent-child theme inheritance hierarchies, with the overlap score displayed on the edges.
3.  **Isolated Communities** (`interactive_communities_only.html`)
    *   *Description:* A clean, isolated layout of Leiden communities (representing workflows/events) with inter-community noise removed.
    *   *Details Included:* Only intra-community edges are rendered. The top 3 central nodes in each community are highlighted as **stars with thick borders** (core representatives), while other nodes are standard circles. Tooltips display community role and a summary of the community's core representative terms.
4.  **Collapsed Communities** (`interactive_collapsed_communities.html`)
    *   *Description:* A high-level macro view that collapses entire Leiden communities (filtered to size >= 3) into single nodes.
    *   *Details Included:* Node size is scaled by the number of entities in that community. The labels display the community ID and lists the mapped master themes from `theme_mapping_clusters.json` along with deduplicated counts showing how many times each theme is represented by nodes in that community. Tooltips show total node count, representative terms, and the detailed theme breakdowns.
5.  **Dual Perspective** (`interactive_dual_perspective.html`)
    *   *Description:* A combined visualization overlaying Leiden modularity communities with Node2Vec structural equivalence clusters.
    *   *Details Included:* Colors map to Leiden community IDs (representing workflows), while node shapes map to Node2Vec cluster IDs (representing structural roles: `dot`, `square`, `triangle`, `diamond`, `star`, etc.). Tooltips show node name, Leiden community ID, Node2Vec cluster shape ID, and PageRank centrality. Renders all edges between the community nodes.
6.  **Node2Vec 2D Embedding Space** (`interactive_node2vec_embeddings.html`)
    *   *Description:* An interactive Plotly 2D scatter plot representing the high-dimensional node embeddings in vector space.
    *   *Details Included:* Maps the 64-dimensional Node2Vec vectors to a 2D coordinate grid using PCA. Points represent nodes, colored by their K-Means structural cluster assignment. Hover tooltips show node names.
7.  **Role-Based Network Graph** (`interactive_structural_clusters_only.html`)
    *   *Description:* A PyVis network graph showing structural equivalences and roles by isolating Node2Vec cluster partitions.
    *   *Details Included:* Nodes are colored by their Node2Vec cluster assignment. Core representative nodes (highest PageRank within the cluster) are styled as stars, and cluster members as dots. Only intra-cluster edges are rendered with customized spacing physics.

## Getting Started & Configuration

SemanticPrism is designed to be highly modular. Before running the pipeline, you must configure your environment and settings.

### 1. Environment Setup
1. Copy the `.env.example` file (if provided) or create a new `.env` file in the root directory.
2. Add your LLM API keys:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```
*(Note: Do not commit the `.env` file to version control. It is ignored by `.gitignore`.)*

### 2. Tuning `config.yaml`
The entire behavior of the pipeline is controlled via the central `config.yaml` file. No Python code needs to be edited to change the pipeline's behavior.

**Key Configuration Parameters to Understand:**

*   **`extraction.domain`**: 
    *   *What it is:* The overarching context given to the extraction LLM (e.g., "Clinical Diagnostics", "Legal Contracts"). 
    *   *Why it matters:* This prevents the LLM from drifting out-of-context when dealing with ambiguous terms in your source documents.
*   **`refinement.clustering_threshold` (Agglomerative Clustering)**:
    *   *What it is:* A decimal value (e.g., `0.5`) that determines how aggressively the script merges similar synonyms. 
    *   *Why it matters:* A lower value (`0.4`) results in very precise, granular schemas. A higher value (`0.8`) forces broad, high-level abstractions by aggressively merging related terms into single entities.
*   **`topology.max_structural_clusters` (Node2Vec)**:
    *   *What it is:* The maximum number of clusters the K-Means algorithm is allowed to search through.
    *   *Why it matters:* The algorithm automatically calculates the optimal number of groups using the Silhouette Score, but this bounds the processing time for massive graphs.
*   **`synthesis.clustering_strategy` (Final Output Target)**:
    *   *What it is:* Accepts either `"leiden"` or `"node2vec"`.
    *   *Why it matters:* This fundamentally changes the final Pydantic output. Set to `"leiden"` to generate schemas representing procedural Workflows and Events (e.g., an assessment timeline). Set to `"node2vec"` to generate schemas representing pure, decoupled Ontological Categories (e.g., a list of distinct symptoms).

### Important Considerations for Users
- **Data Placement:** Place all your unstructured `.txt` or `.md` files into the `inputs/testdocs/` directory. The pipeline will read all files in this folder automatically. 
- **VRAM/Memory Management:** If you are running local models (like Ollama), ensure `manage_vram: true` is set in the config. The pipeline is designed to purge memory between stages to prevent memory leaks during massive batch processing.
- **Iterative Execution:** You do not need to run the entire pipeline at once. If you want to tweak the graph clustering, you only need to re-run Stage 3 (`python3 run_stage_3.py`) and Stage 4. You do not need to re-run the expensive Stage 1 text extraction!
