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
**Purpose:** To map the normalized triplets into a `NetworkX` graph to discover hidden structural patterns and groupings.

- **Spectral Centrality:** Calculates PageRank and Degree Centrality to isolate "Global Hubs" (core concepts) and "Orphans" (highly specific edge cases).
  - *Why:* Removing Hubs and Orphans prevents the graph from collapsing into one massive hairball, allowing distinct clusters to emerge.
- **Leiden Modularity (Community Detection):** Groups nodes based on proximity and connectivity (who talks to whom).
  - *Why:* This clustering strategy successfully discovers chronological or procedural **Clinical Workflows** (e.g., tying a Symptom to an Assessment to an Intervention).
- **Node2Vec Structural Equivalence:** Generates random walks to map the graph into vector embeddings, then dynamically uses the Silhouette Score to group nodes into K-Means clusters based on their *role* in the network.
  - *Why:* This clustering strategy mathematically discovers pure **Ontological Categories** (e.g., grouping all Assessments together, all Symptoms together) regardless of whether they appear in the same sentence.

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
