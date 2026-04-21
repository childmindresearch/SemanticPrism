# SemanticPrism: Internal Module Specifications

This document defines the rigid architectural specifications for each isolated sub-directory component (`src/[module_name]`). When reconstructing SemanticPrism, ensure these mathematical rules and dependencies are strictly enforced.

> [!IMPORTANT]
> **Component Strictness:** You must explicitly guarantee you do NOT cross-contaminate logic spaces. If a module is designed as an "Offline Mathematical Component" (e.g., `embedding` or `topology`), it absolutely must not make lazy LLM calls to solve topological graphing issues. Similarly, LLM modules (`extraction`, `synthesis`) must not calculate geometric clustering. Maintain this strict isolation.

---

## 1. ContextManager (`src/helpers/context_manager.py`)
- **What it is:** A dynamic hardware profiler measuring local GPU limitations natively to prevent crashing localized open-source model execution.
- **Specifications:** Calculates system RAM and VRAM availability. Extracts the `max_chunk_words` mathematically (e.g., matching a safe string size against a hard 8K context token limit).  Produces a safe context size to use for dynamic input into LLM API calls.
- **Parameters Required:** 
  - `model_name` (str)

---

## 2. Extraction Pipeline (`src/extraction/extractor.py`)
- **What it is:** The LLM integration layer operating solely on raw text and Instructor-driven JSON mapping. **[Pure LLM Process]**
- **Specifications:**
  - **Dependencies:** `ollama`, `instructor`, `Pydantic`.
  - Must asynchronously execute two primary chains using Instructor Pydantic mode:
    1. **Theme Discovery** (`ThemeDiscoveryResult`): A dual pass through a LLM to extract overarching domain topics and second pass to consolidate themes into smaller focused set and a single overarching master domain `MasterThemeSynthesisResult`.
    2. **Triple Extraction** (`TripleExtractionResult`): Extracts logical facts mapped as `RawTriple` (Subject, Predicate, Object, Quote, Confidence, Theme) using SemanticChunker blocks bounded by the ContextManager.
    3. LLM Preprocessing Normalization:  Second pass specifically focused on triples, replacing synonyms sequentially.

---

## 3. Embedding Pipeline (`src/embedding/embedding.py`)
- **What it is:** The mathematical entity distillation engine separating string overlap.  **[Pure Offline Process - No LLM]**
- **Specifications:**
  - **Dependencies:** `sentence-transformers`, `scikit-learn`, `numpy`.
  - **Subprocesses:**
    - **A. Text Embedding**: Performs offline embedding using weights downloaded from transformers huggingface. Using the grouped triplets from the extraction pipeline, 3 groups of embeddings are produced (Subjects, Predicates, Objects).  Each group is treated in isolation for the following PCA and clustering tasks. The embedding model is a parameter set in the config file. 
    - **B. Spectral Decomposition**: Performs PCA via `sklearn.decomposition.PCA` on string embeddings, utilizing a dynamically weighted covariance matrix focused on historically frequent terms in the graph. This transforms Euclidean space to prioritize common concepts.
    - **C. Agglomerative Clustering**: Uses `AgglomerativeClustering(metric='cosine', linkage='average')`. 

---

## 4. Hypernym Lifting Pipeline (`src/nlp/hypernyms.py`)
- **What it is:** The conceptual abstraction and entity normalization layer. **[Hybrid Process: 90% Offline Math, 10% LLM Verification]** 
- **Specifications:**
  - **Dependencies:** `ollama`, `instructor`, `Pydantic`.
  - **Subprocesses:**
    - **A. Context Vector Validation**: Provides the identified clusters for each group generated in step 3 back to the LLM individually.  The intent is to ensure clusters of similar words (within each group) are semantically similar in meaning.  If the LLM returns `accuracy_destroyed=True`, the engine splits members back into safely separated individual nodes.
    - **B. Semantic Centering**: For "approved" clusters of previous step, uses cosine centers (`np.argmin`) to find the most "geometrically neutral" label representing the cluster.
    - **C. Taxonomic Lifting**: Passes semantic center identified along with full cluster nodes to the LLM to deduce the "Hypernym" (Formal Is-A relation classification).  Only clusters are sent to the LLM, all other individual text items items are left "as is"


---

## 5. Naming Resolution and Consolidation (`src/nlp/nlp_mapping.py`)
- **What it is:** The organizational layer consolidating the hypernyms extracted with original source triples. **[100% Offline Process]** 
- **Specifications:**
  - **Subprocesses:**
    - **A. NLP Mapping**: Map the taxonomic lifted hypernyms to the original triplets extracted for each subject, object, and predicate.
    - **B. Semantic Replacement**: Where a taxonomic hypernym was identified, replace the original text with the abstracted replacement.  Create a final list of all triples comprised of those with updated hypernyms as well as those that did not have NLP extracted naming assignments (original text).

---

## 6. Topology Pipeline (`src/topology/processor.py`)
- **What it is:** The strict Graph Theory construction framework using `NetworkX`.  Use edges and nodes from the final Semantic Replacement process of step 5 to create graph representation of triples (subjects, objects as nodes and predicates as edges) **[100% Offline Deterministic Math]**
- **Specifications:**
  - **Dependencies:** `networkx`, `scikit-learn`, `community-louvain` or `cdlib`, `pyvis` (HTML output).
  - **Subprocesses:**
    - **A. Edge Weight Setting**: Replaces standard boolean edges with weights derived via inverse-frequency logarithmic distribution based on how often a Predicate historically occurs natively.
    - **B. Spectral Eigengap Modularity**: Using default `spectral` detection, extracts the Normalized Laplacian Matrix, calculates Eigenvalues, measures the `np.diff` gaps between them, and splits the network communities bounded on the structurally largest Eigengap limit (preventing chaotic fragmentation). This is a parameter set in the config file. 
    - **C. Structural Overlap (Jaccard)**: Measures predicate logic Jaccard similarity across independent communities. If structural overlap exceeds 75% (`>= 0.75`), both communities merge.  This is a parameter set in the config file. 
    - **D. Role Geometry**: Measures topological in-degrees and out-degrees. If entirely outbound (`in==0, out>0`), Node = `RootEntity`. If bridged (`in>0, out>0`), Node = `NestedEntity`. If dead-end (`in>0, out==0`), Node = `TerminalAttribute`.
    - **E. Community Characteristics**: For each community that is not an isolated triple (isolate triples only connection is the original subject-predicate-object from the original extract), perform core graph diagnostics including density, degree, cohesion, k-cores, transitivity, edge redundancy and cyclomatic complexity to save as outputs. 


---

## 7. Semantic Reconstruction (`src/nlp/graph_to_text.py`)
- **What it is:** The organizational layer processing the distinctions identified in the graph topology to a semantic text representation. **[100% Offline Process]** 
- **Specifications:**
  - **Subprocesses:**
    - **A. NLP Mapping**: Isolate all independent communities that are not isolated triples into lists of triples.  For all isolated triples that exist outside of a community, merge these into a single list.  Store to disk for reference. 
    - **B. Structural Alignment**: For each community (excluding the isolated triples), convert the graph representation to a nested hierarchy (for example, [child]-[is a]-[patient]-[was administered]-[cognitive test]-[received score]-[85% accuracy] would have a depth level of 4 child-patient-cognitive test-85% accuracy).  Save to disk.
    Packages specific `NetworkX` Graph cluster sub-components explicitly into JSON arrays (`root_classes`, `nested_bridges`, `attributes`).
    

---

## 8. Synthesis Pipeline (`src/synthesis/synthesizer.py`)
- **What it is:** Syntactic code generation mapped precisely off the Community Topology geometry. **[Pure LLM Process]**
- **Specifications:**
  - Passes the organized hierarchical dictionaries for processing via LLM.  Include original full list of all themes extracted for context.
  - Generates executable raw Python strings mapping natively to Pydantic objects preserving relationships.
  - Joins all sub-routines into a cohesive master `outputs/schemas/schemas.py` document.
