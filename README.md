# SemanticPrism: Project Overview and Architecture

## Overall Summary
SemanticPrism is a knowledge graph ontology extractor pipeline that processes multiple raw text documents to construct a mathematical, hierarchical semantic topology. The system translates unstructured text into structured, directed graphs and synthesizes verifiable Python models (Pydantic schemas) representing the isolated ontological communities. 

The pipeline is split into explicit modular stages: Extraction, Syntactic Normalization, Embedding, Hypernym mapping, Topology construction, and Synthesis. Each stage operates strictly on bounded inputs to ensure deterministic behavior, focusing on accuracy, abstraction, and ontology. To ensure robust fault tolerance, the pipeline natively employs **Iterative Diagnostic Logging**, sequentially persisting analytical metrics and runtime traces to disk after every phase.

## Pipeline Architecture and Components

### 1. Extraction Pipeline (`extractor.py`)
**Purpose**: Transforms unstructured text matrices into formal subject-predicate-object triples and discovers overarching domain themes.
**Components**:
- **Multi-Document Theme Consolidation**: Ingests multiple text documents, extracts local themes, weights them by cross-document frequency, and synthesizes a singular master domain context.
- **Triple Extraction**: Extracts logical raw triples using an entity registry to maintain coreference context.

### 2. Syntactic Normalization (`normalize_text.py`)
**Purpose**: Binds raw string tokens to deterministically scrubbed states prior to embedding logic.
**Components**:
- **Explicit Lexical Normalization**: Utilizes constrained JSON decoding to construct strict 1:1 key-value mappings linking raw subject/predicate/object tokens directly to their normalized counterparts without relying on fragile list indexing.

### 3. Embedding Pipeline (`embedding.py`)
**Purpose**: Projects raw string fields into a vector space and reduces redundancy by clustering semantically identical components.
**Components**:
- **Vector Encoding**: Generates embeddings for extracted subjects, objects, and predicates.
- **Weighted PCA**: Duplicates embedding rows by item frequency and performs dimensionality reduction.
- **Agglomerative Clustering**: Groups embeddings mathematically to identify semantic clusters.

### 4. Hypernym Pipeline (`hypernyms.py`)
**Purpose**: Merges mathematical centroids with validation protocols to enforce hierarchical taxonomic structures.
**Components**:
- **Contextual Validation**: Evaluates the coherence of proposed mathematical clusters and splits rejected clusters.
- **Geometric Centroid Calculation**: Computes the central vector for a cluster and maps it to the closest string.
- **Taxonomic Lifting**: Assigns a formal hypernym label to represent the cluster via Chain-of-Thought reasoning.

### 5. Topology Engine (`graph_builder.py`)
**Purpose**: Constructs mathematically defined graphs, partitions them into modular semantic communities, and builds n-ary hypergraph representations for spectral analysis.
**Components**:
- **Directed Graph Construction**: Builds a NetworkX `DiGraph` from the normalized triples, tracking cumulative edge weights and predicate sets.
- **N-ary Hypergraph Grouping**: Groups triples around their `theme_association`, securely tracking local neighborhoods (Identity Guard) to map a bipartite graph connecting entities to thematic hyperedges.
- **Spectral Matrices**: Computes the hypergraph Incidence Matrix ($H$) and Laplacian ($L$) via `numpy` to map high-level entity-theme interactions mathematically.
- **Leiden Community Detection**: Computes modularity partitions for the directed graphs and runs Louvain detection on the bipartite structure for high-fidelity Pyvis visualization outputs.
- **Hierarchy Extraction**: Restructures the graph into isolated subgraphs representing distinct semantic communities.

### 6. Synthesis Engine (`synthesizer.py`)
**Purpose**: Transforms abstract network communities into structured, executable Pydantic schemas.
**Components**:
- **Schema Generation**: Converts graph edges and nodes of each community into programmatic Python class abstractions.
- **Output Export**: Aggregates models and writes the global JSON context (`semantic_prism_master_graph.json`) and executable Python file (`semantic_models.py`).

## LLM vs. Offline Computation

The system strictly divides non-deterministic interpretation (LLM) and deterministic mathematics (Offline computation).

**LLM-Reliant Operations:**
- **Theme and Triple Extraction**: Interpreting raw text to structured models.
- **Lexical String Normalization**: Resolving syntactic and grammatical ambiguities securely via mapped dictionaries.
- **Contextual Validation**: Approving the logical coherence of mathematical clusters.
- **Taxonomic Lifting**: Generating abstract superclasses (hypernyms) for verified sets.
- **Schema Synthesis**: Writing Pydantic Python code from graph representations.

**Offline Computation-Reliant Operations:**
- **Iterative Checkpoint Logging**: Writing diagnostic data footprints safely.
- **Vector Encoding**: Generating numerical representations via SentenceTransformers.
- **Dimensionality Reduction**: Principal Component Analysis (PCA).
- **Clustering**: Agglomerative clustering.
- **Centroid Calculation**: Mean vector and cosine distance operations.
- **Graph Construction**: Managing nodes, edges, and cumulative weights.
- **Spectral Graph Mathematics**: Computing the hypergraph Incidence Matrix ($H$) and Laplacian ($L$).
- **Community Detection**: Executing the Leiden algorithm to find partitions.

## Applied Mathematical and Topological Formulas

**1. Dimensionality Reduction (PCA & Eigengap Analysis)**
- **Formula/Application**: Principal Component Analysis (PCA) is applied to an embedding matrix scaled by the absolute frequency count of identical strings. 
- **Eigengap Heuristic**: The explained variance ratio (eigenvalues) is calculated. The algorithm determines the optimal number of components by finding the maximum gap between adjacent eigenvalues (`argmax(eigenvalues[:-1] - eigenvalues[1:])`).
- **Cumulative Variance Threshold**: If the retained variance from the eigengap is below 0.5, a cumulative variance threshold of 0.85 is utilized as a fallback.

**2. Distance Metrics and Clustering**
- **Formula/Application**: Agglomerative clustering is executed using `cosine` distance and `average` linkage. It stops grouping when the distance between merged clusters exceeds a configured `similarity_threshold`.

**3. Geometric Centroid**
- **Formula/Application**: For a mapped cluster $C$, the mean embedding vector $\bar{v} = \frac{1}{|C|} \sum_{v \in C} v$ is calculated. The algorithm then computes the cosine distance $1 - \frac{v \cdot \bar{v}}{||v|| ||\bar{v}||}$ for all $v \in C$. The cluster's centroid string is the member with the minimum cosine distance to the mean vector.

**4. Network Topology & Modularity Optimization**
- **Formula/Application**: Semantic relationships are modeled as a weighted directed graph $G = (V, E)$. Edge weights are additive $w(u, v) = \sum_{i} 1$, accumulating over occurrences of the same subject-object pairs.
- **Leiden Algorithm**: Resolves graph community structure by optimizing the `ModularityVertexPartition`. Modularity measures the density of edges inside communities compared to edges outside communities. The Leiden algorithm ensures communities are guaranteed to be connected and correctly resolves partitions in directed graphs.

**5. N-ary Hypergraph & Spectral Matrices**
- **Formula/Application**: Complex thematic events are modeled as a bipartite graph $B$ connecting standard entity nodes to thematic hyperedge nodes.
- **Incidence Matrix ($H$)**: An $|V| \times |E|$ binary matrix where $H_{i,j} = 1$ if entity $v_i$ participates in thematic hyperedge $e_j$, else $0$.
- **Laplacian ($L$)**: The graph Laplacian is computed algebraically via $L = D_v - H H^T$, where $D_v$ is the diagonal degree matrix representing the total themes each entity engages with.
