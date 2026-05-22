# SemanticPrism: Project Overview and Architecture

## Overall Summary
SemanticPrism is a knowledge graph ontology extractor pipeline that processes multiple raw text documents to construct a mathematical, hierarchical semantic topology. The system translates unstructured text into structured, directed graphs and synthesizes verifiable Python models (Pydantic schemas) representing the isolated ontological communities. 

The pipeline is split into explicit modular stages: Extraction, Syntactic Normalization, Embedding, Hypernym mapping, Topology construction, and Synthesis. Each stage operates strictly on bounded inputs to ensure deterministic behavior, focusing on accuracy, abstraction, and ontology. To ensure robust fault tolerance, the pipeline natively employs **Iterative Diagnostic Logging**, sequentially persisting analytical metrics and runtime traces to disk after every phase.

## Pipeline Architecture and Components

### 1. Extraction Pipeline (`extractor.py`)
**Purpose**: Transforms unstructured text matrices into formal subject-predicate-object triples and discovers overarching domain themes to guide the extraction process.
**Components**:
- **Multi-Document Theme Consolidation**: Ingests multiple text documents, extracts local themes, weights them by cross-document frequency, and synthesizes a singular master domain context.
- **Triple Extraction**: Extracts logical raw triples using an entity registry to maintain coreference context across the documents.

### 2. Syntactic Normalization (`normalize_text.py`)
**Purpose**: Binds raw string tokens to deterministically scrubbed states prior to embedding logic, eliminating syntactic and grammatical ambiguities without structural loss.
**Components**:
- **Explicit Lexical Normalization**: Utilizes the LLM with constrained JSON decoding to construct strict 1:1 key-value mapping dictionaries. This securely links raw tokens (e.g., `{"dogs": "dog"}`) directly to normalized counterparts without relying on fragile list indexing.

### 3. Embedding Pipeline (`embedding.py`)
**Purpose**: Projects normalized strings into a dense vector space to reduce redundancy by grouping semantically identical components.
**Components**:
- **Theme-based Embedding**: Embeds the original themes via `SentenceTransformers` and maps them mathematically to the consolidated master themes using Cosine Similarity thresholds.
- **Triple Vector Clustering**: Isolates the triples into physical component arrays (Subjects, Predicates, Objects), embeds them into dense vectors, applies explicit L2 normalization onto a spherical manifold, and groups conceptually identical elements using Agglomerative Clustering.

### 4. Hypernym Pipeline (`hypernyms.py`)
**Purpose**: Merges mathematical centroids with LLM validation protocols to enforce hierarchical taxonomic structures (moving from specific entities to abstract superclasses).
**Components**:
- **Contextual Validation**: Evaluates the logical coherence of mathematically proposed clusters and splits rejected sets.
- **Geometric Centroid Calculation**: Computes the mean embedding vector for a verified cluster and maps it to the specific string with the minimum cosine distance.
- **Taxonomic Lifting**: Uses Chain-of-Thought reasoning to assign a formal abstract superclass label (hypernym) that accurately represents the cluster. If the LLM rejects the abstraction, it gracefully falls back to using the geometric centroid.

### 5. Taxonomic Resolution Mapping (`nlp_mapping.py`)
**Purpose**: Condenses the specific localized triple topology into a higher-order abstracted topology.
**Components**:
- **Deterministic String Replacement**: Performs a dictionary-based scan over the normalized triples, deterministically overwriting specific subjects and objects with their newly assigned taxonomic hypernyms, dramatically simplifying the graph structure.

### 6. Topology Engine (`graph_builder.py`)
**Purpose**: Constructs mathematically defined networks, partitions them into modular semantic communities, and builds n-ary hypergraph representations for spectral analysis.
**Components**:
- **Directed Graph Construction**: Builds a NetworkX `DiGraph` from the mapped triples, tracking cumulative edge weights $w(u, v) = \sum_{i} 1$.
- **Leiden Community Detection**: Optimizes the `ModularityVertexPartition` to reliably compute modularity partitions for the directed graphs and runs Louvain detection on the bipartite structure.
- **Hierarchical Extraction Strategy**: 
  - *Standard*: Retains dense communities and explicitly prunes/discards micro-communities.
  - *Hub-and-Spoke*: Identifies the network's super-hub node (highest degree), isolates it as a master component, clusters the remaining subgraph, and pushes micro-communities into an "orphan" pool.
- **N-ary Hypergraph Grouping**: Groups triples around their `theme_association`, securely tracking local neighborhoods (Identity Guard) to map a bipartite graph connecting entities to thematic hyperedges.
- **Spectral Matrices**: Computes the hypergraph Incidence Matrix ($H$) and Laplacian ($L = D_v - H H^T$) via `numpy` to map high-level entity-theme interactions and inheritance mapping mathematically.

### 7. Synthesis Engine (`synthesizer.py`)
**Purpose**: Transforms abstract mathematical network communities into structured, executable programmatic models.
**Components**:
- **LLM Schema Generation**: Passes each identified structural community (alongside the master hub and orphans) to the LLM to dynamically synthesize strictly typed Pydantic `BaseModel`s and duck-typed `Protocol` interfaces.
- **Output Export**: Assembles the generated Python blocks and physically writes the `.py` files. 
  - Standard communities become standalone schema files.
  - Master hubs export to `master_context.py`.
  - Orphaned micro-components are dumped into `global_enums.py` as pure Enums/Literals.

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
- **Spherical Manifold Mapping**: L2-normalization onto a spherical manifold.
- **Clustering**: Agglomerative clustering.
- **Centroid Calculation**: Mean vector and cosine distance operations.
- **Taxonomic Resolution Mapping**: Deterministic dictionary string replacement.
- **Graph Construction**: Managing nodes, edges, and cumulative weights.
- **Spectral Graph Mathematics**: Computing the hypergraph Incidence Matrix ($H$) and Laplacian ($L$).
- **Community Detection**: Executing the Leiden algorithm to find partitions.

## Applied Mathematical and Topological Formulas

**1. Spherical Manifold Mapping**
- **L2-Normalization**: The dense vectors are mathematically normalized ($v / ||v||_2$) onto a spherical manifold, optimizing the stability and reliability of downstream cosine distance grouping.

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
