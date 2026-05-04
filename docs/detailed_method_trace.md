# Detailed Method and Function Trace

This document provides a low-level sequential breakdown of every method, function, internal subroutine, and mathematical operation invoked during the execution of each stage in SemanticPrism.

---

## 0. Pipeline Entrypoint (`run_pipeline.py`)
- **`glob.glob(target_dir, "*.txt")` / `glob.glob(target_dir, "*.md")`**: Scans the designated inputs directory natively for supported file extensions.
- **`open(file_path, "r").read()`**: Standard Python I/O to read raw file bytes into a UTF-8 string array.
- **`SemanticPrismOrchestrator(config_path)`**: Constructor class initializing all pipeline objects natively (`ExtractionPipeline`, `EmbeddingPipeline`, `HypernymPipeline`, `NamingResolutionPipeline`, `TopologyEngine`, `SynthesisEngine`, `SemanticVisualizer`).
- **`execute_knowledge_pipeline(documents)`**: The primary async while/for loop structure that drives data chronologically through the instantiated objects.

---

## Stage 1: Extraction & Theme Consolidation (`ExtractionPipeline`)
### 1.1 `discover_themes`
- **`chunk_text(text, self.theme_max_words, overlap_words=50)`**: Imported from `src.core.chunking`. Splices the raw corpus into string arrays based on token maximums. Employs a sliding window approach, intentionally repeating `overlap_words` (50) backward to ensure contextual phrases on the edge of the text aren't sheared or misinterpreted.
- **`self.llm.safe_api_call_async/sync()`**: The unified LLM router method located in `SemanticLLMClient` (`llm_client.py`).
  - *Internal:* Checks the `LocalLLMProvider` vs `PublicLLMProvider` context size availability.
  - *Internal:* Inspects `connection_protocol`. If `http`, it routes to `execute_http_raw()` which natively POSTs bytes to an endpoint bypassing rigid validation. If `sdk`, it utilizes `client.chat.completions.create()` through the `instructor` library, enforcing rigid JSON extraction based on the `ThemeDiscoveryResult` Pydantic model.
  - *Internal:* Calls `self.provider.release_vram()` explicitly dropping GPU memory references if local Ollama is being used.

### 1.2 `weight_themes`
- **`str.strip().lower()`**: Text normalization standardizing case to properly consolidate duplicate LLM outputs.
- Uses native Python dictionaries to count overlapping theme structures and sorts them by magnitude frequency (`sorted(..., key=lambda item: item[1], reverse=True)`).

### 1.3 `consolidate_themes`
- Generates the unified string layout and passes it exclusively to `self.llm.safe_api_call_async/sync()` mapping directly against the `MasterThemeSynthesisResult` model schema.

### 1.4 `extract_triples`
- **`chunk_text(...)`**: Called a second time utilizing `triple_max_words` (a tighter window).
- Native Python `set()` operations are utilized to manage the `entity_registry`. It slices `list(entity_registry)[-100:]` to provide the LLM with a trailing memory footprint for structural coreference resolution (e.g. converting "He" back to "The Algorithm").
- Relies heavily on Instructor's execution enforcing the `TripleExtractionResult` matrix.

### Stage 2.5: Normalization (`normalize_text.py`)
- **`nlp_preprocess(text)`**:
  - `str.replace('_', ' ')`: Destroys programmatic snake_case layouts.
  - `re.sub(r'[<>/\\|\[\]{}]', '', text)`: Regex engine explicitly stripping XML, HTML, and Array syntax.
  - `str.lower()` and `re.sub(r'\s+', ' ', text).strip()`: Final cleanup pass.
- Organizes the clean strings into list batches of 50.
- **`normalize_triples_strings(batch, master_domain)`**: LLM function mapping tokens rigidly to `NormalizedStrings` schema mapping lists.
- Generates a deep Python `set()` and differential `dict` mapping to log exactly what original text strings were altered by this phase natively.

---

## Stage 2: Offline Embedding (`EmbeddingPipeline`)
### 2.1 `extract_and_group`
- Simple conditional routing checking `.subject`, `.predicate`, `.object` dynamically based on the YAML `compress_fields` matrix, separating the Pydantic array into 3 independent Python lists.

### 2.2 `_process_isolated_group`
- **`collections.Counter()`**: Tallies occurrences of the exact same strings to identify density.
- **`self.encoder.encode(unique_items, convert_to_numpy=True)`**: Utilizes `sentence_transformers.SentenceTransformer` natively mapping strings into floating-point hyper-dimensional vectors.
- Array manipulation mathematically injects duplicates `expanded_embeddings.extend([emb] * count)` back into the space to physically weight high-frequency nodes.
- **`sklearn.decomposition.PCA()`**: 
  - *Eigengap analysis:* Uses `.explained_variance_` to find the largest mathematical gap (`gaps = eigenvalues[:-1] - eigenvalues[1:]`, `np.argmax(gaps) + 1`). This explicitly finds the perfect dynamic component shape to compress the data.
  - Performs a second pass `.fit_transform()` enforcing the calculated optimal dimension.
- **`sklearn.cluster.AgglomerativeClustering(...)`**: Executes hierarchical clustering natively measuring `cosine` distances and grouped by `average` linkage. Yields discrete integer labels.

---

## Stage 3: Taxonomic Lifting (`HypernymPipeline`)
### 3.1 `validate_context_vectors`
- Processes the mathematically defined groups through `self.llm.safe_api_call_async/sync()` against the `ClusterContextualValidation` schema. If rejected (`accuracy_destroyed`), natively destructs the list array using Python comprehension (`[[item] for item in cluster]`).

### 3.2 `taxonomic_lift`
- **`self._find_semantic_center(cluster)`**:
  - Invokes `self.encoder.encode` again.
  - **`np.mean(embeddings, axis=0, keepdims=True)`**: Mathematically locates the direct Euclidean average space across the array.
  - **`sklearn.metrics.pairwise.cosine_distances()`**: Calibrates the angles from the true center.
  - **`np.argmin(distances)`**: Locates the specific string index that lives closest to the centroid.
- Attempts `TaxonomicVerification` via LLM; if the model rejects, it completely bypasses logic and defaults organically natively to the `centroid` calculated above.

---

## Stage 4: Resolution Mapping (`NamingResolutionPipeline`)
### 4.1 `resolve_names`
- Strictly deterministic mapping routine. Uses native python boolean flags (`modified = False`) and `dictionary.get()` replacement functions modifying `.subject`, `.predicate`, and `.object` strictly if a verified taxonomic lift was located natively. Returns updated triple instances cleanly.

---

## Stage 5: Topology Engine (`TopologyEngine`)
### 5.1 `build_graph`
- **`nx.DiGraph()`**: Instantiates NetworkX graph object.
- **`G.has_edge()`**: If an edge exists between nodes, `G[u][v]['weight'] += 1.0` dynamically collapses multiple references into a single heavy structural edge.
- **`set().add()`**: Sub-elements (like predicates) are pushed into native sets inside edge properties to maintain cardinality securely.

### 5.2 `detect_communities`
- **`igraph.Graph(n=..., edges=..., directed=True)`**: C-backed high performance graph construction converting Python NetworkX edges into optimized integers.
- **`leidenalg.find_partition(...)`**: Specifically implements the mathematical `ModularityVertexPartition` layout. Identifies high-modularity functional neighborhoods automatically.

### 5.3 `extract_hierarchy`
- Native dictionary manipulation building a structure matching `{Community_X: {nodes: [], edges: [], sub_graph: nx.DiGraph()}}`. Uses `partition.get(u)` checks to only bridge edges entirely inside the same partition organically securely.

### 5.4 `build_hypergraph_topology`
- Creates a `defaultdict(set)` structure.
- Instantiates **`nx.Graph()`** (Bipartite configuration).
- Synthesizes `HE: theme` label nodes with flag `is_hyperedge=True` directly bridged to isolated entity nodes via `B.add_edge(ent, he_node)`.
- **`H = np.zeros((Entities, Themes))`**: Explicitly matrices the Incidence layout natively using NumPy arrays. Populates structural 1s if an entity interfaces with a hyperedge.
- **`Dv = np.diag(np.sum(H, axis=1))`**: Derives mathematical structural Degree Matrix arrays.
- **`L = Dv - np.dot(H, H.T)`**: Executes final Laplacian vector derivation natively for spectral diagnostics.

---

## Stage 6: Synthesis Generation (`SynthesisEngine`)
### 6.1 `generate_schemas`
- Generates linear stringified representations of network structures natively (`f"{source} -[{predicate}]-> {target}"`).
- Validates the array chunks through `self.llm.safe_api_call_async/sync()` enforcing the explicit nested `GeneratedSchema` syntax.

### 6.2 `build_global_context`
- **`schema.model_dump()`**: Serializes Pydantic to a native dictionary.
- **`json.dump(..., f)`**: Writes `semantic_prism_master_graph.json` to disk.
- Aggregates Python code natively through string joins (`"\n".join(python_blocks)`). Opens `semantic_models.py`, writes hardcoded Python import headers, and writes the concatenated string securely cleanly to disk.
