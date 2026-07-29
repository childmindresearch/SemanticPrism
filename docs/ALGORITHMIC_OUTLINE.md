# SemanticPrism: Algorithmic & Process Outline

## Overview

SemanticPrism is a **six-stage pipeline** that transforms unstructured text documents into structured, hierarchical ontologies expressed as executable Python type schemas (Pydantic models). The central design principle is a **strict separation** between:

- **LLM operations** (non-deterministic, interpretation-heavy steps: extraction, normalization, validation, labeling, code generation)
- **Offline mathematical operations** (deterministic, geometric/graph-theoretic steps: embedding, clustering, community detection, spectral analysis)

Each stage writes its outputs to disk as JSON, enabling **iterative diagnostic logging** — every phase can be inspected, resumed, or replaced independently.

---

## Stage 1: Extraction (LLM)

### 1.1 Theme Discovery

**What:** For each input document, the LLM reads the text in chunks (default 6000-word windows with 50-word overlap) and produces a list of high-level ontological themes.

**Why:** Rather than extracting raw facts first, we first identify the conceptual "skeleton" of each document — the macro-level categories (e.g., "Endocrine Metabolic Disorders" not "Metformin"). This top-down framing constrains downstream extraction, reducing hallucination.

**Key parameters:**
- `theme_chunk_max_words`: 6000 (window size)
- `overlap_words`: 50 (context preservation at chunk boundaries)

**Output:** `List[ThemeDiscoveryResult]` — each containing `List[Theme]` with `title`, `description`, `reasoning`.

### 1.2 Master Theme Consolidation

**What:** All document-level themes are aggregated, **frequency-weighted** (how many documents/chunks mentioned each theme), and passed to the LLM with a prompt to deduplicate, merge, and abstract them into a single **Master Theme List** plus a single **Master Domain** string.

**Why:** Cross-document frequency weighting identifies which themes are globally salient. The LLM then performs semantic deduplication and abstraction — e.g., merging "Database Storage" and "Data Persistence" into "Data Infrastructure" — producing a unified lens for the entire corpus.

**Algorithm detail:** Frequency weighting is a simple count of `normalized_title.lower()` occurrences across all chunks. The top-N weighted themes (sorted descending) are formatted into a prompt block.

**Output:** `MasterThemeSynthesisResult{master_domain: str, master_themes: List[str]}`

### 1.3 Logical Triple Extraction (SVO)

**What:** Each document is re-chunked and the LLM extracts **Subject-Predicate-Object triples** from each chunk, with:
- `subject`, `predicate`, `object`: exact text as-is
- `source_quote`: verbatim evidence string
- `certainty_score`: float in [0,1]
- `theme_association`: which master theme this triple belongs to (or `"Other"`)
- `source_document`: provenance tracking

**Why:** The triple is the atomic unit of the knowledge graph. Requiring a source quote and certainty score provides auditability and a quality gate. The `theme_association` field is critical — it connects every fact to the ontological skeleton, enabling the hypergraph analysis in Stage 5.

**Entity registry:** An in-memory `set()` accumulates all subject/object entities seen so far across chunks (within a document). The last 100 are injected into each subsequent chunk's prompt as "Previously Discovered Entities" for coreference resolution across chunk boundaries.

**Output:** `List[RawTriple]`

### 1.4 Lexical Normalization (Optional)

**What:** A two-pass process:

1. **Rule-based preprocessing:** Convert snake_case to spaces, strip brackets/slashes, lowercase, collapse whitespace.

2. **LLM-based normalization:** Unique strings from each triple component (subject, predicate, object) are sent in batches of 50 to the LLM, which returns a dictionary mapping `original → normalized` form. The LLM is instructed to:
   - Lemmatize (plural→singular, verb normalization)
   - Strip determiners ("the", "a")
   - Remove soft adjectives
   - Expand unambiguous abbreviations
   - Standardize structural predicates ("is a part of" → "part of")
   - Preserve technical named entities ("PostgreSQL" stays "PostgreSQL")

**Why:** The LLM handles ambiguous normalization (e.g., "K8s" → "Kubernetes") that rule-based systems cannot. The dictionary mapping approach (rather than in-place replacement) ensures **determinism** — the same raw token always maps to the same normalized form, which is critical for reproducible graph construction.

**Output:** In-place mutation of `RawTriple` strings + `normalization_mapping_details.json` for audit.

---

## Stage 2: Embedding & Clustering (Offline)

### 2.1 Theme Embedding Mapping

**What:** Each unique original theme (from Stage 1.1) is concatenated with its descriptions and reasonings, then encoded via **SentenceTransformers** (`BAAI/bge-m3`). The master themes (from Stage 1.2) are similarly encoded. For each original theme, cosine similarity is computed against all master themes, and it is assigned to the most similar master theme.

**Why:** This maps the noisy, locally-discovered themes into the clean, master ontology. It also surfaces which master themes are empirically grounded (have many original themes mapped to them) versus which might be spurious.

**Key formula:**
```
cosine_similarity(v1, v2) = (v1 · v2) / (||v1||₂ · ||v2||₂)
```

**Output:** `Dict[master_theme → List[original_themes]]` (saved as `theme_mapping_clusters.json`)

### 2.2 Triple Component Clustering

**What:** Subjects, predicates, and objects from normalized triples are extracted into separate lists. For each component type:

1. **Frequency tracking:** Each unique string's frequency is recorded in a global `FREQUENCY_REGISTRY` (used later for weighted centroid calculations in Stage 3).

2. **Encoding:** Unique strings are encoded via SentenceTransformers.

3. **L2 Normalization:** Each embedding vector is divided by its Euclidean norm: `v̂ = v / ||v||₂`. This projects all vectors onto the **unit sphere** (a spherical manifold).

   **Why:** Cosine similarity on L2-normalized vectors is equivalent to Euclidean distance on the sphere, which is the natural metric for `AgglomerativeClustering` with `metric='cosine'`. This ensures stable, interpretable clustering behavior.

4. **Agglomerative Clustering** with:
   ```
   metric = 'cosine'
   linkage = 'average'   # UPGMA: distance between clusters = average pairwise distance
   distance_threshold = similarity_threshold (default 0.4)
   ```

   **Why average linkage?** Complete linkage is too strict (isolates near-synonyms); single linkage is too permissive (chains unrelated terms). Average linkage (UPGMA) provides the best balance for semantic clustering. The distance threshold directly controls granularity: 0.05 = only perfect synonyms merge; 0.40 = loose grouping of related concepts.

5. **Output:** For each component type, a list of clusters (each cluster is a list of strings).

**Design rationale for separating components:** Subjects, predicates, and objects occupy different semantic spaces. A subject "patient" and an object "patient" may have the same embedding but serve different roles. Clustering them separately prevents cross-role contamination.

**Output:** `Dict[field → List[List[str]]]` — mathematical cluster proposals.

---

## Stage 3: Hypernym Lifting (LLM + Geometry)

This is the **most novel hybrid stage** — it combines geometric centroid computation with LLM validation to produce hierarchically abstracted labels.

### 3.1 Contextual Validation

**What:** Each mathematically-proposed cluster (from Stage 2) is sent to the LLM with the question: *"Does merging these specific terms destroy critical semantic distinctions?"*

The LLM evaluates against three failure conditions:
1. **Hierarchy mixing** (parent-child in same cluster: "Virus" + "COVID-19")
2. **Functional divergence** (different impact/role: "Revenue" + "Profit")
3. **Attribute loss** (general + specific variant: "User" + "Admin User")

Valid merges include lexical variation ("AI" + "Artificial Intelligence") and orthographic noise ("GitHub" + "github.com").

**Why:** Pure geometric clustering groups by embedding proximity, which can conflate related but hierarchically distinct concepts. The LLM acts as a **semantic quality gate**, rejecting mathematically plausible but semantically invalid groupings. If rejected, the cluster is split into singletons.

**Output:** `Dict[field → List[List[str]]]` — verified (potentially split) clusters.

### 3.2 Geometric Centroid Calculation

**What:** For each verified cluster, the **semantic centroid** is computed:

```
mean_vector = (1 / Σf_i) · Σ (f_i · embedding_i)   # weighted by frequency
```

Then cosine distance to each member is computed:
```
distance_i = 1 - (centroid · embedding_i) / (||centroid||₂ · ||embedding_i||₂)
```

The member with the **minimum cosine distance** to the mean vector is selected as the centroid string.

**Why:** The frequency-weighted mean captures the "center of usage gravity" — the most representative usage pattern. Selecting the actual member string (not a synthetic embedding) ensures the centroid is a real, interpretable term. This centroid serves as the fallback if LLM taxonomic lifting fails.

**Output:** A string (the centroid label for each cluster).

### 3.3 Taxonomic Lifting

**What:** The cluster members plus the geometric centroid are sent to the LLM with a prompt to generate a **formal hypernym** — an abstract parent class one level up. The LLM is instructed to:

1. Pass the **"Is-A" test**: every member must be a strict subtype of the proposed hypernym.
2. Provide an **excluded opposite** (negative boundary test): e.g., if hypernym = "Automobile", excluded opposite = "Bicycle".
3. Respect **domain parity**: in a healthcare corpus, "Aspirin" → "Pharmacological Agent", not "Chemical Compound".
4. If members are too heterogeneous, **reject** (`members_verified = False`) and fall back to the geometric centroid.

**Why:** This is a **chain-of-thought constrained generation** — the LLM must reason through membership verification before assigning the label. The excluded opposite forces the LLM to carve clean semantic boundaries. The fallback to geometric centroid ensures the process never fails catastrophically.

**Output:** `Dict[field → Dict[original_string → hypernym_label]]`. For example: `{"subject": {"Toyota": "Car", "Honda": "Car", "Ford": "Car"}}`.

---

## Stage 4: Taxonomic Resolution Mapping (Offline)

**What:** A **purely deterministic dictionary scan** over the normalized triples. For each triple component (subject, predicate, object), the hypernym mapping dictionary is consulted: if the component string exists as a key, it is replaced with the hypernym value.

**Why:** This is the "collapse" operation — individual entity instances (e.g., "PostgreSQL", "MySQL", "SQLite") are uplifted to their shared abstract class ("Database"). This dramatically simplifies the graph, reducing hundreds of specific nodes into tens of abstract nodes, while the original specificity is preserved in the mapping audit log.

**Unmapped strings are left untouched** — no data loss.

**Output:** In-place mutation of triple strings.

---

## Stage 5: Topology Engine (Offline)

### 5.1 Directed Graph Construction

**What:** A `NetworkX.DiGraph` is built from the mapped triples:
```
nodes ← subjects ∪ objects
edges ← subject → object
weight(u,v) ← count of (subject, object) pairs across all triples (cumulative sum)
predicates ← set of all predicate values for each edge
```

**Why:** Edge weights capture frequency of co-occurrence, making the graph a **frequency-weighted semantic network**. Multiple predicates between the same subject-object pair are preserved as a set — this matters for downstream schema synthesis (the LLM sees all relationship types).

### 5.2 Community Detection (Leiden Algorithm)

**What:** The directed, weighted graph is converted to an `igraph.Graph` and partitioned using the **Leiden algorithm** with `RBConfigurationVertexPartition` (a resolution-parameterized variant of modularity).

**Modularity** measures the density of edges inside communities vs. across communities:
```
Q = (1/2m) · Σᵢⱼ [Aᵢⱼ - (kᵢ·kⱼ)/(2m)] · δ(cᵢ, cⱼ)
```
where A is the adjacency matrix, k are degrees, m is total edge weight.

**Why Leiden over Louvain?** The Leiden algorithm guarantees that communities are **connected** (Louvain can produce disconnected communities). It also runs faster and produces higher-quality partitions.

**Resolution parameter** (default 1.0):
- `< 1.0` → fewer, larger communities (coarser ontology)
- `> 1.0` → more, smaller communities (finer granularity)

### 5.3 Hierarchy Extraction Strategies

Two strategies are supported:

**Standard:** Communities below `min_community_size` (default 4 nodes) are **pruned** (discarded). Surviving communities become candidate ontology modules.

**Hub-and-Spoke:** The node with the **highest degree** (most connections) is identified as the "master hub." It is isolated, and Leiden runs on the remaining subgraph. Micro-communities below the size threshold are not discarded but collected into an **"orphan" pool** — they become Enums and Literals rather than full schemas.

**Why two strategies?** Real-world knowledge graphs often have a dominant super-node (e.g., "Patient" in clinical notes, "Document" in knowledge management). The hub-and-spoke strategy preserves this structure explicitly rather than forcing it into a community partition.

### 5.4 N-ary Hypergraph & Spectral Matrices

**What:** A **bipartite graph** is built connecting entity nodes to "hyperedge" nodes representing themes (from the `theme_association` field in triples).

**Incidence Matrix H:**
- Rows = entities, Columns = themes
- `H[i,j] = 1` if entity i participates in theme j, else 0

**Laplacian L (entity space):**
```
Dv = diag(H · 1)           # entity degree matrix (how many themes each entity touches)
L = Dv - H · Hᵀ            # |V| × |V| Laplacian
```

**Theme Overlap Matrix O:**
```
O = Hᵀ · H                 # |E| × |E| — shared entities between theme pairs
```

**Theme Inheritance Map:** For each pair of themes (A, B):
```
if O[i,j] / O[j,j] ≥ overlap_threshold (default 0.75):
    theme_B inherits from / is a subclass of theme_A
```

**Why:** This captures the **entity-level overlap** between themes. If two themes share ≥75% of their entities, they likely represent a hierarchical relationship (one is a specialization of the other), not a mere correlation. This inheritance map is passed to the synthesis engine to guide Protocol/interface generation.

---

## Stage 6: Schema Synthesis (LLM)

### 6.1 Community-to-Schema Translation

**What:** Each community (from Stage 5) is serialized into a structured JSON context containing:
- `community_id`, `domain`, `nodes`, `relationships` (as human-readable edge strings)
- `inheritance_guidelines` (from the hypergraph inheritance map or hub-and-spoke structure)

This is sent to the LLM, which generates a `GeneratedSchema` object containing:
- `title`, `summary`, `core_theme`
- `protocols_code`: Python `typing.Protocol` interfaces for cross-community composition
- `concrete_models_code`: Strictly typed Pydantic v2 `BaseModel` classes
- `key_learnings`, `isolated_facts`: Narrative summaries

**Why two code outputs?** Protocol classes define structural subtyping (Duck Typing) without requiring class inheritance, maintaining loose coupling. Concrete BaseModels provide the actual data structures. The LLM is instructed to abstract field names (e.g., `assessment_score` not `wisc_v_score`) to maximize generalizability.

### 6.2 Hub-and-Spoke Handling

If the hub-and-spoke strategy was used:
- The **master hub** generates a foundational `Protocol`/`BaseModel` (e.g., `PatientContext`) that other schemas reference via composition.
- **Orphans** (micro-communities) generate only `Enum` classes and `Literal` types — no full models.

### 6.3 Code Export

Generated Python code is written to:
- `outputs/schemas/master_context.py` (hub, if applicable)
- `outputs/schemas/global_enums.py` (orphans, if applicable)
- `outputs/schemas/{community_title_snake_case}.py` (each standard community)
- `outputs/semantic_models.py` (all code concatenated)
- `outputs/semantic_prism_master_graph.json` (full structured dump)

---

## LLM Infrastructure

### Pydantic AI & Instructor Pattern

All LLM calls use **Pydantic AI** with structured output schemas. This means:
- The response_model (a Pydantic class) defines the expected JSON structure.
- The LLM is prompted to return valid JSON conforming to that schema.
- Pydantic AI handles retries (up to 3), validation, and parsing.
- Prompt engineering includes explicit anti-pattern rules (e.g., "Do NOT output the JSON Schema definition itself") to avoid common failures.

### Async Concurrency

LLM operations use `asyncio.Semaphore(max_concurrent_llm_calls)` (default 2) to rate-limit concurrent API calls. The Ollama transport layer (`AsyncOllamaTransport`) includes a monkey-patch for the Ollama `null` content bug.

### Dynamic Context Sizing

The `ContextManager` queries `nvidia-smi` to profile free VRAM. If VRAM < 4000 MB, the context window is compressed (max 2000 words per chunk). This prevents out-of-memory crashes on resource-constrained hardware.

---

## Summary: Design Philosophy

| Principle | Implementation |
|---|---|
| **Determinism where possible** | Normalization via 1:1 dictionaries; resolution mapping via dict lookup; graph construction deterministic |
| **LLM as gate, not oracle** | Clustering proposals → LLM validates → fallback to geometric centroid |
| **No silent failure** | Every phase writes diagnostic JSON; `_dump_current_log()` collates all errors and context sizes |
| **Progressive abstraction** | Raw text → Themes → Master themes → Triples → Clusters → Hypernyms → Graph → Code |
| **Composability via Protocols** | Generated schemas use Duck Typing (`typing.Protocol`), not rigid inheritance, for flexibility |
| **Auditability** | Every triple has a source quote and certainty score; every normalization step is logged as a mapping |
