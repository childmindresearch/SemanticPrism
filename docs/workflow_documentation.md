# SemanticPrism Workflow Documentation

This document provides a comprehensive, sequential list of all processes, functions, and transformations that occur during the SemanticPrism pipeline execution.

## 0. Pipeline Entrypoint (Runner)
**Module:** `run_pipeline.py`
**Process:** `execute()`
- **Description:** Automates the discovery of raw text strings from the `inputs/testdocs/` directory. Loads textual content, combines it, and passes it to the orchestrator.
- **Data IN:** Directory path (`inputs/testdocs/*.txt` and `*.md`)
- **Data OUT:** List of raw text strings (`List[str]`)

## 1. Orchestration
**Module:** `src/orchestrator/pipeline.py`
**Class:** `SemanticPrismOrchestrator`
**Process:** `execute_knowledge_pipeline(documents: List[str])`
- **Description:** The master sequence runner. It coordinates all downstream classes and sequentially passes state/data from one pipeline stage to the next.

---

## Stage 1: LLM Extraction & Theme Consolidation
**Module:** `src/extraction/extractor.py`
**Class:** `ExtractionPipeline`

### Step 1.1: Theme Discovery
**Process:** `discover_themes(text: str)`
- **Description:** Splits raw input text into manageable chunks (by max words limit). Uses the LLM to extract high-level themes for each chunk independently.
- **Data IN:** Raw text chunk (`str`)
- **Data OUT:** List of `ThemeDiscoveryResult` Pydantic objects containing `themes` (Title, Description, Reasoning).
- **Schema:** `ThemeDiscoveryResult`
- **Prompts:** `THEME_DISCOVERY_SYSTEM_PROMPT`, `THEME_DISCOVERY_USER_PROMPT`

### Step 1.2: Theme Weighting
**Process:** `weight_themes(themes_list: List[ThemeDiscoveryResult])`
- **Description:** Consolidates and counts the frequency of all extracted themes across all chunks. Formats them into a single ordered string based on frequency.
- **Data IN:** List of all `ThemeDiscoveryResult` objects
- **Data OUT:** A formatted text string listing themes sorted by frequency (`str`)

### Step 1.3: Master Theme Consolidation
**Process:** `consolidate_themes(formatted_themes: str)`
- **Description:** Passes the frequency-weighted themes back to the LLM to synthesize a global "Master Domain" and consolidate overlapping themes into a unified master context.
- **Data IN:** Formatted themes string
- **Data OUT:** `MasterThemeSynthesisResult` Pydantic object (contains `master_domain` and combined `master_themes`)
- **Schema:** `MasterThemeSynthesisResult`
- **Prompts:** `MASTER_THEME_SYSTEM_PROMPT`, `MASTER_THEME_USER_PROMPT`

### Step 1.4: Triple Extraction
**Process:** `extract_triples(text: str, master_theme_context: MasterThemeSynthesisResult)`
- **Description:** Re-processes the raw text in chunks. Extracts logical Subject-Predicate-Object triples using the previously discovered master themes and an entity registry as context to maintain coreference coherence.
- **Data IN:** Raw text chunks, `MasterThemeSynthesisResult`
- **Data OUT:** List of `RawTriple` objects (`List[RawTriple]`)
- **Schema:** `TripleExtractionResult` containing `RawTriple` objects
- **Prompts:** `TRIPLE_EXTRACTION_SYSTEM_PROMPT`, `TRIPLE_EXTRACTION_USER_PROMPT`

---

## Stage 2.5: LLM Text Normalization
**Module:** `src/extraction/normalize_text.py`
**Process:** `execute_normalization_phase(extractor, raw_triples, master_domain, save_state_fn)`
- **Description:** Standardizes the text values of the extracted triplets.
  - **Sub-step A (NLP Preprocessing):** `nlp_preprocess(text)` natively converts snake_case to spaces, removes special brackets (`< > [ ] { }`), lowercases, and deduplicates spaces.
  - **Sub-step B (LLM Normalization):** Aggregates unique subjects, predicates, and objects into batches. Calls `extractor.normalize_triples_strings` to get clean lexical mappings from the LLM based on the `master_domain`.
  - **Sub-step C (Mapping):** Replaces the values in a deep copy of `raw_triples` with the LLM-normalized strings.
- **Data IN:** `List[RawTriple]`, `master_domain` (string)
- **Data OUT:** `normalized_triples` (`List[RawTriple]`), and sets of unique normalized subjects, predicates, and objects.

---

## Stage 2: Offline Embedding & Modularity Proposals
**Module:** `src/embedding/embedding.py`
**Class:** `EmbeddingPipeline`

### Step 2.1: Data Grouping
**Process:** `extract_and_group(triples: List[RawTriple])`
- **Description:** Splits raw triples into isolated distinct lists: "subject", "predicate", and "object" based on `compress_fields` configuration constraints.
- **Data IN:** `normalized_triples` (`List[RawTriple]`)
- **Data OUT:** Dict mapping fields to unclustered string arrays

### Step 2.2: Math Vector Clustering
**Process:** `_process_isolated_group(item_strings: List[str])`
- **Description:** Generates mathematical vector embeddings using a local SentenceTransformer. Dynamically calculates Eigen-gap and applies PCA for dimensionality reduction. Finally, isolates items into clusters using Agglomerative Clustering (Cosine Metric).
- **Data IN:** `List[str]`
- **Data OUT:** A list of mathematically proposed string clusters (`List[List[str]]`)

---

## Stage 3: Hybrid Hypernym Taxonomic Lifting
**Module:** `src/nlp/hypernyms.py`
**Class:** `HypernymPipeline`

### Step 3.1: Contextual Vector Validation
**Process:** `validate_context_vectors(proposals, master_domain)`
- **Description:** Uses the LLM to inspect each mathematical cluster proposed in Stage 2. If the LLM determines the terms are not semantically valid synonyms in the `master_domain`, it safely destroys the cluster and splits it back into isolated single items.
- **Data IN:** `proposed_clusters`, `master_domain`
- **Data OUT:** `verified_clusters` (Safely validated groupings)
- **Schema:** `ClusterContextualValidation`
- **Prompts:** `CONTEXT_VECTOR_VALIDATION_SYSTEM_PROMPT`, `CONTEXT_VECTOR_VALIDATION_USER_PROMPT`

### Step 3.2: Formal Taxonomic Lifting
**Process:** `taxonomic_lift(verified_clusters, master_domain)`
- **Description:** Uses the LLM to select an overarching "Hypernym" label for the cluster. If rejected, mathematically falls back to the geometric centroid vector calculated via `_find_semantic_center`.
- **Data IN:** `verified_clusters`, `master_domain`
- **Data OUT:** `hypernym_mapping` (A dictionary mapping original strings to their verified formal Hypernym label).
- **Schema:** `TaxonomicVerification`
- **Prompts:** `TAXONOMIC_LIFTING_SYSTEM_PROMPT`, `TAXONOMIC_LIFTING_USER_PROMPT`

---

## Stage 4: Taxonomic Resolution Mapping
**Module:** `src/nlp/nlp_mapping.py`
**Class:** `NamingResolutionPipeline`

### Step 4.1: Static Substitution
**Process:** `resolve_names(raw_triples, mapping_dict)`
- **Description:** Iterates through `normalized_triples` and deterministically maps any subject, object, or predicate explicitly replaced in the `hypernym_mapping`. Untampered text nodes are natively ignored avoiding data loss.
- **Data IN:** `normalized_triples`, `hypernym_mapping`
- **Data OUT:** `mapped_triples` (`List[RawTriple]`)

---

## Stage 5: Topological Graph Matrices
**Module:** `src/topology/graph_builder.py`
**Class:** `TopologyEngine`

### Step 5.1: Network Construction
**Process:** `build_graph(triples)`
- **Description:** Parses `mapped_triples` into a NetworkX DiGraph. Edges are mathematically collapsed and weights incremented for matching node pairs.
- **Data IN:** `mapped_triples`
- **Data OUT:** `nx.DiGraph`

### Step 5.2: Modularity Detection
**Process:** `detect_communities(graph)`
- **Description:** Uses the Leiden Algorithm with ModularityVertexPartition to isolate highly interconnected functional subgraph clusters (Communities).
- **Data IN:** `nx.DiGraph`
- **Data OUT:** `partition` (Dictionary mapping node name strings to Integer Community IDs)

### Step 5.3: Hierarchy Slicing
**Process:** `extract_hierarchy(graph, partition)`
- **Description:** Chops the primary graph apart based on community boundaries to provide clean sub-graphs and relationship matrices for each independent cluster.
- **Data IN:** `nx.DiGraph`, `partition`
- **Data OUT:** `hierarchy` (Dictionary mapping `Community_ID` to node arrays and subgraph matrices).

### Step 5.4: Hypergraph Generation (N-Ary)
**Process:** `build_hypergraph_topology(triples)`
- **Description:** Constructs a bipartite graph treating Theme logic elements as bridging N-ary hyperedges. Builds computational arrays Incidence Matrix (H) and Laplacian (L) used for subsequent spectral analysis.
- **Data IN:** `mapped_triples`
- **Data OUT:** Dictionary mapping Bipartite Pyvis components, `H`, and `L`.

---

## Stage 6: Generative Schema Synthesis
**Module:** `src/synthesis/synthesizer.py`
**Class:** `SynthesisEngine`

### Step 6.1: Code Generation
**Process:** `generate_schemas(hierarchy, master_domain)`
- **Description:** Converts the raw subgraph nodes and edges for each community into a textual representation and tasks the LLM with organically mapping out an actionable software structure, generating Python / Pydantic models to mimic the mathematical data layout cleanly.
- **Data IN:** `hierarchy`, `master_domain`
- **Data OUT:** `resolved_schemas` (Dict mapping Community IDs to formal `GeneratedSchema` objects)
- **Schema:** `GeneratedSchema`
- **Prompts:** `PYDANTIC_CODE_GEN_SYSTEM_PROMPT`, `PYDANTIC_CODE_GEN_USER_PROMPT`

### Step 6.2: Payload Global Export
**Process:** `build_global_context(resolved_schemas)`
- **Description:** Extracts the generated schemas and dumps them.
- **Data IN:** `resolved_schemas`
- **Data OUT:**
  - Saves all JSON components to `outputs/semantic_prism_master_graph.json`
  - Concatenates executable programmatic Python blocks to `outputs/semantic_models.py`
  - Returns `file_path`
