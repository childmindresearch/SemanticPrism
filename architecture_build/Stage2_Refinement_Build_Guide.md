# SemanticPrism Stage 2: Refinement Pipeline Build Specification

This document provides explicit instructions for building the Stage 2 Refinement phase. This phase combines the legacy "Embedding" and "NLP" directories into a single unified `refinement/` module, and formally moves Lexical Normalization into this stage.

## 1. Architectural Layout
Construct the module using three cleanly separated files inside `src/refinement/`:
1.  `schemas.py`: Contains strictly Pydantic data models.
2.  `prompts.py`: Contains static text and system prompts as strings.
3.  `refiner.py`: Contains embedding models, clustering math, Pydantic AI Agent definitions, and the execution pipeline class.

---

## 2. Schemas (`src/refinement/schemas.py`)
Implement the following Pydantic models. Ensure strict typing and documentation strings.

### Core Models:
1.  **`NormalizedToken`**: 
    *   Fields: `original` (str), `normalized` (str).
2.  **`NormalizedStrings`**: 
    *   Fields: `tokens` (List[NormalizedToken]).
3.  **`TaxonomicVerification`** (Unified/Simplified from previous split flow):
    *   Fields: `status` (Literal['VERIFIED', 'REJECTED']), `formal_hypernym` (Optional[str]), `fallback_term` (Optional[str]).

---

## 3. Prompts (`src/refinement/prompts.py`)
Define the following strings as constants.

### Required Constants:
*   `LLM_PREPROCESSING_SYSTEM_PROMPT`: Instructions for lexical normalization.
*   `TAXONOMIC_LIFTING_SYSTEM_PROMPT`: Instructions to take a list of terms and assign a formal taxonomic hypernym, or return 'REJECTED' if the terms are fundamentally contradictory. If 'REJECTED', the LLM MUST select the single most representative term from the provided 'Top 3 Centroid Fallbacks' list to serve as the `fallback_term`.

---

## 4. Agent Definitions (`src/refinement/refiner.py`)

Instantiate two distinct `pydantic_ai.Agent` objects at the module level. Ensure the model string is passed dynamically from `config.yaml`.

### Agent 1: Lexical Normalization
*   **System Prompt:** `prompts.LLM_PREPROCESSING_SYSTEM_PROMPT`.
*   **Result Type:** `schemas.NormalizedStrings`.
*   **Dependency Injection:**
    *   Set `deps_type=str` (to pass the master domain).
    *   Use `@norm_agent.system_prompt` decorator to inject: `f"\\nDomain Context: {ctx.deps}"`.

### Agent 2: Taxonomic Lift
*   **System Prompt:** `prompts.TAXONOMIC_LIFTING_SYSTEM_PROMPT`.
*   **Result Type:** `schemas.TaxonomicVerification`.
*   **Dependency Injection:**
    *   Set `deps_type=str` (to pass the master domain).
    *   Use `@lift_agent.system_prompt` decorator to inject: `f"\\nDomain Context: {ctx.deps}"`.

---

## 5. Master Execution Pipeline (`refiner.py`)

Create a master class `RefinementPipeline` to execute the sequence logically. It must accept the global `config` mapping and the `PipelineRunContext`.

### Step 5.1: Lexical Normalization Workflow
1.  **Native Cleaning:** Implement `nlp_preprocess` (replace underscores, strip syntax `[<>/\\|\[\]{}]`, lowercase, strip extra whitespace).
2.  **Preserve Raw Integrity:** Create a deep copy of the extracted `raw_triples` passed from Stage 1 (e.g., `normalized_triples = [t.model_copy() for t in raw_triples]`). **Do not mutate** the original `raw_triples` array.
3.  **Apply Cleaning:** Mutate subjects, predicates, and objects in `normalized_triples` using `nlp_preprocess`.
4.  **Extract Unique Sets:** Collect unique sets of subject, predicate, and object strings.
5.  **Batch Processing:** Loop through each set in batches of 50. Pass batch to `norm_agent.run_sync(batch_json, deps=PipelineRunContext.master_domain, model=config_model)`.
6.  **Re-map Triples:** Build `{original: normalized}` dictionary. Overwrite fields **ONLY** in `normalized_triples`.
7.  **Data Persistence:** Attach `normalized_triples` to `PipelineRunContext`. Save to `outputs/02_refinement/normalized_triplets.json` and output a mapping log.

### Step 5.2: Theme-Based Embedding Mapping
1.  **Initialize Embedding Model:** Load the HuggingFace `SentenceTransformer` defined in `config['refinement']['embedding_model']`.
2.  **Embed Original Themes:** For each original `ThemeDiscoveryResult` from Stage 1, concatenate its title, description, and reasoning. Pass list to `encoder.encode()`.
3.  **Embed Master Themes:** Concatenate each `master_theme` string with the `master_domain`. Pass to `encoder.encode()`.
4.  **Cosine Similarity Mapping:** Loop through original theme vectors and find the maximum cosine similarity score against the master theme vectors natively without using complex matrices. 
5.  **Save Output:** Save the resulting `{master_theme: [original_theme_1, original_theme_2]}` mapping to `outputs/02_refinement/theme_mapping_clusters.json`.

### Step 5.3: Triple Vector Clustering
1.  **Group Triples:** Extract and group unique subjects, predicates, and objects from `normalized_triples`.
2.  **Track Frequencies:** Calculate occurrence frequency for each term and logically store it in the `PipelineRunContext` to be referenced safely globally.
3.  **L2-Normalization:** Embed the unique terms with the `SentenceTransformer`. Apply `sklearn.preprocessing.normalize(embeddings, norm='l2')` to map onto a spherical manifold.
4.  **Agglomerative Clustering:** Cluster the L2-normalized vectors using `sklearn.cluster.AgglomerativeClustering` (metric='cosine', linkage='average', distance_threshold pulled from `config`).
5.  **Save Output:** Save proposed mathematical clusters to JSON.

### Step 5.4: Taxonomic Lifting
1.  **Iterate Clusters:** Loop over each mathematical cluster proposed in Step 5.3.
2.  **Geometric Centroid Fallback:** Calculate the weighted average vector of the cluster (using frequencies from `PipelineRunContext` as weights). Find the top 3 terms closest to this geometric centroid using cosine distance, returned in order from closest to furthest. Save this ordered list as the `mathematical_centroids` fallback.
3.  **LLM Verification:** Format both the `cluster_json` AND the `mathematical_centroids` list into a single payload. Pass this payload to `lift_agent.run_sync(payload, deps=master_domain, model=config_model)`.
4.  **Resolve & Record:** If `status == 'VERIFIED'`, map all cluster members to the LLM-provided `formal_hypernym`. If `'REJECTED'`, map all members to the LLM-selected `fallback_term`. **Crucially, save this 1:1 mapping (Raw Entity -> Hypernym) to a dictionary.**
5.  **Re-Map Triples:** Mutate `normalized_triples` subjects, predicates, and objects using this taxonomic dictionary.
6.  **Data Persistence:** 
    *   Save the fully abstracted `refined_triples` to `outputs/02_refinement/refined_triplets.json`.
    *   Save the forward mapping dictionary to `outputs/02_refinement/taxonomic_map.json`.

---

## 6. Isolated Execution (`run_stage_2.py`)
To ensure strict modularity, this stage must be executable in complete isolation. Create a standalone `run_stage_2.py` execution script that initializes the pipeline, loads the required `outputs/01_extraction/original_triplets.json` and theme outputs from the previous stage, executes the `RefinementPipeline`, and cleanly saves its outputs. This guarantees that the stage can be tested, debugged, and run entirely independently.
