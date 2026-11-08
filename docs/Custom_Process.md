# Custom Multi-Variable Scoring, Ranking & Filtering Processes

This document provides a technical reference for all custom multi-variable scoring, ranking, and filtering algorithms implemented across **Stage 1 (`run_stage_1_extraction.py` / `src/extraction/`)**, **Stage 2 (`run_stage_2_refinement.py` / `src/refinement/`)**, and **Stage 4 (`run_stage_4_synthesis.py` / `src/synthesis/`)**. Single-variable cutoffs are excluded; each process below combines **two or more dynamic variables** to solve specific geometric, statistical, or structural challenges in the SemanticPrism pipeline.

---

## 1. Stage 1 (`src/extraction/`) — Theme Discovery, Clustering & Triple Extraction

### 1.1 Self-Calibrating Manifold Edge Admission & Excess-Weight Scoring
* **Location**: `src/extraction/clustering.py` (`ThemeClusteringEngine.cluster_themes`)
* **Challenge Solved**: Fixed cosine similarity cutoffs fail across diverse embedding backbones—narrow-cone anisotropic models compress all similarities above `0.70` (collapsing the kNN graph into a single giant component), whereas sparse lexical/TF-IDF spaces have lower similarities (shattering communities into singletons).
* **Variables Combined (5)**:
  1. Off-diagonal background similarity mean ($\mu_{\text{bg}}$)
  2. Off-diagonal background similarity standard deviation ($\sigma_{\text{bg}}$)
  3. 20th percentile of top-1 nearest-neighbor similarities ($P_{20}(\mathbf{s}^{(1)})$)
  4. Configured `MANIFOLD_EDGE_THRESHOLD` ($\tau_0$)
  5. Neighbor rank index ($\text{rank}_{ij}$) and pairwise cosine similarity ($w_{ij}$)
* **Process & Formula**:
  1. **Effective Manifold Threshold**:
     $$\tau_{\text{eff}} = \max\!\left(\mu_{\text{bg}} + 0.5\sigma_{\text{bg}},\; \min\!\big(\tau_0,\, P_{20}(\mathbf{s}^{(1)})\big)\right)$$
  2. **Dual-Condition Edge Admission**:
     An undirected kNN edge $(i, j)$ is admitted if:
     $$w_{ij} \ge \tau_{\text{eff}} \quad \text{OR} \quad \big(\text{rank}_{ij} = 0 \;\land\; w_{ij} > \mu_{\text{bg}}\big)$$
     *(The second clause preserves top-1 nearest-neighbor weak ties so valid peripheral themes are not orphaned prior to Leiden community detection.)*
  3. **Excess-Above-Threshold Edge Weighting**:
     $$w^{\text{excess}}_{ij} = \max\!\left(10^{-4},\; w_{ij} - \tau_{\text{eff}} + 0.05\right)$$
     Leiden modularity optimization operates on $w^{\text{excess}}_{ij}$ rather than raw $w_{ij}$, amplifying true local density gradients above the background floor.

---

### 1.2 Leave-One-Out (LOO) Self-Bias-Corrected Centroid Similarity
* **Location**: `src/extraction/clustering.py` (`ThemeClusteringEngine._build_partition_dict`)
* **Challenge Solved**: Because a cluster's spherical centroid $\mathbf{c} = \frac{\sum_{j \in C} \mathbf{e}_j}{\|\sum_{j \in C} \mathbf{e}_j\|_2}$ includes member $\mathbf{e}_i$ itself, raw cosine similarity $s_i = \mathbf{e}_i^\top \mathbf{c}$ has an artificial upward self-bias of $\approx \frac{1}{\sqrt{n_c}}$ in small clusters ($n_c \le 5$).
* **Variables Combined (3)**:
  1. Unnormalized cluster resultant vector length ($R_c = \|\sum_{j \in C} \mathbf{e}_j\|_2$)
  2. Raw theme-to-centroid cosine similarity ($s_i = \mathbf{e}_i^\top \mathbf{c}$)
  3. Cluster cardinality ($n_c$)
* **Process & Formula**:
  For $n_c > 1$ and $R_c > 0$, each theme's exact Leave-One-Out similarity to the centroid of the remaining $n_c - 1$ members ($\mathbf{c}_{-i}$) is computed analytically in $O(1)$ time:
  $$s_i^{\text{LOO}} = \frac{R_c \, s_i - 1}{\sqrt{\max\!\big(10^{-12},\; R_c^2 - 2 R_c s_i + 1\big)}}$$

---

### 1.3 Core-Gated Maximal Marginal Relevance (MMR) Selection
* **Location**: `src/extraction/clustering.py` (`ThemeClusteringEngine._select_mmr_indices`)
* **Challenge Solved**: Selecting top-ranked themes purely by centroid similarity ($s_i$) yields near-duplicate paraphrases of the cluster medoid, starving downstream Map-Reduce synthesis of distinct sub-themes within the cluster. Conversely, unconstrained diversity selection can pull in noisy boundary outliers.
* **Variables Combined (5)**:
  1. Theme centroid similarity ($s_i$)
  2. Cluster similarity mean ($\mu_c$) and standard deviation ($\sigma_c$)
  3. `MANIFOLD_EDGE_THRESHOLD` ($\tau_0$)
  4. Maximum pairwise cosine similarity to already-selected themes ($\max_{j \in S} \mathbf{e}_i^\top \mathbf{e}_j$)
  5. Relevance-vs-diversity trade-off parameter ($\lambda = 0.50$)
* **Process & Formula**:
  1. **Core Candidate Pool Filter**:
     $$\text{CorePool} = \left\{ i \in C \;\middle|\; s_i \ge \min\!\big(\mu_c - 2\max(\sigma_c, 0.05),\; \tau_0\big) \right\}$$
  2. **Medoid Seed + Iterative MMR Ranking**:
     Seeds $S = \{ \arg\max_{i \in \text{CorePool}} s_i \}$ and iteratively selects:
     $$i^* = \arg\max_{i \in \text{CorePool} \setminus S} \left[ \lambda \, s_i - (1 - \lambda) \max_{j \in S} (\mathbf{e}_i^\top \mathbf{e}_j) \right]$$
  * Used to select cluster `exemplars` (`_EXEMPLARS_PER_CLUSTER`), Phase 2A Map themes (`MAP_MAX_THEMES_PER_CLUSTER`), and Phase 2B Reduce anchors (`_REDUCE_ANCHOR_THEME_COUNT`).

---

### 1.4 von Mises-Fisher (vMF) Dual-Gate Theme Confidence Classification
* **Location**: `src/extraction/pipeline.py` (`SemanticPrismStage1Pipeline._export_cluster_theme_mapping`)
* **Challenge Solved**: Classifying cluster membership confidence into `"high"`, `"medium"`, `"low"`, and `"unassigned"` requires adapting to both the cluster's internal dispersion ($\mu_c, \sigma_c$) and finite-sample self-bias ($s_i^{\text{LOO}}$).
* **Variables Combined (5)**:
  1. Raw centroid similarity ($s_i$)
  2. Leave-One-Out similarity ($s_i^{\text{LOO}}$)
  3. Cluster mean similarity ($\mu_c$) and standard deviation ($\sigma_c$)
  4. Effective manifold edge threshold ($\tau_{\text{eff}}$)
  5. Cluster cardinality ($n_c$)
* **Process & Formula**:
  1. **Self-Calibrating Band Cutoffs**:
     $$\tau_{\text{high}} = \max(\mu_c - 10^{-4},\; \tau_{\text{eff}})$$
     $$\tau_{\text{med}} = \max(\mu_c - \max(\sigma_c, 0.05),\; \tau_{\text{eff}})$$
     $$\tau_{\text{low}} = \max(\mu_c - 2\max(\sigma_c, 0.05),\; 0.5\tau_{\text{eff}})$$
  2. **Dual-Gate Classification**:
     * **`"high"`**: $s_i \ge \tau_{\text{high}}$ **and** $(n_c = 1 \;\lor\; s_i^{\text{LOO}} \ge \tau_{\text{eff}})$
     * **`"medium"`**: $s_i \ge \tau_{\text{med}}$ **and** $(n_c = 1 \;\lor\; s_i^{\text{LOO}} \ge 0.5\tau_{\text{eff}})$
     * **`"low"`**: $s_i \ge \tau_{\text{low}}$
     * **`"unassigned"`**: $s_i < \tau_{\text{low}}$

---

## 2. Stage 2 (`src/refinement/`) — SVO Vector Clustering & Taxonomic Lifting

### 2.1 Dimension- & Role-Calibrated Anisotropy Distance Cutoffs (`rescue_tau` & `tau_macro`)
* **Location**: `src/refinement/pipeline.py` (`RefinementPipeline.resolve_clustering_params` & `RefinementPipeline._calibrate_distance_thresholds`)
* **Challenge Solved**: Static cosine distance cutoffs for Tier 1 outlier rescue (`rescue_tau`) and Tier 2 macro-agglomeration (`tau_macro`) break when switching between embedding models with different cone geometries (e.g., narrow-cone `e5`/`nomic` vs wide-cone `bge-large` vs low-dim `MiniLM`) or grammatical roles (`Predicate` verbs vs `Subject`/`Object` nouns).
* **Variables Combined (6)**:
  1. Embedding dimensionality ($D \in \{\le 512,\, 513\text{–}768,\, >768\}$)
  2. Grammatical component role (`Predicate` vs `Subject`/`Object`)
  3. Empirical 1-NN cosine distance percentiles ($P_{35}(\mathbf{d}^{(1)}), P_{60}(\mathbf{d}^{(1)})$)
  4. Background off-diagonal cosine distance mean ($\mu_d$) and standard deviation ($\sigma_d$)
  5. Dimension/role prior cutoffs ($\tau_{\text{rescue}}, \tau_{\text{macro}}$)
* **Process & Formula**:
  1. **Tier 1 Outlier Rescue Cutoff ($\tau_{\text{rescue}}^{\text{eff}}$)**:
     $$\tau_{\text{rescue}}^{\text{emp}} = \min\!\big(\mu_d - 1.0\sigma_d,\; P_{35}(\mathbf{d}^{(1)})\big)$$
     $$\tau_{\text{rescue}}^{\text{eff}} = \text{clip}\!\left(\tau_{\text{rescue}}^{\text{emp}},\; \max(0.04, 0.50\tau_{\text{rescue}}),\; \min(1.35\tau_{\text{rescue}}, \mu_d - 0.75\sigma_d)\right)$$
  2. **Tier 2 Macro-Agglomeration Cutoff ($\tau_{\text{macro}}^{\text{eff}}$)**:
     $$\tau_{\text{macro}}^{\text{emp}} = \min\!\big(\mu_d - 0.65\sigma_d,\; P_{60}(\mathbf{d}^{(1)})\big)$$
     $$\tau_{\text{macro}}^{\text{eff}} = \text{clip}\!\left(\tau_{\text{macro}}^{\text{emp}},\; \max(1.15\tau_{\text{rescue}}^{\text{eff}}, 0.50\tau_{\text{macro}}),\; \min(1.30\tau_{\text{macro}}, \mu_d - 0.50\sigma_d)\right)$$

---

### 2.2 Vocabulary-Scale-Adaptive UMAP Neighborhoods (`n_neighbors` & `n_components`)
* **Location**: `src/refinement/pipeline.py` (`RefinementPipeline._cluster_field`)
* **Challenge Solved**: On small-to-medium vocabularies ($N = 15\text{–}60$), a fixed `n_neighbors = 15` or `20` spans $25\%\text{–}100\%$ of the entire vocabulary, smoothing away local manifold structure.
* **Variables Combined (3)**:
  1. Unique normalized component vocabulary size ($N = \text{num\_terms}$)
  2. Dimension-tier prior `umap_n_neighbors` ($k_{\text{prior}} \in \{15, 20\}$)
  3. Local neighborhood fraction ceiling ($\max(3, \lfloor N / 4 \rfloor)$ for $N \ge 12$)
* **Process & Formula**:
  $$k_{\text{adaptive}} = \lfloor 2 + 2.5\ln(\max(2, N)) \rfloor$$
  $$n_{\text{neighbors}} = \min\!\left(k_{\text{prior}},\; \max\!\big(2,\, \min(k_{\text{adaptive}},\, k_{\text{local\_max}},\, N - 1)\big)\right)$$
  $$n_{\text{components}} = \min\!\left(\text{clip}(\text{round}(\ln(\max(4, N))), 2, 8),\; \max(2, N - 2)\right)$$

---

### 2.3 Frequency-Weighted Spherical Fréchet Mean & LOO Centroid Distance
* **Location**: `src/refinement/pipeline.py` (`RefinementPipeline.compute_spherical_centroid` & `RefinementPipeline._compute_loo_distances`)
* **Challenge Solved**: Selecting a cluster's prototypical exemplar term (`centroid_term`) and `top_3_centroid_fallbacks` must account for both vector centrality and corpus term frequency ($f_i$), without allowing a single high-frequency noisy term to pull the centroid onto itself in small clusters.
* **Variables Combined (3)**:
  1. L2-normalized term embedding ($\mathbf{v}_i$)
  2. Log-dampened corpus frequency weight ($w_i = 1.0 + \ln(f_i)$)
  3. Leave-One-Out weighted resultant vector ($\mathbf{s}_{-i} = \sum_{j \in C} w_j \mathbf{v}_j - w_i \mathbf{v}_i$)
* **Process & Formula**:
  $$\mathbf{c}_{-i} = \frac{\mathbf{s}_{-i}}{\|\mathbf{s}_{-i}\|_2}, \qquad d_i^{\text{LOO}} = \text{clip}\!\left(1.0 - \mathbf{v}_i^\top \mathbf{c}_{-i},\; 0.0,\; 2.0\right)$$
  * Terms with the lowest $d_i^{\text{LOO}}$ are selected as `centroid_term` and `top_3_centroid_fallbacks`.

---

### 2.4 Seeded MMR Payload Ordering & Context-Budget Packing
* **Location**: `src/refinement/pipeline.py` (`RefinementPipeline._select_mmr_order` & `RefinementPipeline._lift_component_clusters`)
* **Challenge Solved**: Ordering or truncating `cluster_terms` alphabetically hides the semantic breadth of large clusters from the LLM during taxonomic hypernym lifting.
* **Variables Combined (4)**:
  1. LOO centrality score ($c_i = 1.0 - d_i^{\text{LOO}}$)
  2. Top-3 centroid fallback seed indices ($S_0 = \text{argsort}(\mathbf{d}^{\text{LOO}})[:3]$)
  3. Maximum pairwise similarity to already-ordered cluster terms ($\max_{j \in S} \mathbf{v}_i^\top \mathbf{v}_j$)
  4. Tokenizer-calibrated `max_payload_tokens` (`ContextBudget`)
* **Process & Formula**:
  Seeds the ordered list with $S_0$, then iteratively appends:
  $$i^* = \arg\max_{i \in C \setminus S} \left[ 0.50 \, c_i - 0.50 \max_{j \in S} (\mathbf{v}_i^\top \mathbf{v}_j) \right]$$
  and greedily packs terms in that MMR order up to `max_payload_tokens`.

---

### 2.5 Contextual Sentence Grounding Fidelity ($\gamma_e$)
* **Location**: `src/refinement/pipeline.py` (`RefinementPipeline._compute_and_cache_sentence_context`)
* **Challenge Solved**: After taxonomic hypernym lifting, some triples may drift semantically from their original source sentence. Downstream graph construction (Stage 3) and schema synthesis (Stage 4) need an empirical grounding score for each triple.
* **Variables Combined (2)**:
  1. L2-normalized embedding of the lifted SVO phrase ($\mathbf{v}_{\text{phrase}} = \text{Embed}(\text{"subject predicate object"})$)
  2. L2-normalized embedding of the verbatim `source_quote` sentence ($\mathbf{v}_{\text{quote}} = \text{Embed}(\text{source\_quote})$)
* **Process & Formula**:
  $$\gamma_e = \text{clip}\!\left(\mathbf{v}_{\text{phrase}}^\top \mathbf{v}_{\text{quote}},\; 0.0,\; 1.0\right)$$

---

## 3. Stage 4 (`src/synthesis/`) — Ontological Schema Synthesis & AST Sanitization

### 3.1 Cross-Cluster C-TF-IDF Domain Term Ranking (`top_domain_terms`)
* **Location**: `src/synthesis/pipeline.py` (`SynthesisPipeline.execute_path_synthesis` $\rightarrow$ `extract_top_domain_terms`)
* **Challenge Solved**: Raw node degree within a community is dominated by generic hub entities (e.g., `"Child"`, `"Clinician"`, `"System"`) that appear across many communities. Even though Leiden assigns each node to a single community's `nodes` list, hub concepts bleed across communities via cross-cluster edges and source sentences.
* **Variables Combined (4)**:
  1. Intra-community triplet degree ($d_{\text{local}}(u)$: count of community triplets where $u$ is subject or object)
  2. Cross-community presence count ($N_{\text{clusters}}(u)$: count of communities where $u$ is a member node, endpoints a cross-community edge in `refined_triplets`, or appears lexically in the community's `source_quote` text)
  3. IDF penalty exponent ($\alpha = \text{DOMAIN\_TERMS\_IDF\_EXPONENT} = 1.5$)
  4. Plural-stem deduplication (`seen_stems`) and closed-class pronoun filtering
* **Process & Formula**:
  $$\text{Score}_{\text{term}}(u) = \frac{d_{\text{local}}(u) + 1.0}{\big(N_{\text{clusters}}(u)\big)^{\alpha}}$$
  * Ranks and deduplicates the top `TOP_DOMAIN_TERMS_COUNT` (`5`) community-specific entities.

---

### 3.2 Fidelity- & Entity-Overlap-Weighted Sentence Ranking (`thematic_observations`)
* **Location**: `src/synthesis/pipeline.py` (`SynthesisPipeline.execute_path_synthesis` $\rightarrow$ `extract_thematic_observations`)
* **Challenge Solved**: A community can reference hundreds of raw sentences, exceeding the LLM context window or diluting schema synthesis with peripheral remarks. Sentences must be ranked by both their semantic grounding quality and their density of community-specific entities.
* **Variables Combined (5)**:
  1. Mean Stage 2 `contextual_fidelity` across all community triplets citing sentence $q$ ($\bar{\gamma}_q = \frac{1}{|T_q|}\sum_{e \in T_q} \gamma_e$)
  2. Distinct community node overlap count in sentence $q$ ($\text{overlap}(q, V_c) = \sum_{u \in V_c} \mathbb{I}[u \subset \text{lower}(q)]$)
  3. Overlap boost weight ($w_{\text{ov}} = \text{OBSERVATION\_OVERLAP\_WEIGHT} = 0.25$)
  4. Overlap saturation cap ($C_{\text{ov}} = \text{OBSERVATION\_OVERLAP\_MAX\_CAP} = 4$)
  5. Minimum sentence length filter (`MIN_SENTENCE_LENGTH = 15`)
* **Process & Formula**:
  $$\text{Score}_{\text{sent}}(q) = \bar{\gamma}_q \times \left(1.0 + w_{\text{ov}} \cdot \min\!\big(C_{\text{ov}},\, \text{overlap}(q, V_c)\big)\right)$$
  * Selects the top `MAX_SENTENCES_PER_TARGET` (`200`) sentences, groups them by `theme_association`, and deterministically places `'Other'` last.

---

### 3.3 Multi-Criteria AST Field Filtering, Entropy Collapse & Priority Ranking
* **Location**: `src/synthesis/sanitizers.py` (`_sanitize_ast_models`)
* **Challenge Solved**: LLMs frequently flatten nested sub-entities into long runs of prefixed fields (`sub_entity_name`, `sub_entity_role`, `sub_entity_status`), emit boolean checklist dumps (`outcome_a_achieved: bool`, `outcome_b_achieved: bool`), or exceed the 15-field cognitive/schema cap (`MAX_BASEMODEL_FIELDS_PER_CLASS`).
* **Variables Combined (5)**:
  1. Shared 2-token `snake_case` prefix frequency (`raw_prefix2_counts[p1_p2]`) for $\ge 3$-token field names
  2. Shared non-exempt suffix token frequency (`raw_suffix_counts[suffix]`) for $\ge 2$-token field names
  3. Boolean type indicator ($\mathbb{I}[\text{is\_bool}]$)
  4. Identifier token length ($\text{len}(\text{parts})$)
  5. Original AST declaration index ($\text{idx}$)
* **Process & Formula**:
  1. **Pass 1 (Structural Prefix/Suffix Entropy Collapse)**:
     * If $\text{raw\_prefix2\_counts}[p_1\_p_2] \ge 3$ (or $\text{raw\_suffix\_counts}[\text{suffix}] \ge 3$ for non-exempt suffixes), all matching fields are collapsed into a single pluralized collection field (`<prefix2>s: Optional[List[str]]` or `<suffix>s: Optional[List[str]]`).
     * Remaining fields sharing a 2-token prefix or non-exempt suffix are capped at `2`.
  2. **Pass 2 (Lexicographic Priority Ranking when $> 15$ fields survive)**:
     Surviving fields are sorted ascending by the 4-tuple priority key:
     $$\text{PriorityKey}(f) = \Big(\mathbb{I}[\text{is\_bool}],\; \mathbb{I}\big[\text{len}(\text{parts}) \ge 3 \land \text{raw\_prefix2\_counts}[p_1\_p_2] > 1\big],\; \mathbb{I}\big[\text{len}(\text{parts}) \ge 3\big],\; \text{idx}\Big)$$
     * Retains the top `MAX_BASEMODEL_FIELDS_PER_CLASS` (`15`) core/typed/collection fields (restored to their original declaration order) and strips lower-priority boolean/compound overflow fields.

---

### 3.4 Multi-Signal Module Name Disambiguation
* **Location**: `src/synthesis/sanitizers.py` (`_resolve_unique_module_name`)
* **Challenge Solved**: Multiple graph communities in the same domain often receive identical or generic `module_name` strings from the LLM (e.g., `clinical_assessment` or `community_schema`), which would overwrite or collide when saved to `01_<module>.py`, `02_<module>.py`, etc.
* **Variables Combined (4)**:
  1. Sanitized LLM `module_name` (`candidate`) and generic-name detection
  2. AST-extracted `BaseModel` class names in `snake_case` (`model_names`)
  3. Community C-TF-IDF `top_domain_terms` (`term_slugs`)
  4. Set of already-assigned module names across the run (`used_names`)
* **Process & Resolution Cascade**:
  Evaluates a 4-tier cascade against `used_names`:
  1. Non-generic LLM `candidate` if unused
  2. Primary AST `BaseModel` class name (`model_names[0]`, `model_names[1]`, ...)
  3. Non-overlapping token combination of top C-TF-IDF term + primary `BaseModel` name (`<term_tokens>_<model_name>`) or `<candidate>_<term_tokens>`
  4. Compound pair of top two `BaseModel` names (`<model_0>_<model_1>`) or deterministic `<base>_community_<id>` fallback.

