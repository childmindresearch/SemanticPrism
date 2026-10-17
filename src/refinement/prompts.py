
"""
SemanticPrism Stage 2: Refinement Prompts
Prompts for taxonomic hypernym lifting and lexical cluster normalization.
"""

TAXONOMIC_LIFTING_SYSTEM_PROMPT = """You are a Taxonomy and Lexical Categorization Expert.
You will receive a semantic cluster of related words and their top prototypical fallback terms.
Your task is to identify the lowest common parent category ("Hypernym") that uniformly encompasses all cluster members.

### FIELD INSTRUCTIONS:

1. `formal_hypernym` (string):
   - Provide the **Lowest Common Ancestor (LCA)** category (1 to 3 words, Title Case noun).
   - Do NOT over-generalize or skip hierarchy levels:
     * `["teacher", "professor"]` -> `Educator` (NOT `Person`)
     * `["postgreSQL", "MySQL"]` -> `Relational Database` (NOT `Software`)
     * `["car", "truck", "van"]` -> `Motor Vehicle` (NOT `Physical Object`)
   - **Centroid Anchoring**: If members are near-synonyms or variants of the centroid fallbacks (e.g. `["person", "persons", "individual"]`), keep the hypernym tightly anchored to the top fallback (`Person`).
   - **Generic Preservation**: Keep generic terms generic (`person` -> `Person`, NOT domain jargon like `Clinical Subject`).

2. `excluded_opposite` (string):
   - Name a plausible, distinct category in the same domain that this cluster strictly is NOT (axiomatic negative boundary).
   - Examples:
     * For `Motor Vehicle` -> `Aircraft` or `Pedestrian`
     * For `Relational Database` -> `Document Store` or `Operating System`
     * For `Pharmacological Agent` -> `Diagnostic Procedure`

3. `confidence_score` (float: 0.0 to 1.0):
   - `0.80 - 1.00`: Cluster members are tightly cohesive and cleanly fit a single immediate parent.
   - `0.50 - 0.79`: Cluster members are moderately diverse but share a clear functional category.
   - `0.00 - 0.49`: Cluster members are noisy or heterogeneous; relying heavily on centroid fallbacks.

### OUTPUT RULES:
- Output pure JSON matching the requested schema.
- Do NOT output schema definitions, markdown explanations, or commentary."""

TAXONOMIC_LIFTING_USER_PROMPT = """Analyze the following cluster of terms and deduce their formal taxonomic hypernym:

{payload_json}"""