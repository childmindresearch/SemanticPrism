# HiTOP Clinical Ontology Integration Plan

## Overview

HiTOP (Hierarchical Taxonomy of Psychopathology) is a dimensional, top-down clinical classification system that organizes mental health psychopathology into a multi-level hierarchy: **Spectra** (Internalizing, Externalizing, Thought Disorder, Somatoform, etc.) → **Subfactors** (Distress, Fear, etc.) → **Syndromes/Components** (Major Depression, Social Anxiety, etc.) → **Symptoms/Traits**.

SemanticPrism's pipeline is entirely **bottom-up**: unstructured text → themes → triples → clusters → hypernyms → graph communities → Pydantic schemas. The ontology that emerges is shaped purely by corpus statistics and LLM abstraction, with no external clinical knowledge imposed.

This plan bridges the two: it maps bottom-up discovered schemas into HiTOP's clinically defined structure, adding a clinical grounding layer without discarding the data-driven discovery.

---

## Design Philosophy Alignment

SemanticPrism's existing patterns dictate how HiTOP integration must be implemented:

| SemanticPrism Principle | HiTOP Integration Application |
|---|---|
| **"LLM as gate, not oracle"** | Geometric matching (embedding similarity) proposes HiTOP bins; LLM validates or rejects the proposal |
| **"Determinism where possible"** | HiTOP hierarchy encoded as a static, versioned knowledge asset (YAML) — not re-discovered each run |
| **"Every phase writes diagnostic JSON"** | HiTOP mapping output logged with confidence scores, provenance, and unmapped items |
| **"Progressive abstraction"** | HiTOP mapping layers in at the community level (not raw text), preserving the abstraction pipeline |
| **"Fallback to geometric centroid"** | Communities that fail LLM validation for HiTOP mapping retain their data-derived label — never discarded |
| **"No silent failure"** | Binning errors log and degrade gracefully; synthesis proceeds without clinical annotations |

---

## Architecture

### Why Post-Community (Stage 5.5) as Primary Integration Point

Communities are the atomic unit of generated schemas — one community produces one schema module. Binning at the community level means every schema receives a single clinical classification. Binning post-synthesis would require re-parsing generated Python code to extract and classify schemas, which is fragile, violates the "determinism where possible" principle, and introduces a new failure mode (code parsing errors).

The post-community, pre-synthesis slot is optimal because:
- Community membership is already resolved and validated
- The semantic content of each community is fully abstracted (hypernyms applied)
- The mapping feeds directly into the synthesis prompt, enriching generated code
- No downstream stages depend on HiTOP — it's purely additive

### Why LLM Validation of Geometric Proposals

Pure embedding similarity correctly maps "depressed mood" to "Major Depressive Disorder," but may mis-map "antidepressant medication" to a clinical syndrome rather than a pharmacological category, or map "insurance billing" to Externalizing because of co-occurring entities. The LLM gate catches these semantic boundary violations — identical to the pattern established in Stage 3.1 (cluster contextual validation).

### Why Not Force Every Community into a HiTOP Bin

HiTOP covers psychopathology. A clinical corpus may produce communities about "electrocardiogram results," "hospital scheduling," or "billing codes" that have no HiTOP mapping. Forcing them would create false clinical signal. Unmapped communities are a feature, not a bug — they reveal which parts of the corpus extend beyond the clinical ontology.

### Shared Encoder Instance

HiTOP node embeddings must be computed with the same `SentenceTransformer` model (`BAAI/bge-m3`) used in Stage 2 for theme and triple embeddings. Loading a second instance of the ~1GB model wastes memory. The encoder is shared via the existing `EmbeddingPipeline` or a module-level singleton.

---

## Implementation Plan

### Prerequisite: HiTOP Knowledge Asset

**File**: `src/hitop/hitop_hierarchy.yaml`

A structured, versioned YAML representation of the HiTOP hierarchy. Each node includes:

```yaml
spectra:
  - name: "Internalizing"
    description: "Disorders of emotion, mood, and anxiety"
    subfactors:
      - name: "Distress"
        description: "Emotional suffering and internal distress"
        syndromes:
          - name: "Major Depressive Disorder"
            description: "Persistent depressed mood and anhedonia"
            synonyms: ["Depression", "MDD"]
            icd_mappings: ["F32", "F33"]
          - name: "Generalized Anxiety Disorder"
            synonyms: ["GAD", "Free-floating anxiety"]
            icd_mappings: ["F41.1"]
      - name: "Fear"
        syndromes:
          - name: "Panic Disorder"
          - name: "Social Anxiety Disorder"
          - name: "Specific Phobia"
  - name: "Externalizing"
    subfactors:
      - name: "Antagonism"
      - name: "Disinhibition"
  - name: "Thought Disorder"
    subfactors:
      - name: "Psychoticism"
  - name: "Somatoform"
```

Each node gets an embedding pre-computed at pipeline startup (using the shared encoder) and persisted to disk so re-embedding only happens when the YAML asset changes.

---

### Task 1: Create HiTOP Knowledge Asset

- **Goal**: Encode the HiTOP hierarchy as a structured YAML file.
- **New files**: `src/hitop/hitop_hierarchy.yaml`, `src/hitop/__init__.py`
- **Modified files**: none
- **Parallel group**: A (no dependencies)
- **Definition of Done**:
  - [ ] YAML contains all major HiTOP spectra (Internalizing, Externalizing, Thought Disorder, Somatoform, and others)
  - [ ] Each spectrum has ≥2 subfactors where defined by the HiTOP model
  - [ ] Each subfactor has ≥2 example syndromes or components
  - [ ] Each node has `name`, `description`, and `synonyms` fields
  - [ ] File passes `yaml.safe_load()` without error
  - [ ] Indentation and comment conventions match `config.yaml`
  - [ ] `__init__.py` marks `src/hitop/` as a package

---

### Task 2: Build HiTOP Embedding Index Module

- **Goal**: Load the YAML hierarchy, embed all nodes via the shared SentenceTransformer, persist embeddings, and support fast cosine-similarity queries.
- **New files**: `src/hitop/embedding_index.py`
- **Modified files**: `src/embedding/embedding.py` — add a method or factory to share the `SentenceTransformer` encoder instance
- **Parallel group**: B (depends on Task 1)
- **Definition of Done**:
  - [ ] `HiTOPEmbeddingIndex` class loads `hitop_hierarchy.yaml` and flattens all nodes into a list of hierarchy-aware text strings: `"Spectrum: {name}. Subfactor: {name}. Syndrome: {name}. Description: {desc}. Synonyms: {syns}"`
  - [ ] Embeddings computed using the same model as `EmbeddingPipeline` (shared instance, not a second load)
  - [ ] Embeddings cached to `outputs/embeddings/hitop_embeddings.npy` with corresponding `outputs/embeddings/hitop_node_index.json` mapping array indices to full node metadata (name, level, parent path)
  - [ ] `query(text, top_k=5)` method accepts a text string, embeds it, returns the top-k closest HiTOP nodes with cosine similarity scores and full metadata
  - [ ] Unit test: querying `"depressed mood anhedonia loss of interest"` returns Internalizing → Distress → Major Depressive Disorder within top-3 candidates with similarity > 0.3
  - [ ] Embeddings are lazy-loaded: first access triggers computation and persistence; subsequent accesses load from disk unless the YAML file has been modified (check via modification timestamp)

---

### Task 3: Implement Stage 5.5 — Clinical Binning Module

- **Goal**: Map each Stage 5 community to a HiTOP classification via geometric proposal + LLM validation.
- **New files**: `src/hitop/binning.py`, `src/hitop/schemas.py`
- **Modified files**:
  - `src/orchestrator/pipeline.py` — insert Stage 5.5 call between topology block and synthesis block
  - `config.yaml` — add `hitop:` config section
- **Parallel group**: C (depends on Task 2)
- **Definition of Done**:
  - [ ] `binning.py` defines `ClinicalBinningPipeline` with `async bin_communities(hierarchy_payload, master_domain)` method
  - [ ] Community composite embedding uses same frequency-weighted mean formula as Stage 3.2 geometric centroid: `mean_vector = (1 / Σf_i) · Σ(f_i · embedding_i)`, reusing `FREQUENCY_REGISTRY`
  - [ ] Top-3 HiTOP candidates retrieved via `HiTOPEmbeddingIndex.query()`
  - [ ] LLM validation prompt structured with: community member entity list, relationships, HiTOP candidate name + description. Asks: *"Does this community of concepts fit within the HiTOP category '{name}'? If yes, at which level (spectrum/subfactor/syndrome)? If no, suggest a better HiTOP fit or mark unmapped. Provide reasoning."*
  - [ ] LLM response parsed via Pydantic model `HiTOPMappingValidation` with fields: `mapped: bool`, `hitop_spectrum: str`, `hitop_subfactor: Optional[str]`, `hitop_syndrome: Optional[str]`, `confidence: float`, `reasoning: str`
  - [ ] If `mapped == False`, community recorded in `unmapped_communities` list — no data loss, proceeds to synthesis without HiTOP annotation
  - [ ] Output written to `outputs/05_topology/hitop_binning.json` with structure per community: `{community_id, community_title, hitop_classification, confidence, validation_status, member_entities, alternative_mappings}`
  - [ ] Pipeline integration: new stage inserted between topology and synthesis in `pipeline.py`, with try/except that logs error and skips binning (passes `None` to synthesis) if the entire binning stage fails
  - [ ] Config section added to `config.yaml`:
    ```yaml
    hitop:
      enabled: true
      mapping_confidence_threshold: 0.5
      require_llm_validation: true
      enable_theme_check: true
      theme_consistency_threshold: 0.3
    ```

---

### Task 4: Enrich Stage 6 Schema Synthesis with HiTOP Annotations

- **Goal**: Inject HiTOP classification context into the synthesis prompt; annotate generated schemas with clinical metadata.
- **Modified files**:
  - `src/synthesis/schemas.py` — add `hitop_classification` field to `GeneratedSchema`
  - `src/synthesis/prompts.py` — add HiTOP context injection block
  - `src/synthesis/synthesizer.py` — pass HiTOP mapping into prompt construction, handle empty/no-mapping gracefully
  - `src/orchestrator/pipeline.py` — pass binning result into `generate_schemas()` call
- **Parallel group**: C (depends on Task 3)
- **Definition of Done**:
  - [ ] `GeneratedSchema` gains new optional field: `hitop_classification: Optional[Dict[str, Any]] = None`
  - [ ] Synthesis prompt includes a `"=== HiTOP Clinical Context ==="` section when a mapping exists for the community
  - [ ] Prompt instructs LLM to include `# HiTOP: {spectrum} > {subfactor} > {syndrome}` in generated code docstrings
  - [ ] If no mapping exists, prompt omits HiTOP context entirely — graceful degradation, no empty fields
  - [ ] `SynthesisEngine.generate_schemas()` accepts optional `hitop_mappings` parameter (type: `Optional[List[Dict]]`)
  - [ ] Pipeline passes `hitop_binning.json` data into synthesis call
  - [ ] Master context output file (`semantic_prism_master_graph.json`) includes `hitop_mappings` summary section
  - [ ] Unit test: when `hitop_classification` is provided, generated schema includes the metadata; when absent, schema generates identically to pre-Integration behavior

---

### Task 5: Optional Stage 1.5 — Theme-to-HiTOP Early Consistency Check

- **Goal**: Diagnostic check after master theme consolidation — determine whether the corpus is clinically relevant by checking if master themes map to any HiTOP spectrum.
- **New files**: `src/hitop/theme_check.py`
- **Modified files**:
  - `src/orchestrator/pipeline.py` — insert diagnostic call after theme consolidation, before triple extraction
- **Parallel group**: C (depends on Task 2; can run in parallel with Tasks 3 and 4)
- **Definition of Done**:
  - [ ] `theme_check.py` exports `async check_theme_hitop_consistency(master_themes, master_domain, threshold=0.3)`
  - [ ] For each master theme, composite text `"Theme: {theme}. Domain: {domain}"` is embedded and compared against HiTOP spectrum-level embeddings
  - [ ] Results written to `outputs/01_extraction/hitop_theme_consistency.json` with structure: `{"domain_relevance": "clinical" | "non_clinical" | "mixed", "theme_mappings": [{theme, best_hitop_spectrum, similarity}]}`
  - [ ] Logged at INFO level: `"HiTOP consistency: {count_mapped}/{total} master themes map to clinical spectra"`
  - [ ] Check runs silently on error — does not block or alter downstream pipeline behavior
  - [ ] Config boolean `hitop.enable_theme_check` gates whether this check runs

---

## Execution Order

```
Phase A: Task 1 (HiTOP YAML asset)
Phase B: Task 2 (Embedding Index)
Phase C: Task 3 (Binning Module) + Task 4 (Synthesis Enrichment) + Task 5 (Theme Check)
  └── Tasks 3 and 4 are sequential (synthesis depends on binning output)
  └── Task 5 is independent and can run parallel to 3 and 4
```

---

## Risk Assessment

| Risk | Mitigation |
|---|---|
| HiTOP YAML is incomplete or outdated | Version the YAML asset; design for extensibility (easy to add syndromes/subfactors) |
| Embedding index fails to load | Lazy-load with fallback; if index unavailable, binning is skipped (graceful degradation) |
| LLM rejects all mappings | Unmapped communities proceed to synthesis unchanged — no pipeline block |
| Shared encoder causes race conditions | EmbeddingPipeline loads encoder before binning stage runs; HiTOP index loads after |
| HiTOP mappings are inaccurate | Confidence scores logged; LLM validation reasoning recorded; unmapped items flagged for review |

---

## Success Criteria

1. **Pipeline completes without regression** — all existing stages produce identical output when HiTOP binning is disabled (`hitop.enabled: false`)
2. **Clinical binning output** — `hitop_binning.json` contains mappings with confidence scores and validation reasoning for each community
3. **Schema enrichment** — generated Pydantic models include HiTOP classification metadata in docstrings when a mapping exists
4. **Diagnostic visibility** — theme consistency check reports whether the corpus is clinically relevant
5. **Zero data loss** — unmapped communities are never discarded; they proceed through synthesis without clinical annotation
6. **Configurable** — all HiTOP behavior controlled via `config.yaml`; disabling HiTOP restores pre-integration pipeline behavior
