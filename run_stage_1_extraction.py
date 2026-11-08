#!/usr/bin/env python3
"""
SemanticPrism Stage 1: Full Extraction Pipeline (Powered by PrismRunner)
========================================================================
Performs the exact Stage 1 Extraction workflow from:
    https://github.com/childmindresearch/SemanticPrism

Workflow Phases:
  - Phase 1   : Global Theme Discovery (per-document chunks -> themes/{doc}_themes.json)
  - Phase 1.5 : Theme Aggregation (-> all_themes.json)
  - Phase 1.8 : Empirical Vector Theme Clustering (Dynamic kNN/Leiden -> clustering_summary.json)
  - Phase 2   : Master Theme Synthesis (Map-Reduce -> cluster_candidates.json, master_themes.json, theme_mapping_clusters.json)
  - Phase 3   : Global SVO Triple Extraction with Master Theme Association (-> triples/{doc}_triplets.json)
  - Phase 3.5 : Triple Aggregation (-> original_triplets.json, triplet_counts.csv)

All LLM, input file/columns, output locations, chunking, and clustering parameters
are configured as global variables at the top of this file.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# Prevent PyTorch CUDA allocator fragmentation when switching between LLM and embedding stages
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

# Ensure repository root is importable when executed directly
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prismrunner import (
    ContextBudget,
    PrismPipeline,
    clear_device_memory,
    get_optimal_device,
    resolve_dynamic_batch_size,
)

# =============================================================================
# GLOBAL CONFIGURATION VARIABLES
# =============================================================================

# --- 1. LLM, Embedding & Hardware Settings ---
MODEL_NAME: str = "Qwen/Qwen2.5-3B-Instruct"            # Generative LLM for Theme Discovery, Master Synthesis & Triple Extraction
THEME_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"         # SentenceTransformer model for Phase 2 theme clustering
DEVICE: str = "auto"                                    # "auto", "cuda", "mps", or "cpu"
TORCH_DTYPE: str = "auto"                               # "auto", "bfloat16", "float16", or "float32"
LOAD_IN_4BIT: bool = False                              # Enable 4-bit NF4 quantization (CUDA bitsandbytes)
LOAD_IN_8BIT: bool = False                              # Enable 8-bit quantization (CUDA bitsandbytes)
ATTN_IMPLEMENTATION: str = "auto"                       # "auto", "flash_attention_2", "sdpa", or "eager"
BATCH_SIZE: Union[int, str] = "auto"                    # "auto" (hardware-aware) or explicit integer (e.g., 8)
CONTEXT_WINDOW_CAP: Union[int, str] = "auto"            # "auto" (native model context limit) or integer token cap
TEMPERATURE: float = 0.0                                # 0.0 for deterministic greedy decoding
CONSTRAIN_JSON: bool = True                             # Force JSON opening bracket & bracket-balanced early stopping
MANAGE_VRAM: bool = True                                # Purge GPU/MPS cache between major stage transitions
DRY_RUN: bool = False                                   # Set True to preview prompts without running LLM generation

# --- 2. Input Dataset & Output Settings ---
INPUT_PATH: str = "inputs/testdocs.parquet"             # Path to input file (.parquet, .csv, .json, .jsonl) or folder of .txt files
ID_COLUMN: str = "unique_val"                           # Column containing unique document IDs (SemanticPrism default: "unique_val")
TEXT_COLUMN: str = "report"                             # Column containing raw document text (SemanticPrism default: "report")
DOCUMENT_LIMIT: Optional[int] = None                    # Optional cap on number of documents to process (None = all)
OUTPUTS_DIR: str = "outputs"                            # Root outputs directory (writes to outputs/01_extraction, outputs/logs, etc.)
SAVE_LLM_PAYLOADS: bool = True                          # Save sample LLM prompts to outputs/payload_text
RESUME_MODE: str = "skip"                               # "skip" (reuse existing per-doc outputs) or "overwrite" (re-run all)
PRINT_CHECKPOINTS: bool = True                          # Print real-time chunk, document, and stage checkpoint progress to console

# --- 3. Chunking, Token & Theme Clustering Settings ---
THEME_CHUNK_MAX_WORDS: int = 2000                       # Max words per chunk during Phase 1 Theme Discovery
TRIPLE_CHUNK_MAX_WORDS: int = 400                       # Max words per chunk during Phase 3 SVO Triple Extraction
THEME_MAX_NEW_TOKENS: int = 512                         # Max generated tokens per theme discovery chunk
MASTER_SYNTHESIS_MAX_NEW_TOKENS: int = 512              # Max generated tokens for cluster map/reduce synthesis
TRIPLE_MAX_NEW_TOKENS: int = 768                        # Max generated tokens per triple extraction chunk
MAP_MAX_THEMES_PER_CLUSTER: int = 150                   # Max themes sent to Map specialist per cluster
MANIFOLD_EDGE_THRESHOLD: float = 0.35                   # Minimum cosine similarity for kNN theme graph edges
MICRO_CLUSTER_RATIO: float = 0.01                       # Clusters smaller than 1% of N are absorbed into nearest dominant centroid


# =============================================================================
# 1. SEMANTICPRISM STAGE 1 PYDANTIC SCHEMAS
# =============================================================================

class Theme(BaseModel):
    """Represents an isolated, high-level structural theme."""

    title: str = Field(description="The formal title of the extracted theme.")
    description: Optional[str] = Field(default="", description="A brief description of the theme.")
    reasoning: Optional[str] = Field(default="", description="Reasoning for classifying this as a critical theme.")

    @model_validator(mode="before")
    @classmethod
    def remap_and_clean(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"title": data.strip(), "description": "", "reasoning": ""}
        if isinstance(data, dict):
            title = data.get("title") or data.get("name") or data.get("theme") or data.get("theme_title") or ""
            desc = data.get("description") or data.get("desc") or data.get("summary") or data.get("scope") or ""
            reason = data.get("reasoning") or data.get("rationale") or data.get("justification") or data.get("reason") or ""
            return {
                "title": str(title).strip(),
                "description": str(desc).strip(),
                "reasoning": str(reason).strip(),
            }
        return data

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("Theme title cannot be empty")
        return v.strip()


class ThemeDiscoveryResult(BaseModel):
    """The aggregate output of the Theme Discovery phase."""

    themes: List[Theme] = Field(default_factory=list, description="List of themes discovered in the text.")

    @model_validator(mode="before")
    @classmethod
    def handle_raw_payloads(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"themes": data}
        if isinstance(data, dict):
            if "themes" not in data:
                for alt_key in ["discovered_themes", "theme_list", "raw_themes", "categories", "items", "results"]:
                    if alt_key in data and isinstance(data[alt_key], list):
                        data["themes"] = data[alt_key]
                        break
                else:
                    if any(k in data for k in ("title", "name", "theme", "theme_title")):
                        return {"themes": [data]}
        return data

    @field_validator("themes", mode="before")
    @classmethod
    def filter_and_salvage_themes(cls, v: Any) -> List[Any]:
        if not isinstance(v, list):
            return []
        valid_themes = []
        for item in v:
            try:
                if isinstance(item, str) and item.strip():
                    valid_themes.append(Theme(title=item.strip()))
                elif isinstance(item, dict):
                    valid_themes.append(Theme.model_validate(item))
                elif isinstance(item, Theme):
                    valid_themes.append(item)
            except Exception:
                pass
        return valid_themes


class ClusterCandidateSynthesis(BaseModel):
    """Intermediate candidate synthesis for an individual cluster produced during the Map phase."""

    cluster_id: int = Field(default=0, description="The numeric ID of the cluster.")
    candidate_title: str = Field(description="A concise, authoritative candidate title (3 to 6 words).")
    scope_definition: str = Field(default="", description="1-2 sentences defining what this cluster strictly covers.")
    key_inclusions: List[str] = Field(default_factory=list, description="Key concepts or mechanisms that strictly belong to this theme.")
    key_exclusions: List[str] = Field(default_factory=list, description="Concepts or mechanisms that might seem related but belong elsewhere.")
    confidence: Optional[str] = Field(default="high", description="Assessment of cluster coherence: high, medium, or low.")


class MasterThemeSynthesisResult(BaseModel):
    """The normalized global output consolidating multiple document themes."""

    master_domain: str = Field(description="The singular, overarching domain describing the entire corpus.")
    master_themes: List[str] = Field(description="The consolidated, deduplicated list of formal theme names.")


class RawTriple(BaseModel):
    """Extracts a high-fidelity subject-predicate-object relationship."""

    subject: str = Field(..., description="The primary entity or concept exactly as it appears in the source text.")
    predicate: str = Field(..., description="The relationship or action connecting subject and object.")
    object: str = Field(..., description="The target entity, concept, or value.")
    source_quote: str = Field(default="", description="The EXACT, verbatim substring from the text confirming this relationship.")
    theme_association: Optional[str] = Field(default="Other", description="The master theme this triple most closely aligns with.")

    @model_validator(mode="before")
    @classmethod
    def align_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "subject" not in data:
                for k in ("subj", "head", "entity", "source"):
                    if k in data:
                        data["subject"] = data[k]
                        break
            if "predicate" not in data:
                for k in ("pred", "relation", "relationship", "edge", "action"):
                    if k in data:
                        data["predicate"] = data[k]
                        break
            if "object" not in data:
                for k in ("obj", "tail", "target", "value"):
                    if k in data:
                        data["object"] = data[k]
                        break
            if "source_quote" not in data:
                for k in ("quote", "evidence", "sentence", "context"):
                    if k in data:
                        data["source_quote"] = data[k]
                        break
        return data

    @field_validator("subject", "predicate", "object")
    @classmethod
    def prevent_empty(cls, v: Any) -> str:
        if not v or len(str(v).strip()) == 0:
            raise ValueError("Fields cannot be empty strings")
        return str(v).strip()


class TripleExtractionResult(BaseModel):
    """The aggregate result grouping multiple triples from a single text block."""

    triples: List[RawTriple] = Field(default_factory=list, description="The complete list of extracted node-edge relationships.")

    @model_validator(mode="before")
    @classmethod
    def align_triples(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"triples": data}
        if isinstance(data, dict):
            if "triples" not in data:
                for alt_key in ("triplets", "relationships", "items", "results", "extractions"):
                    if alt_key in data and isinstance(data[alt_key], list):
                        data["triples"] = data[alt_key]
                        break
                else:
                    if any(k in data for k in ("subject", "subj", "head")):
                        return {"triples": [data]}
        return data

    @field_validator("triples", mode="before")
    @classmethod
    def drop_invalid_triples(cls, v: Any) -> List[Any]:
        if not isinstance(v, list):
            return []
        valid_chunk = []
        for item in v:
            try:
                if isinstance(item, dict):
                    valid_chunk.append(RawTriple.model_validate(item))
                elif isinstance(item, RawTriple):
                    valid_chunk.append(item)
            except Exception:
                pass
        return valid_chunk


# =============================================================================
# 2. SEMANTICPRISM STAGE 1 SYSTEM & USER PROMPTS
# =============================================================================

THEME_DISCOVERY_SYSTEM_PROMPT = """You are an Ontological Concept Engine.
Your objective is to read the provided text and strictly discover the most critical underlying abstractions or structural 'forms' of information it represents, moving explicitly away from isolated specific subjects and maintaining distinction across themes.
Examples:
1. Rather than mapping specific instances (e.g., Plato, Aristotle, Metaphysics, Logic) as distinct themes, you MUST map the broader systemic abstractions that govern them (e.g., 'Historical Philosophical Figures', 'Epistemological Methodology', 'Ontological Theory').
2. Rather than mapping specific instances (e.g., Metformin, HbA1c testing, Type 2 Diabetes, Insulin resistance), you MUST map the broader systemic abstractions that govern them (e.g., 'Endocrine Metabolic Disorders', 'Pharmacotherapeutic Interventions', 'Biomarker Diagnostic Metrics').
3. Rather than mapping specific instances (e.g., Phishing email, CrowdStrike sensor, Ransomware, ISO 27001 compliance), you MUST map the broader systemic abstractions that govern them (e.g., 'Threat Vector Vectors', 'Endpoint Telemetry Mechanisms', 'Malicious Payload Typologies', 'Information Security Governance Frameworks').
Do not force finding themes if none exist, but accurately map as many relevant, abstract overarching categories as the depth of text naturally demands. Maintaining efficiency, the exact amount of categories should be dictated dynamically by the text's content.
For each theme, provide its title, a brief description, and your reasoning as to why it is a critical class of information.

CRITICAL INSTRUCTION: You must return a single, valid JSON data object matching the schema exactly. Do NOT include conversational filler, markdown formatting blocks, or the JSON Schema definition itself."""

THEME_DISCOVERY_USER_PROMPT = """Read this text and list the most critical overarching themes/classes of information based on the text:

<source_text>
{text_content}
</source_text>"""

CLUSTER_MAP_SYSTEM_PROMPT = """You are a Cluster Domain Specialist.
You will receive an empirical cluster of themes discovered through vector embedding clustering.
The themes are sorted in descending order of geometric proximity to the cluster centroid (most prototypical themes first).

Your objective is to perform deep conceptual synthesis for this single cluster:
1. Analyze the core themes (especially the top centroid prototypes) to discover the true semantic center of this cluster.
2. Formulate ONE concise, authoritative candidate title (3 to 6 words) that captures the core essence of this domain.
3. Write a clear, 1-2 sentence scope definition defining what this cluster strictly covers.
4. List 3 to 5 key inclusions (concepts or mechanisms that strictly belong inside this theme).
5. List 2 to 4 key exclusions (concepts that might seem superficially related but belong elsewhere).
6. Provide a qualitative confidence assessment ("high", "medium", "low") of how cohesive the cluster is.

CRITICAL: Return a single JSON object matching the ClusterCandidateSynthesis schema. Do NOT output markdown code fences or schema definitions."""

CLUSTER_MAP_USER_PROMPT = """Synthesize the conceptual identity for Cluster #{cluster_id} ({theme_count} total themes in cluster, showing {displayed_count}):

<cluster_themes>
{theme_lines}
</cluster_themes>"""

CLUSTER_REDUCE_SYSTEM_PROMPT = """You are the Chief Ontological Synthesizer.
You will receive candidate theme syntheses produced from the Map stage across all clusters discovered in the corpus.
For each cluster, you also receive the Top Exemplar Themes closest to the mathematical centroid as empirical ground-truth anchors.

Your objective is to harmonize all candidate themes into a unified, authoritative Master Ontology:
1. Formulate ONE single overarching `master_domain` string capturing the root ontological domain of the entire corpus.
2. Disambiguate and sharpen boundaries between any overlapping candidate titles. Use the empirical anchor themes to ensure each title faithfully reflects its cluster's centroid reality.
3. Ensure all titles are mutually distinct, non-overlapping, and follow a consistent grammatical convention (e.g. Title Case noun phrases).
4. The titles in the `master_themes` list MUST strictly correspond 1:1 to the cluster groups provided, maintaining the exact sequence (Cluster #1, Cluster #2, etc.).

CRITICAL: Return a single JSON object containing:
- `master_domain`: string (the unified root domain of the corpus)
- `master_themes`: list of strings (the finalized master theme titles in exact 1:1 order with the clusters)
Do NOT output the JSON Schema definition itself."""

CLUSTER_REDUCE_USER_PROMPT = """Review the following {cluster_count} cluster candidates and their empirical anchor themes. Harmonize them into an authoritative Master Ontology:

{cluster_blocks}"""

TRIPLE_EXTRACTION_SYSTEM_PROMPT = """You are a High-Precision Triple Extractor Agent running schema-mapped knowledge discovery.
Your objective is to comprehensively extract all factual assertions, clinical findings, behaviors, diagnoses, relationships, and actions from the source text into atomic (Subject, Predicate, Object) triples.

### EXTRACTION OBJECTIVES:
1. Comprehensive Recall & Graph Density: Extract EVERY meaningful relationship across the entire text chunk. Do not stop after 2 or 3 triples. A rich, informative document chunk typically contains 10 to 30 valid factual triples.
2. Atomic Entities: Keep subjects and objects clear and concise (e.g., "patient", "WISC-V evaluation", "ADHD", "morning drop-offs", "processing speed bottleneck") so they form clean, connected knowledge graph nodes.
3. Descriptive Predicates: Use active, meaningful relationships (e.g., "was diagnosed with", "demonstrates", "yielded Full Scale IQ of", "impairs", "resides with", "was referred by", "presents with"). Never empty.
4. Source Quote (Verbatim): For EVERY extracted relationship, you MUST provide the exact, verbatim substring from the source text confirming it. Do NOT summarize, shorten, or alter the quote.
5. Theme Association: Set `theme_association` to the exact matching string from the 'Allowed theme_association values' list. If none fit cleanly, set `theme_association` to "Other".

### EXAMPLE STRUCTURE:
```json
{
  "triples": [
    {
      "subject": "patient",
      "predicate": "meets diagnostic criteria for",
      "object": "Separation Anxiety Disorder",
      "source_quote": "Patient meets diagnostic criteria for Attention-Deficit/Hyperactivity Disorder, Combined Presentation, and Separation Anxiety Disorder.",
      "theme_association": "Clinical Diagnoses"
    },
    {
      "subject": "WISC-V evaluation",
      "predicate": "yielded Full Scale IQ of",
      "object": "102",
      "source_quote": "Psychometric evaluation utilizing the Wechsler Intelligence Scale for Children, Fifth Edition (WISC-V) yielded a Full Scale IQ of 102.",
      "theme_association": "Psychometric Testing"
    }
  ]
}
```

### CRITICAL RULES:
- Output ALL extracted triples in the `triples` array. Do NOT summarize, abbreviate, or use ellipses ("...").
- Do not invent facts not stated in the text. Every triple must be grounded by its verbatim `source_quote`."""

TRIPLE_EXTRACTION_USER_PROMPT = """Extract all atomic relationship triples from the following text:

<source_text>
{text_content}
</source_text>"""


# =============================================================================
# 3. EMPIRICAL VECTOR THEME CLUSTERING
# =============================================================================

_EXEMPLARS_PER_CLUSTER: int = 10
_REDUCE_ANCHOR_THEME_COUNT: int = 5
_FALLBACK_DOMAIN: str = "General Domain"


def _derive_fallback_domain(titles: Optional[List[str]] = None) -> str:
    """Derive a descriptive corpus domain label from top cluster/theme titles when LLM domain synthesis is unavailable."""
    if not titles:
        return _FALLBACK_DOMAIN
    cleaned: List[str] = []
    for raw in titles:
        c = re.sub(r"^(?:Cluster\s*#?\d+[\s:.-]*|Master\s*Theme\s*#?\d+[\s:.-]*|\d+[\s:.-]+)", "", str(raw or "")).strip()
        if c and c not in cleaned:
            cleaned.append(c)
        if len(cleaned) >= 3:
            break
    if not cleaned:
        return _FALLBACK_DOMAIN
    return f"Synthesized Domain: {', '.join(cleaned)}"


def format_theme_embedding_text(theme: Any, mode: str = "adaptive") -> str:
    """
    Format a Theme model or dict into text for vector embedding or Map prompts.
    In 'adaptive' mode (default), uses 'title: description' when description is substantive (>= 5 words)
    to avoid generic LLM reasoning meta-boilerplate inflating cross-theme cosine similarity, and
    only appends reasoning when description is missing or terse (< 5 words).
    """
    if hasattr(theme, "title"):
        title = str(theme.title or "").strip()
        desc = str(getattr(theme, "description", "") or "").strip()
        reason = str(getattr(theme, "reasoning", "") or "").strip()
    elif isinstance(theme, dict):
        title = str(theme.get("title", "") or "").strip()
        desc = str(theme.get("description", "") or "").strip()
        reason = str(theme.get("reasoning", "") or "").strip()
    else:
        return str(theme).strip()

    if mode == "title":
        return title
    if mode == "title_desc":
        return f"{title}: {desc}" if desc else title
    if mode == "adaptive":
        if len(desc.split()) >= 5:
            return f"{title}: {desc}"
        if desc and reason:
            return f"{title}: {desc} ({reason})"
        if reason:
            return f"{title}: {reason}"
        return f"{title}: {desc}" if desc else title
    if desc and reason:
        return f"{title}: {desc} (Reasoning: {reason})"
    if desc:
        return f"{title}: {desc}"
    return title


class ThemeClusteringEngine:
    """
    Empirical vector theme clustering engine matching SemanticPrism's ThemeClusteringEngine.
    Uses SentenceTransformer embeddings (with deterministic TF-IDF cosine fallback if offline),
    computes dynamic kNN/Leiden modularity partitions, ranks themes by centroid cosine similarity,
    and exports 'clustering_summary.json'.
    """

    def __init__(
        self,
        themes_path: Path,
        output_dir: Path,
        embedding_model_name: str = THEME_EMBEDDING_MODEL,
        exemplar_count: int = _EXEMPLARS_PER_CLUSTER,
        map_max_themes: int = MAP_MAX_THEMES_PER_CLUSTER,
        reduce_anchor_count: int = _REDUCE_ANCHOR_THEME_COUNT,
    ):
        self.themes_path = Path(themes_path)
        self.output_dir = Path(output_dir)
        self.master_themes_path = self.output_dir / "master_themes.json"
        self.embedding_model_name = embedding_model_name
        self.exemplar_count = exemplar_count
        self.map_max_themes = map_max_themes
        self.reduce_anchor_count = reduce_anchor_count

        self.raw_themes: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.partitions: Dict[str, Any] = {}
        self.active_exemplars: List[Dict[str, Any]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute L2-normalized embeddings via SentenceTransformer or TF-IDF fallback, freeing VRAM immediately."""
        try:
            import torch
            from sentence_transformers import SentenceTransformer
            models_dir = REPO_ROOT / "models" / "embeddings"
            models_dir.mkdir(parents=True, exist_ok=True)
            model = None
            emb = None
            try:
                try:
                    # First try default HuggingFace cache offline, then repo models_dir offline, then online
                    model = SentenceTransformer(self.embedding_model_name, local_files_only=True)
                except Exception:
                    try:
                        model = SentenceTransformer(self.embedding_model_name, cache_folder=str(models_dir), local_files_only=True)
                    except Exception:
                        model = SentenceTransformer(self.embedding_model_name, cache_folder=str(models_dir), local_files_only=False)
                emb_batch_size = resolve_dynamic_batch_size(
                    device=get_optimal_device(DEVICE),
                    requested_batch_size=BATCH_SIZE,
                    items=[{"prompt": t} for t in texts],
                )
                with torch.inference_mode():
                    emb = model.encode(
                        texts,
                        normalize_embeddings=True,
                        batch_size=emb_batch_size,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                arr = np.asarray(emb, dtype=np.float32)
                return arr
            finally:
                if model is not None:
                    try:
                        model.to("cpu")
                    except Exception:
                        pass
                del model, emb
                clear_device_memory()
        except Exception:
            vocab: Dict[str, int] = {}
            tokenized: List[List[str]] = []
            for t in texts:
                raw_words = [w for w in re.findall(r"[a-z0-9]{2,}", t.lower())]
                # Include 5-char prefix stems alongside full words to capture morphological variants offline
                feats = list(raw_words) + [f"#{w[:5]}" for w in raw_words if len(w) >= 5]
                tokenized.append(feats)
                for w in set(feats):
                    vocab[w] = vocab.get(w, 0) + 1
            terms = sorted(vocab.keys())
            if not terms:
                return np.eye(max(1, len(texts)), dtype=np.float32)
            term_idx = {w: i for i, w in enumerate(terms)}
            n_docs = len(texts)
            mat = np.zeros((n_docs, len(terms)), dtype=np.float32)
            for d_i, words in enumerate(tokenized):
                for w in words:
                    if w in term_idx:
                        mat[d_i, term_idx[w]] += 1.0
            idf = np.log((1.0 + n_docs) / (1.0 + np.array([vocab[w] for w in terms], dtype=np.float32))) + 1.0
            mat = mat * idf
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return mat / norms

    @staticmethod
    def _select_mmr_indices(
        cluster_embs: np.ndarray,
        sims_to_centroid: np.ndarray,
        top_n: int,
        lambda_param: float = 0.50,
    ) -> List[int]:
        """
        Select up to top_n local indices from a cluster using Maximal Marginal Relevance (MMR):
          MMR(i) = lambda * sim(i, centroid) - (1 - lambda) * max_{j in S} cos(e_i, e_j)
        Starts with the exact cluster medoid (highest centroid similarity) and restricts the initial
        candidate pool to non-outlier cluster members (>= min(mu - 2*sigma, MANIFOLD_EDGE_THRESHOLD)).
        """
        n_c = len(sims_to_centroid)
        if n_c == 0 or top_n <= 0:
            return []
        ranked = [int(i) for i in np.argsort(sims_to_centroid)[::-1]]
        if n_c <= top_n:
            return ranked

        mu_c = float(np.mean(sims_to_centroid))
        sigma_c = float(np.std(sims_to_centroid)) if n_c > 1 else 0.0
        core_floor = min(mu_c - 2.0 * max(sigma_c, 0.05), float(MANIFOLD_EDGE_THRESHOLD))
        core_pool = [i for i in ranked if float(sims_to_centroid[i]) >= core_floor]
        if len(core_pool) < top_n:
            core_pool = ranked

        selected: List[int] = [core_pool[0]]
        remaining: List[int] = [i for i in core_pool[1:]]
        pair_sims = np.clip(cluster_embs @ cluster_embs.T, -1.0, 1.0)

        while len(selected) < min(top_n, len(core_pool)) and remaining:
            best_idx = remaining[0]
            best_score = -float("inf")
            for cand_i in remaining:
                rel = float(sims_to_centroid[cand_i])
                max_red = max(float(pair_sims[cand_i, sel_j]) for sel_j in selected)
                score = (lambda_param * rel) - ((1.0 - lambda_param) * max_red)
                if score > best_score:
                    best_score = score
                    best_idx = cand_i
            selected.append(best_idx)
            remaining.remove(best_idx)

        return selected

    @staticmethod
    def _spherical_centroid_fallback(embeddings: np.ndarray, k: int) -> np.ndarray:
        """
        Deterministic farthest-first spherical centroid clustering fallback when networkx is unavailable
        or the similarity graph is degenerate. Avoids arbitrary modulo index assignment.
        """
        n = len(embeddings)
        k = max(1, min(k, n))
        if k == 1:
            return np.zeros(n, dtype=int)

        mean_vec = embeddings.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        mean_vec = mean_vec / norm if norm > 0 else mean_vec
        first_seed = int(np.argmax(embeddings @ mean_vec))
        centers = [embeddings[first_seed]]
        min_sims = embeddings @ centers[0]

        for _ in range(1, k):
            next_seed = int(np.argmin(min_sims))
            centers.append(embeddings[next_seed])
            min_sims = np.maximum(min_sims, embeddings @ centers[-1])

        cent_mat = np.vstack(centers)
        labels = np.argmax(embeddings @ cent_mat.T, axis=1).astype(int)

        for _ in range(3):
            new_centers = []
            for cid in range(k):
                mask = labels == cid
                if np.any(mask):
                    c_vec = embeddings[mask].mean(axis=0)
                    c_norm = np.linalg.norm(c_vec)
                    new_centers.append(c_vec / c_norm if c_norm > 0 else c_vec)
                else:
                    new_centers.append(cent_mat[cid])
            cent_mat = np.vstack(new_centers)
            labels = np.argmax(embeddings @ cent_mat.T, axis=1).astype(int)

        remap = {old_id: new_id for new_id, old_id in enumerate(sorted(set(int(x) for x in labels)))}
        return np.array([remap[int(x)] for x in labels], dtype=int)

    def _build_partition_dict(
        self,
        name: str,
        labels: np.ndarray,
        raw_themes: List[Dict[str, Any]],
        embeddings: np.ndarray,
        effective_edge_threshold: float = MANIFOLD_EDGE_THRESHOLD,
    ) -> Dict[str, Any]:
        unique_ids = sorted(set(int(x) for x in labels))
        clusters: Dict[int, Dict[str, Any]] = {}
        assignments: List[Dict[str, Any]] = []
        n_total = max(1, len(raw_themes))

        for new_cid, orig_cid in enumerate(unique_ids):
            idxs = [i for i, lbl in enumerate(labels) if int(lbl) == orig_cid]
            n_c = len(idxs)
            cluster_embs = embeddings[idxs]
            sum_vec = cluster_embs.sum(axis=0)
            r_sum = float(np.linalg.norm(sum_vec))
            centroid = (sum_vec / r_sum) if r_sum > 0 else cluster_embs.mean(axis=0)

            sims = cluster_embs @ centroid
            # Leave-One-Out (LOO) cosine similarity to remove 1/sqrt(n_c) finite-sample self-bias:
            # s_i^LOO = (||S_c|| * s_i - 1) / sqrt(||S_c||^2 - 2 * ||S_c|| * s_i + 1)
            if n_c > 1 and r_sum > 0:
                loo_denom_sq = np.maximum(1e-12, (r_sum ** 2) - (2.0 * r_sum * sims) + 1.0)
                loo_sims = ((r_sum * sims) - 1.0) / np.sqrt(loo_denom_sq)
                loo_sims = np.clip(loo_sims, -1.0, 1.0)
            else:
                loo_sims = sims.copy()

            mean_sim = float(np.mean(sims)) if n_c > 0 else 0.0
            std_sim = float(np.std(sims)) if n_c > 1 else 0.0

            ranked_order = np.argsort(sims)[::-1]
            local_to_item: Dict[int, Dict[str, Any]] = {}

            all_cluster_themes = []
            for local_pos in ranked_order:
                l_idx = int(local_pos)
                r_idx = idxs[l_idx]
                sim_val = float(sims[l_idx])
                loo_val = float(loo_sims[l_idx])
                t_obj = raw_themes[r_idx]
                item_dict = {
                    "theme_index": int(r_idx),
                    "title": t_obj.get("title", ""),
                    "description": t_obj.get("description", ""),
                    "reasoning": t_obj.get("reasoning", ""),
                    "similarity_to_centroid": round(sim_val, 4),
                    "loo_similarity_to_centroid": round(loo_val, 4),
                }
                all_cluster_themes.append(item_dict)
                local_to_item[l_idx] = item_dict

            mmr_exemplar_locals = self._select_mmr_indices(cluster_embs, sims, self.exemplar_count, lambda_param=0.50)
            top_exemplars = [local_to_item[i] for i in mmr_exemplar_locals if i in local_to_item]
            mmr_map_locals = self._select_mmr_indices(cluster_embs, sims, min(n_c, max(self.map_max_themes, self.exemplar_count)), lambda_param=0.50)
            mmr_themes = [local_to_item[i] for i in mmr_map_locals if i in local_to_item]

            lead_title = top_exemplars[0]["title"] if top_exemplars else (all_cluster_themes[0]["title"] if all_cluster_themes else f"Cluster #{new_cid + 1}")
            clusters[new_cid] = {
                "cluster_id": new_cid,
                "label": f"Cluster #{new_cid + 1}: {lead_title}",
                "count": n_c,
                "percentage": round(n_c * 100.0 / n_total, 2),
                "mean_similarity": round(mean_sim, 4),
                "std_similarity": round(std_sim, 4),
                "exemplars": top_exemplars,
                "mmr_themes": mmr_themes,
                "all_themes": all_cluster_themes,
            }

            for item in all_cluster_themes:
                assignments.append({
                    "theme_index": item["theme_index"],
                    "title": item["title"],
                    "cluster_id": new_cid,
                    "cluster_label": clusters[new_cid]["label"],
                    "similarity_to_centroid": item["similarity_to_centroid"],
                })

        return {
            "name": name,
            "cluster_count": len(clusters),
            "effective_edge_threshold": round(float(effective_edge_threshold), 4),
            "clusters": clusters,
            "assignments": sorted(assignments, key=lambda x: x["theme_index"]),
        }

    def cluster_themes(self) -> Dict[str, Any]:
        if not self.themes_path.exists():
            raise FileNotFoundError(f"Missing themes file: {self.themes_path}")
        with open(self.themes_path, "r", encoding="utf-8") as f:
            self.raw_themes = json.load(f)

        n = len(self.raw_themes)
        if n == 0:
            self.partitions = {}
            self.active_exemplars = []
            return {"partitions": {}, "active_exemplars": [], "raw_themes": []}

        texts = [format_theme_embedding_text(t, mode="adaptive") for t in self.raw_themes]
        self.embeddings = self._compute_embeddings(texts)

        # Dynamic Partition (Self-calibrating kNN/Leiden modularity community detection)
        fallback_k = max(1, min(8, int(round(math.sqrt(n / 2.0))) or 1))
        dyn_labels = np.zeros(n, dtype=int)
        tau_eff = float(MANIFOLD_EDGE_THRESHOLD)
        if n > 2:
            try:
                import networkx as nx
                sim_matrix = np.clip(self.embeddings @ self.embeddings.T, -1.0, 1.0)
                off_diag = sim_matrix[~np.eye(n, dtype=bool)]
                mu_bg = float(np.mean(off_diag)) if off_diag.size > 0 else 0.0
                sigma_bg = float(np.std(off_diag)) if off_diag.size > 1 else 0.0

                k_neighbors = int(np.clip(round(2 * math.log2(max(2, n))), 2, min(30, n - 1)))
                top1_sims: List[float] = []
                top_nbr_rows: List[np.ndarray] = []
                for i in range(n):
                    row = sim_matrix[i]
                    nbrs = np.argsort(row)[::-1]
                    nbrs = nbrs[nbrs != i][:k_neighbors]
                    top_nbr_rows.append(nbrs)
                    if nbrs.size > 0:
                        top1_sims.append(float(row[nbrs[0]]))

                top1_arr = np.array(top1_sims, dtype=np.float32) if top1_sims else np.array([MANIFOLD_EDGE_THRESHOLD], dtype=np.float32)
                p20_top1 = float(np.percentile(top1_arr, 20))
                # Self-calibrating manifold threshold across dense, anisotropic, and sparse TF-IDF spaces:
                # - Uses MANIFOLD_EDGE_THRESHOLD in standard dense spaces (where P20(1-NN) >= MANIFOLD_EDGE_THRESHOLD)
                # - Tightens to mu_bg + 0.5*sigma_bg in narrow anisotropic cones
                # - Relaxes to P20(1-NN) in sparse/lexical spaces so communities do not shatter into singletons
                tau_eff = max(mu_bg + 0.5 * sigma_bg, min(float(MANIFOLD_EDGE_THRESHOLD), p20_top1))

                graph = nx.Graph()
                for i in range(n):
                    graph.add_node(i)
                    row = sim_matrix[i]
                    nbrs = top_nbr_rows[i]
                    for rank_idx, j in enumerate(nbrs):
                        w = float(row[j])
                        # Admit edges >= tau_eff, or guarantee top-1 nearest neighbor weak-tie if above background
                        if w >= tau_eff or (rank_idx == 0 and w > mu_bg):
                            excess_w = max(1e-4, w - tau_eff + 0.05)
                            if graph.has_edge(i, int(j)):
                                if excess_w > graph[i][int(j)].get("weight", 0.0):
                                    graph[i][int(j)]["weight"] = excess_w
                            else:
                                graph.add_edge(i, int(j), weight=excess_w)

                if graph.number_of_edges() > 0:
                    communities = list(nx.community.greedy_modularity_communities(graph, weight="weight"))
                    for c_idx, comm in enumerate(communities):
                        for node_idx in comm:
                            dyn_labels[int(node_idx)] = c_idx
                else:
                    dyn_labels = self._spherical_centroid_fallback(self.embeddings, fallback_k)
                    communities = [set(np.where(dyn_labels == cid)[0]) for cid in set(int(x) for x in dyn_labels)]

                # Absorb micro-clusters (< MICRO_CLUSTER_RATIO of corpus or singletons) into nearest dominant centroid
                if len(communities) > 2:
                    min_cluster_size = max(2, int(round(n * MICRO_CLUSTER_RATIO)))
                    unique_c, counts_c = np.unique(dyn_labels, return_counts=True)
                    dom_ids = [int(cid) for cid, cnt in zip(unique_c, counts_c) if cnt >= min_cluster_size]
                    if not dom_ids:
                        top_order = np.argsort(counts_c)[::-1][: max(2, min(8, len(unique_c)))]
                        dom_ids = [int(unique_c[idx]) for idx in top_order]

                    if 0 < len(dom_ids) < len(unique_c):
                        dom_centroids = []
                        for cid in dom_ids:
                            c_embs = self.embeddings[dyn_labels == cid]
                            cent = c_embs.mean(axis=0)
                            norm = np.linalg.norm(cent)
                            dom_centroids.append(cent / norm if norm > 0 else cent)
                        dom_mat = np.vstack(dom_centroids)

                        dom_set = set(dom_ids)
                        for idx in range(n):
                            if int(dyn_labels[idx]) not in dom_set:
                                sims = dom_mat @ self.embeddings[idx]
                                best_dom = dom_ids[int(np.argmax(sims))]
                                dyn_labels[idx] = best_dom

                        remap = {old_id: new_id for new_id, old_id in enumerate(sorted(set(int(x) for x in dyn_labels)))}
                        dyn_labels = np.array([remap[int(x)] for x in dyn_labels], dtype=int)
            except Exception:
                dyn_labels = self._spherical_centroid_fallback(self.embeddings, fallback_k)

        self.partitions = {
            "dynamic_leiden": self._build_partition_dict(
                "Dynamic Topological Leiden",
                dyn_labels,
                self.raw_themes,
                self.embeddings,
                effective_edge_threshold=tau_eff,
            ),
        }

        active_part = self.get_active_partition(self.partitions)
        self.active_exemplars = [
            {
                "cluster_id": c_id,
                "theme_count": c_data["count"],
                "percentage": c_data["percentage"],
                "exemplars": c_data["exemplars"],
            }
            for c_id, c_data in sorted(active_part["clusters"].items())
        ]
        self.embeddings = None

        return {
            "partitions": self.partitions,
            "active_exemplars": self.active_exemplars,
            "raw_themes": self.raw_themes,
        }

    def get_active_partition_key(self) -> str:
        return "dynamic_leiden"

    def get_active_partition(self, partitions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        parts = partitions or self.partitions
        key = self.get_active_partition_key()
        return parts.get(key) or next(iter(parts.values()), {"clusters": {}, "assignments": []})

    def get_map_payload(
        self,
        max_themes_per_cluster: int = MAP_MAX_THEMES_PER_CLUSTER,
        payload_format: str = "title_desc",
        max_words_budget: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        active_part = self.get_active_partition()
        payloads = []
        for c_id, c_data in sorted(active_part.get("clusters", {}).items()):
            all_t = c_data.get("all_themes", [])
            # Prefer MMR-diversified ordering when truncating a large cluster so all sub-facets are represented
            source_themes = c_data.get("mmr_themes") or all_t
            candidates = source_themes[:max_themes_per_cluster]
            lines = []
            running_words = 0
            for idx, t in enumerate(candidates, start=1):
                formatted = format_theme_embedding_text(t, mode=payload_format)
                line = f"{idx}. {formatted} (sim={t.get('similarity_to_centroid', 0.0):.3f})"
                line_words = len(line.split())
                if max_words_budget and lines and (running_words + line_words > max_words_budget):
                    break
                lines.append(line)
                running_words += line_words
            payloads.append({
                "cluster_id": c_id,
                "theme_count": len(all_t),
                "displayed_count": len(lines),
                "theme_lines": "\n".join(lines),
            })
        return payloads

    def get_reduce_anchors(self, top_n: int = _REDUCE_ANCHOR_THEME_COUNT) -> Dict[int, List[Dict[str, Any]]]:
        active_part = self.get_active_partition()
        anchors: Dict[int, List[Dict[str, Any]]] = {}
        for c_id, c_data in sorted(active_part.get("clusters", {}).items()):
            source_themes = c_data.get("exemplars") or c_data.get("mmr_themes") or c_data.get("all_themes", [])
            anchors[c_id] = source_themes[:top_n]
        return anchors

    def export_clustering_summary(
        self,
        master_themes: Optional[List[str]] = None,
        master_domain: Optional[str] = None,
    ) -> Path:
        if not self.partitions:
            self.cluster_themes()

        active_key = self.get_active_partition_key()
        active_part = self.partitions.get(active_key, {})
        fallback_titles = master_themes or [
            c_data.get("label", "") for _, c_data in sorted(active_part.get("clusters", {}).items())
        ]
        resolved_domain = master_domain or _derive_fallback_domain(fallback_titles)

        if active_part and master_themes:
            active_part["master_domain"] = resolved_domain
            for idx, (c_id, c_data) in enumerate(sorted(active_part.get("clusters", {}).items())):
                if idx < len(master_themes):
                    clean_name = re.sub(r"^(?:Cluster\s*\d+[\s:.-]*|\d+[\s:.-]+)", "", str(master_themes[idx])).strip()
                    c_data["master_theme_name"] = clean_name
                    c_data["label"] = f"Master Theme #{idx + 1}: {clean_name}"

        summary_path = self.output_dir / "clustering_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "master_domain": resolved_domain,
                    "active_partition": active_key,
                    "total_themes": len(self.raw_themes),
                    "partitions": self.partitions,
                },
                f,
                indent=2,
            )
        return summary_path


# =============================================================================
# 4. SEMANTICPRISM STAGE 1 PIPELINE ORCHESTRATOR
# =============================================================================

class PipelineRunContext:
    """State tracking object persisting discovered themes, master ontology, and SVO triples."""

    def __init__(self):
        self.all_discovered_themes: List[Theme] = []
        self.cluster_candidates: List[ClusterCandidateSynthesis] = []
        self.master_themes: Optional[MasterThemeSynthesisResult] = None
        self.raw_triples: List[Dict[str, Any]] = []
        self.processed_documents: Set[str] = set()


class SemanticPrismStage1Pipeline:
    """
    Executes the complete SemanticPrism Stage 1 Extraction workflow using PrismRunner
    for local hardware-optimized batch LLM inference.
    """

    def __init__(
        self,
        context: Optional[PipelineRunContext] = None,
        prism_pipeline: Optional[PrismPipeline] = None,
    ):
        self.context = context or PipelineRunContext()
        root_out = Path(OUTPUTS_DIR)
        self.out_dir = root_out / "01_extraction"
        self.themes_dir = self.out_dir / "themes"
        self.triples_dir = self.out_dir / "triples"
        self.log_dir = root_out / "logs"
        self.payload_dir = root_out / "payload_text"

        for d in (self.out_dir, self.themes_dir, self.triples_dir, self.log_dir, self.payload_dir):
            d.mkdir(parents=True, exist_ok=True)

        if prism_pipeline is not None:
            self.prism = prism_pipeline
        else:
            self.prism = PrismPipeline(
                model_name=MODEL_NAME,
                device=DEVICE,
                batch_size=BATCH_SIZE,
                config={
                    "model": {
                        "name": MODEL_NAME,
                        "device": DEVICE,
                        "torch_dtype": TORCH_DTYPE,
                        "load_in_4bit": LOAD_IN_4BIT,
                        "load_in_8bit": LOAD_IN_8BIT,
                        "attn_implementation": ATTN_IMPLEMENTATION,
                    },
                    "inference": {
                        "batch_size": BATCH_SIZE,
                        "temperature": TEMPERATURE,
                        "constrain_json": CONSTRAIN_JSON,
                    },
                },
            )

        if self.context.master_themes is None:
            self.load_master_themes()

    @staticmethod
    def sanitize_filename(name: str) -> str:
        """Sanitize document identifiers to be filesystem-safe."""
        return re.sub(r'[\\/:*?"<>|]', "_", str(name))

    @staticmethod
    def _log_checkpoint(phase: str, message: str) -> None:
        """Print timestamped checkpoint progress to the console when PRINT_CHECKPOINTS is enabled."""
        if PRINT_CHECKPOINTS:
            ts = time.strftime("%H:%M:%S")
            print(f"[{ts} | Checkpoint | {phase}] {message}", flush=True)

    def clear_stage_logs(self, phase: Optional[str] = None) -> None:
        log_map = {
            "themes": ["stage_01_theme_discovery_errors.json"],
            "master": ["stage_01_master_theme_errors.json"],
            "triples": ["stage_01_triple_extraction_errors.json"],
        }
        targets = log_map.get(phase, [
            "stage_01_theme_discovery_errors.json",
            "stage_01_master_theme_errors.json",
            "stage_01_triple_extraction_errors.json",
        ])
        for log_file in targets:
            p = self.log_dir / log_file
            if p.exists():
                p.unlink()

    def _save_payload_text(self, filename: str, content: str) -> None:
        if not SAVE_LLM_PAYLOADS:
            return
        target_path = self.payload_dir / filename
        if not target_path.exists():
            target_path.write_text(content, encoding="utf-8")

    @staticmethod
    def chunk_text(text: str, max_words: int) -> List[Tuple[str, int, int]]:
        """
        SemanticPrism's word-based sentence-boundary lookahead chunker with a bounded
        lookahead cap so unpunctuated sections/tables cannot merge into giant chunks.
        Returns a list of tuples: (chunk_text, start_word_index, end_word_index).
        """
        words = text.split()
        chunks: List[Tuple[str, int, int]] = []
        start = 0
        sentence_endings = (".", "!", "?", '."', '!"', '?"', ".'", "!'", "?'", ".)", "!)")
        max_lookahead = max(100, max_words // 5)
        while start < len(words):
            base_end = min(start + max_words, len(words))
            end = base_end
            if end < len(words):
                lookahead_limit = min(len(words), base_end + max_lookahead)
                while end < lookahead_limit and not words[end - 1].endswith(sentence_endings):
                    end += 1
                if end == lookahead_limit and not words[end - 1].endswith(sentence_endings):
                    end = base_end
            chunk = " ".join(words[start:end])
            chunks.append((chunk, start, end))
            if end >= len(words):
                break
            start = end
        return chunks

    @staticmethod
    def load_input_documents(
        input_path: Optional[Union[str, Path, pd.DataFrame, List[Dict[str, Any]]]] = None,
        id_column: str = ID_COLUMN,
        text_column: str = TEXT_COLUMN,
        limit: Optional[int] = DOCUMENT_LIMIT,
    ) -> List[Dict[str, str]]:
        """
        Load input documents using global settings (auto-detects Parquet, CSV, JSON/JSONL, DataFrame, or directory of .txt files).
        Returns a list of dicts: [{'id': str, 'text': str}].
        """
        target = input_path if input_path is not None else INPUT_PATH
        docs: List[Dict[str, str]] = []

        if isinstance(target, pd.DataFrame):
            df = target
        elif isinstance(target, list):
            for i, item in enumerate(target):
                if isinstance(item, dict):
                    doc_id = str(item.get(id_column) or item.get("id") or f"doc_{i:04d}")
                    txt = str(item.get(text_column) or item.get("text") or "")
                    if txt.strip():
                        docs.append({"id": doc_id, "text": txt})
                elif isinstance(item, str) and item.strip():
                    docs.append({"id": f"doc_{i:04d}", "text": item})
            return docs[:limit] if limit else docs
        else:
            p = Path(target)
            if not p.exists() and (REPO_ROOT / p).exists():
                p = REPO_ROOT / p
            if not p.exists():
                raise FileNotFoundError(f"Input path not found at: {target}")

            if p.is_dir():
                print(f"Loading input documents from Directory: {p}")
                for txt_file in sorted(p.glob("*.txt")):
                    text = txt_file.read_text(encoding="utf-8")
                    if text.strip():
                        docs.append({"id": txt_file.stem, "text": text})
                return docs[:limit] if limit else docs

            if p.suffix.lower() in (".parquet", ".pq"):
                print(f"Loading input documents from Parquet file: {p}")
                df = pd.read_parquet(p)
            elif p.suffix.lower() == ".csv":
                print(f"Loading input documents from CSV file: {p}")
                df = pd.read_csv(p)
            elif p.suffix.lower() in (".json", ".jsonl"):
                print(f"Loading input documents from JSON file: {p}")
                df = pd.read_json(p, lines=p.suffix.lower() == ".jsonl")
            else:
                raise ValueError(f"Unsupported input file extension: {p.suffix}")

        # Resolve ID and text columns (with fallback if custom column names differ)
        col_map = {str(c).lower().strip(): c for c in df.columns}
        resolved_id = id_column if id_column in df.columns else col_map.get(id_column.lower())
        if not resolved_id:
            for cand in ("unique_val", "id", "doc_id", "record_id", "identifier", "patient_id"):
                if cand in col_map:
                    resolved_id = col_map[cand]
                    break

        resolved_text = text_column if text_column in df.columns else col_map.get(text_column.lower())
        if not resolved_text:
            for cand in ("report", "text", "anonymized", "content", "clinical_report", "note", "body"):
                if cand in col_map:
                    resolved_text = col_map[cand]
                    break

        if not resolved_text:
            raise ValueError(f"text_column '{text_column}' not found in columns: {list(df.columns)}")

        for idx, row in df.iterrows():
            doc_id = str(row[resolved_id]) if resolved_id and pd.notna(row[resolved_id]) else f"doc_{idx:04d}"
            text = str(row[resolved_text]) if pd.notna(row[resolved_text]) else ""
            if text.strip() and text.strip().lower() != "nan":
                docs.append({"id": doc_id, "text": text})

        return docs[:limit] if limit else docs

    def load_master_themes(self) -> Optional[MasterThemeSynthesisResult]:
        """Load master_themes.json from disk into context if it exists."""
        master_path = self.out_dir / "master_themes.json"
        if master_path.exists():
            try:
                with open(master_path, "r", encoding="utf-8") as f:
                    self.context.master_themes = MasterThemeSynthesisResult(**json.load(f))
            except Exception as e:
                print(f"Warning: Failed to load master themes from {master_path}: {e}")
        return self.context.master_themes

    # =========================================================================
    # PHASE 1: GLOBAL THEME DISCOVERY & AGGREGATION
    # =========================================================================

    def _apply_context_window_cap(self) -> int:
        """
        Dynamically resolve the effective context window from the loaded model's config
        (or apply CONTEXT_WINDOW_CAP if an explicit integer cap is set).
        """
        native_ctx = getattr(self.prism, "context_window", 4096) or 4096
        if isinstance(CONTEXT_WINDOW_CAP, int) and CONTEXT_WINDOW_CAP > 0:
            effective_ctx = min(native_ctx, CONTEXT_WINDOW_CAP)
        else:
            effective_ctx = native_ctx
        self.prism.context_window = effective_ctx
        if self.prism.engine is not None:
            self.prism.engine.context_window = effective_ctx
        return effective_ctx

    def _resolve_stage_max_words(
        self,
        requested_max_words: int,
        max_new_tokens: int,
        system_prompt: Optional[str] = None,
        template_text: Optional[str] = None,
    ) -> int:
        """
        Dynamically bound word chunk length using ContextBudget.from_tokenizer so it always fits
        within the model's context budget (accounting for actual system prompt, schema, and max_new_tokens).
        """
        effective_ctx = self._apply_context_window_cap()
        try:
            budget = ContextBudget.from_tokenizer(
                tokenizer=self.prism.tokenizer,
                context_limit=effective_ctx,
                system_prompt=system_prompt,
                template_text=template_text,
                max_new_tokens=max_new_tokens,
            )
            available_input_tokens = budget.max_input_tokens
        except ValueError:
            available_input_tokens = max(1, effective_ctx // 2)
        safe_max_words = max(1, int(available_input_tokens * 0.7))
        return min(requested_max_words, safe_max_words)

    def discover_themes_across_documents(
        self,
        documents: List[Dict[str, str]],
        dry_run: bool = DRY_RUN,
    ) -> List[Theme]:
        """
        Phase 1: Discover abstract overarching themes across all documents concurrently.
        Delegates dynamic payload-aware batch sizing and windowed length-bucketing to
        PrismPipeline.run(), flushing each document's {doc}_themes.json to disk as soon
        as its chunks finish.
        """
        print(f"=== Phase 1: Global Theme Discovery (Model: {MODEL_NAME}) ===")
        total_docs = len(documents)
        if total_docs == 0:
            return self.aggregate_themes()

        theme_template = {
            "version": "1.0",
            "task": "theme_discovery",
            "system_prompt": THEME_DISCOVERY_SYSTEM_PROMPT,
            "prompt": f"{THEME_DISCOVERY_USER_PROMPT.replace('{text_content}', '{{ text }}')}\n\n{{schema_block}}",
        }

        def _extract_theme_objects(items: List[Any]) -> List[Theme]:
            extracted: List[Theme] = []
            for item in items:
                if isinstance(item, ThemeDiscoveryResult):
                    extracted.extend(item.themes)
                elif isinstance(item, Theme):
                    extracted.append(item)
                elif isinstance(item, dict):
                    try:
                        parsed_res = ThemeDiscoveryResult.model_validate(item)
                        extracted.extend(parsed_res.themes)
                    except Exception:
                        pass
            return extracted

        if not dry_run and self.prism.engine is None:
            self._log_checkpoint("Phase 1: Themes", f"Loading model '{MODEL_NAME}' onto device...")
            self.prism.init_engine(
                batch_size=BATCH_SIZE,
                max_new_tokens=THEME_MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
            )
            eff_ctx = self._apply_context_window_cap()
            self._log_checkpoint(
                "Phase 1: Themes",
                f"Model ready on device '{self.prism.device}' (effective context window: {eff_ctx} tokens)",
            )

        effective_theme_words = self._resolve_stage_max_words(
            THEME_CHUNK_MAX_WORDS,
            THEME_MAX_NEW_TOKENS,
            system_prompt=THEME_DISCOVERY_SYSTEM_PROMPT,
            template_text=THEME_DISCOVERY_USER_PROMPT,
        )
        chunk_rows: List[Dict[str, Any]] = []
        chunk_meta_by_uid: Dict[str, Dict[str, Any]] = {}
        doc_expected_chunks: Dict[str, int] = {}

        for doc_idx, doc in enumerate(documents, start=1):
            doc_id = doc["id"]
            safe_name = self.sanitize_filename(doc_id)
            theme_file_path = self.themes_dir / f"{safe_name}_themes.json"

            if RESUME_MODE == "skip" and theme_file_path.exists():
                print(f"-> [{doc_idx}/{total_docs}] Skipping theme discovery for: {doc_id} (already exists on disk)")
                try:
                    cached = json.loads(theme_file_path.read_text(encoding="utf-8"))
                    for t in cached:
                        self.context.all_discovered_themes.append(Theme.model_validate(t))
                except Exception:
                    pass
                continue

            doc_chunks = self.chunk_text(doc["text"], effective_theme_words)
            doc_expected_chunks[doc_id] = len(doc_chunks)

            for chunk_idx, (chunk_str, start_w, end_w) in enumerate(doc_chunks):
                user_prompt = THEME_DISCOVERY_USER_PROMPT.format(text_content=chunk_str)
                full_prompt = f"=== SYSTEM PROMPT ===\n{THEME_DISCOVERY_SYSTEM_PROMPT}\n\n=== USER PROMPT ===\n{user_prompt}"
                self._save_payload_text("stage_01_theme_discovery.txt", full_prompt)
                chunk_uid = f"{doc_id}__chunk_{chunk_idx}"
                row_meta = {
                    "chunk_uid": chunk_uid,
                    "doc_id": doc_id,
                    "chunk_idx": chunk_idx,
                    "start_word": start_w,
                    "end_word": end_w,
                    "text": chunk_str,
                }
                chunk_rows.append(row_meta)
                chunk_meta_by_uid[chunk_uid] = row_meta

        if not chunk_rows:
            return self.aggregate_themes()

        total_pending_docs = len(doc_expected_chunks)
        save_lock = threading.Lock()
        themes_by_doc: Dict[str, List[Theme]] = {doc_id: [] for doc_id in doc_expected_chunks}
        completed_chunks_by_doc: Dict[str, int] = {doc_id: 0 for doc_id in doc_expected_chunks}
        saved_doc_ids: Set[str] = set()

        def _flush_doc_themes(doc_id: str) -> None:
            if doc_id in saved_doc_ids:
                return
            saved_doc_ids.add(doc_id)
            doc_themes = themes_by_doc.get(doc_id, [])
            safe_name = self.sanitize_filename(doc_id)
            out_file = self.themes_dir / f"{safe_name}_themes.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump([t.model_dump() for t in doc_themes], f, indent=2)
            self.context.all_discovered_themes.extend(doc_themes)
            self._log_checkpoint(
                "Phase 1: Themes",
                f"Saved '{out_file.name}' ({len(doc_themes)} themes) — "
                f"Document {len(saved_doc_ids)}/{total_pending_docs} complete",
            )

        def _on_theme_batch_start(batch_idx: int, total_batches: int, batch_items: List[Dict[str, Any]]) -> None:
            batch_docs = list(dict.fromkeys(
                chunk_meta_by_uid[str(it.get("record_id") or "")]["doc_id"]
                for it in batch_items
                if str(it.get("record_id") or "") in chunk_meta_by_uid
            ))
            self._log_checkpoint(
                "Phase 1: Themes",
                f"Running concurrent batch {batch_idx}/{total_batches}: {len(batch_items)} chunk(s) "
                f"across {len(batch_docs)} doc(s) {batch_docs[:4]}{'...' if len(batch_docs) > 4 else ''}",
            )

        def _on_theme_chunk(chunk_res: Dict[str, Any]) -> None:
            chunk_uid = str(chunk_res.get("record_id") or chunk_res.get("id") or "")
            row_meta = chunk_meta_by_uid.get(chunk_uid)
            if not row_meta:
                return
            doc_id = row_meta["doc_id"]
            items = chunk_res.get("parsed_items") or chunk_res.get("extracted_items") or []

            with save_lock:
                if not items and not chunk_res.get("parse_success", True):
                    err_rec = {
                        "source_document": doc_id,
                        "start_word": row_meta["start_word"],
                        "end_word": row_meta["end_word"],
                        "error": chunk_res.get("parse_error", "Validation error"),
                        "chunk_text": row_meta["text"],
                    }
                    with open(self.log_dir / "stage_01_theme_discovery_errors.json", "a", encoding="utf-8") as ef:
                        ef.write(json.dumps(err_rec) + "\n")
                    new_themes: List[Theme] = []
                else:
                    new_themes = _extract_theme_objects(items)
                    themes_by_doc[doc_id].extend(new_themes)

                completed_chunks_by_doc[doc_id] += 1
                self._log_checkpoint(
                    "Phase 1: Themes",
                    f"Doc '{doc_id}' — chunk {completed_chunks_by_doc[doc_id]}/{doc_expected_chunks[doc_id]} "
                    f"processed (+{len(new_themes)} themes)",
                )
                if completed_chunks_by_doc[doc_id] >= doc_expected_chunks[doc_id]:
                    _flush_doc_themes(doc_id)

        results = self.prism.run(
            task="extract",
            input_data=chunk_rows,
            text_column="text",
            id_column="chunk_uid",
            schema_cls=ThemeDiscoveryResult,
            prompt_template=theme_template,
            system_prompt=THEME_DISCOVERY_SYSTEM_PROMPT,
            chunk=False,
            as_array=False,
            batch_size=BATCH_SIZE,
            max_new_tokens=THEME_MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            constrain_json=CONSTRAIN_JSON,
            dry_run=dry_run,
            on_batch_start=_on_theme_batch_start,
            on_chunk_processed=_on_theme_chunk,
        )

        if dry_run:
            return self.context.all_discovered_themes

        for row_meta, res in zip(chunk_rows, results):
            doc_id = row_meta["doc_id"]
            if doc_id not in saved_doc_ids:
                items = res.get("extracted_items") or res.get("parsed_items") or []
                themes_by_doc[doc_id].extend(_extract_theme_objects(items))

        for doc_id in doc_expected_chunks:
            if doc_id not in saved_doc_ids:
                _flush_doc_themes(doc_id)

        return self.aggregate_themes()

    def aggregate_themes(self) -> List[Theme]:
        """Phase 1.5: Consolidate individual *_themes.json files into all_themes.json."""
        print("\n=== Phase 1.5: Theme Aggregation ===")
        all_themes: List[Dict[str, Any]] = []
        for fpath in sorted(self.themes_dir.glob("*_themes.json")):
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                all_themes.extend(data)
            except Exception as e:
                print(f"Warning: Failed to load theme file {fpath.name}: {e}")

        if not all_themes and (self.out_dir / "all_themes.json").exists():
            all_themes = json.loads((self.out_dir / "all_themes.json").read_text(encoding="utf-8"))

        self.context.all_discovered_themes = [Theme.model_validate(t) for t in all_themes]
        if all_themes:
            out_all = self.out_dir / "all_themes.json"
            with open(out_all, "w", encoding="utf-8") as f:
                json.dump([t.model_dump() for t in self.context.all_discovered_themes], f, indent=2)
            self._log_checkpoint(
                "Phase 1.5: Aggregation",
                f"Saved '{out_all.name}' ({len(self.context.all_discovered_themes)} total themes)",
            )
        return self.context.all_discovered_themes

    # =========================================================================
    # PHASE 2: EMPIRICAL CLUSTERING & MASTER THEME SYNTHESIS
    # =========================================================================

    def _export_cluster_theme_mapping(self, active_partition: Dict[str, Any], synthesized_master_names: List[str]) -> None:
        theme_mapping: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        clusters = active_partition.get("clusters", {})
        cluster_items = sorted(clusters.items()) if isinstance(clusters, dict) else list(enumerate(clusters))

        for idx, (c_id, c_data) in enumerate(cluster_items):
            if idx < len(synthesized_master_names):
                raw_val = str(synthesized_master_names[idx]).strip()
                cleaned = re.sub(r"^(?:Cluster\s*\d+[\s:.-]*|\d+[\s:.-]+)", "", raw_val).strip()
                master_name = cleaned if cleaned else raw_val
            else:
                master_name = c_data.get("label", f"Cluster_{c_id}")

            cluster_themes = c_data.get("all_themes", [])
            n_c = len(cluster_themes)
            if cluster_themes:
                sims_arr = np.array([float(t.get("similarity_to_centroid", 0.0)) for t in cluster_themes], dtype=np.float32)
                mu_c = float(c_data.get("mean_similarity", float(np.mean(sims_arr))))
                sigma_c = float(c_data.get("std_similarity", float(np.std(sims_arr)) if n_c > 1 else 0.0))
            else:
                mu_c, sigma_c = MANIFOLD_EDGE_THRESHOLD, 0.0

            # Self-calibrating von Mises-Fisher (vMF) mean resultant length & sigma bands
            # bounded below by the partition's effective manifold edge threshold (anisotropy- and dimension-invariant):
            tau_edge = float(active_partition.get("effective_edge_threshold", MANIFOLD_EDGE_THRESHOLD))
            tau_high = max(mu_c - 1e-4, tau_edge)
            tau_medium = max(mu_c - max(sigma_c, 0.05), tau_edge)
            tau_low = max(mu_c - 2.0 * max(sigma_c, 0.05), 0.5 * tau_edge)

            for t_item in cluster_themes:
                sim = float(t_item.get("similarity_to_centroid", 0.0))
                loo_sim = float(t_item.get("loo_similarity_to_centroid", sim))
                # Guard against small-cluster 1/sqrt(n_c) self-bias by requiring LOO similarity >= tau_edge for "high"
                if sim >= tau_high and (n_c == 1 or loo_sim >= tau_edge):
                    conf = "high"
                elif sim >= tau_medium and (n_c == 1 or loo_sim >= 0.5 * tau_edge):
                    conf = "medium"
                elif sim >= tau_low:
                    conf = "low"
                else:
                    conf = "unassigned"

                theme_mapping[master_name].append({
                    "raw_theme": t_item["title"],
                    "confidence": conf,
                    "similarity_score": round(sim, 4),
                    "loo_similarity_score": round(loo_sim, 4),
                })

        mapping_path = self.out_dir / "theme_mapping_clusters.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(theme_mapping, f, indent=2)
        print(f"-> Exported theme consolidation mapping directly to: {mapping_path}")

    def synthesize_master_themes(self, force: bool = False, dry_run: bool = DRY_RUN) -> Optional[MasterThemeSynthesisResult]:
        """
        Phase 2: Cluster global aggregated themes with ThemeClusteringEngine and
        synthesize the authoritative Master Ontology via Map-Reduce or Single-Stage.
        """
        print(f"\n=== Phase 2: Master Theme Synthesis (Model: {MODEL_NAME}) ===")
        self.aggregate_themes()
        if not self.context.all_discovered_themes:
            print("Warning: No themes found in themes directory or all_themes.json. Skipping master theme synthesis.")
            return None

        master_path = self.out_dir / "master_themes.json"
        mapping_path = self.out_dir / "theme_mapping_clusters.json"

        if not force and RESUME_MODE == "skip" and master_path.exists() and mapping_path.exists():
            print("-> Skipping master theme synthesis (already exists on disk). Loading existing master themes...")
            return self.load_master_themes()

        if MANAGE_VRAM:
            clear_device_memory()

        self._log_checkpoint(
            "Phase 2: Clustering",
            f"Embedding and clustering {len(self.context.all_discovered_themes)} discovered themes (dynamic Leiden)...",
        )
        cluster_engine = ThemeClusteringEngine(
            themes_path=self.out_dir / "all_themes.json",
            output_dir=self.out_dir,
        )
        cluster_results = cluster_engine.cluster_themes()
        if MANAGE_VRAM:
            clear_device_memory()
        partitions = cluster_results["partitions"]
        exemplar_groups = cluster_results["active_exemplars"]
        active_partition = cluster_engine.get_active_partition(partitions)
        self._log_checkpoint(
            "Phase 2: Clustering",
            f"Formed {len(exemplar_groups)} empirical theme cluster(s)",
        )

        if not dry_run and self.prism.engine is None:
            self._log_checkpoint("Phase 2: Master Synthesis", f"Loading model '{MODEL_NAME}' onto device...")
            self.prism.init_engine(
                batch_size=BATCH_SIZE,
                max_new_tokens=MASTER_SYNTHESIS_MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
            )
            self._apply_context_window_cap()

        map_template = {
            "version": "1.0",
            "task": "cluster_map",
            "system_prompt": CLUSTER_MAP_SYSTEM_PROMPT,
            "prompt": "{{ text }}\n\n{schema_block}",
        }
        map_max_words = self._resolve_stage_max_words(
            requested_max_words=MAP_MAX_THEMES_PER_CLUSTER * 40,
            max_new_tokens=MASTER_SYNTHESIS_MAX_NEW_TOKENS,
            system_prompt=CLUSTER_MAP_SYSTEM_PROMPT,
            template_text=CLUSTER_MAP_USER_PROMPT,
        )
        map_payloads = cluster_engine.get_map_payload(
            max_themes_per_cluster=MAP_MAX_THEMES_PER_CLUSTER,
            payload_format="title_desc",
            max_words_budget=map_max_words,
        )
        map_inputs = []
        for c_info in map_payloads:
            c_id = c_info["cluster_id"]
            u_prompt = CLUSTER_MAP_USER_PROMPT.format(
                cluster_id=c_id + 1,
                theme_count=c_info["theme_count"],
                displayed_count=c_info["displayed_count"],
                theme_lines=c_info["theme_lines"],
            )
            self._save_payload_text(
                "stage_01_master_theme_map.txt",
                f"=== SYSTEM PROMPT ===\n{CLUSTER_MAP_SYSTEM_PROMPT}\n\n=== USER PROMPT ===\n{u_prompt}",
            )
            map_inputs.append({"id": f"cluster_{c_id}", "cluster_id": c_id, "text": u_prompt, "theme_lines": c_info["theme_lines"]})

        self._log_checkpoint(
            "Phase 2A: Cluster Map",
            f"Synthesizing candidate definitions for {len(map_inputs)} cluster(s)...",
        )
        map_results = self.prism.run(
            task="extract",
            input_data=map_inputs,
            text_column="text",
            id_column="id",
            schema_cls=ClusterCandidateSynthesis,
            prompt_template=map_template,
            system_prompt=CLUSTER_MAP_SYSTEM_PROMPT,
            chunk=False,
            as_array=False,
            batch_size=BATCH_SIZE,
            max_new_tokens=MASTER_SYNTHESIS_MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            constrain_json=CONSTRAIN_JSON,
            dry_run=dry_run,
            on_batch_start=lambda b_idx, b_tot, b_items: self._log_checkpoint(
                "Phase 2A: Cluster Map",
                f"Running concurrent Map batch {b_idx}/{b_tot} ({len(b_items)} cluster(s))",
            ),
        )
        if dry_run:
            return None

        map_candidates: List[ClusterCandidateSynthesis] = []
        for m_in, m_res in zip(map_inputs, map_results):
            c_id = m_in["cluster_id"]
            items = m_res.get("extracted_items") or m_res.get("parsed_items") or []
            cand: Optional[ClusterCandidateSynthesis] = None
            for it in items:
                if isinstance(it, ClusterCandidateSynthesis):
                    cand = it
                    break
                if isinstance(it, dict):
                    try:
                        cand = ClusterCandidateSynthesis.model_validate(it)
                        break
                    except Exception:
                        pass
            if cand is None:
                first_line = m_in["theme_lines"].split("\n")[0] if m_in["theme_lines"] else f"Cluster #{c_id + 1}"
                clean_title = first_line.split(".", 1)[-1].split(":", 1)[0].strip() or f"Theme Domain {c_id + 1}"
                cand = ClusterCandidateSynthesis(
                    cluster_id=c_id,
                    candidate_title=clean_title,
                    scope_definition=f"Domain encompassing core conceptual themes centered on {clean_title}.",
                    key_inclusions=[clean_title],
                    key_exclusions=[],
                    confidence="low",
                )
            cand.cluster_id = c_id
            map_candidates.append(cand)

        self.context.cluster_candidates = map_candidates
        cand_path = self.out_dir / "cluster_candidates.json"
        with open(cand_path, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in map_candidates], f, indent=2)
        self._log_checkpoint(
            "Phase 2A: Cluster Map",
            f"Saved '{cand_path.name}' ({len(map_candidates)} cluster candidate syntheses)",
        )
        del map_results
        if MANAGE_VRAM:
            clear_device_memory()

        reduce_sys_prompt = CLUSTER_REDUCE_SYSTEM_PROMPT
        reduce_max_words = self._resolve_stage_max_words(
            requested_max_words=12000,
            max_new_tokens=MASTER_SYNTHESIS_MAX_NEW_TOKENS,
            system_prompt=reduce_sys_prompt,
            template_text=CLUSTER_REDUCE_USER_PROMPT,
        )

        # Dynamically adapt anchor count and description length so the unified Reduce prompt fits ContextBudget
        cluster_blocks: List[str] = []
        for anchor_cap in range(_REDUCE_ANCHOR_THEME_COUNT, 0, -1):
            reduce_anchors = cluster_engine.get_reduce_anchors(top_n=anchor_cap)
            for include_desc in (True, False):
                trial_blocks: List[str] = []
                for cand in map_candidates:
                    anchors = reduce_anchors.get(cand.cluster_id, [])
                    if include_desc:
                        anchor_lines = [f"     - {a['title']}: {a['description']}" if a.get("description") else f"     - {a['title']}" for a in anchors]
                    else:
                        anchor_lines = [f"     - {a['title']}" for a in anchors]
                    trial_blocks.append(
                        f"### Cluster #{cand.cluster_id + 1}:\n"
                        f"- Proposed Master Theme Title: {cand.candidate_title}\n"
                        f"- Scope Definition: {cand.scope_definition}\n"
                        f"- Key Inclusions: {', '.join(cand.key_inclusions) or 'None specified'}\n"
                        f"- Key Exclusions: {', '.join(cand.key_exclusions) or 'None specified'}\n"
                        f"- Cohesion Confidence: {cand.confidence or 'high'}\n"
                        f"- Top {len(anchors)} Exemplar Themes (Empirical Anchors):\n" + "\n".join(anchor_lines)
                    )
                cluster_blocks = trial_blocks
                if len("\n\n".join(cluster_blocks).split()) <= reduce_max_words:
                    break
            if len("\n\n".join(cluster_blocks).split()) <= reduce_max_words:
                break

        reduce_user_prompt = CLUSTER_REDUCE_USER_PROMPT.format(
            cluster_count=len(map_candidates),
            cluster_blocks="\n\n".join(cluster_blocks),
        )
        self._save_payload_text(
            "stage_01_master_theme_reduce.txt",
            f"=== SYSTEM PROMPT ===\n{reduce_sys_prompt}\n\n=== USER PROMPT ===\n{reduce_user_prompt}",
        )

        reduce_template = {
            "version": "1.0",
            "task": "master_reduce",
            "system_prompt": reduce_sys_prompt,
            "prompt": "{{ text }}\n\n{schema_block}",
        }
        self._log_checkpoint("Phase 2B: Master Ontology", "Running Reduce synthesis for unified Master Ontology...")
        reduce_results = self.prism.run(
            task="extract",
            input_data=reduce_user_prompt,
            schema_cls=MasterThemeSynthesisResult,
            prompt_template=reduce_template,
            system_prompt=reduce_sys_prompt,
            chunk=False,
            as_array=False,
            batch_size=1,
            max_new_tokens=MASTER_SYNTHESIS_MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            constrain_json=CONSTRAIN_JSON,
            dry_run=dry_run,
        )
        if dry_run:
            return None

        master_res: Optional[MasterThemeSynthesisResult] = None
        for r in reduce_results:
            for it in (r.get("extracted_items") or r.get("parsed_items") or []):
                if isinstance(it, MasterThemeSynthesisResult):
                    master_res = it
                    break
                if isinstance(it, dict):
                    try:
                        master_res = MasterThemeSynthesisResult.model_validate(it)
                        break
                    except Exception:
                        pass

        if master_res is None:
            fallback_titles = [
                c.candidate_title for c in self.context.cluster_candidates
            ] or [
                grp["exemplars"][0]["title"] if grp["exemplars"] else f"Theme {grp['cluster_id'] + 1}"
                for grp in exemplar_groups
            ]
            master_res = MasterThemeSynthesisResult(
                master_domain=_derive_fallback_domain(fallback_titles),
                master_themes=fallback_titles,
            )
        elif not str(master_res.master_domain or "").strip() or master_res.master_domain == _FALLBACK_DOMAIN:
            master_res.master_domain = _derive_fallback_domain(master_res.master_themes)

        cleaned_themes = []
        for t in master_res.master_themes:
            c_str = re.sub(r"^(?:Cluster\s*\d+[\s:.-]*|\d+[\s:.-]+)", "", str(t)).strip()
            cleaned_themes.append(c_str if c_str else str(t))
        master_res.master_themes = cleaned_themes
        self.context.master_themes = master_res

        with open(master_path, "w", encoding="utf-8") as f:
            f.write(master_res.model_dump_json(indent=2))
        self._log_checkpoint(
            "Phase 2B: Master Ontology",
            f"Saved '{master_path.name}' ({len(master_res.master_themes)} master themes in domain '{master_res.master_domain}')",
        )

        summary_path = cluster_engine.export_clustering_summary(
            master_themes=master_res.master_themes,
            master_domain=master_res.master_domain,
        )
        self._log_checkpoint("Phase 2.5: Clustering Summary", f"Saved '{summary_path.name}'")
        self._export_cluster_theme_mapping(active_partition, master_res.master_themes)
        del cluster_engine, cluster_results, partitions, exemplar_groups, active_partition, reduce_results
        if MANAGE_VRAM:
            clear_device_memory()
        return master_res

    # =========================================================================
    # PHASE 3: GLOBAL SVO TRIPLE EXTRACTION & AGGREGATION
    # =========================================================================

    def extract_triples_across_documents(
        self,
        documents: List[Dict[str, str]],
        dry_run: bool = DRY_RUN,
    ) -> List[Dict[str, Any]]:
        """
        Phase 3: Extract atomic (Subject, Predicate, Object) triples document-by-document
        using Master Themes context and verbatim source_quote grounding.
        Saves each document's {doc}_triplets.json immediately upon completing that document.
        """
        print(f"\n=== Phase 3: Global Triple Extraction (Model: {MODEL_NAME}) ===")
        total_docs = len(documents)
        if total_docs == 0:
            return self.aggregate_triplets()

        if not self.context.master_themes:
            self.load_master_themes()

        if self.context.master_themes and self.context.master_themes.master_themes:
            theme_list = ", ".join([f'"{t}"' for t in self.context.master_themes.master_themes])
            sys_context = f'\nAllowed theme_association values: [{theme_list}, "Other"]'
        else:
            sys_context = '\nAllowed theme_association values: ["Other"]'

        full_sys_prompt = f"{TRIPLE_EXTRACTION_SYSTEM_PROMPT}\n{sys_context}"
        triple_template = {
            "version": "1.0",
            "task": "triple_extraction",
            "system_prompt": full_sys_prompt,
            "prompt": f"{TRIPLE_EXTRACTION_USER_PROMPT.replace('{text_content}', '{{ text }}')}\n\n{{schema_block}}",
        }

        allowed_themes = (
            set(self.context.master_themes.master_themes)
            if (self.context.master_themes and self.context.master_themes.master_themes)
            else set()
        )

        def _extract_triple_dicts(items: List[Any], doc_id: str) -> List[Dict[str, Any]]:
            extracted_triples: List[RawTriple] = []
            for item in items:
                if isinstance(item, TripleExtractionResult):
                    extracted_triples.extend(item.triples)
                elif isinstance(item, RawTriple):
                    extracted_triples.append(item)
                elif isinstance(item, dict):
                    try:
                        parsed_tr = TripleExtractionResult.model_validate(item)
                        extracted_triples.extend(parsed_tr.triples)
                    except Exception:
                        pass

            dicts: List[Dict[str, Any]] = []
            for t in extracted_triples:
                t_dict = t.model_dump()
                if allowed_themes and t_dict.get("theme_association") not in allowed_themes:
                    t_dict["theme_association"] = "Other"
                t_dict["source_document"] = doc_id
                dicts.append(t_dict)
            return dicts

        if not dry_run and self.prism.engine is None:
            self._log_checkpoint("Phase 3: Triples", f"Loading model '{MODEL_NAME}' onto device...")
            self.prism.init_engine(
                batch_size=BATCH_SIZE,
                max_new_tokens=TRIPLE_MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
            )
            self._apply_context_window_cap()

        effective_triple_words = self._resolve_stage_max_words(
            TRIPLE_CHUNK_MAX_WORDS,
            TRIPLE_MAX_NEW_TOKENS,
            system_prompt=full_sys_prompt,
            template_text=TRIPLE_EXTRACTION_USER_PROMPT,
        )
        chunk_rows: List[Dict[str, Any]] = []
        chunk_meta_by_uid: Dict[str, Dict[str, Any]] = {}
        doc_expected_chunks: Dict[str, int] = {}

        for doc_idx, doc in enumerate(documents, start=1):
            doc_id = doc["id"]
            self.context.processed_documents.add(doc_id)
            safe_name = self.sanitize_filename(doc_id)
            triple_file_path = self.triples_dir / f"{safe_name}_triplets.json"

            if RESUME_MODE == "skip" and triple_file_path.exists():
                print(f"-> [{doc_idx}/{total_docs}] Skipping triple extraction for: {doc_id} (already exists on disk)")
                continue

            triple_chunks = self.chunk_text(doc["text"], effective_triple_words)
            doc_expected_chunks[doc_id] = len(triple_chunks)

            for chunk_idx, (chunk_str, start_w, end_w) in enumerate(triple_chunks):
                user_prompt = TRIPLE_EXTRACTION_USER_PROMPT.format(text_content=chunk_str)
                self._save_payload_text(
                    "stage_01_triple_extraction.txt",
                    f"=== SYSTEM PROMPT ===\n{full_sys_prompt}\n\n=== USER PROMPT ===\n{user_prompt}",
                )
                chunk_uid = f"{doc_id}__tchunk_{chunk_idx}"
                row_meta = {
                    "chunk_uid": chunk_uid,
                    "doc_id": doc_id,
                    "chunk_idx": chunk_idx,
                    "start_word": start_w,
                    "end_word": end_w,
                    "text": chunk_str,
                }
                chunk_rows.append(row_meta)
                chunk_meta_by_uid[chunk_uid] = row_meta

        if not chunk_rows:
            return self.aggregate_triplets()

        total_pending_docs = len(doc_expected_chunks)
        save_lock = threading.Lock()
        triples_by_doc: Dict[str, List[Dict[str, Any]]] = {doc_id: [] for doc_id in doc_expected_chunks}
        completed_chunks_by_doc: Dict[str, int] = {doc_id: 0 for doc_id in doc_expected_chunks}
        saved_doc_ids: Set[str] = set()

        def _flush_doc_triples(doc_id: str) -> None:
            if doc_id in saved_doc_ids:
                return
            saved_doc_ids.add(doc_id)
            doc_triples = triples_by_doc.get(doc_id, [])
            safe_name = self.sanitize_filename(doc_id)
            out_file = self.triples_dir / f"{safe_name}_triplets.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(doc_triples, f, indent=2)
            self._log_checkpoint(
                "Phase 3: Triples",
                f"Saved '{out_file.name}' ({len(doc_triples)} triples) — "
                f"Document {len(saved_doc_ids)}/{total_pending_docs} complete",
            )

        def _on_triple_batch_start(batch_idx: int, total_batches: int, batch_items: List[Dict[str, Any]]) -> None:
            batch_docs = list(dict.fromkeys(
                chunk_meta_by_uid[str(it.get("record_id") or "")]["doc_id"]
                for it in batch_items
                if str(it.get("record_id") or "") in chunk_meta_by_uid
            ))
            self._log_checkpoint(
                "Phase 3: Triples",
                f"Running concurrent batch {batch_idx}/{total_batches}: {len(batch_items)} chunk(s) "
                f"across {len(batch_docs)} doc(s) {batch_docs[:4]}{'...' if len(batch_docs) > 4 else ''}",
            )

        def _on_triple_chunk(chunk_res: Dict[str, Any]) -> None:
            chunk_uid = str(chunk_res.get("record_id") or chunk_res.get("id") or "")
            row_meta = chunk_meta_by_uid.get(chunk_uid)
            if not row_meta:
                return
            doc_id = row_meta["doc_id"]
            items = chunk_res.get("parsed_items") or chunk_res.get("extracted_items") or []

            with save_lock:
                if not items and not chunk_res.get("parse_success", True):
                    err_rec = {
                        "phase": "initial_triple_extraction_failed",
                        "source_document": doc_id,
                        "start_word": row_meta["start_word"],
                        "end_word": row_meta["end_word"],
                        "error_message": chunk_res.get("parse_error", "Validation error"),
                        "malformed_output": chunk_res.get("raw_output", ""),
                        "chunk_text": row_meta["text"],
                    }
                    with open(self.log_dir / "stage_01_triple_extraction_errors.json", "a", encoding="utf-8") as ef:
                        ef.write(json.dumps(err_rec) + "\n")
                    new_triples: List[Dict[str, Any]] = []
                else:
                    new_triples = _extract_triple_dicts(items, doc_id)
                    triples_by_doc[doc_id].extend(new_triples)

                completed_chunks_by_doc[doc_id] += 1
                self._log_checkpoint(
                    "Phase 3: Triples",
                    f"Doc '{doc_id}' — chunk {completed_chunks_by_doc[doc_id]}/{doc_expected_chunks[doc_id]} "
                    f"processed (+{len(new_triples)} triples)",
                )
                if completed_chunks_by_doc[doc_id] >= doc_expected_chunks[doc_id]:
                    _flush_doc_triples(doc_id)

        results = self.prism.run(
            task="extract",
            input_data=chunk_rows,
            text_column="text",
            id_column="chunk_uid",
            schema_cls=TripleExtractionResult,
            prompt_template=triple_template,
            system_prompt=full_sys_prompt,
            chunk=False,
            as_array=False,
            batch_size=BATCH_SIZE,
            max_new_tokens=TRIPLE_MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            constrain_json=CONSTRAIN_JSON,
            dry_run=dry_run,
            on_batch_start=_on_triple_batch_start,
            on_chunk_processed=_on_triple_chunk,
        )

        if dry_run:
            return self.context.raw_triples

        for row_meta, res in zip(chunk_rows, results):
            doc_id = row_meta["doc_id"]
            if doc_id not in saved_doc_ids:
                items = res.get("extracted_items") or res.get("parsed_items") or []
                triples_by_doc[doc_id].extend(_extract_triple_dicts(items, doc_id))

        for doc_id in doc_expected_chunks:
            if doc_id not in saved_doc_ids:
                _flush_doc_triples(doc_id)

        return self.aggregate_triplets()

    def aggregate_triplets(self) -> List[Dict[str, Any]]:
        """Phase 3.5: Consolidate all *_triplets.json files into original_triplets.json and triplet_counts.csv."""
        print("\n=== Phase 3.5: Triple Aggregation ===")
        all_triples: List[Dict[str, Any]] = []
        processed_docs: Set[str] = set()

        for fpath in sorted(self.triples_dir.glob("*_triplets.json")):
            try:
                triples_data = json.loads(fpath.read_text(encoding="utf-8"))
                all_triples.extend(triples_data)
                for t in triples_data:
                    doc = t.get("source_document")
                    if doc:
                        processed_docs.add(doc)
            except Exception as e:
                print(f"Warning: Failed to load triplet file {fpath.name}: {e}")

        self.context.raw_triples = all_triples
        self.context.processed_documents = processed_docs

        with open(self.out_dir / "original_triplets.json", "w", encoding="utf-8") as f:
            json.dump(all_triples, f, indent=2)

        counts = {doc: 0 for doc in processed_docs}
        for t in all_triples:
            doc = t.get("source_document", "unknown")
            counts[doc] = counts.get(doc, 0) + 1

        csv_path = self.out_dir / "triplet_counts.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("document_name,triplet_count\n")
            for doc, count in sorted(counts.items()):
                f.write(f"{doc},{count}\n")

        self._log_checkpoint(
            "Phase 3.5: Aggregation",
            f"Saved 'original_triplets.json' ({len(all_triples)} total triples) and '{csv_path.name}' ({len(counts)} docs)",
        )
        return all_triples

    # =========================================================================
    # FULL STAGE 1 ORCHESTRATOR
    # =========================================================================

    def run_stage_1(
        self,
        input_data: Optional[Union[str, Path, pd.DataFrame, List[Dict[str, Any]]]] = None,
        dry_run: bool = DRY_RUN,
    ) -> Dict[str, Any]:
        """Execute Stage 1 Part 1 (Themes & Master Synthesis) and Stage 1 Part 2 (SVO Triples)."""
        print("=== SemanticPrism Stage 1: Full Extraction Pipeline ===")
        stage_start = time.time()

        if RESUME_MODE != "skip":
            self.clear_stage_logs()

        documents = self.load_input_documents(input_path=input_data)
        print(f"Found {len(documents)} document(s) to process.\n")

        # Part 1: Theme Discovery, Aggregation, Empirical Clustering & Master Synthesis
        self.discover_themes_across_documents(documents, dry_run=dry_run)
        self.synthesize_master_themes(dry_run=dry_run)
        if MANAGE_VRAM:
            clear_device_memory()

        # Part 2: SVO Triple Extraction & Aggregation
        self.extract_triples_across_documents(documents, dry_run=dry_run)
        if MANAGE_VRAM:
            clear_device_memory()

        total_duration = time.time() - stage_start
        print("\n" + "=" * 60)
        print("STAGE 1 FULL EXTRACTION PIPELINE COMPLETE!")
        print(f"Total Stage 1 Time: {total_duration:.2f} seconds")
        print(f"Outputs saved in '{self.out_dir}':")
        print(" - all_themes.json")
        print(" - clustering_summary.json")
        print(" - cluster_candidates.json")
        print(" - master_themes.json")
        print(" - theme_mapping_clusters.json")
        print(" - original_triplets.json")
        print(" - triplet_counts.csv")
        print("=" * 60 + "\n")

        return {
            "themes_count": len(self.context.all_discovered_themes),
            "master_themes": self.context.master_themes.model_dump() if self.context.master_themes else None,
            "triples_count": len(self.context.raw_triples),
            "output_dir": str(self.out_dir),
        }


def main() -> None:
    pipeline = SemanticPrismStage1Pipeline()
    pipeline.run_stage_1()


if __name__ == "__main__":
    main()
