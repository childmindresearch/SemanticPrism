"""
SemanticPrism Stage 1: Full Extraction Pipeline Orchestrator (Powered by PrismRunner)
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_SIBLING_PRISMRUNNER = REPO_ROOT.parent / "PrismRunner"
if _SIBLING_PRISMRUNNER.exists() and str(_SIBLING_PRISMRUNNER) not in sys.path:
    sys.path.insert(0, str(_SIBLING_PRISMRUNNER))

try:
    from model_config import (
        EMBEDDING_MODEL as _DEFAULT_EMBEDDING_MODEL,
        MODEL_NAME as _DEFAULT_MODEL_NAME,
    )
except Exception:
    _DEFAULT_MODEL_NAME = "Qwen/Qwen3.6-27B"
    _DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

from prismrunner import (
    ContextBudget,
    PrismPipeline,
    clear_device_memory,
)

from src.extraction.clustering import (
    ThemeClusteringEngine,
    _FALLBACK_DOMAIN,
    _REDUCE_ANCHOR_THEME_COUNT,
    _derive_fallback_domain,
)
from src.extraction.prompts import (
    CLUSTER_MAP_SYSTEM_PROMPT,
    CLUSTER_MAP_USER_PROMPT,
    CLUSTER_REDUCE_SYSTEM_PROMPT,
    CLUSTER_REDUCE_USER_PROMPT,
    THEME_DISCOVERY_SYSTEM_PROMPT,
    THEME_DISCOVERY_USER_PROMPT,
    TRIPLE_EXTRACTION_SYSTEM_PROMPT,
    TRIPLE_EXTRACTION_USER_PROMPT,
)
from src.extraction.schemas import (
    ClusterCandidateSynthesis,
    MasterThemeSynthesisResult,
    RawTriple,
    Theme,
    ThemeDiscoveryResult,
    TripleExtractionResult,
    salvage_partial_triples,
)


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
        model_name: str = _DEFAULT_MODEL_NAME,
        theme_embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
        device: str = "auto",
        torch_dtype: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        attn_implementation: str = "auto",
        batch_size: Union[int, str] = "auto",
        context_window_cap: Union[int, str] = "auto",
        temperature: float = 0.0,
        constrain_json: bool = True,
        manage_vram: bool = True,
        dry_run: bool = False,
        input_path: str = "inputs/testdocs.parquet",
        id_column: str = "unique_val",
        text_column: str = "report",
        document_limit: Optional[int] = None,
        outputs_dir: str = "outputs",
        save_llm_payloads: bool = True,
        resume_mode: str = "skip",
        print_checkpoints: bool = True,
        theme_chunk_max_words: int = 1800,
        triple_chunk_max_words: Union[int, str] = "auto",
        triple_chunk_overlap_words: Union[int, str] = "auto",
        theme_max_new_tokens: int = 512,
        master_synthesis_max_new_tokens: int = 512,
        triple_max_new_tokens: Union[int, str] = "auto",
        map_max_themes_per_cluster: int = 150,
        manifold_edge_threshold: float = 0.35,
        micro_cluster_ratio: float = 0.01,
    ):
        self.context = context or PipelineRunContext()
        self.model_name = model_name
        self.theme_embedding_model = theme_embedding_model
        self.device = device
        self.torch_dtype = torch_dtype
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.attn_implementation = attn_implementation
        self.batch_size = batch_size
        self.context_window_cap = context_window_cap
        self.temperature = temperature
        self.constrain_json = constrain_json
        self.manage_vram = manage_vram
        self.dry_run = dry_run
        self.input_path = input_path
        self.id_column = id_column
        self.text_column = text_column
        self.document_limit = document_limit
        self.outputs_dir = outputs_dir
        self.save_llm_payloads = save_llm_payloads
        self.resume_mode = resume_mode
        self.print_checkpoints = print_checkpoints
        self.theme_chunk_max_words = theme_chunk_max_words
        self.triple_chunk_max_words = triple_chunk_max_words
        self.triple_chunk_overlap_words = triple_chunk_overlap_words
        self.theme_max_new_tokens = theme_max_new_tokens
        self.master_synthesis_max_new_tokens = master_synthesis_max_new_tokens
        self.triple_max_new_tokens = triple_max_new_tokens
        self.map_max_themes_per_cluster = map_max_themes_per_cluster
        self.manifold_edge_threshold = manifold_edge_threshold
        self.micro_cluster_ratio = micro_cluster_ratio

        root_out = Path(self.outputs_dir)
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
                model_name=self.model_name,
                device=self.device,
                batch_size=self.batch_size,
                config={
                    "model": {
                        "name": self.model_name,
                        "device": self.device,
                        "torch_dtype": self.torch_dtype,
                        "load_in_4bit": self.load_in_4bit,
                        "load_in_8bit": self.load_in_8bit,
                        "attn_implementation": self.attn_implementation,
                    },
                    "inference": {
                        "batch_size": self.batch_size,
                        "temperature": self.temperature,
                        "constrain_json": self.constrain_json,
                    },
                },
            )

        if self.context.master_themes is None:
            self.load_master_themes()

        init_profile = self._resolve_triple_extraction_profile()
        try:
            with open(self.out_dir / "model_execution_metadata.json", "w", encoding="utf-8") as mf:
                json.dump(
                    {
                        "stage": "stage_1_extraction",
                        "configured_model": self.model_name,
                        "active_model": self.model_name,
                        "embedding_model": self.theme_embedding_model,
                        "capability_tier": init_profile["tier"],
                        "triple_chunk_target_words": init_profile["chunk_words"],
                        "triple_chunk_overlap_words": init_profile["overlap_words"],
                        "triple_max_new_tokens": init_profile["max_new_tokens"],
                        "quantization_or_dtype": (
                            "4-bit NF4"
                            if self.load_in_4bit
                            else ("8-bit" if self.load_in_8bit else f"native ({self.torch_dtype})")
                        ),
                        "fallback_used": False,
                        "fallback_reason": None,
                    },
                    mf,
                    indent=2,
                )
        except Exception:
            pass

    @staticmethod
    def _estimate_model_billions(
        model_name: str,
        prism_obj: Any = None,
        load_in_4bit: bool = False,
    ) -> Optional[float]:
        """Estimate effective model parameter scale (in billions) from model_name or loaded engine."""
        name_str = str(model_name or "")

        # 1. Active-parameter MoE notation (e.g., "Qwen3-30B-A3B")
        moe_active = re.search(r"(\d+(?:\.\d+)?)\s*[bB]-[aA](\d+(?:\.\d+)?)\s*[bB]\b", name_str)
        if moe_active:
            total_b = float(moe_active.group(1))
            active_b = float(moe_active.group(2))
            return round(math.sqrt(max(0.1, total_b * active_b)), 2)

        # 2. Expert-count MoE notation (e.g., "Mixtral-8x7B", "8x22B")
        moe_experts = re.search(r"(\d+)\s*[xX]\s*(\d+(?:\.\d+)?)\s*[bB]\b", name_str)
        if moe_experts:
            n_exp = float(moe_experts.group(1))
            exp_b = float(moe_experts.group(2))
            return round(min(n_exp * exp_b, 2.0 * exp_b), 2)

        # 3. Standard billions suffix (e.g., "3B", "7B", "14B", "27B", "32B", "1.5B")
        b_matches = re.findall(r"(?<![a-zA-Z0-9.])(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z0-9])", name_str)
        if b_matches:
            return float(b_matches[-1])

        # 4. Millions suffix (e.g., "360M", "500M")
        m_matches = re.findall(r"(?<![a-zA-Z0-9.])(\d{2,4})\s*[mM](?![a-zA-Z0-9])", name_str)
        if m_matches:
            return round(float(m_matches[-1]) / 1000.0, 3)

        # 5. Inspect loaded Hugging Face model parameter count if engine is initialized
        engine = getattr(prism_obj, "engine", None) if prism_obj is not None else None
        hf_model = getattr(engine, "model", None) or getattr(prism_obj, "model", None)
        if hf_model is not None and callable(getattr(hf_model, "num_parameters", None)):
            try:
                raw_params = float(hf_model.num_parameters())
                if raw_params > 0:
                    scale = 1.8 if load_in_4bit else 1.0
                    return round((raw_params / 1e9) * scale, 2)
            except Exception:
                pass

        return None

    def _detect_model_capability_tier(self) -> Tuple[str, Optional[float], Optional[int]]:
        """
        Classify the active extraction LLM into 'compact', 'mid', or 'large' so chunk density
        and output token budgets adapt automatically across 3B laptop models and 27B+ server models.
        """
        est_b = self._estimate_model_billions(self.model_name, self.prism, self.load_in_4bit)
        if getattr(self.prism, "engine", None) is not None:
            known_ctx: Optional[int] = self._apply_context_window_cap()
        elif isinstance(self.context_window_cap, int) and self.context_window_cap > 0:
            known_ctx = self.context_window_cap
        else:
            known_ctx = None

        if (est_b is not None and est_b <= 4.5) or (known_ctx is not None and known_ctx <= 4096):
            return "compact", est_b, known_ctx
        if (
            est_b is None
            or est_b <= 16.5
            or (self.load_in_4bit and est_b < 24.0)
            or (known_ctx is not None and known_ctx < 16384)
        ):
            return "mid", est_b, known_ctx
        return "large", est_b, known_ctx

    def _resolve_triple_extraction_profile(self) -> Dict[str, Any]:
        """
        Resolve Phase 3 target chunk words, overlap words, and max_new_tokens dynamically
        when configured as 'auto', while honoring explicit integer overrides.
        """
        tier, est_b, known_ctx = self._detect_model_capability_tier()
        tier_defaults = {
            "compact": {"chunk_words": 240, "overlap_words": 30, "max_new_tokens": 1280},
            "mid": {"chunk_words": 380, "overlap_words": 40, "max_new_tokens": 1792},
            "large": {"chunk_words": 600, "overlap_words": 50, "max_new_tokens": 2560},
        }[tier]

        if isinstance(self.triple_chunk_max_words, int) and self.triple_chunk_max_words > 0:
            chunk_words = self.triple_chunk_max_words
            explicit_chunk = True
        else:
            chunk_words = tier_defaults["chunk_words"]
            explicit_chunk = False

        if isinstance(self.triple_chunk_overlap_words, int) and self.triple_chunk_overlap_words >= 0:
            overlap_words = min(self.triple_chunk_overlap_words, max(0, chunk_words // 3))
        elif explicit_chunk:
            overlap_words = min(50, max(15, chunk_words // 10)) if chunk_words >= 120 else 0
        else:
            overlap_words = tier_defaults["overlap_words"]

        if isinstance(self.triple_max_new_tokens, int) and self.triple_max_new_tokens > 0:
            max_new_tokens = self.triple_max_new_tokens
        else:
            max_new_tokens = tier_defaults["max_new_tokens"]
            if known_ctx is not None and known_ctx <= 4096:
                max_new_tokens = min(max_new_tokens, max(512, int(known_ctx * 0.45)))

        return {
            "tier": tier,
            "estimated_billions": est_b,
            "known_context_window": known_ctx,
            "chunk_words": chunk_words,
            "overlap_words": overlap_words,
            "max_new_tokens": max_new_tokens,
        }

    @staticmethod
    def sanitize_filename(name: str) -> str:
        """Sanitize document identifiers to be filesystem-safe."""
        return re.sub(r'[\\/:*?"<>|]', "_", str(name))

    def _log_checkpoint(self, phase: str, message: str) -> None:
        """Print timestamped checkpoint progress to the console when print_checkpoints is enabled."""
        if self.print_checkpoints:
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
        if not self.save_llm_payloads:
            return
        target_path = self.payload_dir / filename
        if not target_path.exists():
            target_path.write_text(content, encoding="utf-8")

    @staticmethod
    def _is_standalone_header_line(line: str) -> bool:
        """Detect standalone section headers (Markdown, ALL-CAPS, or short colon headings) so they stay attached to their section."""
        stripped = line.strip()
        if not stripped or len(stripped) > 95:
            return False
        words = stripped.split()
        if len(words) > 11:
            return False
        if re.match(r"^(?:[-*•]|\d+[.)])\s+", stripped) or "|" in stripped:
            return False
        if stripped.startswith("#"):
            return True
        if stripped.endswith(":") and not any( stripped.endswith(p) for p in (".:", "!:", "?:") ):
            # Exclude key-value pairs with values already on the same line, only match pure headers ending with ':'
            return True
        alpha_chars = [ch for ch in stripped if ch.isalpha()]
        if len(alpha_chars) >= 4 and all(ch.isupper() for ch in alpha_chars) and not stripped.endswith((".", "!", "?")):
            return True
        return False

    @classmethod
    def chunk_text(
        cls,
        text: str,
        max_words: int,
        overlap_words: int = 0,
        preserve_structure: bool = True,
    ) -> List[Tuple[str, int, int]]:
        """
        Structure-preserving adaptive chunker with whole-document fast-path, section-header
        forward binding, hierarchy-first splitting (paragraphs -> lines/bullets/tables ->
        sentences -> clauses), small-tail absorption, and optional sentence/line overlap.
        Returns a list of tuples: (chunk_text, start_word_index, end_word_index).
        """
        if not text or not text.strip():
            return []

        max_words = max(1, int(max_words))
        overlap_words = max(0, min(int(overlap_words), max_words // 2))

        normalized = re.sub(r"\r\n?", "\n", text)
        normalized = "\n".join(ln.rstrip() for ln in normalized.split("\n")).strip()
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)

        if not preserve_structure:
            flat_words = normalized.split()
            if len(flat_words) <= max_words:
                return [(" ".join(flat_words), 0, len(flat_words))]
            chunks: List[Tuple[str, int, int]] = []
            start = 0
            while start < len(flat_words):
                end = min(len(flat_words), start + max_words)
                chunks.append((" ".join(flat_words[start:end]), start, end))
                if end >= len(flat_words):
                    break
                start = max(start + 1, end - overlap_words)
            return chunks

        total_words = len(normalized.split())
        if total_words <= max_words:
            return [(normalized, 0, total_words)]

        # Step 1: Parse non-empty lines while recording paragraph ("\n\n") vs single-line ("\n") separators
        raw_lines = normalized.split("\n")
        line_blocks: List[Dict[str, Any]] = []
        saw_blank = False
        for raw_ln in raw_lines:
            if not raw_ln.strip():
                saw_blank = True
                continue
            sep = "\n\n" if (saw_blank and line_blocks) else ("\n" if line_blocks else "")
            line_blocks.append({"text": raw_ln.strip(), "sep": sep})
            saw_blank = False

        # Step 2: Bind standalone section headers forward to the first line beneath them
        merged_blocks: List[Dict[str, Any]] = []
        idx = 0
        while idx < len(line_blocks):
            cur = line_blocks[idx]
            if (
                idx + 1 < len(line_blocks)
                and cls._is_standalone_header_line(cur["text"])
                and len(cur["text"].split()) + len(line_blocks[idx + 1]["text"].split()) <= max_words
            ):
                nxt = line_blocks[idx + 1]
                merged_blocks.append({
                    "text": f"{cur['text']}\n{nxt['text']}",
                    "sep": cur["sep"],
                })
                idx += 2
            else:
                merged_blocks.append(cur)
                idx += 1

        # Step 3: Split long prose blocks into sentence/clause units while keeping bullets/tables intact
        sentence_split_re = re.compile(r'(?<=[.!?])(?:["\')\]]+)?\s+(?=[A-Z0-9"\'(\[])')
        clause_split_re = re.compile(r"(?<=[;:])\s+|(?<=,)\s+")
        units: List[Dict[str, Any]] = []
        cursor_w = 0

        def _append_hard_wrapped(seg_text: str, first_sep: str) -> None:
            nonlocal cursor_w
            seg_words = seg_text.split()
            if not seg_words:
                return
            if len(seg_words) <= max_words:
                w_len = len(seg_words)
                units.append({
                    "text": seg_text,
                    "sep": first_sep if units else "",
                    "w_count": w_len,
                    "start_w": cursor_w,
                    "end_w": cursor_w + w_len,
                })
                cursor_w += w_len
                return

            # Try clause-level split before hard word slicing
            clauses = [c.strip() for c in clause_split_re.split(seg_text) if c.strip()]
            if len(clauses) > 1 and max(len(c.split()) for c in clauses) < len(seg_words):
                buf_clauses: List[str] = []
                buf_w = 0
                used_first = False
                for cl in clauses:
                    cl_w = len(cl.split())
                    if cl_w > max_words:
                        if buf_clauses:
                            joined = " ".join(buf_clauses)
                            units.append({
                                "text": joined,
                                "sep": (first_sep if not used_first else " ") if units else "",
                                "w_count": buf_w,
                                "start_w": cursor_w,
                                "end_w": cursor_w + buf_w,
                            })
                            cursor_w += buf_w
                            used_first = True
                            buf_clauses, buf_w = [], 0
                        cl_words = cl.split()
                        for s_i in range(0, len(cl_words), max_words):
                            piece = cl_words[s_i : s_i + max_words]
                            p_len = len(piece)
                            units.append({
                                "text": " ".join(piece),
                                "sep": (first_sep if not used_first else " ") if units else "",
                                "w_count": p_len,
                                "start_w": cursor_w,
                                "end_w": cursor_w + p_len,
                            })
                            cursor_w += p_len
                            used_first = True
                    elif buf_w + cl_w <= max_words:
                        buf_clauses.append(cl)
                        buf_w += cl_w
                    else:
                        joined = " ".join(buf_clauses)
                        units.append({
                            "text": joined,
                            "sep": (first_sep if not used_first else " ") if units else "",
                            "w_count": buf_w,
                            "start_w": cursor_w,
                            "end_w": cursor_w + buf_w,
                        })
                        cursor_w += buf_w
                        used_first = True
                        buf_clauses, buf_w = [cl], cl_w
                if buf_clauses:
                    joined = " ".join(buf_clauses)
                    units.append({
                        "text": joined,
                        "sep": (first_sep if not used_first else " ") if units else "",
                        "w_count": buf_w,
                        "start_w": cursor_w,
                        "end_w": cursor_w + buf_w,
                    })
                    cursor_w += buf_w
                return

            for s_i in range(0, len(seg_words), max_words):
                piece = seg_words[s_i : s_i + max_words]
                p_len = len(piece)
                units.append({
                    "text": " ".join(piece),
                    "sep": (first_sep if s_i == 0 else " ") if units else "",
                    "w_count": p_len,
                    "start_w": cursor_w,
                    "end_w": cursor_w + p_len,
                })
                cursor_w += p_len

        sentence_split_threshold = max(60, int(max_words * 0.55))
        for blk in merged_blocks:
            blk_text = blk["text"]
            blk_sep = blk["sep"]
            blk_words = len(blk_text.split())
            if blk_words <= sentence_split_threshold:
                units.append({
                    "text": blk_text,
                    "sep": blk_sep if units else "",
                    "w_count": blk_words,
                    "start_w": cursor_w,
                    "end_w": cursor_w + blk_words,
                })
                cursor_w += blk_words
                continue

            # Preserve explicit internal newline if a header was bound to a long paragraph
            if "\n" in blk_text:
                hdr_part, body_part = blk_text.split("\n", 1)
                sents = [s.strip() for s in sentence_split_re.split(body_part) if s.strip()]
                if sents:
                    sents[0] = f"{hdr_part}\n{sents[0]}"
                else:
                    sents = [hdr_part]
            else:
                sents = [s.strip() for s in sentence_split_re.split(blk_text) if s.strip()]

            for s_idx, sent in enumerate(sents):
                s_sep = blk_sep if s_idx == 0 else " "
                _append_hard_wrapped(sent, s_sep)

        if not units:
            return []

        # Step 4: Pack units into structure-aligned chunks with paragraph break preference
        grouped_units: List[List[Dict[str, Any]]] = []
        current_group: List[Dict[str, Any]] = []
        current_words = 0
        para_soft_floor = int(max_words * 0.78)

        for u in units:
            u_w = u["w_count"]
            if not current_group:
                current_group.append(u)
                current_words = u_w
                continue

            starts_new_para = u["sep"] == "\n\n"
            if current_words + u_w > max_words or (
                starts_new_para and current_words >= para_soft_floor and u_w >= max(25, max_words // 8)
            ):
                grouped_units.append(current_group)
                current_group = [u]
                current_words = u_w
            else:
                current_group.append(u)
                current_words += u_w

        if current_group:
            grouped_units.append(current_group)

        # Step 5: Absorb tiny trailing orphan chunk into previous chunk when proportional
        if len(grouped_units) >= 2:
            tail_words = sum(u["w_count"] for u in grouped_units[-1])
            prev_words = sum(u["w_count"] for u in grouped_units[-2])
            tail_floor = max(40, max_words // 4)
            if tail_words <= tail_floor and (prev_words + tail_words) <= int(max_words * 1.25):
                grouped_units[-2].extend(grouped_units[-1])
                grouped_units.pop()

        def _render_unit_slice(u_slice: List[Dict[str, Any]]) -> str:
            parts: List[str] = []
            for idx_u, item_u in enumerate(u_slice):
                if idx_u == 0:
                    parts.append(item_u["text"])
                else:
                    parts.append(f"{item_u['sep'] or ' '}{item_u['text']}")
            return "".join(parts).strip()

        # Step 6: Build final chunks with clean unit-aligned overlap
        final_chunks: List[Tuple[str, int, int]] = []
        for g_idx, grp in enumerate(grouped_units):
            if g_idx > 0 and overlap_words > 0:
                prev_grp = grouped_units[g_idx - 1]
                overlap_prefix: List[Dict[str, Any]] = []
                ov_count = 0
                for prev_u in reversed(prev_grp):
                    pw = prev_u["w_count"]
                    limit_w = overlap_words if overlap_prefix else max(overlap_words, int(overlap_words * 1.5))
                    if ov_count + pw <= limit_w:
                        overlap_prefix.append(prev_u)
                        ov_count += pw
                    else:
                        break
                overlap_prefix.reverse()
                combined = overlap_prefix + grp
            else:
                combined = grp

            rendered = _render_unit_slice(combined)
            start_w = combined[0]["start_w"]
            end_w = combined[-1]["end_w"]
            final_chunks.append((rendered, start_w, end_w))

        return final_chunks

    @classmethod
    def _build_sticky_context_anchor(cls, full_doc_text: str, chunk_text: str, start_word: int) -> str:
        """
        Build a lightweight 1-2 line context anchor for continuation chunks (chunk_idx > 0)
        carrying the document lead subject and most recent active section header.
        """
        if start_word <= 0 or not full_doc_text.strip():
            return ""

        norm_doc = re.sub(r"\r\n?", "\n", full_doc_text).strip()
        lines = [ln.strip() for ln in norm_doc.split("\n") if ln.strip()]
        if not lines:
            return ""

        # 1. Concise document lead (first substantive line/sentence, capped at 24 words)
        lead_candidates: List[str] = []
        for ln in lines[:4]:
            if cls._is_standalone_header_line(ln) and len(lines) > 1:
                continue
            first_sent = re.split(r"(?<=[.!?])\s+", ln, maxsplit=1)[0].strip()
            lead_words = first_sent.split()
            lead_candidates.append(" ".join(lead_words[:24]) + ("..." if len(lead_words) > 24 else ""))
            break
        doc_lead = lead_candidates[0] if lead_candidates else " ".join(lines[0].split()[:20])

        # 2. Most recent standalone section header preceding this chunk (unless chunk starts with a header)
        chunk_first_line = next((ln.strip() for ln in chunk_text.split("\n") if ln.strip()), "")
        active_header = ""
        if not cls._is_standalone_header_line(chunk_first_line):
            running_words = 0
            for ln in lines:
                ln_w = len(ln.split())
                if running_words >= start_word:
                    break
                if cls._is_standalone_header_line(ln):
                    active_header = ln.lstrip("#").strip().rstrip(":")
                running_words += ln_w

        anchor_lines = [f"Document Lead: {doc_lead}"]
        if active_header and active_header.lower() not in doc_lead.lower():
            anchor_lines.append(f"Active Section: {active_header}")
        return "\n".join(anchor_lines)

    @staticmethod
    def _format_triple_chunk_prompt(chunk_text: str, context_anchor: str = "") -> str:
        """Format the Phase 3 user prompt body, optionally prepending a non-extracted context anchor."""
        if not context_anchor:
            return TRIPLE_EXTRACTION_USER_PROMPT.format(text_content=chunk_text)
        return (
            "Extract all atomic relationship triples from <source_text> below "
            "(use <context_anchor> only to resolve pronouns or section scope; "
            "do NOT extract triples solely from <context_anchor>):\n\n"
            f"<context_anchor>\n{context_anchor}\n</context_anchor>\n\n"
            f"<source_text>\n{chunk_text}\n</source_text>"
        )

    @staticmethod
    def _deduplicate_doc_triples(triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate overlapping (subject, predicate, object) triples within a single document
        while dropping degenerate self-loops and retaining the richest verbatim source_quote
        and most specific theme_association.
        """
        if not triples:
            return []

        def _norm_part(val: Any) -> str:
            return re.sub(r"\s+", " ", str(val or "").lower()).strip(" \t\r\n.,:;-")

        deduped: List[Dict[str, Any]] = []
        key_to_idx: Dict[Tuple[str, str, str], int] = {}

        for tr in triples:
            s_norm = _norm_part(tr.get("subject"))
            p_norm = _norm_part(tr.get("predicate"))
            o_norm = _norm_part(tr.get("object"))
            if not s_norm or not p_norm or not o_norm:
                continue
            if s_norm == o_norm:
                continue

            key = (s_norm, p_norm, o_norm)
            if key not in key_to_idx:
                key_to_idx[key] = len(deduped)
                deduped.append(tr)
            else:
                existing_idx = key_to_idx[key]
                existing = deduped[existing_idx]
                ex_theme_spec = 0 if str(existing.get("theme_association") or "Other") == "Other" else 1
                nw_theme_spec = 0 if str(tr.get("theme_association") or "Other") == "Other" else 1
                ex_quote_len = len(str(existing.get("source_quote") or "").strip())
                nw_quote_len = len(str(tr.get("source_quote") or "").strip())
                if (nw_theme_spec, nw_quote_len) > (ex_theme_spec, ex_quote_len):
                    deduped[existing_idx] = tr

        return deduped

    def load_input_documents(
        self,
        input_path: Optional[Union[str, Path, pd.DataFrame, List[Dict[str, Any]]]] = None,
        id_column: Optional[str] = None,
        text_column: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Load input documents using pipeline settings (auto-detects Parquet, CSV, JSON/JSONL, DataFrame, or directory of .txt files).
        Returns a list of dicts: [{'id': str, 'text': str}].
        """
        target = input_path if input_path is not None else self.input_path
        id_col = id_column if id_column is not None else self.id_column
        txt_col = text_column if text_column is not None else self.text_column
        doc_limit = limit if limit is not None else self.document_limit
        docs: List[Dict[str, str]] = []

        if isinstance(target, pd.DataFrame):
            df = target
        elif isinstance(target, list):
            for i, item in enumerate(target):
                if isinstance(item, dict):
                    doc_id = str(item.get(id_col) or item.get("id") or f"doc_{i:04d}")
                    txt = str(item.get(txt_col) or item.get("text") or "")
                    if txt.strip():
                        docs.append({"id": doc_id, "text": txt})
                elif isinstance(item, str) and item.strip():
                    docs.append({"id": f"doc_{i:04d}", "text": item})
            return docs[:doc_limit] if doc_limit else docs
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
                return docs[:doc_limit] if doc_limit else docs

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

        col_map = {str(c).lower().strip(): c for c in df.columns}
        resolved_id = id_col if id_col in df.columns else col_map.get(id_col.lower())
        if not resolved_id:
            for cand in ("unique_val", "id", "doc_id", "record_id", "identifier", "patient_id"):
                if cand in col_map:
                    resolved_id = col_map[cand]
                    break

        resolved_text = txt_col if txt_col in df.columns else col_map.get(txt_col.lower())
        if not resolved_text:
            for cand in ("report", "text", "anonymized", "content", "clinical_report", "note", "body"):
                if cand in col_map:
                    resolved_text = col_map[cand]
                    break

        if not resolved_text:
            raise ValueError(f"text_column '{txt_col}' not found in columns: {list(df.columns)}")

        for idx, row in df.iterrows():
            doc_id = str(row[resolved_id]) if resolved_id and pd.notna(row[resolved_id]) else f"doc_{idx:04d}"
            text = str(row[resolved_text]) if pd.notna(row[resolved_text]) else ""
            if text.strip() and text.strip().lower() != "nan":
                docs.append({"id": doc_id, "text": text})

        return docs[:doc_limit] if doc_limit else docs

    def load_master_themes(self) -> Optional[MasterThemeSynthesisResult]:
        """Load master_themes.json (and cluster_candidates.json if available) from disk into context."""
        master_path = self.out_dir / "master_themes.json"
        if master_path.exists():
            try:
                with open(master_path, "r", encoding="utf-8") as f:
                    self.context.master_themes = MasterThemeSynthesisResult(**json.load(f))
            except Exception as e:
                print(f"Warning: Failed to load master themes from {master_path}: {e}")
        cand_path = self.out_dir / "cluster_candidates.json"
        if not self.context.cluster_candidates and cand_path.exists():
            try:
                raw_cands = json.loads(cand_path.read_text(encoding="utf-8"))
                if isinstance(raw_cands, list):
                    self.context.cluster_candidates = [
                        ClusterCandidateSynthesis.model_validate(c) for c in raw_cands if isinstance(c, dict)
                    ]
            except Exception:
                pass
        return self.context.master_themes

    # =========================================================================
    # PHASE 1: GLOBAL THEME DISCOVERY & AGGREGATION
    # =========================================================================

    def _apply_context_window_cap(self) -> int:
        native_ctx = getattr(self.prism, "context_window", 4096) or 4096
        if isinstance(self.context_window_cap, int) and self.context_window_cap > 0:
            effective_ctx = min(native_ctx, self.context_window_cap)
        else:
            effective_ctx = native_ctx
        self.prism.context_window = effective_ctx
        if self.prism.engine is not None:
            try:
                self.prism.engine.context_window = effective_ctx
            except AttributeError:
                pass
        return effective_ctx

    def _resolve_stage_max_words(
        self,
        requested_max_words: int,
        max_new_tokens: int,
        system_prompt: Optional[str] = None,
        template_text: Optional[str] = None,
    ) -> int:
        if getattr(self.prism, "engine", None) is not None or (
            isinstance(self.context_window_cap, int) and self.context_window_cap > 0
        ):
            effective_ctx = self._apply_context_window_cap()
        else:
            tier, _, _ = self._detect_model_capability_tier()
            native_ctx = getattr(self.prism, "context_window", 4096) or 4096
            effective_ctx = max(native_ctx, 8192 if tier == "compact" else 32768)
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
        return min(int(requested_max_words), safe_max_words)

    def discover_themes_across_documents(
        self,
        documents: List[Dict[str, str]],
        dry_run: Optional[bool] = None,
    ) -> List[Theme]:
        """Phase 1: Discover abstract overarching themes across all documents concurrently."""
        is_dry = self.dry_run if dry_run is None else dry_run
        print(f"=== Phase 1: Global Theme Discovery (Model: {self.model_name}) ===")
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

        if not is_dry and self.prism.engine is None:
            self._log_checkpoint("Phase 1: Themes", f"Loading model '{self.model_name}' onto device...")
            self.prism.init_engine(
                batch_size=self.batch_size,
                max_new_tokens=self.theme_max_new_tokens,
                temperature=self.temperature,
            )
            eff_ctx = self._apply_context_window_cap()
            self._log_checkpoint(
                "Phase 1: Themes",
                f"Model ready on device '{self.prism.device}' (effective context window: {eff_ctx} tokens)",
            )

        effective_theme_words = self._resolve_stage_max_words(
            self.theme_chunk_max_words,
            self.theme_max_new_tokens,
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

            if self.resume_mode == "skip" and theme_file_path.exists():
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
        processed_theme_chunk_uids: Set[str] = set()
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
                if chunk_uid in processed_theme_chunk_uids:
                    return
                processed_theme_chunk_uids.add(chunk_uid)
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
            batch_size=self.batch_size,
            max_new_tokens=self.theme_max_new_tokens,
            temperature=self.temperature,
            constrain_json=self.constrain_json,
            dry_run=is_dry,
            on_batch_start=_on_theme_batch_start,
            on_chunk_processed=_on_theme_chunk,
        )

        if is_dry:
            return self.context.all_discovered_themes

        for res in results:
            chunk_uid = str(res.get("record_id") or res.get("id") or "")
            if not chunk_uid or chunk_uid in processed_theme_chunk_uids:
                continue
            row_meta = chunk_meta_by_uid.get(chunk_uid)
            if not row_meta:
                continue
            doc_id = row_meta["doc_id"]
            if doc_id not in saved_doc_ids:
                processed_theme_chunk_uids.add(chunk_uid)
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
                mu_c, sigma_c = self.manifold_edge_threshold, 0.0

            tau_edge = float(active_partition.get("effective_edge_threshold", self.manifold_edge_threshold))
            tau_high = max(mu_c - 1e-4, tau_edge)
            tau_medium = max(mu_c - max(sigma_c, 0.05), tau_edge)
            tau_low = max(mu_c - 2.0 * max(sigma_c, 0.05), 0.5 * tau_edge)

            for t_item in cluster_themes:
                sim = float(t_item.get("similarity_to_centroid", 0.0))
                loo_sim = float(t_item.get("loo_similarity_to_centroid", sim))
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

    def synthesize_master_themes(self, force: bool = False, dry_run: Optional[bool] = None) -> Optional[MasterThemeSynthesisResult]:
        """
        Phase 2: Cluster global aggregated themes with ThemeClusteringEngine and
        synthesize the authoritative Master Ontology via Map-Reduce.
        """
        is_dry = self.dry_run if dry_run is None else dry_run
        print(f"\n=== Phase 2: Master Theme Synthesis (Model: {self.model_name}) ===")
        self.aggregate_themes()
        if not self.context.all_discovered_themes:
            print("Warning: No themes found in themes directory or all_themes.json. Skipping master theme synthesis.")
            return None

        master_path = self.out_dir / "master_themes.json"
        mapping_path = self.out_dir / "theme_mapping_clusters.json"

        if not force and self.resume_mode == "skip" and master_path.exists() and mapping_path.exists():
            print("-> Skipping master theme synthesis (already exists on disk). Loading existing master themes...")
            return self.load_master_themes()

        if self.manage_vram:
            clear_device_memory()

        self._log_checkpoint(
            "Phase 2: Clustering",
            f"Embedding and clustering {len(self.context.all_discovered_themes)} discovered themes (dynamic Leiden)...",
        )
        cluster_engine = ThemeClusteringEngine(
            themes_path=self.out_dir / "all_themes.json",
            output_dir=self.out_dir,
            embedding_model_name=self.theme_embedding_model,
            map_max_themes=self.map_max_themes_per_cluster,
            manifold_edge_threshold=self.manifold_edge_threshold,
            micro_cluster_ratio=self.micro_cluster_ratio,
            device=self.device,
            batch_size=self.batch_size,
        )
        cluster_results = cluster_engine.cluster_themes()
        if self.manage_vram:
            clear_device_memory()
        partitions = cluster_results["partitions"]
        exemplar_groups = cluster_results["active_exemplars"]
        active_partition = cluster_engine.get_active_partition(partitions)
        self._log_checkpoint(
            "Phase 2: Clustering",
            f"Formed {len(exemplar_groups)} empirical theme cluster(s)",
        )

        if not is_dry and self.prism.engine is None:
            self._log_checkpoint("Phase 2: Master Synthesis", f"Loading model '{self.model_name}' onto device...")
            self.prism.init_engine(
                batch_size=self.batch_size,
                max_new_tokens=self.master_synthesis_max_new_tokens,
                temperature=self.temperature,
            )
            self._apply_context_window_cap()

        map_template = {
            "version": "1.0",
            "task": "cluster_map",
            "system_prompt": CLUSTER_MAP_SYSTEM_PROMPT,
            "prompt": "{{ text }}\n\n{schema_block}",
        }
        map_max_words = self._resolve_stage_max_words(
            requested_max_words=self.map_max_themes_per_cluster * 40,
            max_new_tokens=self.master_synthesis_max_new_tokens,
            system_prompt=CLUSTER_MAP_SYSTEM_PROMPT,
            template_text=CLUSTER_MAP_USER_PROMPT,
        )
        map_payloads = cluster_engine.get_map_payload(
            max_themes_per_cluster=self.map_max_themes_per_cluster,
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
            batch_size=self.batch_size,
            max_new_tokens=self.master_synthesis_max_new_tokens,
            temperature=self.temperature,
            constrain_json=self.constrain_json,
            dry_run=is_dry,
            on_batch_start=lambda b_idx, b_tot, b_items: self._log_checkpoint(
                "Phase 2A: Cluster Map",
                f"Running concurrent Map batch {b_idx}/{b_tot} ({len(b_items)} cluster(s))",
            ),
        )
        if is_dry:
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
        if self.manage_vram:
            clear_device_memory()

        reduce_sys_prompt = CLUSTER_REDUCE_SYSTEM_PROMPT
        reduce_max_words = self._resolve_stage_max_words(
            requested_max_words=12000,
            max_new_tokens=self.master_synthesis_max_new_tokens,
            system_prompt=reduce_sys_prompt,
            template_text=CLUSTER_REDUCE_USER_PROMPT,
        )

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
            max_new_tokens=self.master_synthesis_max_new_tokens,
            temperature=self.temperature,
            constrain_json=self.constrain_json,
            dry_run=is_dry,
        )
        if is_dry:
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
        if self.manage_vram:
            clear_device_memory()
        return master_res

    # =========================================================================
    # PHASE 3: GLOBAL SVO TRIPLE EXTRACTION & AGGREGATION
    # =========================================================================

    def extract_triples_across_documents(
        self,
        documents: List[Dict[str, str]],
        dry_run: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Phase 3: Extract atomic (Subject, Predicate, Object) triples across documents."""
        is_dry = self.dry_run if dry_run is None else dry_run
        print(f"\n=== Phase 3: Global Triple Extraction (Model: {self.model_name}) ===")
        total_docs = len(documents)
        if total_docs == 0:
            return self.aggregate_triplets()

        if not self.context.master_themes:
            self.load_master_themes()

        sys_context_lines: List[str] = []
        if self.context.master_themes:
            m_domain = str(self.context.master_themes.master_domain or "").strip()
            if m_domain and m_domain != _FALLBACK_DOMAIN:
                sys_context_lines.append(f'Corpus Master Domain: "{m_domain}"')

        if self.context.master_themes and self.context.master_themes.master_themes:
            m_themes = self.context.master_themes.master_themes
            theme_list = ", ".join([f'"{t}"' for t in m_themes])
            sys_context_lines.append(f'Allowed theme_association values: [{theme_list}, "Other"]')
            cands = self.context.cluster_candidates
            if cands and len(cands) == len(m_themes) and len(m_themes) <= 15:
                hint_lines: List[str] = []
                for t_title, cand in zip(m_themes, cands):
                    scope_str = re.split(r"(?<=[.!?])\s+", str(cand.scope_definition or "").strip(), maxsplit=1)[0]
                    scope_words = scope_str.split()
                    if scope_words:
                        short_scope = " ".join(scope_words[:20]) + ("..." if len(scope_words) > 20 else "")
                        hint_lines.append(f'- "{t_title}": {short_scope}')
                if hint_lines:
                    sys_context_lines.append("Master Theme Scope Guide:\n" + "\n".join(hint_lines))
        else:
            sys_context_lines.append('Allowed theme_association values: ["Other"]')

        sys_context = "\n" + "\n".join(sys_context_lines)
        full_sys_prompt = f"{TRIPLE_EXTRACTION_SYSTEM_PROMPT}\n{sys_context}"
        triple_template = {
            "version": "1.0",
            "task": "triple_extraction",
            "system_prompt": full_sys_prompt,
            "prompt": "{{ text }}\n\n{schema_block}",
        }

        allowed_themes = (
            set(self.context.master_themes.master_themes)
            if (self.context.master_themes and self.context.master_themes.master_themes)
            else set()
        )
        allowed_theme_ci = {t.lower(): t for t in allowed_themes}

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
                raw_assoc = str(t_dict.get("theme_association") or "Other").strip()
                if allowed_themes:
                    if raw_assoc in allowed_themes:
                        t_dict["theme_association"] = raw_assoc
                    elif raw_assoc.lower() in allowed_theme_ci:
                        t_dict["theme_association"] = allowed_theme_ci[raw_assoc.lower()]
                    else:
                        t_dict["theme_association"] = "Other"
                else:
                    t_dict["theme_association"] = "Other"
                t_dict["source_document"] = doc_id
                dicts.append(t_dict)
            return dicts

        pre_profile = self._resolve_triple_extraction_profile()
        if not is_dry and self.prism.engine is None:
            self._log_checkpoint("Phase 3: Triples", f"Loading model '{self.model_name}' onto device...")
            self.prism.init_engine(
                batch_size=self.batch_size,
                max_new_tokens=pre_profile["max_new_tokens"],
                temperature=self.temperature,
            )
            self._apply_context_window_cap()

        triple_profile = self._resolve_triple_extraction_profile()
        effective_max_tokens = int(triple_profile["max_new_tokens"])
        effective_triple_words = self._resolve_stage_max_words(
            triple_profile["chunk_words"],
            effective_max_tokens,
            system_prompt=full_sys_prompt,
            template_text=TRIPLE_EXTRACTION_USER_PROMPT,
        )
        effective_overlap_words = min(
            int(triple_profile["overlap_words"]),
            max(0, effective_triple_words // 3),
        )
        self._log_checkpoint(
            "Phase 3: Triples",
            f"Adaptive profile [{triple_profile['tier']} tier]: target={effective_triple_words} words, "
            f"overlap={effective_overlap_words} words, max_new_tokens={effective_max_tokens}",
        )

        chunk_rows: List[Dict[str, Any]] = []
        chunk_meta_by_uid: Dict[str, Dict[str, Any]] = {}
        doc_expected_chunks: Dict[str, int] = {}
        doc_text_by_id: Dict[str, str] = {}

        for doc_idx, doc in enumerate(documents, start=1):
            doc_id = doc["id"]
            self.context.processed_documents.add(doc_id)
            safe_name = self.sanitize_filename(doc_id)
            triple_file_path = self.triples_dir / f"{safe_name}_triplets.json"

            if self.resume_mode == "skip" and triple_file_path.exists():
                print(f"-> [{doc_idx}/{total_docs}] Skipping triple extraction for: {doc_id} (already exists on disk)")
                continue

            doc_text_by_id[doc_id] = doc["text"]
            triple_chunks = self.chunk_text(
                doc["text"],
                effective_triple_words,
                overlap_words=effective_overlap_words,
                preserve_structure=True,
            )
            doc_expected_chunks[doc_id] = len(triple_chunks)

            for chunk_idx, (chunk_str, start_w, end_w) in enumerate(triple_chunks):
                context_anchor = (
                    self._build_sticky_context_anchor(doc["text"], chunk_str, start_w)
                    if chunk_idx > 0
                    else ""
                )
                user_prompt = self._format_triple_chunk_prompt(chunk_str, context_anchor)
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
                    "raw_chunk_text": chunk_str,
                    "text": user_prompt,
                }
                chunk_rows.append(row_meta)
                chunk_meta_by_uid[chunk_uid] = row_meta

        if not chunk_rows:
            return self.aggregate_triplets()

        total_pending_docs = len(doc_expected_chunks)
        save_lock = threading.Lock()
        triples_by_doc: Dict[str, List[Dict[str, Any]]] = {doc_id: [] for doc_id in doc_expected_chunks}
        completed_chunks_by_doc: Dict[str, int] = {doc_id: 0 for doc_id in doc_expected_chunks}
        processed_triple_chunk_uids: Set[str] = set()
        docs_pending_retry: Set[str] = set()
        failed_chunks_for_retry: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        saved_doc_ids: Set[str] = set()

        def _log_triple_failure(phase_label: str, row_meta: Dict[str, Any], chunk_res: Dict[str, Any]) -> None:
            err_rec = {
                "phase": phase_label,
                "source_document": row_meta["doc_id"],
                "start_word": row_meta["start_word"],
                "end_word": row_meta["end_word"],
                "error_message": chunk_res.get("parse_error", "Validation error"),
                "malformed_output": chunk_res.get("raw_output", ""),
                "chunk_text": row_meta.get("raw_chunk_text") or row_meta["text"],
            }
            with open(self.log_dir / "stage_01_triple_extraction_errors.json", "a", encoding="utf-8") as ef:
                ef.write(json.dumps(err_rec) + "\n")

        def _flush_doc_triples(doc_id: str) -> None:
            if doc_id in saved_doc_ids:
                return
            saved_doc_ids.add(doc_id)
            raw_doc_triples = triples_by_doc.get(doc_id, [])
            doc_triples = self._deduplicate_doc_triples(raw_doc_triples)
            triples_by_doc[doc_id] = doc_triples
            safe_name = self.sanitize_filename(doc_id)
            out_file = self.triples_dir / f"{safe_name}_triplets.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(doc_triples, f, indent=2)
            self._log_checkpoint(
                "Phase 3: Triples",
                f"Saved '{out_file.name}' ({len(doc_triples)} deduplicated triples) — "
                f"Document {len(saved_doc_ids)}/{total_pending_docs} complete",
            )

        def _handle_chunk_outcome(row_meta: Dict[str, Any], chunk_res: Dict[str, Any]) -> None:
            doc_id = row_meta["doc_id"]
            items = chunk_res.get("parsed_items") or chunk_res.get("extracted_items") or []
            new_triples = _extract_triple_dicts(items, doc_id)
            salvaged_used = False

            if not new_triples:
                raw_out = str(chunk_res.get("raw_output") or "")
                salvaged_objs = salvage_partial_triples(raw_out)
                if salvaged_objs:
                    new_triples = _extract_triple_dicts(salvaged_objs, doc_id)
                    salvaged_used = bool(new_triples)

            if new_triples:
                triples_by_doc[doc_id].extend(new_triples)
                if salvaged_used:
                    self._log_checkpoint(
                        "Phase 3: Triples",
                        f"Doc '{doc_id}' — salvaged {len(new_triples)} valid triple(s) from partial/truncated JSON",
                    )
            elif not chunk_res.get("parse_success", True):
                raw_chunk_words = len(str(row_meta.get("raw_chunk_text") or "").split())
                if raw_chunk_words > 120:
                    docs_pending_retry.add(doc_id)
                    failed_chunks_for_retry.append((row_meta, chunk_res))
                    self._log_checkpoint(
                        "Phase 3: Triples",
                        f"Doc '{doc_id}' — queued malformed {raw_chunk_words}-word chunk for automatic half-chunk split retry",
                    )
                else:
                    _log_triple_failure("initial_triple_extraction_failed", row_meta, chunk_res)

            completed_chunks_by_doc[doc_id] += 1
            self._log_checkpoint(
                "Phase 3: Triples",
                f"Doc '{doc_id}' — chunk {completed_chunks_by_doc[doc_id]}/{doc_expected_chunks[doc_id]} "
                f"processed (+{len(new_triples)} triples)",
            )
            if (
                completed_chunks_by_doc[doc_id] >= doc_expected_chunks[doc_id]
                and doc_id not in docs_pending_retry
            ):
                _flush_doc_triples(doc_id)

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
            with save_lock:
                if chunk_uid in processed_triple_chunk_uids:
                    return
                processed_triple_chunk_uids.add(chunk_uid)
                _handle_chunk_outcome(row_meta, chunk_res)

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
            batch_size=self.batch_size,
            max_new_tokens=effective_max_tokens,
            temperature=self.temperature,
            constrain_json=self.constrain_json,
            dry_run=is_dry,
            on_batch_start=_on_triple_batch_start,
            on_chunk_processed=_on_triple_chunk,
        )

        if is_dry:
            return self.context.raw_triples

        for res in results:
            chunk_uid = str(res.get("record_id") or res.get("id") or "")
            if not chunk_uid or chunk_uid in processed_triple_chunk_uids:
                continue
            row_meta = chunk_meta_by_uid.get(chunk_uid)
            if not row_meta:
                continue
            with save_lock:
                if chunk_uid not in processed_triple_chunk_uids and row_meta["doc_id"] not in saved_doc_ids:
                    processed_triple_chunk_uids.add(chunk_uid)
                    _handle_chunk_outcome(row_meta, res)

        # Automatic half-chunk split-retry safety net for smaller models on dense/malformed chunks
        if failed_chunks_for_retry:
            retry_rows: List[Dict[str, Any]] = []
            retry_meta_by_uid: Dict[str, Dict[str, Any]] = {}
            parent_recovered_counts: Dict[str, int] = {}

            for parent_meta, _ in failed_chunks_for_retry:
                parent_uid = parent_meta["chunk_uid"]
                parent_recovered_counts[parent_uid] = 0
                doc_id = parent_meta["doc_id"]
                raw_txt = str(parent_meta.get("raw_chunk_text") or "")
                raw_w = len(raw_txt.split())
                sub_target = max(75, (raw_w + 1) // 2)
                sub_overlap = min(15, max(0, sub_target // 6))
                sub_chunks = self.chunk_text(
                    raw_txt,
                    sub_target,
                    overlap_words=sub_overlap,
                    preserve_structure=True,
                )
                if len(sub_chunks) <= 1:
                    words_list = raw_txt.split()
                    mid_pt = max(1, len(words_list) // 2)
                    sub_chunks = [
                        (" ".join(words_list[:mid_pt]), 0, mid_pt),
                        (" ".join(words_list[mid_pt:]), mid_pt, len(words_list)),
                    ]

                full_doc_txt = doc_text_by_id.get(doc_id, raw_txt)
                base_start = int(parent_meta["start_word"])
                for sub_idx, (sub_str, sub_s, sub_e) in enumerate(sub_chunks):
                    abs_start = base_start + sub_s
                    abs_end = base_start + sub_e
                    anchor = self._build_sticky_context_anchor(full_doc_txt, sub_str, max(1, abs_start))
                    sub_prompt = self._format_triple_chunk_prompt(sub_str, anchor)
                    sub_uid = f"{parent_uid}__retry_{sub_idx}"
                    sub_meta = {
                        "chunk_uid": sub_uid,
                        "parent_uid": parent_uid,
                        "doc_id": doc_id,
                        "start_word": abs_start,
                        "end_word": abs_end,
                        "raw_chunk_text": sub_str,
                        "text": sub_prompt,
                    }
                    retry_rows.append(sub_meta)
                    retry_meta_by_uid[sub_uid] = sub_meta

            if retry_rows:
                self._log_checkpoint(
                    "Phase 3: Retry",
                    f"Retrying {len(failed_chunks_for_retry)} failed chunk(s) as {len(retry_rows)} smaller sub-chunk(s)...",
                )
                retry_results = self.prism.run(
                    task="extract",
                    input_data=retry_rows,
                    text_column="text",
                    id_column="chunk_uid",
                    schema_cls=TripleExtractionResult,
                    prompt_template=triple_template,
                    system_prompt=full_sys_prompt,
                    chunk=False,
                    as_array=False,
                    batch_size=self.batch_size,
                    max_new_tokens=effective_max_tokens,
                    temperature=self.temperature,
                    constrain_json=self.constrain_json,
                    dry_run=False,
                )
                for r_res in retry_results:
                    r_uid = str(r_res.get("record_id") or r_res.get("id") or "")
                    r_meta = retry_meta_by_uid.get(r_uid)
                    if not r_meta:
                        continue
                    doc_id = r_meta["doc_id"]
                    r_items = r_res.get("parsed_items") or r_res.get("extracted_items") or []
                    rec_triples = _extract_triple_dicts(r_items, doc_id)
                    if not rec_triples:
                        salvaged_retry = salvage_partial_triples(str(r_res.get("raw_output") or ""))
                        if salvaged_retry:
                            rec_triples = _extract_triple_dicts(salvaged_retry, doc_id)
                    if rec_triples:
                        triples_by_doc[doc_id].extend(rec_triples)
                        parent_recovered_counts[r_meta["parent_uid"]] += len(rec_triples)

            for parent_meta, orig_res in failed_chunks_for_retry:
                p_uid = parent_meta["chunk_uid"]
                rec_n = parent_recovered_counts.get(p_uid, 0)
                if rec_n > 0:
                    self._log_checkpoint(
                        "Phase 3: Retry",
                        f"Doc '{parent_meta['doc_id']}' — recovered {rec_n} triple(s) via half-chunk split retry",
                    )
                else:
                    _log_triple_failure("split_retry_triple_extraction_failed", parent_meta, orig_res)

            docs_pending_retry.clear()

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
        dry_run: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Execute Stage 1 Part 1 (Themes & Master Synthesis) and Stage 1 Part 2 (SVO Triples)."""
        is_dry = self.dry_run if dry_run is None else dry_run
        print("=== SemanticPrism Stage 1: Full Extraction Pipeline ===")
        stage_start = time.time()

        if self.resume_mode != "skip":
            self.clear_stage_logs()

        documents = self.load_input_documents(input_path=input_data)
        print(f"Found {len(documents)} document(s) to process.\n")

        self.discover_themes_across_documents(documents, dry_run=is_dry)
        self.synthesize_master_themes(dry_run=is_dry)
        if self.manage_vram:
            clear_device_memory()

        self.extract_triples_across_documents(documents, dry_run=is_dry)
        if self.manage_vram:
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

    def run(
        self,
        input_data: Optional[Union[str, Path, pd.DataFrame, List[Dict[str, Any]]]] = None,
        dry_run: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Convenience alias for run_stage_1()."""
        return self.run_stage_1(input_data=input_data, dry_run=dry_run)
