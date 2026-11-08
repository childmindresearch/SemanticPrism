"""
SemanticPrism Stage 1: Full Extraction Pipeline Orchestrator (Powered by PrismRunner)
"""

from __future__ import annotations

from collections import defaultdict
import json
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
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        theme_embedding_model: str = "all-MiniLM-L6-v2",
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
        theme_chunk_max_words: int = 2000,
        triple_chunk_max_words: int = 400,
        theme_max_new_tokens: int = 512,
        master_synthesis_max_new_tokens: int = 512,
        triple_max_new_tokens: int = 768,
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
        native_ctx = getattr(self.prism, "context_window", 4096) or 4096
        if isinstance(self.context_window_cap, int) and self.context_window_cap > 0:
            effective_ctx = min(native_ctx, self.context_window_cap)
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

        if not is_dry and self.prism.engine is None:
            self._log_checkpoint("Phase 3: Triples", f"Loading model '{self.model_name}' onto device...")
            self.prism.init_engine(
                batch_size=self.batch_size,
                max_new_tokens=self.triple_max_new_tokens,
                temperature=self.temperature,
            )
            self._apply_context_window_cap()

        effective_triple_words = self._resolve_stage_max_words(
            self.triple_chunk_max_words,
            self.triple_max_new_tokens,
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

            if self.resume_mode == "skip" and triple_file_path.exists():
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
            batch_size=self.batch_size,
            max_new_tokens=self.triple_max_new_tokens,
            temperature=self.temperature,
            constrain_json=self.constrain_json,
            dry_run=is_dry,
            on_batch_start=_on_triple_batch_start,
            on_chunk_processed=_on_triple_chunk,
        )

        if is_dry:
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
