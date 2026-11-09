#!/usr/bin/env python3
"""
SemanticPrism Master Pipeline Execution Script
This script sequentially executes Stages 1 through 5 of the SemanticPrism pipeline.
It utilizes subprocesses to ensure memory and VRAM are cleanly garbage collected
between each intensive LLM and mathematical operation, and writes a final end-to-end
pipeline summary report to `outputs/pipeline_summary.txt` and `outputs/pipeline_summary.json`.
"""

from __future__ import annotations

import argparse
import ast
import csv
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

STAGES: List[Tuple[int, str, str, bool]] = [
    (1, "run_stage_1_extraction.py", "Stage 1: Triplet Extraction & Theme Discovery", True),
    (2, "run_stage_2_refinement.py", "Stage 2: Vector Clustering & Taxonomic Lifting", True),
    (3, "run_stage_3_topology.py", "Stage 3: Topology Graphing & Structural Clustering", False),
    (4, "run_stage_4_synthesis.py", "Stage 4: Pydantic Schema Synthesis", True),
    (5, "run_stage_5_mapping.py", "Stage 5: Schema-Driven Extraction & Aggregation", True),
]


def _load_json_safe(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_ast_classes(py_path: Path) -> Tuple[List[str], List[str]]:
    """Returns (basemodel_names, enum_names) parsed from a Python schema file."""
    basemodels: List[str] = []
    enums: List[str] = []
    if not py_path.exists():
        return basemodels, enums
    try:
        source = py_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                base_names = []
                for b in node.bases:
                    if isinstance(b, ast.Name):
                        base_names.append(b.id)
                    elif isinstance(b, ast.Attribute):
                        base_names.append(b.attr)
                if "BaseModel" in base_names:
                    basemodels.append(node.name)
                elif any("Enum" in bn for bn in base_names):
                    enums.append(node.name)
    except Exception:
        pass
    return basemodels, enums


def generate_pipeline_summary(
    outputs_dir: Path = Path("outputs"),
    total_duration: Optional[float] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Collects key metrics and artifacts across Stages 1–4 and writes:
      - outputs/pipeline_summary.txt
      - outputs/pipeline_summary.json
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STAGE 1 METRICS
    # =========================================================================
    s1_dir = outputs_dir / "01_extraction"
    all_themes = _load_json_safe(s1_dir / "all_themes.json") or []
    master_themes_data = _load_json_safe(s1_dir / "master_themes.json") or {}
    original_triplets = _load_json_safe(s1_dir / "original_triplets.json") or []

    # Count ingested documents across triplet_counts.csv, themes/, triples/, and JSON records
    doc_ids = set()
    triplet_counts_csv = s1_dir / "triplet_counts.csv"
    if triplet_counts_csv.exists():
        try:
            with open(triplet_counts_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("document_name"):
                        doc_ids.add(row["document_name"])
        except Exception:
            pass
    for t in original_triplets:
        if isinstance(t, dict) and t.get("source_document"):
            doc_ids.add(str(t["source_document"]))
    if (s1_dir / "themes").exists():
        for tf in (s1_dir / "themes").glob("*_themes.json"):
            doc_ids.add(tf.stem[: -len("_themes")])

    docs_ingested_count = len(doc_ids)
    themes_identified_count = len(all_themes) if isinstance(all_themes, list) else 0
    master_themes_list: List[str] = (
        master_themes_data.get("master_themes", [])
        if isinstance(master_themes_data, dict)
        else []
    )
    master_domain: str = (
        str(master_themes_data.get("master_domain", "")).strip()
        if isinstance(master_themes_data, dict)
        else ""
    )
    triples_extracted_count = len(original_triplets) if isinstance(original_triplets, list) else 0

    # =========================================================================
    # STAGE 2 METRICS
    # =========================================================================
    s2_dir = outputs_dir / "02_refinement"
    subj_tax_map = _load_json_safe(s2_dir / "subject_taxonomic_map.json") or {}
    pred_tax_map = _load_json_safe(s2_dir / "predicate_taxonomic_map.json") or {}
    obj_tax_map = _load_json_safe(s2_dir / "object_taxonomic_map.json") or {}
    refined_triplets = _load_json_safe(s2_dir / "refined_triplets.json") or []

    subjects_lifted_count = len(subj_tax_map) if isinstance(subj_tax_map, dict) else 0
    subjects_unique_hypernyms = len(set(subj_tax_map.values())) if isinstance(subj_tax_map, dict) else 0
    predicates_lifted_count = len(pred_tax_map) if isinstance(pred_tax_map, dict) else 0
    predicates_unique_hypernyms = len(set(pred_tax_map.values())) if isinstance(pred_tax_map, dict) else 0
    objects_lifted_count = len(obj_tax_map) if isinstance(obj_tax_map, dict) else 0
    objects_unique_hypernyms = len(set(obj_tax_map.values())) if isinstance(obj_tax_map, dict) else 0

    # =========================================================================
    # STAGE 3 METRICS
    # =========================================================================
    s3_dir = outputs_dir / "03_topology"
    raw_diag = _load_json_safe(s3_dir / "01_raw_graph_evaluation" / "raw_graph_diagnostics.json") or {}
    hubs_reg = _load_json_safe(s3_dir / "01_raw_graph_evaluation" / "global_hubs_registry.json") or {}
    core_diag = _load_json_safe(s3_dir / "02_core_separation" / "core_graph_diagnostics.json") or {}
    comm_diag = _load_json_safe(s3_dir / "03_results" / "community_path_diagnostics.json") or {}
    baseline_parts = _load_json_safe(s3_dir / "03_results" / "topology_partitions_baseline.json") or {}
    final_parts = _load_json_safe(s3_dir / "03_results" / "topology_partitions.json") or {}
    agglom_report = _load_json_safe(s3_dir / "03_results" / "community_agglomeration_report.json") or {}

    raw_scale = raw_diag.get("graph_scale", {})
    raw_nodes = raw_scale.get("num_nodes", 0)
    raw_dir_edges = raw_scale.get("num_directed_edges", 0)
    raw_undir_edges = raw_scale.get("num_undirected_edges", 0)
    raw_density = raw_scale.get("density", 0.0)
    raw_total_graph_size = raw_nodes + raw_dir_edges
    global_hubs_removed = hubs_reg.get("total_global_hubs_removed", len(final_parts.get("global_hubs", [])))

    core_scale = core_diag.get("graph_scale", {})
    final_sub_nodes = core_scale.get("num_nodes", comm_diag.get("total_partitioned_nodes", 0))
    final_sub_edges = core_scale.get("num_edges", 0)
    dense_cluster_nodes = core_scale.get("cluster_subgraph_nodes", comm_diag.get("dense_core_nodes", 0))
    dense_cluster_edges = core_scale.get("cluster_subgraph_edges", 0)

    final_clusters_count = comm_diag.get(
        "total_communities",
        len(final_parts.get("communities", [])) if isinstance(final_parts, dict) else 0,
    )
    if isinstance(baseline_parts, dict) and "communities" in baseline_parts:
        initial_clusters_count = len(baseline_parts["communities"])
    else:
        total_merges = agglom_report.get("total_merges", 0) if isinstance(agglom_report, dict) else 0
        initial_clusters_count = final_clusters_count + total_merges

    # =========================================================================
    # STAGE 4 METRICS
    # =========================================================================
    s4_comm_dir = outputs_dir / "04_synthesis" / "schemas" / "community"
    schema_files: List[Path] = []
    if s4_comm_dir.exists():
        schema_files = sorted([
            p for p in s4_comm_dir.glob("*.py")
            if p.name not in ("__init__.py", "master_ontology.py", "comprehensive_ontology.py")
        ])

    schema_names = [p.name for p in schema_files]

    # Collect BaseModel and Enum classes across individual schema modules and consolidated ontology
    per_schema_basemodels: List[str] = []
    per_schema_enums: List[str] = []
    seen_bm = set()
    seen_en = set()
    for pf in schema_files:
        bms, ens = _extract_ast_classes(pf)
        for bm in bms:
            if bm not in seen_bm:
                seen_bm.add(bm)
                per_schema_basemodels.append(bm)
        for en in ens:
            if en not in seen_en:
                seen_en.add(en)
                per_schema_enums.append(en)

    comp_ontology_path = s4_comm_dir / "comprehensive_ontology.py"
    if not comp_ontology_path.exists():
        comp_ontology_path = s4_comm_dir / "master_ontology.py"
    consolidated_basemodels, consolidated_enums = _extract_ast_classes(comp_ontology_path)
    final_basemodels = consolidated_basemodels if consolidated_basemodels else per_schema_basemodels
    final_enums = consolidated_enums if consolidated_enums else per_schema_enums

    # =========================================================================
    # STAGE 5 METRICS
    # =========================================================================
    s5_dir = outputs_dir / "05_extraction"
    s5_extracts_dir = s5_dir / "extracts"
    s5_tables_dir = s5_dir / "aggregated_tables"

    s5_docs_processed = 0
    s5_total_extractions = 0
    s5_populated_extractions = 0
    if s5_extracts_dir.exists():
        doc_dirs = [d for d in s5_extracts_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        s5_docs_processed = len(doc_dirs)
        extract_jsons = [
            f for f in s5_extracts_dir.rglob("*.json")
            if not f.name.startswith(".") and "chunks" not in f.parts
        ]
        s5_total_extractions = len(extract_jsons)
        for jf in extract_jsons:
            data = _load_json_safe(jf)
            if data and (
                (isinstance(data, dict) and any(v is not None and v != [] and v != {} and v != "" for v in data.values()))
                or (isinstance(data, list) and len(data) > 0)
            ):
                s5_populated_extractions += 1

    s5_csv_tables: List[str] = []
    s5_wide_csv_exists = False
    s5_wide_parquet_exists = False
    s5_sqlite_exists = False
    s5_wide_rows = 0
    s5_wide_cols = 0
    if s5_tables_dir.exists():
        for csv_f in sorted(s5_tables_dir.glob("*.csv")):
            if csv_f.name == "consolidated_wide.csv":
                s5_wide_csv_exists = True
                try:
                    with open(csv_f, "r", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        header = next(reader, None)
                        if header:
                            s5_wide_cols = len(header)
                            s5_wide_rows = sum(1 for _ in reader)
                except Exception:
                    pass
            else:
                s5_csv_tables.append(csv_f.name)
        s5_wide_parquet_exists = (s5_tables_dir / "consolidated_wide.parquet").exists()
        s5_sqlite_exists = (s5_tables_dir / "aggregated_data.db").exists()

    # =========================================================================
    # BUILD HUMAN-READABLE & JSON SUMMARY
    # =========================================================================
    summary_data: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "total_execution_seconds": round(total_duration, 2) if total_duration is not None else None,
        "stage_1_extraction": {
            "documents_ingested": docs_ingested_count,
            "themes_identified": themes_identified_count,
            "master_themes_consolidated_count": len(master_themes_list),
            "master_themes_consolidated": master_themes_list,
            "master_theme_domain": master_domain,
            "triples_extracted": triples_extracted_count,
        },
        "stage_2_refinement": {
            "subjects_hypernym_lifted": subjects_lifted_count,
            "subjects_unique_hypernyms": subjects_unique_hypernyms,
            "predicates_hypernym_lifted": predicates_lifted_count,
            "predicates_unique_hypernyms": predicates_unique_hypernyms,
            "objects_hypernym_lifted": objects_lifted_count,
            "objects_unique_hypernyms": objects_unique_hypernyms,
            "refined_triplets_total": len(refined_triplets) if isinstance(refined_triplets, list) else 0,
        },
        "stage_3_topology": {
            "raw_graph": {
                "total_nodes": raw_nodes,
                "total_directed_edges": raw_dir_edges,
                "total_undirected_edges": raw_undir_edges,
                "total_graph_size": raw_total_graph_size,
                "density": raw_density,
                "global_hubs_removed": global_hubs_removed,
            },
            "final_subgraph": {
                "core_nodes": final_sub_nodes,
                "core_edges": final_sub_edges,
                "dense_clustering_nodes": dense_cluster_nodes,
                "dense_clustering_edges": dense_cluster_edges,
                "clusters_identified_pre_merge": initial_clusters_count,
                "final_clusters_after_merging": final_clusters_count,
            },
        },
        "stage_4_synthesis": {
            "final_schemas_count": len(schema_names),
            "final_schemas": schema_names,
            "final_basemodels_count": len(final_basemodels),
            "final_basemodels": final_basemodels,
            "consolidated_enums_count": len(final_enums),
            "consolidated_enums": final_enums,
        },
        "stage_5_mapping": {
            "documents_processed": s5_docs_processed,
            "total_schema_extractions": s5_total_extractions,
            "populated_schema_extractions": s5_populated_extractions,
            "relational_csv_tables_count": len(s5_csv_tables),
            "relational_csv_tables": s5_csv_tables,
            "consolidated_wide_csv": s5_wide_csv_exists,
            "consolidated_wide_parquet": s5_wide_parquet_exists,
            "consolidated_wide_rows": s5_wide_rows,
            "consolidated_wide_columns": s5_wide_cols,
            "sqlite_database": s5_sqlite_exists,
        },
    }

    lines: List[str] = [
        "=" * 70,
        "SEMANTIC PRISM — END-TO-END PIPELINE EXECUTION SUMMARY",
        "=" * 70,
    ]
    if total_duration is not None:
        lines.append(f"Total Execution Time: {total_duration:.2f} seconds")
    lines.append(f"Generated At: {summary_data['generated_at']}")
    lines.append("")

    # Stage 1
    lines.append("Stage 1: Extraction")
    lines.append(f"  - {docs_ingested_count} documents ingested")
    lines.append(f"  - {themes_identified_count} themes identified")
    lines.append(f"  - {len(master_themes_list)} master themes consolidated:")
    if master_themes_list:
        for idx, mt in enumerate(master_themes_list, start=1):
            lines.append(f"      {idx}. {mt}")
    else:
        lines.append("      (none)")
    lines.append(f"  - 1 master theme (domain): {master_domain or '(not available)'}")
    lines.append(f"  - {triples_extracted_count} triples extracted")
    lines.append("")

    # Stage 2
    lines.append("Stage 2: Refinement")
    lines.append(
        f"  - {subjects_lifted_count} Subjects hypernym lifted "
        f"(-> {subjects_unique_hypernyms} canonical concepts)"
    )
    if predicates_lifted_count > 0:
        lines.append(
            f"  - {predicates_lifted_count} Predicates hypernym lifted "
            f"(-> {predicates_unique_hypernyms} canonical relations)"
        )
    lines.append(
        f"  - {objects_lifted_count} Objects hypernym lifted "
        f"(-> {objects_unique_hypernyms} canonical concepts)"
    )
    lines.append("")

    # Stage 3
    lines.append("Stage 3: Topology")
    lines.append("  Original raw graph details:")
    lines.append(f"    - {raw_nodes} total nodes, {raw_dir_edges} total edges ({raw_undir_edges} undirected)")
    lines.append(
        f"    - Total Graph size: {raw_total_graph_size} (nodes + edges) | "
        f"Density: {raw_density} | Global hubs separated: {global_hubs_removed}"
    )
    lines.append("  Final subgraph details:")
    lines.append(
        f"    - {final_sub_nodes} core nodes, {final_sub_edges} core edges "
        f"(dense clustering core: {dense_cluster_nodes} nodes, {dense_cluster_edges} edges)"
    )
    lines.append(f"    - {initial_clusters_count} clusters identified")
    lines.append(f"    - {final_clusters_count} final clusters after merging")
    lines.append("")

    # Stage 4
    lines.append("Stage 4: Synthesis")
    lines.append(f"  - {len(schema_names)} final schemas created:")
    if schema_names:
        for s_name in schema_names:
            lines.append(f"      * {s_name}")
    else:
        lines.append("      (none)")
    lines.append(f"  - {len(final_basemodels)} final BaseModels created:")
    if final_basemodels:
        for bm_name in final_basemodels:
            lines.append(f"      * {bm_name}")
    else:
        lines.append("      (none)")
    lines.append("")

    # Stage 5
    lines.append("Stage 5: Schema-Driven Extraction & Aggregation")
    lines.append(f"  - {s5_docs_processed} documents processed")
    lines.append(
        f"  - {s5_populated_extractions}/{s5_total_extractions} populated (document, schema) extraction files"
    )
    lines.append(f"  - {len(s5_csv_tables)} relational CSV tables generated")
    lines.append(
        f"  - Consolidated wide table: {s5_wide_rows} rows x {s5_wide_cols} columns "
        f"(CSV: {s5_wide_csv_exists}, Parquet: {s5_wide_parquet_exists}, SQLite DB: {s5_sqlite_exists})"
    )
    lines.append("=" * 70)

    report_text = "\n".join(lines) + "\n"

    txt_path = outputs_dir / "pipeline_summary.txt"
    json_path = outputs_dir / "pipeline_summary.json"
    txt_path.write_text(report_text, encoding="utf-8")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return report_text, summary_data


def run_stage(
    stage_num: int,
    script_name: str,
    stage_desc: str,
    extra_args: List[str] | None = None,
) -> None:
    print("\n" + "=" * 60, flush=True)
    print(f"🚀 INITIATING {stage_desc.upper()}", flush=True)
    print("=" * 60, flush=True)

    start_time = time.time()
    cmd = [sys.executable, "-u", script_name]
    if extra_args:
        cmd.extend(extra_args)

    log_dir = Path("outputs") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stage_log_path = log_dir / f"stage_{stage_num:02d}_console.log"

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    recent_lines: deque[str] = deque(maxlen=40)

    try:
        with open(stage_log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
                recent_lines.append(line.rstrip("\n"))
            return_code = proc.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)
    except subprocess.CalledProcessError as err:
        print(f"\n❌ CRITICAL ERROR in {script_name} (exit code {err.returncode})!", flush=True)
        if recent_lines:
            print("--- Last output lines from stage ---", flush=True)
            for rl in recent_lines:
                print(f"  {rl}", flush=True)
            print(f"Full stage console log saved to: {stage_log_path}", flush=True)
        print(f"Pipeline halted at {stage_desc}.", flush=True)
        sys.exit(1)

    duration = time.time() - start_time
    print(f"\n✅ {stage_desc} completed in {duration:.2f} seconds.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SemanticPrism Master Pipeline Orchestrator (Stages 1-5)"
    )
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4, 5],
        default=[1, 2, 3, 4, 5],
        help="Stages to execute in order (default: 1 2 3 4 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to LLM stages (Stages 1, 2, 4, and 5)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Generate and save the pipeline summary from existing outputs without running stages",
    )
    args = parser.parse_args()

    if args.summary_only:
        report_text, _ = generate_pipeline_summary()
        print("\n" + report_text)
        print("Saved summary to 'outputs/pipeline_summary.txt' and 'outputs/pipeline_summary.json'.")
        return

    print("Welcome to SemanticPrism.")
    print(f"Starting pipeline for stages: {args.stages}...\n")

    global_start = time.time()

    for stage_num, script, desc, supports_dry_run in STAGES:
        if stage_num not in args.stages:
            continue
        stage_args = ["--dry-run"] if (args.dry_run and supports_dry_run) else []
        run_stage(stage_num, script, desc, stage_args)

    global_duration = time.time() - global_start
    report_text, _ = generate_pipeline_summary(total_duration=global_duration)
    print("\n" + report_text)
    print("🎉 SEMANTIC PRISM PIPELINE COMPLETE!")
    print("Summary saved to: 'outputs/pipeline_summary.txt' and 'outputs/pipeline_summary.json'.")


if __name__ == "__main__":
    main()
