"""
Stage 5 Phase 4: Relational & Consolidated Wide-Table Aggregation (CSV, Parquet, SQLite).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

logger = logging.getLogger("SemanticPrism5")


def _sanitize_identifier(name: str) -> str:
    """Sanitizes a string to be safe for filesystem paths, filenames, and SQLite identifiers."""
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", str(name))


def _log_agg_checkpoint(phase: str, message: str, print_checkpoints: bool = True) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    if print_checkpoints:
        print(f"[{ts} | Checkpoint | {phase}] {message}", flush=True)
    logger.info(f"[{phase}] {message}")


def is_scalar(val: Any) -> bool:
    """Returns True if the value is a primitive scalar (or None)."""
    return val is None or isinstance(val, (str, int, float, bool))


def infer_sqlite_type(values: List[Any]) -> str:
    """Infers the appropriate SQLite column affinity (INTEGER, REAL, TEXT) from sample values."""
    non_null = [v for v in values if v is not None]
    if not non_null:
        return "TEXT"
    if all(isinstance(v, bool) for v in non_null):
        return "INTEGER"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in non_null):
        return "INTEGER"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null):
        return "REAL"
    return "TEXT"


def flatten_dict(
    data: Dict[str, Any],
    parent_key: str = "",
    sep: str = "__",
) -> Dict[str, Any]:
    """
    Flattens nested dictionaries into `parent__child` keys while keeping lists at the top level
    so they can be routed either to child relational tables or JSON-serialized columns.
    """
    items: List[Tuple[str, Any]] = []
    for k, v in data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) and v:
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, dict) and not v:
            items.append((new_key, None))
        else:
            items.append((new_key, v))
    return dict(items)


def extract_relational_tables(
    records: List[Dict[str, Any]],
    root_table_name: str,
    multi_table: bool = True,
    drop_empty_records: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Transforms extracted JSON records for a schema into normalized relational tables:
      - Parent table `<root_table_name>`
      - Child tables `<root_table_name>_<list_field>` linked via `source_id`, `source_file`,
        `parent_item_index`, and `item_index`.
    """
    tables: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for rec in records:
        source_id = rec.get("_source_id", "unknown")
        source_file = rec.get("_source_file", "unknown")
        parent_idx = rec.get("_item_index", 0)

        payload = {k: v for k, v in rec.items() if not k.startswith("_")}
        flat = flatten_dict(payload)

        if drop_empty_records:
            has_data = any(
                v is not None and v != [] and v != {} and v != ""
                for v in flat.values()
            )
            if not has_data:
                continue

        parent_row: Dict[str, Any] = {
            "source_id": source_id,
            "source_file": source_file,
            "item_index": parent_idx,
        }

        for key, val in flat.items():
            if isinstance(val, list):
                if not multi_table:
                    parent_row[key] = json.dumps(val, ensure_ascii=False) if val else None
                else:
                    child_table_name = f"{root_table_name}_{key}"
                    for child_idx, item in enumerate(val):
                        if item is None or item == "" or item == {} or item == []:
                            continue
                        child_row: Dict[str, Any] = {
                            "source_id": source_id,
                            "source_file": source_file,
                            "parent_item_index": parent_idx,
                            "item_index": child_idx,
                        }
                        if isinstance(item, dict):
                            flat_child = flatten_dict(item)
                            if drop_empty_records and not any(
                                cv is not None and cv != [] and cv != {} and cv != ""
                                for cv in flat_child.values()
                            ):
                                continue
                            for ck, cv in flat_child.items():
                                if isinstance(cv, (list, dict)):
                                    child_row[ck] = json.dumps(cv, ensure_ascii=False) if cv else None
                                else:
                                    child_row[ck] = cv
                        else:
                            child_row["value"] = item
                        tables[child_table_name].append(child_row)
            else:
                parent_row[key] = val

        tables[root_table_name].append(parent_row)

    return tables


def build_consolidated_wide_table(
    grouped_records: Dict[Tuple[str, str], List[Dict[str, Any]]],
    source_texts: Optional[Dict[str, str]] = None,
    include_source_text: bool = True,
    drop_empty_records: bool = True,
) -> List[Dict[str, Any]]:
    """
    Consolidates all extracted schemas across the corpus into a single wide table where
    each row corresponds to one unique document (`source_id`):
      - Scalar fields -> `<Schema>__<Field>`
      - List of scalars -> JSON array in `<Schema>__<Field>` + count in `<Schema>__<Field>__count`
      - List of child objects -> JSON array in `<Schema>__<Field>` + count in `<Schema>__<Field>__count`
        + exploded child attributes `<Schema>__<Field>__<attr>`
    """
    doc_rows: Dict[str, Dict[str, Any]] = {}
    all_columns: List[str] = ["source_id"]
    if include_source_text and source_texts:
        all_columns.append("source_text")
    seen_columns: Set[str] = set(all_columns)

    def register_col(col_name: str) -> None:
        if col_name not in seen_columns:
            seen_columns.add(col_name)
            all_columns.append(col_name)

    if source_texts:
        for doc_id, text in source_texts.items():
            doc_rows[str(doc_id)] = {"source_id": str(doc_id)}
            if include_source_text:
                doc_rows[str(doc_id)]["source_text"] = text

    schema_name_counts: Dict[str, int] = defaultdict(int)
    for mod_name, sch_name in grouped_records.keys():
        schema_name_counts[sch_name] += 1

    for (module_name, schema_name), records in sorted(grouped_records.items()):
        schema_prefix = (
            f"{module_name}__{schema_name}"
            if schema_name_counts[schema_name] > 1
            else schema_name
        )

        by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rec in records:
            doc_id = str(rec.get("_source_id", "unknown"))
            payload = {k: v for k, v in rec.items() if not k.startswith("_")}
            flat = flatten_dict(payload)

            if drop_empty_records:
                has_data = any(
                    v is not None and v != [] and v != {} and v != ""
                    for v in flat.values()
                )
                if not has_data:
                    continue

            by_doc[doc_id].append(flat)

        for doc_id, flat_items in by_doc.items():
            if doc_id not in doc_rows:
                doc_rows[doc_id] = {"source_id": doc_id}
                if include_source_text and source_texts is not None:
                    doc_rows[doc_id]["source_text"] = source_texts.get(doc_id)

            row = doc_rows[doc_id]

            all_keys: List[str] = []
            for item in flat_items:
                for k in item.keys():
                    if k not in all_keys:
                        all_keys.append(k)

            for key in all_keys:
                col_base = f"{schema_prefix}__{key}"
                vals = [
                    item.get(key)
                    for item in flat_items
                    if item.get(key) is not None and item.get(key) != "" and item.get(key) != []
                ]

                if not vals:
                    register_col(col_base)
                    row.setdefault(col_base, None)
                    continue

                first_val = vals[0]

                if isinstance(first_val, list):
                    combined_list: List[Any] = []
                    for v in vals:
                        if isinstance(v, list):
                            for elem in v:
                                if elem is not None and elem != "" and elem != {} and elem != []:
                                    combined_list.append(elem)

                    register_col(col_base)
                    count_col = f"{col_base}__count"
                    register_col(count_col)

                    if not combined_list:
                        row[col_base] = None
                        row[count_col] = 0
                        continue

                    row[col_base] = json.dumps(combined_list, ensure_ascii=False)
                    row[count_col] = len(combined_list)

                    dict_elems = [e for e in combined_list if isinstance(e, dict)]
                    if dict_elems:
                        sub_keys: List[str] = []
                        flat_dicts = [flatten_dict(e) for e in dict_elems]
                        for fd in flat_dicts:
                            for sk in fd.keys():
                                if sk not in sub_keys:
                                    sub_keys.append(sk)

                        for sk in sub_keys:
                            sub_col = f"{col_base}__{sk}"
                            register_col(sub_col)
                            sub_vals = [
                                fd.get(sk)
                                for fd in flat_dicts
                                if fd.get(sk) is not None and fd.get(sk) != "" and fd.get(sk) != []
                            ]
                            if not sub_vals:
                                row[sub_col] = None
                            elif len(sub_vals) == 1 and is_scalar(sub_vals[0]):
                                row[sub_col] = sub_vals[0]
                            else:
                                unique_sub: List[Any] = []
                                for sv in sub_vals:
                                    if sv not in unique_sub:
                                        unique_sub.append(sv)
                                row[sub_col] = (
                                    unique_sub[0]
                                    if (len(unique_sub) == 1 and is_scalar(unique_sub[0]))
                                    else json.dumps(unique_sub, ensure_ascii=False)
                                )
                else:
                    register_col(col_base)
                    if len(vals) == 1:
                        row[col_base] = vals[0]
                    else:
                        unique_vals: List[Any] = []
                        for v in vals:
                            if v not in unique_vals:
                                unique_vals.append(v)
                        row[col_base] = (
                            unique_vals[0]
                            if len(unique_vals) == 1
                            else json.dumps(unique_vals, ensure_ascii=False)
                        )

    final_rows: List[Dict[str, Any]] = []
    for doc_id in sorted(doc_rows.keys()):
        raw_row = doc_rows[doc_id]
        if drop_empty_records:
            has_extracted_cols = any(
                k not in ("source_id", "source_text")
                and not k.endswith("__count")
                and v is not None
                and v != ""
                and v != "[]"
                for k, v in raw_row.items()
            )
            if not has_extracted_cols:
                continue

        standardized = {col: raw_row.get(col, None) for col in all_columns}
        final_rows.append(standardized)

    return final_rows


class DataAggregator:
    """
    Reads per-document JSON extraction files from `outputs/05_extraction/extracts/` and writes:
      1. Normalized relational CSV tables & SQLite tables (`multi_table_relational = True`)
      2. Consolidated wide table (`consolidated_wide.csv`, `consolidated_wide.parquet`, and
         the `consolidated_wide` table inside `aggregated_data.db`).
    """

    def __init__(
        self,
        extracts_dir: Union[str, Path],
        tables_dir: Union[str, Path],
        generate_csv: bool = True,
        generate_sqlite: bool = True,
        generate_parquet: bool = True,
        db_filename: str = "aggregated_data.db",
        multi_table_relational: bool = True,
        consolidated_wide_table: bool = True,
        wide_table_name: str = "consolidated_wide",
        include_source_text: bool = True,
        drop_empty_records: bool = True,
        drop_empty_tables: bool = True,
        print_checkpoints: bool = True,
    ):
        self.extracts_dir = Path(extracts_dir)
        self.tables_dir = Path(tables_dir)
        self.generate_csv = generate_csv
        self.generate_sqlite = generate_sqlite
        self.generate_parquet = generate_parquet
        self.db_filename = db_filename
        self.multi_table_relational = multi_table_relational
        self.consolidated_wide_table = consolidated_wide_table
        self.wide_table_name = wide_table_name
        self.include_source_text = include_source_text
        self.drop_empty_records = drop_empty_records
        self.drop_empty_tables = drop_empty_tables
        self.print_checkpoints = print_checkpoints

    @staticmethod
    def parse_extraction_filename(file_path: Path) -> Tuple[str, str, str]:
        """Parses `<doc_id>__<module_name>__<class_name>.json`."""
        parts = file_path.stem.split("__")
        if len(parts) >= 3:
            doc_id = "__".join(parts[:-2])
            module_name = parts[-2]
            schema_name = parts[-1]
            return doc_id, module_name, schema_name
        return "unknown", "default", file_path.stem

    def collect_extractions(self) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
        """Scans `extracts_dir` for document-level `.json` extraction files (excluding `chunks/` and `.completed.json`)."""
        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

        if not self.extracts_dir.exists():
            logger.warning(f"Extracts directory does not exist: {self.extracts_dir}")
            return grouped

        json_files = sorted(
            f for f in self.extracts_dir.rglob("*.json")
            if not f.name.startswith(".") and "chunks" not in f.parts
        )

        for jf in json_files:
            doc_id, module_name, schema_name = self.parse_extraction_filename(jf)
            try:
                content = json.loads(jf.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"Failed to read JSON file {jf}: {e}")
                continue

            if not content:
                continue

            items = content if isinstance(content, list) else [content]
            for idx, item in enumerate(items):
                if isinstance(item, dict) and item:
                    record: Dict[str, Any] = {
                        "_source_id": doc_id,
                        "_source_file": jf.name,
                        "_item_index": idx,
                    }
                    record.update(item)
                    grouped[(module_name, schema_name)].append(record)

        return grouped

    def run(self, source_texts: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Executes relational and wide-table aggregation, returning summary statistics."""
        _log_agg_checkpoint(
            "Phase 4: Aggregation",
            f"Scanning extraction outputs in '{self.extracts_dir}'...",
            self.print_checkpoints,
        )
        grouped = self.collect_extractions()

        self.tables_dir.mkdir(parents=True, exist_ok=True)
        all_relational_tables: Dict[str, List[Dict[str, Any]]] = {}

        schema_name_counts: Dict[str, int] = defaultdict(int)
        for mod_name, sch_name in grouped.keys():
            schema_name_counts[sch_name] += 1

        for (module_name, schema_name), records in grouped.items():
            if not records:
                continue

            root_table_name = (
                f"{module_name}_{schema_name}"
                if schema_name_counts[schema_name] > 1
                else schema_name
            )
            extracted_tables = extract_relational_tables(
                records=records,
                root_table_name=root_table_name,
                multi_table=self.multi_table_relational,
                drop_empty_records=self.drop_empty_records,
            )

            for t_name, t_rows in extracted_tables.items():
                if self.drop_empty_tables and not t_rows:
                    continue
                if t_name in all_relational_tables:
                    all_relational_tables[t_name].extend(t_rows)
                else:
                    all_relational_tables[t_name] = t_rows

        wide_rows: List[Dict[str, Any]] = []
        if self.consolidated_wide_table and (grouped or source_texts):
            wide_rows = build_consolidated_wide_table(
                grouped_records=grouped,
                source_texts=source_texts,
                include_source_text=self.include_source_text,
                drop_empty_records=self.drop_empty_records,
            )
            if wide_rows or not self.drop_empty_tables:
                all_relational_tables[self.wide_table_name] = wide_rows
                _log_agg_checkpoint(
                    "Phase 4: Aggregation",
                    f"Built consolidated wide table '{self.wide_table_name}' with {len(wide_rows)} document row(s) "
                    f"and {len(wide_rows[0]) if wide_rows else 0} column(s).",
                    self.print_checkpoints,
                )

        if not all_relational_tables:
            _log_agg_checkpoint(
                "Phase 4: Aggregation",
                "No non-empty records found to aggregate.",
                self.print_checkpoints,
            )
            return {"tables_created": 0, "table_names": [], "wide_rows": 0, "wide_columns": 0}

        if self.generate_csv:
            self._export_csvs(all_relational_tables)

        if self.generate_parquet and wide_rows:
            self._export_wide_parquet(wide_rows)

        if self.generate_sqlite:
            self._export_sqlite(all_relational_tables)

        return {
            "tables_created": len(all_relational_tables),
            "table_names": list(all_relational_tables.keys()),
            "wide_rows": len(wide_rows),
            "wide_columns": len(wide_rows[0]) if wide_rows else 0,
        }

    def _export_csvs(self, all_tables: Dict[str, List[Dict[str, Any]]]) -> None:
        for table_name, rows in all_tables.items():
            if not rows:
                continue
            safe_name = _sanitize_identifier(table_name)
            csv_path = self.tables_dir / f"{safe_name}.csv"
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False, encoding="utf-8")
            _log_agg_checkpoint(
                "Phase 4: CSV Export",
                f"Saved '{csv_path.name}' ({len(df)} rows, {len(df.columns)} columns).",
                self.print_checkpoints,
            )

    def _export_wide_parquet(self, wide_rows: List[Dict[str, Any]]) -> None:
        safe_name = _sanitize_identifier(self.wide_table_name)
        pq_path = self.tables_dir / f"{safe_name}.parquet"
        try:
            df = pd.DataFrame(wide_rows)
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].apply(
                        lambda x: json.dumps(x, ensure_ascii=False)
                        if isinstance(x, (list, dict))
                        else (str(x) if x is not None and not isinstance(x, str) else x)
                    )
            df.to_parquet(pq_path, index=False)
            _log_agg_checkpoint(
                "Phase 4: Parquet Export",
                f"Saved '{pq_path.name}' ({len(df)} rows, {len(df.columns)} columns).",
                self.print_checkpoints,
            )
        except Exception as e:
            logger.warning(f"Could not export Parquet file {pq_path}: {e}")

    def _export_sqlite(self, all_tables: Dict[str, List[Dict[str, Any]]]) -> None:
        db_path = self.tables_dir / self.db_filename
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            for raw_table_name, rows in all_tables.items():
                if not rows:
                    continue

                table_name = _sanitize_identifier(raw_table_name)
                columns: List[str] = []
                for r in rows:
                    for k in r.keys():
                        if k not in columns:
                            columns.append(k)

                col_defs: List[str] = []
                for col in columns:
                    sample_vals = [r.get(col) for r in rows]
                    col_type = infer_sqlite_type(sample_vals)
                    safe_col = col.replace('"', '""')
                    col_defs.append(f'"{safe_col}" {col_type}')

                cursor.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                create_stmt = f'CREATE TABLE "{table_name}" ({", ".join(col_defs)})'
                cursor.execute(create_stmt)

                placeholders = ", ".join(["?"] * len(columns))
                quoted_cols = ", ".join([f'"{c.replace(chr(34), chr(34)*2)}"' for c in columns])
                insert_stmt = f'INSERT INTO "{table_name}" ({quoted_cols}) VALUES ({placeholders})'

                values_to_insert: List[Tuple[Any, ...]] = []
                for r in rows:
                    row_tuple: List[Any] = []
                    for col in columns:
                        val = r.get(col)
                        if isinstance(val, bool):
                            val = int(val)
                        elif isinstance(val, (list, dict)):
                            val = json.dumps(val, ensure_ascii=False)
                        row_tuple.append(val)
                    values_to_insert.append(tuple(row_tuple))

                cursor.executemany(insert_stmt, values_to_insert)

                if "source_id" in columns:
                    idx_name = _sanitize_identifier(f"idx_{table_name}_source_id")
                    cursor.execute(
                        f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table_name}" ("source_id")'
                    )

            conn.commit()
            _log_agg_checkpoint(
                "Phase 4: SQLite Export",
                f"Saved relational & wide SQLite database to '{db_path}' ({len(all_tables)} tables).",
                self.print_checkpoints,
            )
        except Exception as e:
            logger.error(f"SQLite export failed: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
