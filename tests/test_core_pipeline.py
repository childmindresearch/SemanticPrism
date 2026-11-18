"""
Fast, zero-GPU CPU unit test suite for SemanticPrism core invariants:
  1. Stage 4 AST Sanitizer, Enum `Other` fallback injection, complexity caps & class merging
  2. Stage 5 prose-free (`include_source_text=False`) wide-table & relational SQLite/CSV/Parquet export
  3. Master pipeline summary generation & official 5-stage naming alignment
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from run_pipeline import generate_ontology_build_summary, generate_pipeline_summary
from src.mapping.aggregator import DataAggregator, build_consolidated_wide_table
from src.synthesis.sanitizers import (
    clean_python_code,
    deduplicate_consolidated_classes,
    validate_python_syntax,
)


class TestStage4ASTSanitizer(unittest.TestCase):
    def test_clean_and_validate_enum_and_reserved_keywords(self) -> None:
        raw_llm_code = '''```python
from enum import Enum
from typing import Optional
from pydantic import BaseModel

class MedicationCategory(Enum):
    STIMULANT = "Stimulant"
    SSRI = "SSRI"

class ClinicalVisit(BaseModel):
    from: Optional[str] = None
    follow-up_plan: Optional[MedicationCategory] = None
```'''
        cleaned = clean_python_code(raw_llm_code)
        sanitized = deduplicate_consolidated_classes(cleaned, context_label="unit_test")
        validate_python_syntax(sanitized)

        namespace: dict = {}
        exec(sanitized, namespace)
        enum_cls = namespace["MedicationCategory"]
        model_cls = namespace["ClinicalVisit"]

        # Verify (str, Enum) inheritance & mandatory OTHER = 'Other' placed last
        self.assertTrue(issubclass(enum_cls, str))
        members = list(enum_cls)
        self.assertEqual(members[-1].name, "OTHER")
        self.assertEqual(members[-1].value, "Other")

        # Verify reserved keyword & hyphenated field repair
        self.assertIn("from_field", model_cls.model_fields)
        self.assertIn("follow_up_plan", model_cls.model_fields)

    def test_duplicate_class_consolidation_and_field_cap(self) -> None:
        fields_a = "\n".join(f"    metric_{i}: Optional[int] = None" for i in range(10))
        fields_b = "\n".join(f"    metric_{i}: Optional[int] = None" for i in range(8, 20))
        combined_raw = f"""
from typing import Optional
from pydantic import BaseModel

class CognitivePanel(BaseModel):
    \"\"\"First community panel.\"\"\"
{fields_a}

class CognitivePanel(BaseModel):
    \"\"\"Second community panel.\"\"\"
{fields_b}
"""
        merged = deduplicate_consolidated_classes(combined_raw, context_label="merge_test")
        validate_python_syntax(merged)

        ns: dict = {}
        exec(merged, ns)
        panel_cls = ns["CognitivePanel"]
        self.assertLessEqual(len(panel_cls.model_fields), 15)


class TestStage5PrivacyAndAggregation(unittest.TestCase):
    def test_consolidated_wide_table_omits_source_text_by_default(self) -> None:
        grouped = {
            ("01_pharmacology", "PatientMedication"): [
                {
                    "_source_id": "doc_001",
                    "_source_file": "doc_001__01_pharmacology__PatientMedication.json",
                    "_item_index": 0,
                    "medication_type": "Stimulant",
                    "rebound_symptoms": True,
                }
            ]
        }
        source_texts = {"doc_001": "CONFIDENTIAL RAW PATIENT PROSE THAT MUST NOT LEAK."}

        # Default privacy-safe behavior: prose-free wide table
        rows_default = build_consolidated_wide_table(grouped, source_texts=source_texts)
        self.assertEqual(len(rows_default), 1)
        self.assertNotIn("source_text", rows_default[0])
        self.assertEqual(rows_default[0]["PatientMedication__medication_type"], "Stimulant")
        self.assertTrue(rows_default[0]["PatientMedication__rebound_symptoms"])

        # Explicit opt-in includes source_text
        rows_opt_in = build_consolidated_wide_table(
            grouped, source_texts=source_texts, include_source_text=True
        )
        self.assertIn("source_text", rows_opt_in[0])

    def test_data_aggregator_exports_csv_parquet_and_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ext_dir = root / "extracts" / "doc_1"
            tbl_dir = root / "tables"
            ext_dir.mkdir(parents=True)

            sample_json = ext_dir / "doc_1__mod__SleepManagement.json"
            sample_json.write_text(
                json.dumps({"sleep_onset_latency": 35.0, "avoid_stimulants": True}),
                encoding="utf-8",
            )

            agg = DataAggregator(extracts_dir=root / "extracts", tables_dir=tbl_dir)
            stats = agg.run(source_texts={"doc_1": "Verbatim clinical note"})

            self.assertGreaterEqual(stats["tables_created"], 2)
            self.assertTrue((tbl_dir / "consolidated_wide.csv").exists())
            self.assertTrue((tbl_dir / "consolidated_wide.parquet").exists())
            self.assertTrue((tbl_dir / "aggregated_data.db").exists())

            with sqlite3.connect(tbl_dir / "aggregated_data.db") as conn:
                cols = [r[1] for r in conn.execute("PRAGMA table_info(consolidated_wide)").fetchall()]
                self.assertNotIn("source_text", cols)


class TestPipelineSummaries(unittest.TestCase):
    def test_official_five_stage_names_in_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            ont_txt, _ = generate_ontology_build_summary(outputs_dir=out_dir, total_duration=12.5)
            full_txt, _ = generate_pipeline_summary(outputs_dir=out_dir, total_duration=20.0)

            for label in [
                "Stage 1: Relational Extraction",
                "Stage 2: Hypernym Lifting",
                "Stage 3: Community Partitioning",
                "Stage 4: Schema Synthesis",
            ]:
                self.assertIn(label, ont_txt)
                self.assertIn(label, full_txt)

            self.assertNotIn("Stage 5: Tabular Projection", ont_txt)
            self.assertIn("Stage 5: Tabular Projection", full_txt)


if __name__ == "__main__":
    unittest.main()

