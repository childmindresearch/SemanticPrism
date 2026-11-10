"""
SemanticPrism Stage 5: Schema-Driven Extraction & Relational Aggregation Package.
"""

from src.mapping.aggregator import (
    DataAggregator,
    build_consolidated_wide_table,
    extract_relational_tables,
    flatten_dict,
    infer_sqlite_type,
    is_scalar,
)
from src.mapping.pipeline import (
    ResumeTracker,
    SafeJSONEncoder,
    SemanticPrismStage5Pipeline,
    chunk_text,
    configure_logging,
    estimate_tokens,
    has_extracted_content,
    load_input_documents,
    log_checkpoint,
    merge_pydantic_instances,
    sanitize_filename,
)
from src.mapping.prompts import (
    EXTRACTION_RULES,
    TRIAGE_RULES,
    build_extraction_prompt,
    build_triage_prompt,
    build_unified_extraction_prompt,
    format_schema_field_instructions,
)
from src.mapping.schemas import (
    CONSOLIDATED_ONTOLOGY_FILENAMES,
    SchemaMeta,
    TriageResult,
    build_unified_envelope_model,
    extract_referenced_models,
    get_child_schemas,
    get_top_level_schemas,
    load_schemas,
    load_schemas_from_file,
    resolve_schema_files,
    resolve_schema_path_With_fallbacks,
)

__all__ = [
    "CONSOLIDATED_ONTOLOGY_FILENAMES",
    "SchemaMeta",
    "TriageResult",
    "build_unified_envelope_model",
    "extract_referenced_models",
    "resolve_schema_path_With_fallbacks",
    "resolve_schema_files",
    "load_schemas_from_file",
    "load_schemas",
    "get_top_level_schemas",
    "get_child_schemas",
    "EXTRACTION_RULES",
    "TRIAGE_RULES",
    "format_schema_field_instructions",
    "build_triage_prompt",
    "build_extraction_prompt",
    "build_unified_extraction_prompt",
    "is_scalar",
    "infer_sqlite_type",
    "flatten_dict",
    "extract_relational_tables",
    "build_consolidated_wide_table",
    "DataAggregator",
    "estimate_tokens",
    "configure_logging",
    "log_checkpoint",
    "load_input_documents",
    "chunk_text",
    "SafeJSONEncoder",
    "sanitize_filename",
    "ResumeTracker",
    "has_extracted_content",
    "merge_pydantic_instances",
    "SemanticPrismStage5Pipeline",
]

