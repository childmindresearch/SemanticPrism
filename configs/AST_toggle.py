"""
Stage 4 Synthesis: AST Sanitization & Code Cleanup Configuration (`configs/AST_toggle.py`)
==========================================================================================
Main configuration file controlling all Pre-AST text normalization, Enum AST sanitization,
BaseModel AST sanitization, and Consolidated Ontology deduplication rules in Stage 4.

- Set any rule toggle to `True` to enable it, or `False` to bypass it.
- Edit the numerical/value parameters below (e.g., `MAX_ENUM_MEMBERS_PER_CLASS`,
  `MAX_BASEMODEL_FIELDS_PER_CLASS`) to adjust cardinality caps and thresholds globally
  across both Stage 4 LLM prompts and AST post-processors.
"""

from __future__ import annotations

from typing import Tuple

# ==============================================================================
# 1. PRE-AST TEXT NORMALIZATION (`clean_python_code`)
# ==============================================================================

# 1.1 Strip leading ```python / ``` and trailing ``` markdown code fences
STRIP_MARKDOWN_CODE_FENCES: bool = True

# 1.2 Unescape over-escaped quotes (\" -> ") and literal \n sequences outside raw strings
UNESCAPE_QUOTES_AND_NEWLINES: bool = True

# 1.3 Normalize invalid spaces or commas on the LHS of '=' assignments into underscores (_)
FIX_ASSIGNMENT_LHS_IDENTIFIERS: bool = True

# 1.4 Replace hyphens in type-annotated field names (':' LHS) with underscores (_)
FIX_FIELD_HYPHENS: bool = True

# 1.5 Rename field names that collide with Python reserved keywords (e.g. 'from' -> 'from_field')
RENAME_PYTHON_RESERVED_KEYWORDS: bool = True

# 1.6 Strip non-ASCII / CJK character leakage ([\u4e00-\u9fff]+) in comments or docstrings
STRIP_CJK_CHARACTERS: bool = True

# 1.7 Strip accidental classes that model the prompt/JSON envelope itself
STRIP_META_MODEL_CLASSES: bool = True
BANNED_META_MODEL_CLASSES: Tuple[str, ...] = (
    "ThemeObservation",
    "ThematicObservation",
    "GeneratedModule",
)


# ==============================================================================
# 2. ENUM AST SANITIZER (`_sanitize_ast_enums`)
# ==============================================================================

# Master switch for the Enum AST Sanitizer pass
ENABLE_ENUM_AST_SANITIZER: bool = True

# 2.1 Strip verbose / non-atomic Enum members exceeding word or character bounds
STRIP_VERBOSE_ENUM_MEMBERS: bool = True
ENUM_MEMBER_NAME_MAX_WORDS: int = 4          # Max '_'-separated words in Enum member identifier
ENUM_MEMBER_NAME_MAX_CHARS: int = 35         # Max characters in Enum member identifier
ENUM_MEMBER_VALUE_MAX_WORDS: int = 5         # Max words in Enum member string value
ENUM_MEMBER_VALUE_MAX_CHARS: int = 40        # Max characters in Enum member string value

# 2.2 Strip duplicate member identifiers within the same Enum class
STRIP_DUPLICATE_ENUM_NAMES: bool = True

# 2.3 Strip members with duplicate normalized lowercase string values within the same Enum class
STRIP_DUPLICATE_ENUM_VALUES: bool = True

# 2.4 Cap maximum number of members allowed per Enum class
ENFORCE_MAX_ENUM_MEMBERS_CAP: bool = True
MAX_ENUM_MEMBERS_PER_CLASS: int = 15         # Main config cap for max Enum members per class

# 2.5 Automatically move 'OTHER' / 'Other' member to the final position of the Enum class
PLACE_ENUM_OTHER_LAST: bool = True

# 2.6 Remove Enum classes with 0 valid members remaining and rewrite referencing fields to 'str'
REMOVE_EMPTY_ENUMS_AND_REWRITE_REFS: bool = True


# ==============================================================================
# 3. BASEMODEL AST SANITIZER (`_sanitize_ast_models`)
# ==============================================================================

# Master switch for the BaseModel AST Sanitizer pass
ENABLE_BASEMODEL_AST_SANITIZER: bool = True

# 3.1 Rewrite undefined/dangling field type references (outside Literal[...]) to 'str'
REWRITE_DANGLING_TYPE_REFERENCES: bool = True

# 3.2 Rename fields shadowing reserved Pydantic BaseModel attributes or Python keywords to '<name>_field'
RENAME_SHADOWED_BASEMODEL_ATTRIBUTES: bool = True

# 3.3 Strip duplicate field names (case-insensitive) within the same BaseModel class
STRIP_DUPLICATE_BASEMODEL_FIELDS: bool = True

# 3.4 Shared 2-token snake_case prefix collapse & cap (for fields with >= PREFIX_MIN_TOKENS tokens)
COLLAPSE_SHARED_PREFIX_FIELDS: bool = True
PREFIX_MIN_TOKENS: int = 3                   # Minimum underscore-separated tokens to check 2-token prefix
SHARED_PREFIX_COLLAPSE_THRESHOLD: int = 3    # Collapse into Optional[List[str]] when >= N fields share prefix

ENFORCE_SHARED_PREFIX_CAP: bool = True
MAX_SHARED_PREFIX_FIELDS: int = 2            # Max surviving fields allowed to share the same 2-token prefix

# 3.5 Shared non-exempt suffix collapse & cap (for fields with >= SUFFIX_MIN_TOKENS tokens)
COLLAPSE_SHARED_SUFFIX_FIELDS: bool = True
SUFFIX_MIN_TOKENS: int = 2                   # Minimum underscore-separated tokens to check final suffix token
SHARED_SUFFIX_COLLAPSE_THRESHOLD: int = 3    # Collapse into Optional[List[str]] when >= N fields share suffix

ENFORCE_SHARED_SUFFIX_CAP: bool = True
MAX_SHARED_SUFFIX_FIELDS: int = 2            # Max surviving fields allowed to share a non-exempt suffix
EXEMPT_FIELD_SUFFIXES: Tuple[str, ...] = (
    "id", "name", "type", "types", "status", "level", "score", "range",
    "date", "time", "plan", "notes", "summary", "description", "details",
    "count", "rate", "list", "items", "category", "mode", "source",
)

# 3.6 Cap maximum number of fields allowed per Pydantic BaseModel class (with 4-tuple priority ranking)
ENFORCE_MAX_BASEMODEL_FIELDS_CAP: bool = True
MAX_BASEMODEL_FIELDS_PER_CLASS: int = 15     # Main config cap for max BaseModel fields per class

# 3.7 Guarantee every BaseModel class begins with a triple-quoted docstring ("""...""")
ENSURE_BASEMODEL_DOCSTRING: bool = True

# 3.8 Order top-level declarations (__future__ -> imports -> constants -> Enums -> topological BaseModels)
CANONICAL_TOPOLOGICAL_CLASS_ORDERING: bool = True

# 3.9 Quote self-referential or cyclic BaseModel forward references so Pydantic v2 never raises NameError
QUOTE_CYCLIC_FORWARD_REFERENCES: bool = True

# 3.10 Insert 'pass' if a BaseModel class has 0 fields remaining so syntax stays valid
ENSURE_EMPTY_CLASS_PASS_FALLBACK: bool = True


# ==============================================================================
# 4. CONSOLIDATED ONTOLOGY AST MERGER (`deduplicate_consolidated_classes`)
# ==============================================================================

# Master switch for Phase 2 consolidated ontology AST deduplication/merging
ENABLE_CONSOLIDATED_ONTOLOGY_MERGER: bool = True

# 4.1 Merge duplicate Enum classes with the same name by unioning unique members
MERGE_DUPLICATE_ENUM_CLASSES: bool = True

# 4.2 Merge duplicate BaseModel classes with the same name by unioning unique fields
MERGE_DUPLICATE_BASEMODEL_CLASSES: bool = True

# 4.3 Disambiguate Enum vs. BaseModel name collisions by appending COLLISION_RENAME_SUFFIX to the BaseModel
DISAMBIGUATE_ENUM_BASEMODEL_COLLISIONS: bool = True
COLLISION_RENAME_SUFFIX: str = "Detail"

# 4.4 Re-run _sanitize_ast_enums and _sanitize_ast_models on the merged ontology
RESANITIZE_MERGED_ONTOLOGY: bool = True


# ==============================================================================
# 5. SANITIZATION LOGGING & CONSOLE CHECKPOINTS
# ==============================================================================

# Write AST sanitization audit events to outputs/04_synthesis/logs/schema_sanitization.log
LOG_SANITIZATION_TO_FILE: bool = True

# Print AST sanitization summaries to console when PRINT_CHECKPOINTS is enabled
PRINT_SANITIZATION_CHECKPOINTS: bool = True
