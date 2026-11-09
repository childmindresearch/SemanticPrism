"""
Stage 5 Phase 1: Dynamic Schema Discovery, DAG Classification & Triage Models.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import importlib.util
import inspect
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union, get_args, get_origin

from pydantic import BaseModel, Field
from pydantic._internal._decorators import Decorator, ModelValidatorDecoratorInfo

logger = logging.getLogger("SemanticPrism5")

# Consolidated ontology filenames automatically skipped in directory mode when individual community .py files exist
CONSOLIDATED_ONTOLOGY_FILENAMES: Set[str] = {
    "master_ontology.py",
    "comprehensive_ontology.py",
}


def _log_discovery(message: str, print_checkpoints: bool = True) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    if print_checkpoints:
        print(f"[{ts} | Checkpoint | Schema Discovery] {message}", flush=True)
    logger.info(f"[Schema Discovery] {message}")


@dataclass
class SchemaMeta:
    """Metadata and classification for a dynamically loaded Pydantic BaseModel schema."""

    name: str
    schema_cls: Type[BaseModel]
    source_file: Path
    module_name: str
    is_top_level: bool
    is_child: bool
    referenced_models: Set[str]

    @property
    def qualified_name(self) -> str:
        """Unique identifier combining module name and class name."""
        return f"{self.module_name}.{self.name}"


class TriageResult(BaseModel):
    """Structured output for fast multi-schema relevance triage on a text chunk."""

    relevant_schemas: List[str] = Field(
        default_factory=list,
        description="List of exact schema class names that have explicit factual evidence present in the text excerpt.",
    )


def extract_referenced_models(annotation: Any) -> Set[Type[BaseModel]]:
    """
    Recursively inspects a type annotation to find all Pydantic BaseModel subclasses
    referenced directly or inside generic containers (Optional, Union, List, Set, Tuple, Dict).
    """
    found: Set[Type[BaseModel]] = set()
    if annotation is None:
        return found

    if inspect.isclass(annotation) and issubclass(annotation, BaseModel) and annotation is not BaseModel:
        found.add(annotation)
        return found

    origin = get_origin(annotation)
    if origin is not None:
        for arg in get_args(annotation):
            found.update(extract_referenced_models(arg))

    return found


_NULL_LIKE_ENUM_STRINGS: Set[str] = {
    "", "null", "none", "n/a", "na", "nil", "unspecified", "not_applicable", "not_mentioned",
}


def _norm_enum_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def _coerce_single_enum_value(raw_val: Any, enum_cls: Type[Enum]) -> Any:
    """
    Normalizes a raw LLM output value against `enum_cls`:
      1. Exact match on `.value` or `.name`.
      2. Case-insensitive / punctuation-normalized slug match on `.value` or `.name`.
      3. Unambiguous substring / token containment match against a single non-OTHER member.
      4. Safe fallback to the Enum's `OTHER` member (`'Other'`) if present, or `None` if null-like.
    """
    if raw_val is None or isinstance(raw_val, enum_cls):
        return raw_val

    s = str(raw_val).strip()
    if not s:
        return None
    s_lower = s.lower()
    s_slug = _norm_enum_slug(s)

    members = list(enum_cls)
    if not members:
        return raw_val

    # 1. Exact value or name match
    for m in members:
        if s == str(m.value) or s == m.name:
            return m.value

    # 2. Case-insensitive or normalized slug match
    for m in members:
        m_val_str = str(m.value).strip()
        if s_lower == m_val_str.lower() or s_lower == m.name.lower():
            return m.value
        if s_slug and (s_slug == _norm_enum_slug(m_val_str) or s_slug == _norm_enum_slug(m.name)):
            return m.value

    if s_lower in _NULL_LIKE_ENUM_STRINGS or s_slug in _NULL_LIKE_ENUM_STRINGS:
        return None

    # 3. Unambiguous substring/token containment against non-OTHER members
    other_member: Optional[Enum] = None
    containment_matches: List[Enum] = []
    for m in members:
        m_val_str = str(m.value).strip()
        if m.name.strip().upper() == "OTHER" or m_val_str.lower() == "other":
            other_member = m
            continue
        m_slug = _norm_enum_slug(m_val_str)
        if len(m_slug) >= 3 and len(s_slug) >= 3:
            if m_slug in s_slug or s_slug in m_slug:
                containment_matches.append(m)

    if len(containment_matches) == 1:
        return containment_matches[0].value

    # 4. Safe fallback to OTHER member when a non-empty categorical value was extracted
    if other_member is not None:
        return other_member.value
    return None


def _coerce_single_bool_value(raw_val: Any) -> Any:
    """Coerces string boolean indicators ('yes', 'present', 'no', 'absent', 'n/a') for bool fields."""
    if raw_val is None or isinstance(raw_val, bool):
        return raw_val
    if isinstance(raw_val, (int, float)):
        if raw_val == 1:
            return True
        if raw_val == 0:
            return False
    s_slug = _norm_enum_slug(str(raw_val))
    if s_slug in {"true", "yes", "y", "present", "positive", "confirmed", "active", "1"}:
        return True
    if s_slug in {"false", "no", "n", "absent", "negative", "none_reported", "inactive", "0"}:
        return False
    if s_slug in _NULL_LIKE_ENUM_STRINGS or s_slug in {"unknown", "unclear", "indeterminate"}:
        return None
    return raw_val


def _coerce_value_for_annotation(raw_val: Any, annotation: Any) -> Any:
    """Recursively coerces raw dict/list/scalar values for annotations containing Enum or bool types."""
    if raw_val is None or annotation is None:
        return raw_val

    if annotation is bool:
        return _coerce_single_bool_value(raw_val)

    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        return _coerce_single_enum_value(raw_val, annotation)

    origin = get_origin(annotation)
    if origin is Union:
        non_none_args = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none_args) == 1:
            return _coerce_value_for_annotation(raw_val, non_none_args[0])
        for arg in non_none_args:
            if inspect.isclass(arg) and issubclass(arg, Enum):
                coerced = _coerce_single_enum_value(raw_val, arg)
                if coerced is not None:
                    return coerced
        return raw_val

    if origin in (list, List, set, Set, tuple, Tuple):
        args = get_args(annotation)
        inner_ann = args[0] if args else None
        if inner_ann is None:
            return raw_val
        # Check if inner_ann is or contains an Enum
        inner_origin = get_origin(inner_ann)
        inner_enum: Optional[Type[Enum]] = None
        if inspect.isclass(inner_ann) and issubclass(inner_ann, Enum):
            inner_enum = inner_ann
        elif inner_origin is Union:
            for a in get_args(inner_ann):
                if inspect.isclass(a) and issubclass(a, Enum):
                    inner_enum = a
                    break
        if inner_enum is not None:
            items = raw_val if isinstance(raw_val, (list, tuple, set)) else [raw_val]
            coerced_list: List[Any] = []
            seen_vals: Set[Any] = set()
            for item in items:
                c_item = _coerce_single_enum_value(item, inner_enum)
                if c_item is not None and c_item not in seen_vals:
                    seen_vals.add(c_item)
                    coerced_list.append(c_item)
            return coerced_list

    return raw_val


def _sp5_coerce_enum_fields_impl(cls: Type[BaseModel], data: Any) -> Any:
    """Before-mode model validator implementation attached to dynamically loaded Stage 5 schemas."""
    if not isinstance(data, dict) or not hasattr(cls, "model_fields"):
        return data
    out = dict(data)
    for field_name, field_info in cls.model_fields.items():
        if field_name in out:
            out[field_name] = _coerce_value_for_annotation(out[field_name], field_info.annotation)
    return out


def resolve_schema_path_with_fallbacks(
    schema_path: Union[str, Path],
    outputs_dir: Union[str, Path] = "outputs",
) -> Path:
    """
    Resolves `schema_path` with smart fallback checks across:
      - `outputs/04_synthesis/schemas/community/`
      - `outputs/04_synthesis/schemas/`
      - Repository root
    If a consolidated single file (`master_ontology.py` or `comprehensive_ontology.py`) is
    requested but does not exist yet while individual community `.py` schemas exist in
    `outputs/04_synthesis/schemas/community/`, it automatically falls back to the `community/` directory.
    """
    p = Path(schema_path)
    if p.exists():
        return p

    repo_root = Path(__file__).resolve().parent.parent.parent
    out_base = Path(outputs_dir)
    alt_candidates: List[Path] = [
        repo_root / p,
        out_base / "04_synthesis" / "schemas" / p.name,
        out_base / "04_synthesis" / "schemas" / "community" / p.name,
        repo_root / p.name,
    ]
    if p.suffix == ".py":
        alt_candidates.extend(
            [
                p.parent / "community" / p.name,
                p.parent.parent / p.name if p.parent.name == "community" else p.parent / p.name,
            ]
        )
        # Fallback to community directory if consolidated ontology file wasn't generated
        if p.name in CONSOLIDATED_ONTOLOGY_FILENAMES:
            alt_candidates.extend(
                [
                    out_base / "04_synthesis" / "schemas" / "community",
                    repo_root / "outputs" / "04_synthesis" / "schemas" / "community",
                ]
            )

    for cand in alt_candidates:
        if cand.exists():
            _log_discovery(f"Resolved '{schema_path}' -> '{cand}' via fallback lookup.")
            return cand

    raise FileNotFoundError(f"Schema path does not exist: {schema_path}")


# Backward-compatible alias
resolve_schema_path_With_fallbacks = resolve_schema_path_with_fallbacks


def resolve_schema_files(
    schema_path: Union[str, Path],
    outputs_dir: Union[str, Path] = "outputs",
) -> List[Path]:
    """
    Resolves `schema_path` into a list of `.py` files:
      - Option 1 (Single `.py` file): returns `[file_path]`.
      - Option 2 (Directory of `.py` files): returns sorted `.py` files in the directory
        (excluding `__init__.py` and private `_*.py` files). If individual community `.py`
        files exist alongside `master_ontology.py` or `comprehensive_ontology.py`, the
        consolidated files are automatically skipped to prevent duplicate extraction.
    """
    p = resolve_schema_path_with_fallbacks(schema_path, outputs_dir=outputs_dir)

    if p.is_file():
        if p.suffix != ".py":
            raise ValueError(f"Expected a .py file, got: {p}")
        return [p]

    if p.is_dir():
        py_files = sorted(
            f for f in p.glob("*.py")
            if f.is_file() and not f.name.startswith("_")
        )
        if not py_files:
            comm_dir = p / "community"
            if comm_dir.is_dir():
                py_files = sorted(
                    f for f in comm_dir.glob("*.py")
                    if f.is_file() and not f.name.startswith("_")
                )
        if not py_files:
            raise ValueError(f"No valid .py schema files found in directory: {p}")

        # Smart directory exclusion guard: if non-consolidated .py files exist in this directory,
        # exclude master_ontology.py and comprehensive_ontology.py to avoid redundant extraction.
        non_consolidated = [
            f for f in py_files if f.name not in CONSOLIDATED_ONTOLOGY_FILENAMES
        ]
        if non_consolidated and len(non_consolidated) < len(py_files):
            skipped = [f.name for f in py_files if f.name in CONSOLIDATED_ONTOLOGY_FILENAMES]
            _log_discovery(
                f"Directory mode detected {len(non_consolidated)} individual schema file(s); "
                f"skipping consolidated ontology file(s): {skipped}"
            )
            return non_consolidated

        return py_files

    raise ValueError(f"Invalid schema path: {p}")


def load_schemas_from_file(py_file: Path) -> List[SchemaMeta]:
    """
    Dynamically imports a Python module and inspects all Pydantic BaseModel subclasses,
    classifying each as `is_top_level` (root entity) or `is_child` (referenced inside another BaseModel).
    """
    module_name = py_file.stem
    unique_mod_key = f"_sp5_schema_mod_{module_name}_{abs(hash(str(py_file.resolve())))}"
    spec = importlib.util.spec_from_file_location(unique_mod_key, py_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {py_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_mod_key] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        logger.error(f"Error executing schema module {py_file}: {e}")
        raise

    defined_models: Dict[str, Type[BaseModel]] = {}
    module_namespace: Dict[str, Any] = dict(vars(module))
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseModel) and obj is not BaseModel:
            if getattr(obj, "__module__", None) == unique_mod_key:
                defined_models[name] = obj

    if not defined_models:
        logger.warning(f"No Pydantic BaseModel subclasses found in {py_file}")
        return []

    # Attach resilient Enum pre-validator and rebuild models across two passes so forward references resolve cleanly
    for cls in defined_models.values():
        try:
            setattr(cls, "_sp5_coerce_enum_fields", classmethod(_sp5_coerce_enum_fields_impl))
            cls.__pydantic_decorators__.model_validators["_sp5_coerce_enum_fields"] = Decorator.build(
                cls,
                cls_var_name="_sp5_coerce_enum_fields",
                shim=None,
                info=ModelValidatorDecoratorInfo(mode="before"),
            )
        except Exception:
            pass

    for _ in range(2):
        for cls in defined_models.values():
            try:
                cls.model_rebuild(_types_namespace=module_namespace, force=True)
            except Exception:
                pass

    # Analyze field references to build parent -> child dependency graph
    referenced_by_others: Set[str] = set()
    model_references: Dict[str, Set[str]] = {}

    for name, cls in defined_models.items():
        refs: Set[str] = set()
        for field_name, field_info in cls.model_fields.items():
            ref_classes = extract_referenced_models(field_info.annotation)
            for ref_cls in ref_classes:
                if ref_cls.__name__ in defined_models and ref_cls.__name__ != name:
                    refs.add(ref_cls.__name__)
                    referenced_by_others.add(ref_cls.__name__)
        model_references[name] = refs

    schemas: List[SchemaMeta] = []
    for name, cls in defined_models.items():
        is_child = name in referenced_by_others
        is_top_level = not is_child
        schemas.append(
            SchemaMeta(
                name=name,
                schema_cls=cls,
                source_file=py_file,
                module_name=module_name,
                is_top_level=is_top_level,
                is_child=is_child,
                referenced_models=model_references[name],
            )
        )

    return schemas


def load_schemas(
    schema_path: Union[str, Path],
    target_class: Optional[str] = None,
    outputs_dir: Union[str, Path] = "outputs",
) -> List[SchemaMeta]:
    """
    Loads all schemas from a single `.py` file or directory of `.py` files.
    Optionally filters by a specific `target_class` name.
    """
    py_files = resolve_schema_files(schema_path, outputs_dir=outputs_dir)
    all_schemas: List[SchemaMeta] = []

    for py_file in py_files:
        try:
            file_schemas = load_schemas_from_file(py_file)
            all_schemas.extend(file_schemas)
        except Exception as e:
            logger.error(f"Failed to load schemas from {py_file}: {e}")

    if target_class:
        filtered = [s for s in all_schemas if s.name == target_class]
        if not filtered:
            available = ", ".join(s.name for s in all_schemas)
            raise ValueError(
                f"Target class '{target_class}' not found in {schema_path}. "
                f"Available classes: [{available}]"
            )
        return filtered

    return all_schemas


def get_top_level_schemas(schemas: List[SchemaMeta]) -> List[SchemaMeta]:
    """Returns only top-level (root) schemas that are not nested exclusively inside other schemas."""
    return [s for s in schemas if s.is_top_level]


def get_child_schemas(schemas: List[SchemaMeta]) -> List[SchemaMeta]:
    """Returns only child (sub-component) schemas referenced by other schemas."""
    return [s for s in schemas if s.is_child]
