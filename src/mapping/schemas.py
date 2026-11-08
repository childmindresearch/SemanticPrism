"""
Stage 5 Phase 1: Dynamic Schema Discovery, DAG Classification & Triage Models.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import importlib.util
import inspect
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Type, Union, get_args, get_origin

from pydantic import BaseModel, Field

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
        out_base / "04_synthesis" / "schemas" / "community" / p.name,
        out_base / "04_synthesis" / "schemas" / p.name,
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

    # Rebuild models across two passes so forward references resolve cleanly
    for _ in range(2):
        for cls in defined_models.values():
            try:
                cls.model_rebuild(_types_namespace=module_namespace)
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
