"""
Domain-Agnostic Structural AST Sanitizers & Deduplication Helpers for Stage 4
=============================================================================
"""

from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime
import keyword
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

try:
    import tiktoken
    _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")

    def estimate_tokens(text: str) -> int:
        """Counts exact tokens using tiktoken (cl100k_base)."""
        if not text:
            return 0
        return len(_TIKTOKEN_ENCODER.encode(text))
except ImportError:
    def estimate_tokens(text: str) -> int:
        """Heuristic-based estimation: 1 word ~ 1.3 tokens."""
        if not text:
            return 0
        return int(len(text.split()) * 1.3)


import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import configs.AST_toggle as ast_cfg

# Module-level configurable defaults (synced with configs/AST_toggle.py)
DEFAULT_MAX_ENUM_MEMBERS_PER_CLASS: int = ast_cfg.MAX_ENUM_MEMBERS_PER_CLASS
DEFAULT_MAX_BASEMODEL_FIELDS_PER_CLASS: int = ast_cfg.MAX_BASEMODEL_FIELDS_PER_CLASS
DEFAULT_OUTPUTS_DIR: str = "outputs"
DEFAULT_PRINT_CHECKPOINTS: bool = True


def configure_sanitizer_defaults(
    max_enum_members_per_class: Optional[int] = None,
    max_basemodel_fields_per_class: Optional[int] = None,
    outputs_dir: str = "outputs",
    print_checkpoints: bool = True,
) -> None:
    """Updates module-level sanitizer settings to match runtime configuration."""
    global DEFAULT_MAX_ENUM_MEMBERS_PER_CLASS
    global DEFAULT_MAX_BASEMODEL_FIELDS_PER_CLASS
    global DEFAULT_OUTPUTS_DIR
    global DEFAULT_PRINT_CHECKPOINTS
    if max_enum_members_per_class is not None:
        ast_cfg.MAX_ENUM_MEMBERS_PER_CLASS = max_enum_members_per_class
    if max_basemodel_fields_per_class is not None:
        ast_cfg.MAX_BASEMODEL_FIELDS_PER_CLASS = max_basemodel_fields_per_class
    DEFAULT_MAX_ENUM_MEMBERS_PER_CLASS = ast_cfg.MAX_ENUM_MEMBERS_PER_CLASS
    DEFAULT_MAX_BASEMODEL_FIELDS_PER_CLASS = ast_cfg.MAX_BASEMODEL_FIELDS_PER_CLASS
    DEFAULT_OUTPUTS_DIR = outputs_dir
    DEFAULT_PRINT_CHECKPOINTS = print_checkpoints


def _log_sanitization_detail(
    context_label: str,
    class_name: str,
    class_kind: str,
    kept_count: int,
    stripped_items: List[str],
    collapsed_items: Optional[List[str]] = None,
    extra_note: Optional[str] = None,
    outputs_dir: Optional[str] = None,
    print_checkpoints: Optional[bool] = None,
) -> None:
    """
    Logs detailed post-processing AST sanitization events (stripped or collapsed Enum members
    and BaseModel fields) to both the console and `outputs/04_synthesis/logs/schema_sanitization.log`.
    """
    if not stripped_items and not collapsed_items and not extra_note:
        return

    out_dir = outputs_dir if outputs_dir is not None else DEFAULT_OUTPUTS_DIR
    do_print = (
        (print_checkpoints if print_checkpoints is not None else DEFAULT_PRINT_CHECKPOINTS)
        and ast_cfg.PRINT_SANITIZATION_CHECKPOINTS
    )

    ts = datetime.now().strftime("%H:%M:%S")
    iso_ts = datetime.now().isoformat()
    header = (
        f"[{ts} | AST Sanitizer | {context_label}] {class_kind} '{class_name}' "
        f"(kept={kept_count}, stripped={len(stripped_items)}"
        f"{f', collapsed_groups={len(collapsed_items)}' if collapsed_items else ''})"
    )
    detail_lines: List[str] = [header]
    if extra_note:
        detail_lines.append(f"      * Note: {extra_note}")
    if collapsed_items:
        for c_item in collapsed_items:
            detail_lines.append(f"      ~ Collapsed: {c_item}")
    for s_item in stripped_items:
        detail_lines.append(f"      - Stripped : {s_item}")

    full_msg = "\n".join(detail_lines)
    if do_print:
        print(full_msg, flush=True)

    if ast_cfg.LOG_SANITIZATION_TO_FILE:
        try:
            log_dir = Path(out_dir) / "04_synthesis" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            audit_log_path = log_dir / "schema_sanitization.log"
            with open(audit_log_path, "a", encoding="utf-8") as lf:
                lf.write(f"[{iso_ts}] {full_msg}\n")
        except Exception:
            pass


def _is_verbose_or_runaway_enum_member(member_name: str, member_val: Optional[str] = None) -> bool:
    """
    Domain-agnostic structural check: returns True if an Enum member identifier or string value
    exceeds atomic token/character bounds configured in `configs/AST_toggle.py`.
    """
    max_words = ast_cfg.ENUM_MEMBER_NAME_MAX_WORDS
    max_chars = ast_cfg.ENUM_MEMBER_NAME_MAX_CHARS
    val_max_words = ast_cfg.ENUM_MEMBER_VALUE_MAX_WORDS
    val_max_chars = ast_cfg.ENUM_MEMBER_VALUE_MAX_CHARS
    ident = str(member_name).strip().upper()
    if not ident:
        return True
    words = [w for w in ident.split("_") if w]
    if len(words) > max_words or len(ident) > max_chars:
        return True
    if member_val is not None:
        val_str = str(member_val).strip()
        val_words = [w for w in re.split(r"[\s_-]+", val_str) if w]
        if len(val_words) > val_max_words or len(val_str) > val_max_chars:
            return True
    return False


def _sanitize_ast_enums(
    code: str,
    context_label: str = "schema",
    max_enum_members_per_class: Optional[int] = None,
) -> str:
    """
    Domain-agnostic AST post-processor for Enum classes controlled by `configs/AST_toggle.py`:
      1. Strips non-atomic Enum members (`STRIP_VERBOSE_ENUM_MEMBERS`).
      2. Deduplicates Enum member names (`STRIP_DUPLICATE_ENUM_NAMES`) and values (`STRIP_DUPLICATE_ENUM_VALUES`).
      3. Caps each Enum class at `MAX_ENUM_MEMBERS_PER_CLASS` (`ENFORCE_MAX_ENUM_MEMBERS_CAP`).
      4. Places `OTHER` / `'Other'` last (`PLACE_ENUM_OTHER_LAST`).
      5. Removes empty Enums and rewrites referencing BaseModel fields to `str` (`REMOVE_EMPTY_ENUMS_AND_REWRITE_REFS`).
    """
    if not ast_cfg.ENABLE_ENUM_AST_SANITIZER:
        return code

    max_members = (
        max_enum_members_per_class
        if max_enum_members_per_class is not None
        else ast_cfg.MAX_ENUM_MEMBERS_PER_CLASS
    )
    try:
        tree = ast.parse(code)
    except Exception:
        return code

    def is_enum_class(node: ast.ClassDef) -> bool:
        for b in node.bases:
            if isinstance(b, ast.Name) and "Enum" in b.id:
                return True
            if isinstance(b, ast.Attribute) and "Enum" in b.attr:
                return True
        return False

    new_body: List[ast.AST] = []
    dropped_enum_names: set = set()
    modified = False

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and is_enum_class(node):
            kept_stmts: List[ast.stmt] = []
            seen_names: set = set()
            seen_vals: set = set()
            member_count = 0
            stripped_details: List[str] = []

            for stmt in node.body:
                m_name = None
                m_val = None
                if isinstance(stmt, ast.Assign) and stmt.targets and isinstance(stmt.targets[0], ast.Name):
                    m_name = stmt.targets[0].id
                    if isinstance(stmt.value, ast.Constant):
                        m_val = stmt.value.value
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    m_name = stmt.target.id
                    if stmt.value and isinstance(stmt.value, ast.Constant):
                        m_val = stmt.value.value

                if m_name is not None:
                    reason: Optional[str] = None
                    if ast_cfg.STRIP_VERBOSE_ENUM_MEMBERS and _is_verbose_or_runaway_enum_member(m_name, m_val):
                        reason = (
                            f"verbose/non-atomic member (>{ast_cfg.ENUM_MEMBER_NAME_MAX_WORDS} words "
                            f"or >{ast_cfg.ENUM_MEMBER_NAME_MAX_CHARS} chars)"
                        )
                    elif ast_cfg.STRIP_DUPLICATE_ENUM_NAMES and m_name in seen_names:
                        reason = "duplicate member name"
                    elif (
                        ast_cfg.STRIP_DUPLICATE_ENUM_VALUES
                        and m_val is not None
                        and str(m_val).lower().strip() in seen_vals
                    ):
                        reason = "duplicate member value"
                    elif ast_cfg.ENFORCE_MAX_ENUM_MEMBERS_CAP and member_count >= max_members:
                        reason = f"exceeded MAX_ENUM_MEMBERS_PER_CLASS ({max_members})"

                    if reason is not None:
                        modified = True
                        stripped_details.append(f"{m_name} = {m_val!r} [reason: {reason}]")
                        continue

                    seen_names.add(m_name)
                    if m_val is not None:
                        seen_vals.add(str(m_val).lower().strip())
                    member_count += 1
                    kept_stmts.append(stmt)
                else:
                    kept_stmts.append(stmt)

            if member_count > 0:
                if ast_cfg.PLACE_ENUM_OTHER_LAST:
                    non_field_stmts: List[ast.stmt] = []
                    regular_member_stmts: List[ast.stmt] = []
                    other_member_stmts: List[ast.stmt] = []
                    for k_stmt in kept_stmts:
                        k_name = None
                        k_val = None
                        if isinstance(k_stmt, ast.Assign) and k_stmt.targets and isinstance(k_stmt.targets[0], ast.Name):
                            k_name = k_stmt.targets[0].id
                            if isinstance(k_stmt.value, ast.Constant):
                                k_val = k_stmt.value.value
                        elif isinstance(k_stmt, ast.AnnAssign) and isinstance(k_stmt.target, ast.Name):
                            k_name = k_stmt.target.id
                            if k_stmt.value and isinstance(k_stmt.value, ast.Constant):
                                k_val = k_stmt.value.value
                        if k_name is None:
                            non_field_stmts.append(k_stmt)
                        elif k_name.strip().upper() == "OTHER" or (k_val is not None and str(k_val).strip().lower() == "other"):
                            other_member_stmts.append(k_stmt)
                        else:
                            regular_member_stmts.append(k_stmt)
                    reordered_stmts = non_field_stmts + regular_member_stmts + other_member_stmts
                    if reordered_stmts != kept_stmts:
                        modified = True
                    node.body = reordered_stmts
                else:
                    node.body = kept_stmts
                new_body.append(node)
                if stripped_details:
                    _log_sanitization_detail(
                        context_label=context_label,
                        class_name=node.name,
                        class_kind="Enum",
                        kept_count=member_count,
                        stripped_items=stripped_details,
                    )
            elif ast_cfg.REMOVE_EMPTY_ENUMS_AND_REWRITE_REFS:
                dropped_enum_names.add(node.name)
                modified = True
                _log_sanitization_detail(
                    context_label=context_label,
                    class_name=node.name,
                    class_kind="Enum",
                    kept_count=0,
                    stripped_items=stripped_details,
                    extra_note="Enum class had 0 valid members remaining and was removed (field references rewritten to 'str').",
                )
            else:
                node.body = kept_stmts + [ast.Pass()]
                modified = True
                new_body.append(node)
        else:
            new_body.append(node)

    if dropped_enum_names and ast_cfg.REMOVE_EMPTY_ENUMS_AND_REWRITE_REFS:
        class _DroppedEnumRewriter(ast.NodeTransformer):
            def visit_Name(self, n: ast.Name) -> ast.AST:
                if n.id in dropped_enum_names and isinstance(n.ctx, ast.Load):
                    return ast.copy_location(ast.Name(id="str", ctx=ast.Load()), n)
                return n

        rewriter = _DroppedEnumRewriter()
        new_body = [rewriter.visit(n) for n in new_body]

    if not modified:
        return code

    tree.body = new_body
    try:
        return ast.unparse(tree)
    except Exception:
        return code


def _sanitize_ast_models(
    code: str,
    context_label: str = "schema",
    max_basemodel_fields_per_class: Optional[int] = None,
) -> str:
    """
    Domain-agnostic AST post-processor for Pydantic BaseModel classes that:
      1. Renames fields shadowing reserved Pydantic BaseModel attributes or Python keywords.
      2. Rewrites undefined/dangling non-builtin type annotations to `str`.
      3. Deduplicates fields within each BaseModel.
      4. Caps/collapses shared 2-token snake_case prefixes (prevents flattened sub-entity attribute dumps).
      5. Caps/collapses shared non-generic suffix tokens.
      6. Enforces MAX_BASEMODEL_FIELDS_PER_CLASS per BaseModel, prioritizing core/typed/collection fields.
      7. Guarantees every BaseModel has a descriptive triple-quoted docstring (\"\"\"...\"\"\").
      8. Ensures `from __future__ import annotations` and canonical top-level ordering (Imports -> Enums -> Topological BaseModels).
    """
    if not getattr(ast_cfg, "ENABLE_BASEMODEL_AST_SANITIZER", True):
        return code

    max_fields = (
        max_basemodel_fields_per_class
        if max_basemodel_fields_per_class is not None
        else getattr(ast_cfg, "MAX_BASEMODEL_FIELDS_PER_CLASS", DEFAULT_MAX_BASEMODEL_FIELDS_PER_CLASS)
    )
    max_shared_prefix = getattr(ast_cfg, "MAX_SHARED_PREFIX_FIELDS", 2)
    max_shared_suffix = getattr(ast_cfg, "MAX_SHARED_SUFFIX_FIELDS", 2)
    prefix_min_tokens = getattr(ast_cfg, "PREFIX_MIN_TOKENS", 3)
    prefix_collapse_threshold = getattr(ast_cfg, "SHARED_PREFIX_COLLAPSE_THRESHOLD", 3)
    suffix_min_tokens = getattr(ast_cfg, "SUFFIX_MIN_TOKENS", 2)
    suffix_collapse_threshold = getattr(ast_cfg, "SHARED_SUFFIX_COLLAPSE_THRESHOLD", 3)
    try:
        tree = ast.parse(code)
    except Exception:
        return code

    def is_enum_class(node: ast.ClassDef) -> bool:
        for b in node.bases:
            if isinstance(b, ast.Name) and "Enum" in b.id:
                return True
            if isinstance(b, ast.Attribute) and "Enum" in b.attr:
                return True
        return False

    exempt_suffixes = set(
        getattr(
            ast_cfg,
            "EXEMPT_FIELD_SUFFIXES",
            (
                "id", "name", "type", "types", "status", "level", "score", "range",
                "date", "time", "plan", "notes", "summary", "description", "details",
                "count", "rate", "list", "items", "category", "mode", "source",
            ),
        )
    )

    pydantic_reserved_attrs = {
        "schema", "json", "copy", "dict", "validate", "fields",
        "model_config", "model_fields", "model_computed_fields",
        "model_extra", "model_fields_set", "model_dump",
        "model_dump_json", "model_validate", "model_json_schema",
        "model_construct", "model_copy", "model_rebuild",
        "update_forward_refs", "parse_obj", "parse_raw",
        "parse_file", "from_orm", "construct",
    }
    python_keywords = set(keyword.kwlist) | {"async", "await"}

    # Collect all defined or imported type names to guard against dangling type references
    known_type_names: set = {
        "str", "int", "float", "bool", "bytes", "list", "dict", "set", "tuple",
        "object", "None", "Any", "Optional", "List", "Dict", "Set", "Tuple",
        "Union", "Literal", "Callable", "Sequence", "Mapping", "Iterable",
        "Type", "ClassVar", "Final", "Annotated", "BaseModel", "Field", "Enum",
        "date", "datetime", "time", "timedelta", "UUID", "Decimal", "Path",
    }
    for top_node in tree.body:
        if isinstance(top_node, ast.ClassDef):
            known_type_names.add(top_node.name)
        elif isinstance(top_node, ast.Import):
            for alias in top_node.names:
                known_type_names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(top_node, ast.ImportFrom):
            for alias in top_node.names:
                known_type_names.add(alias.asname or alias.name)
        elif isinstance(top_node, ast.Assign):
            for t in top_node.targets:
                if isinstance(t, ast.Name):
                    known_type_names.add(t.id)

    class _DanglingTypeRewriter(ast.NodeTransformer):
        def __init__(self) -> None:
            self.replaced: List[str] = []

        def visit_Subscript(self, sub_node: ast.Subscript) -> ast.AST:
            # Do not rewrite string/identifier arguments inside Literal[...]
            if isinstance(sub_node.value, ast.Name) and sub_node.value.id == "Literal":
                return sub_node
            if isinstance(sub_node.value, ast.Attribute) and sub_node.value.attr == "Literal":
                return sub_node
            return self.generic_visit(sub_node)

        def visit_Name(self, name_node: ast.Name) -> ast.AST:
            if isinstance(name_node.ctx, ast.Load) and name_node.id not in known_type_names:
                self.replaced.append(name_node.id)
                return ast.copy_location(ast.Name(id="str", ctx=ast.Load()), name_node)
            return name_node

    modified = False
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or is_enum_class(node):
            continue

        non_field_prefix_stmts: List[ast.stmt] = []
        non_field_suffix_stmts: List[ast.stmt] = []
        field_entries: List[Tuple[int, str, ast.stmt, bool, List[str], str]] = []
        seen_fields: set = set()
        stripped_details: List[str] = []
        collapsed_details: List[str] = []

        for idx, stmt in enumerate(node.body):
            f_name: Optional[str] = None
            ann_str = ""
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                f_name = stmt.target.id
                if getattr(ast_cfg, "REWRITE_DANGLING_TYPE_REFERENCES", True):
                    # Rewrite any dangling/undefined type references in field annotation to `str`
                    rewriter = _DanglingTypeRewriter()
                    stmt.annotation = rewriter.visit(stmt.annotation)
                    if rewriter.replaced:
                        modified = True
                        try:
                            ann_str = ast.unparse(stmt.annotation)
                        except Exception:
                            ann_str = ""
                        collapsed_details.append(
                            f"'{f_name}' type annotation -> '{ann_str}' "
                            f"[reason: rewrote undefined type reference(s) {sorted(set(rewriter.replaced))} to 'str']"
                        )
                if not ann_str:
                    try:
                        ann_str = ast.unparse(stmt.annotation)
                    except Exception:
                        ann_str = ""
            elif isinstance(stmt, ast.Assign) and stmt.targets and isinstance(stmt.targets[0], ast.Name):
                f_name = stmt.targets[0].id

            if f_name is None:
                if not field_entries:
                    non_field_prefix_stmts.append(stmt)
                else:
                    non_field_suffix_stmts.append(stmt)
                continue

            norm_name = f_name.strip().lower()
            # Rename fields that shadow Pydantic BaseModel attributes or Python keywords
            if getattr(ast_cfg, "RENAME_SHADOWED_BASEMODEL_ATTRIBUTES", True) and (
                norm_name in pydantic_reserved_attrs or norm_name in python_keywords
            ):
                new_f_name = f"{norm_name}_field"
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    stmt.target.id = new_f_name
                elif isinstance(stmt, ast.Assign) and stmt.targets and isinstance(stmt.targets[0], ast.Name):
                    stmt.targets[0].id = new_f_name
                modified = True
                collapsed_details.append(
                    f"'{f_name}' -> '{new_f_name}' [reason: renamed field shadowing reserved keyword/BaseModel attribute]"
                )
                f_name = new_f_name
                norm_name = new_f_name

            if getattr(ast_cfg, "STRIP_DUPLICATE_BASEMODEL_FIELDS", True) and norm_name in seen_fields:
                modified = True
                stripped_details.append(f"{f_name}: {ann_str or 'Any'} [reason: duplicate field name]")
                continue
            seen_fields.add(norm_name)

            is_bool = bool(re.search(r"\bbool\b", ann_str))
            parts = [p for p in norm_name.split("_") if p]
            field_entries.append((idx, norm_name, stmt, is_bool, parts, ann_str or "Any"))

        if not field_entries:
            continue

        # Pass 1: Structural prefix-entropy & suffix-repetition filtering
        raw_prefix2_counts: Counter = Counter(
            f"{e[4][0]}_{e[4][1]}" for e in field_entries if len(e[4]) >= prefix_min_tokens
        )
        raw_suffix_counts: Counter = Counter(
            e[4][-1] for e in field_entries if len(e[4]) >= suffix_min_tokens
        )
        prefix2_counts: Counter = Counter()
        suffix_counts: Counter = Counter()
        filtered_entries: List[Tuple[int, str, ast.stmt, bool, List[str], str]] = []

        for entry in field_entries:
            idx, norm_name, stmt, is_bool, parts, ann_str = entry

            # 1a. Shared 2-token prefix collapse/cap (collapses 3+ flattened sub-entity fields into a List[str] field)
            if len(parts) >= prefix_min_tokens:
                prefix2 = f"{parts[0]}_{parts[1]}"
                if getattr(ast_cfg, "COLLAPSE_SHARED_PREFIX_FIELDS", True) and raw_prefix2_counts[prefix2] >= prefix_collapse_threshold:
                    modified = True
                    if prefix2_counts[prefix2] == 0:
                        collapsed_name = prefix2 if prefix2.endswith("s") else f"{prefix2}s"
                        grouped_names = [
                            e[1] for e in field_entries if len(e[4]) >= prefix_min_tokens and f"{e[4][0]}_{e[4][1]}" == prefix2
                        ]
                        if collapsed_name not in seen_fields:
                            seen_fields.add(collapsed_name)
                            human_label = collapsed_name.replace("_", " ")
                            stmt = ast.parse(
                                f'{collapsed_name}: Optional[List[str]] = Field(default=None, description="Associated {human_label}.")'
                            ).body[0]
                            norm_name = collapsed_name
                            is_bool = False
                            parts = [p for p in collapsed_name.split("_") if p]
                            ann_str = "Optional[List[str]]"
                        prefix2_counts[prefix2] += 1
                        filtered_entries.append((idx, norm_name, stmt, is_bool, parts, ann_str))
                        collapsed_details.append(
                            f"{grouped_names} -> '{norm_name}: Optional[List[str]]' "
                            f"[reason: collapsed {len(grouped_names)} fields sharing prefix '{prefix2}_*']"
                        )
                    continue
                if getattr(ast_cfg, "ENFORCE_SHARED_PREFIX_CAP", True) and prefix2_counts[prefix2] >= max_shared_prefix:
                    modified = True
                    stripped_details.append(
                        f"{norm_name}: {ann_str} [reason: exceeded shared prefix cap ({max_shared_prefix}) for '{prefix2}_*']"
                    )
                    continue
            else:
                prefix2 = None

            # 1b. Shared suffix collapse/cap (collapses 3+ repeated suffix/checklist fields into a List[str] field)
            suffix1 = parts[-1] if len(parts) >= suffix_min_tokens else None
            if suffix1 and suffix1 not in exempt_suffixes:
                if getattr(ast_cfg, "COLLAPSE_SHARED_SUFFIX_FIELDS", True) and raw_suffix_counts[suffix1] >= suffix_collapse_threshold:
                    modified = True
                    if suffix_counts[suffix1] == 0:
                        collapsed_name = suffix1 if suffix1.endswith("s") else f"{suffix1}s"
                        grouped_names = [
                            e[1] for e in field_entries if len(e[4]) >= suffix_min_tokens and e[4][-1] == suffix1
                        ]
                        if collapsed_name not in seen_fields:
                            seen_fields.add(collapsed_name)
                            human_label = collapsed_name.replace("_", " ")
                            stmt = ast.parse(
                                f'{collapsed_name}: Optional[List[str]] = Field(default=None, description="Associated {human_label}.")'
                            ).body[0]
                            norm_name = collapsed_name
                            is_bool = False
                            parts = [p for p in collapsed_name.split("_") if p]
                            ann_str = "Optional[List[str]]"
                        suffix_counts[suffix1] += 1
                        filtered_entries.append((idx, norm_name, stmt, is_bool, parts, ann_str))
                        collapsed_details.append(
                            f"{grouped_names} -> '{norm_name}: Optional[List[str]]' "
                            f"[reason: collapsed {len(grouped_names)} fields sharing suffix '*_{suffix1}']"
                        )
                    continue
                if getattr(ast_cfg, "ENFORCE_SHARED_SUFFIX_CAP", True) and suffix_counts[suffix1] >= max_shared_suffix:
                    modified = True
                    stripped_details.append(
                        f"{norm_name}: {ann_str} [reason: exceeded shared suffix cap ({max_shared_suffix}) for '*_{suffix1}']"
                    )
                    continue

            if prefix2:
                prefix2_counts[prefix2] += 1
            if suffix1 and suffix1 not in exempt_suffixes:
                suffix_counts[suffix1] += 1

            filtered_entries.append((idx, norm_name, stmt, is_bool, parts, ann_str))

        # Pass 2: Enforce hard cap of MAX_BASEMODEL_FIELDS_PER_CLASS
        if getattr(ast_cfg, "ENFORCE_MAX_BASEMODEL_FIELDS_CAP", True) and len(filtered_entries) > max_fields:
            modified = True
            scored = sorted(
                filtered_entries,
                key=lambda item: (
                    1 if item[3] else 0,                 # non-bool before bool
                    1 if (len(item[4]) >= 3 and raw_prefix2_counts[f"{item[4][0]}_{item[4][1]}"] > 1) else 0,
                    1 if len(item[4]) >= 3 else 0,       # 1-2 token core fields before 3+ token compound fields
                    item[0],                             # original declaration order
                ),
            )
            kept_top = scored[:max_fields]
            dropped_overflow = sorted(scored[max_fields:], key=lambda item: item[0])
            for d_item in dropped_overflow:
                stripped_details.append(
                    f"{d_item[1]}: {d_item[5]} [reason: exceeded MAX_BASEMODEL_FIELDS_PER_CLASS ({max_fields})]"
                )
            kept_top.sort(key=lambda item: item[0])
            filtered_entries = kept_top

        # Guarantee every BaseModel class begins with a descriptive triple-quoted docstring
        has_docstring = (
            bool(non_field_prefix_stmts)
            and isinstance(non_field_prefix_stmts[0], ast.Expr)
            and isinstance(non_field_prefix_stmts[0].value, ast.Constant)
            and isinstance(non_field_prefix_stmts[0].value.value, str)
            and bool(str(non_field_prefix_stmts[0].value.value).strip())
        )
        if getattr(ast_cfg, "ENSURE_BASEMODEL_DOCSTRING", True) and not has_docstring:
            human_cls = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", node.name).strip()
            sample_fields = [e[1].replace("_", " ") for e in filtered_entries[:4]]
            if sample_fields:
                synth_doc = f"Represents {human_cls} domain attributes, including {', '.join(sample_fields)}."
            else:
                synth_doc = f"Represents {human_cls} domain entity and its associated attributes."
            non_field_prefix_stmts.insert(0, ast.Expr(value=ast.Constant(value=synth_doc)))
            modified = True

        if stripped_details or collapsed_details:
            _log_sanitization_detail(
                context_label=context_label,
                class_name=node.name,
                class_kind="BaseModel",
                kept_count=len(filtered_entries),
                stripped_items=stripped_details,
                collapsed_items=collapsed_details,
            )

        if filtered_entries:
            node.body = non_field_prefix_stmts + [e[2] for e in filtered_entries] + non_field_suffix_stmts
        elif getattr(ast_cfg, "ENSURE_EMPTY_CLASS_PASS_FALLBACK", True):
            node.body = non_field_prefix_stmts + [ast.Pass()] + non_field_suffix_stmts
            modified = True

    if getattr(ast_cfg, "CANONICAL_TOPOLOGICAL_CLASS_ORDERING", True):
        # Canonical top-level ordering:
        # (__future__ (if any) -> standard imports -> constants/aliases -> Enums -> topologically ordered BaseModels)
        future_nodes: List[ast.AST] = []
        import_nodes: List[ast.AST] = []
        enum_nodes: List[ast.ClassDef] = []
        model_nodes: List[ast.ClassDef] = []
        other_top_nodes: List[ast.AST] = []

        for top_node in tree.body:
            if isinstance(top_node, ast.ImportFrom) and top_node.module == "__future__":
                future_nodes.append(top_node)
            elif isinstance(top_node, (ast.Import, ast.ImportFrom)):
                import_nodes.append(top_node)
            elif isinstance(top_node, ast.ClassDef):
                if is_enum_class(top_node):
                    enum_nodes.append(top_node)
                else:
                    model_nodes.append(top_node)
            else:
                other_top_nodes.append(top_node)

        # Topologically order BaseModel classes so referenced child BaseModels precede parent BaseModels
        if len(model_nodes) >= 1:
            model_map = {m.name: m for m in model_nodes}
            deps_by_model: Dict[str, List[str]] = {}
            for m in model_nodes:
                refs: List[str] = []
                for st in m.body:
                    if isinstance(st, ast.AnnAssign):
                        for sub in ast.walk(st.annotation):
                            if isinstance(sub, ast.Name) and sub.id in model_map and sub.id != m.name and sub.id not in refs:
                                refs.append(sub.id)
                deps_by_model[m.name] = refs

            ordered_models: List[ast.ClassDef] = []
            visited_models: set = set()
            visiting_models: set = set()

            def _visit_model(m_name: str) -> None:
                if m_name in visited_models or m_name in visiting_models:
                    return
                visiting_models.add(m_name)
                for dep in deps_by_model.get(m_name, []):
                    if dep in model_map:
                        _visit_model(dep)
                visiting_models.remove(m_name)
                visited_models.add(m_name)
                ordered_models.append(model_map[m_name])

            for m in model_nodes:
                _visit_model(m.name)
            if [m.name for m in ordered_models] != [m.name for m in model_nodes]:
                modified = True
            model_nodes = ordered_models

            # Quote any remaining self-reference or cyclic forward reference so class creation never raises NameError
            if getattr(ast_cfg, "QUOTE_CYCLIC_FORWARD_REFERENCES", True):
                defined_so_far: set = set()
                for m in model_nodes:
                    class _ForwardRefQuoter(ast.NodeTransformer):
                        def __init__(self, current_cls: str, ready: set) -> None:
                            self.current_cls = current_cls
                            self.ready = ready
                            self.quoted = False

                        def visit_Name(self, n: ast.Name) -> ast.AST:
                            if isinstance(n.ctx, ast.Load) and n.id in model_map and (n.id == self.current_cls or n.id not in self.ready):
                                self.quoted = True
                                return ast.copy_location(ast.Constant(value=n.id), n)
                            return n

                    quoter = _ForwardRefQuoter(m.name, defined_so_far)
                    for st in m.body:
                        if isinstance(st, ast.AnnAssign):
                            st.annotation = quoter.visit(st.annotation)
                    if quoter.quoted:
                        modified = True
                    defined_so_far.add(m.name)

        new_tree_body = future_nodes + import_nodes + other_top_nodes + enum_nodes + model_nodes
        if new_tree_body != tree.body:
            tree.body = new_tree_body
            modified = True
    elif getattr(ast_cfg, "QUOTE_CYCLIC_FORWARD_REFERENCES", True):
        model_nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and not is_enum_class(n)]
        model_map = {m.name: m for m in model_nodes}
        defined_so_far = set()
        for m in model_nodes:
            class _ForwardRefQuoterOnly(ast.NodeTransformer):
                def __init__(self, current_cls: str, ready: set) -> None:
                    self.current_cls = current_cls
                    self.ready = ready
                    self.quoted = False

                def visit_Name(self, n: ast.Name) -> ast.AST:
                    if isinstance(n.ctx, ast.Load) and n.id in model_map and (n.id == self.current_cls or n.id not in self.ready):
                        self.quoted = True
                        return ast.copy_location(ast.Constant(value=n.id), n)
                    return n

            quoter = _ForwardRefQuoterOnly(m.name, defined_so_far)
            for st in m.body:
                if isinstance(st, ast.AnnAssign):
                    st.annotation = quoter.visit(st.annotation)
            if quoter.quoted:
                modified = True
            defined_so_far.add(m.name)

    if not modified and "'''" not in code:
        return code

    try:
        return ast.unparse(tree)
    except Exception:
        return code


def clean_python_code(code: str, context_label: str = "schema") -> str:
    """Removes markdown formatting, invalid characters, and escapes that LLMs often inject."""
    if not code:
        return ""
    code = code.strip()
    if getattr(ast_cfg, "STRIP_MARKDOWN_CODE_FENCES", True):
        fenced_py = re.search(r"```(?:python|py)?\s*([\s\S]*?)\s*```", code, flags=re.IGNORECASE)
        if fenced_py and ("class " in fenced_py.group(1) or "import " in fenced_py.group(1)):
            code = fenced_py.group(1).strip()
        else:
            if code.startswith("```python"):
                code = code[9:]
            elif code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]
            code = code.strip()

    if getattr(ast_cfg, "UNESCAPE_QUOTES_AND_NEWLINES", True):
        # Only unescape literal \n and \" if the code is single-line JSON-escaped or fails ast.parse as-is
        needs_unescape = "\n" not in code and "\\n" in code
        if not needs_unescape and ("\\n" in code or '\\"' in code):
            try:
                ast.parse(code)
            except SyntaxError:
                needs_unescape = True

        if needs_unescape:
            candidate_unescaped = code.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "    ").replace('\\"', '"')
            try:
                ast.parse(candidate_unescaped)
                code = candidate_unescaped
            except SyntaxError:
                # If only \n was escaped outside strings, try unescaping \n without breaking inner \"
                line_unescaped = code.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "    ")
                try:
                    ast.parse(line_unescaped)
                    code = line_unescaped
                except SyntaxError:
                    code = candidate_unescaped

    python_keywords = set(keyword.kwlist) | {"async", "await"}
    keyword_renames: List[str] = []
    fix_lhs = getattr(ast_cfg, "FIX_ASSIGNMENT_LHS_IDENTIFIERS", True)
    fix_hyphens = getattr(ast_cfg, "FIX_FIELD_HYPHENS", True)
    fix_keywords = getattr(ast_cfg, "RENAME_PYTHON_RESERVED_KEYWORDS", True)

    def fix_line(line: str) -> str:
        # 1. Assignment '=' fix
        if fix_lhs:
            m = re.match(r"^(\s+)([a-zA-Z0-9_, ]+?)(\s*=\s*.*)$", line)
            if m:
                indent, var, rest = m.groups()
                stripped_var = var.strip()
                if ("," in stripped_var or " " in stripped_var) and not stripped_var.startswith(
                    ("def ", "class ", "return ", "if ", "elif ")
                ):
                    clean_var = re.sub(r"[, ]+", "_", stripped_var)
                    line = f"{indent}{clean_var}{rest}"

        # 2. Type annotation ':' fix for hyphens & Python reserved keywords
        if fix_hyphens or fix_keywords:
            m_ann = re.match(r"^(\s+)([a-zA-Z_][a-zA-Z0-9_-]*)(\s*:\s*.*)$", line)
            if m_ann:
                indent, var, rest = m_ann.groups()
                clean_var = var.replace("-", "_") if (fix_hyphens and "-" in var) else var
                if fix_keywords and clean_var in python_keywords:
                    renamed_kw = f"{clean_var}_field"
                    keyword_renames.append(f"'{clean_var}' -> '{renamed_kw}' [reason: Python reserved keyword used as field name]")
                    clean_var = renamed_kw
                if clean_var != var:
                    line = f"{indent}{clean_var}{rest}"
        return line

    code = "\n".join(fix_line(line_str) for line_str in code.splitlines())
    if keyword_renames:
        _log_sanitization_detail(
            context_label=context_label,
            class_name="<module>",
            class_kind="KeywordFieldFix",
            kept_count=len(keyword_renames),
            stripped_items=[],
            collapsed_items=keyword_renames,
        )

    # Strip rogue CJK / non-ASCII character leakage from bilingual models in comments or docstrings
    if getattr(ast_cfg, "STRIP_CJK_CHARACTERS", True):
        code = re.sub(r"[\u4e00-\u9fff]+", "", code)

    # Strip any accidental meta-modeling classes that mirror the prompt/JSON wrapper
    if getattr(ast_cfg, "STRIP_META_MODEL_CLASSES", True):
        banned_classes = getattr(
            ast_cfg,
            "BANNED_META_MODEL_CLASSES",
            ("ThemeObservation", "ThematicObservation", "GeneratedModule"),
        )
        if banned_classes:
            cls_pattern = "|".join(re.escape(c) for c in banned_classes)
            meta_matches = re.findall(
                rf"(?m)^class\s+({cls_pattern})\s*\([^)]*\):",
                code,
            )
            if meta_matches:
                code = re.sub(
                    rf"(?m)^class\s+({cls_pattern})\s*\([^)]*\):.*?(?=(?:\nclass\s+|\n[A-Za-z0-9_]+\s*=|\Z))",
                    "",
                    code,
                    flags=re.DOTALL,
                )
                for m_cls in meta_matches:
                    _log_sanitization_detail(
                        context_label=context_label,
                        class_name=m_cls,
                        class_kind="MetaModel",
                        kept_count=0,
                        stripped_items=[f"class {m_cls} [reason: banned prompt-wrapper meta-model]"],
                        extra_note="Meta-model class mirroring prompt envelope was stripped completely.",
                    )
    code = _sanitize_ast_enums(code.strip(), context_label=context_label)
    code = _sanitize_ast_models(code.strip(), context_label=context_label)
    return code.strip()


def deduplicate_consolidated_classes(code: str, context_label: str = "master_ontology") -> str:
    """
    Programmatic AST post-processor for consolidated ontologies (master_ontology.py and comprehensive_ontology.py).
    Merges duplicate Enums and BaseModels, disambiguates Enum vs BaseModel naming collisions,
    and enforces MAX_ENUM_MEMBERS_PER_CLASS and MAX_BASEMODEL_FIELDS_PER_CLASS.
    """
    clean_code = clean_python_code(code, context_label=context_label)
    if not getattr(ast_cfg, "ENABLE_CONSOLIDATED_ONTOLOGY_MERGER", True):
        return clean_code
    try:
        tree = ast.parse(clean_code)
    except Exception:
        return clean_code

    def is_enum_class(node: ast.ClassDef) -> bool:
        for b in node.bases:
            if isinstance(b, ast.Name) and "Enum" in b.id:
                return True
            if isinstance(b, ast.Attribute) and "Enum" in b.attr:
                return True
        return False

    def get_assigned_name(stmt: ast.AST) -> Optional[str]:
        if isinstance(stmt, ast.Assign) and stmt.targets and isinstance(stmt.targets[0], ast.Name):
            return stmt.targets[0].id
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            return stmt.target.id
        return None

    enums_by_name: Dict[str, ast.ClassDef] = {}
    models_by_name: Dict[str, ast.ClassDef] = {}
    other_nodes: List[ast.AST] = []
    merge_enums = getattr(ast_cfg, "MERGE_DUPLICATE_ENUM_CLASSES", True)
    merge_models = getattr(ast_cfg, "MERGE_DUPLICATE_BASEMODEL_CLASSES", True)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            name = node.name
            if is_enum_class(node):
                if name not in enums_by_name:
                    enums_by_name[name] = node
                elif merge_enums:
                    existing_members = {
                        get_assigned_name(stmt)
                        for stmt in enums_by_name[name].body
                        if get_assigned_name(stmt) is not None
                    }
                    added_members, skipped_members = [], []
                    for stmt in node.body:
                        m_name = get_assigned_name(stmt)
                        if m_name:
                            if m_name not in existing_members:
                                enums_by_name[name].body.append(stmt)
                                existing_members.add(m_name)
                                added_members.append(m_name)
                            else:
                                skipped_members.append(f"{m_name} [reason: duplicate member across merged Enum '{name}']")
                    _log_sanitization_detail(
                        context_label=context_label,
                        class_name=name,
                        class_kind="Enum (Consolidation Merge)",
                        kept_count=len(existing_members),
                        stripped_items=skipped_members,
                        extra_note=f"Merged duplicate Enum class '{name}' (added {len(added_members)} unique member(s)).",
                    )
                else:
                    other_nodes.append(node)
            else:
                if name not in models_by_name:
                    models_by_name[name] = node
                elif merge_models:
                    # If the existing model lacks a docstring and the merged one has a docstring, adopt it
                    existing_has_doc = (
                        bool(models_by_name[name].body)
                        and isinstance(models_by_name[name].body[0], ast.Expr)
                        and isinstance(models_by_name[name].body[0].value, ast.Constant)
                        and isinstance(models_by_name[name].body[0].value.value, str)
                    )
                    incoming_has_doc = (
                        bool(node.body)
                        and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)
                    )
                    if not existing_has_doc and incoming_has_doc:
                        models_by_name[name].body.insert(0, node.body[0])

                    existing_fields = {
                        get_assigned_name(stmt)
                        for stmt in models_by_name[name].body
                        if get_assigned_name(stmt) is not None
                    }
                    added_fields, skipped_fields = [], []
                    for stmt in node.body:
                        f_name = get_assigned_name(stmt)
                        if f_name:
                            if f_name not in existing_fields:
                                models_by_name[name].body.append(stmt)
                                existing_fields.add(f_name)
                                added_fields.append(f_name)
                            else:
                                skipped_fields.append(f"{f_name} [reason: duplicate field across merged BaseModel '{name}']")
                    _log_sanitization_detail(
                        context_label=context_label,
                        class_name=name,
                        class_kind="BaseModel (Consolidation Merge)",
                        kept_count=len(existing_fields),
                        stripped_items=skipped_fields,
                        extra_note=f"Merged duplicate BaseModel class '{name}' (added {len(added_fields)} unique field(s)).",
                    )
                else:
                    other_nodes.append(node)
        else:
            other_nodes.append(node)

    # Disambiguate Enum vs BaseModel name collisions
    if getattr(ast_cfg, "DISAMBIGUATE_ENUM_BASEMODEL_COLLISIONS", True):
        collision_suffix = getattr(ast_cfg, "COLLISION_RENAME_SUFFIX", "Detail") or "Detail"
        for m_name in list(models_by_name.keys()):
            if m_name in enums_by_name:
                new_name = f"{m_name}{collision_suffix}"
                model_node = models_by_name.pop(m_name)
                model_node.name = new_name
                models_by_name[new_name] = model_node
                _log_sanitization_detail(
                    context_label=context_label,
                    class_name=m_name,
                    class_kind="BaseModel (Collision Rename)",
                    kept_count=len(model_node.body),
                    stripped_items=[],
                    extra_note=f"Renamed BaseModel '{m_name}' -> '{new_name}' to disambiguate collision with Enum '{m_name}'.",
                )

    tree.body = other_nodes + list(enums_by_name.values()) + list(models_by_name.values())
    try:
        merged_code = ast.unparse(tree)
        if getattr(ast_cfg, "RESANITIZE_MERGED_ONTOLOGY", True):
            merged_code = _sanitize_ast_enums(merged_code, context_label=f"{context_label} (merged)")
            merged_code = _sanitize_ast_models(merged_code, context_label=f"{context_label} (merged)")
        return merged_code
    except Exception:
        return clean_code


def sanitize_module_name(name: str, fallback: str = "domain_schema") -> str:
    """Normalizes a module_name string into a valid snake_case Python module identifier."""
    if not name:
        return fallback
    cleaned = re.sub(r"\.py$", "", str(name).strip().lower())
    cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned).strip("_")
    stripped_suffix = re.sub(r"_(?:schema|schemas|module|ontology|models|model)$", "", cleaned).strip("_")
    if stripped_suffix and stripped_suffix not in ("domain", "community", "master", "comprehensive"):
        cleaned = stripped_suffix
    return cleaned or fallback


def _camel_to_snake(name: str) -> str:
    """Converts a PascalCase / CamelCase identifier into snake_case."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", str(name).strip())
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
    return re.sub(r"[^a-z0-9_]+", "_", s2).strip("_")


def _extract_ast_basemodel_names(code: str) -> List[str]:
    """Returns snake_case names of all non-Enum BaseModel classes defined in `code`."""
    try:
        tree = ast.parse(code)
    except Exception:
        return []

    def is_enum_class(node: ast.ClassDef) -> bool:
        for b in node.bases:
            if isinstance(b, ast.Name) and "Enum" in b.id:
                return True
            if isinstance(b, ast.Attribute) and "Enum" in b.attr:
                return True
        return False

    names: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not is_enum_class(node):
            snake = sanitize_module_name(_camel_to_snake(node.name), fallback="")
            if snake and snake not in names:
                names.append(snake)
    return names


def _resolve_unique_module_name(
    raw_mod_name: str,
    code: str,
    top_terms: Optional[List[str]],
    used_names: set,
    fallback: str,
) -> str:
    """
    Ensures each community `.py` file receives a distinct, specific snake_case module name.
    If the LLM's `raw_mod_name` is generic or collides with an already-used module name in
    `used_names`, disambiguates using the module's AST BaseModel class names and C-TF-IDF
    `top_domain_terms`.
    """
    candidate = sanitize_module_name(raw_mod_name, fallback=fallback)
    is_generic = candidate in ("domain_schema", "community_schema", fallback) or candidate.startswith("community_")
    model_names = _extract_ast_basemodel_names(code)
    term_slugs = [
        sanitize_module_name(t, fallback="")
        for t in (top_terms or [])
        if sanitize_module_name(t, fallback="")
    ]

    if not is_generic and candidate not in used_names:
        used_names.add(candidate)
        return candidate

    # 1. Try primary BaseModel class names directly (in snake_case)
    for m_name in model_names:
        if m_name not in used_names:
            used_names.add(m_name)
            return m_name

    # 2. Try combining the top distinctive C-TF-IDF term with the primary BaseModel name or candidate
    for t_slug in term_slugs:
        if model_names:
            t_tokens = [tok for tok in t_slug.split("_") if tok and tok not in model_names[0].split("_")]
            if t_tokens:
                combined_model = f"{'_'.join(t_tokens[:2])}_{model_names[0]}"
                if combined_model not in used_names:
                    used_names.add(combined_model)
                    return combined_model
        if not is_generic:
            t_tokens = [tok for tok in t_slug.split("_") if tok and tok not in candidate.split("_")]
            if t_tokens:
                combined_cand = f"{candidate}_{'_'.join(t_tokens[:2])}"
                if combined_cand not in used_names:
                    used_names.add(combined_cand)
                    return combined_cand
        if len(t_slug.split("_")) >= 2 and t_slug not in used_names:
            used_names.add(t_slug)
            return t_slug

    # 3. Try combining top two BaseModel names
    if len(model_names) >= 2:
        pair_name = f"{model_names[0]}_{model_names[1]}"
        if pair_name not in used_names:
            used_names.add(pair_name)
            return pair_name

    # 4. Deterministic fallback with target suffix
    base = model_names[0] if model_names else (term_slugs[0] if term_slugs else candidate)
    unique_cand = f"{base}_{fallback}" if base != fallback else fallback
    used_names.add(unique_cand)
    return unique_cand


def validate_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Checks whether cleaned Python code parses into a non-empty AST with at least one class."""
    if not code or not code.strip():
        return False, "Empty source_code returned."
    try:
        tree = ast.parse(code)
        has_class = any(isinstance(node, ast.ClassDef) for node in tree.body)
        if not has_class:
            return False, "Generated source_code contains no Python class definitions (BaseModel or Enum)."
        return True, None
    except SyntaxError as err:
        return False, f"SyntaxError: {err}"
    except Exception as err:
        return False, str(err)
