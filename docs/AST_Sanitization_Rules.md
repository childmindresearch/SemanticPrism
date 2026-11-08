# Stage 4 Synthesis AST Sanitization & Code Cleanup Rules

This document catalogs all pre-AST text normalization, Enum AST sanitization, `BaseModel` AST sanitization, and consolidated ontology deduplication rules implemented in [`src/synthesis/sanitizers.py`](../src/synthesis/sanitizers.py) (configured via [`run_stage_4_synthesis.py`](../run_stage_4_synthesis.py)).

All AST sanitization events (stripped or collapsed Enum members and `BaseModel` fields) are logged in real time to the console (when `PRINT_CHECKPOINTS = True`) and persisted to:
- `outputs/04_synthesis/logs/schema_sanitization.log` via `_log_sanitization_detail` in [`src/synthesis/sanitizers.py`](../src/synthesis/sanitizers.py).

---

## 1. Pre-AST Text Normalization (`clean_python_code`)

Executed on raw LLM-generated `source_code` before parsing into a Python AST:

1. **Markdown Code Fence Stripping**
   - Strips leading ```` ```python ```` or ```` ``` ```` and trailing ```` ``` ```` wrappers.
2. **Escape Sequence Unescaping**
   - Replaces over-escaped quotes (`\"` -> `"`) and literal `\n` sequences with actual newlines.
3. **Invalid Python Identifier & Reserved Keyword Fixes (`clean_python_code`)**
   - **Assignment LHS (`=`)**: Replaces spaces or commas inside variable/member identifiers with underscores (`_`).
   - **Type Annotation LHS (`:`)**: Replaces hyphens in field names with underscores (e.g., `help-seeking_barriers: ...` -> `help_seeking_barriers: ...`).
   - **Python Reserved Keyword Rename**: Rewrites field names that are Python reserved keywords (`from`, `class`, `return`, `global`, `lambda`, `async`, `def`, etc.) to `<keyword>_field` before `ast.parse()` so syntax parsing never fails, and logs the rename to `schema_sanitization.log`.
4. **Non-ASCII / CJK Character Stripping**
   - Removes CJK characters (`[\u4e00-\u9fff]+`) that bilingual LLMs occasionally emit in comments or docstrings.
5. **Meta-Model Class Regex Removal**
   - Strips any accidental classes that model the prompt/JSON envelope itself (`ThemeObservation`, `ThematicObservation`, `GeneratedModule`) and logs the removal to `schema_sanitization.log`.

---

## 2. Enum AST Sanitizer (`_sanitize_ast_enums`)

Applied to every `Enum` class definition in the parsed AST:

1. **Verbose / Non-Atomic Member Stripping (`_is_verbose_or_runaway_enum_member`)**
   - Strips any `Enum` member whose identifier exceeds **4 words** (`_`-separated) or **35 characters**.
   - Strips any `Enum` member whose string value exceeds **5 words** or **40 characters**.
   - **Log reason**: `verbose/non-atomic member (>4 words or >35 chars)`
2. **Duplicate Member Name Stripping**
   - Strips repeated member identifiers within the same `Enum` class.
   - **Log reason**: `duplicate member name`
3. **Duplicate Member Value Stripping**
   - Strips members whose normalized lowercase string value duplicates an earlier member in the same `Enum` class.
   - **Log reason**: `duplicate member value`
4. **Max Enum Member Cap**
   - Caps each `Enum` class at `MAX_ENUM_MEMBERS_PER_CLASS` (**15** members), stripping any subsequent members.
   - **Log reason**: `exceeded MAX_ENUM_MEMBERS_PER_CLASS (15)`
5. **`OTHER` / `'Other'` Member Placed Last**
   - If an `Enum` class includes an `OTHER` member (or a member with string value `'Other'`), it is automatically moved to the final position of the `Enum` class regardless of how many members are included.
6. **Empty Enum Removal & Type Annotation Rewriting**
   - If an `Enum` class has `0` valid members remaining after filtering, the `Enum` class is removed from the AST and `_DroppedEnumRewriter` rewrites any `BaseModel` field type annotations referencing that `Enum` to `str`.
   - **Log note**: `Enum class had 0 valid members remaining and was removed (field references rewritten to 'str').`

---

## 3. BaseModel AST Sanitizer (`_sanitize_ast_models`)

Applied to every non-`Enum` class (`BaseModel`) in the parsed AST:

1. **Dangling / Undefined Type Reference Rewriting (`_DanglingTypeRewriter`)**
   - Inspects every field type annotation (outside `Literal[...]`) and rewrites any identifier that is neither a Python/typing/Pydantic built-in nor a class/import defined in the module to `str`.
   - **Log reason**: `rewrote undefined type reference(s) [...] to 'str'`
2. **Shadowed Pydantic `BaseModel` Attribute & Keyword Rename**
   - If a field name shadows a reserved Pydantic `BaseModel` method/attribute (`schema`, `json`, `copy`, `dict`, `validate`, `fields`, `model_config`, `model_fields`, `model_dump`, etc.) or Python keyword, renames it to `<field_name>_field`.
   - **Log reason**: `renamed field shadowing reserved keyword/BaseModel attribute`
3. **Duplicate Field Name Stripping**
   - Strips fields with duplicate normalized lowercase names within the same `BaseModel` class.
   - **Log reason**: `duplicate field name`
4. **Shared 2-Token Prefix Collapse / Cap**
   - Applies to fields with `>= 3` underscore-separated tokens (`parts[0]_parts[1]_...`):
   - **Collapse (`>= 3` fields sharing a 2-token prefix)**: Collapses all fields sharing `prefix2` into a single `Optional[List[str]]` field named `<prefix2>s` (with `Field(default=None, description="Associated <prefix2>s.")`).
     - **Log reason**: `collapsed N fields sharing prefix '<prefix2>_*'`
   - **Cap (`max_shared_prefix = 2`)**: If a 2-token prefix has already appeared twice on the class, subsequent fields sharing that prefix are stripped.
     - **Log reason**: `exceeded shared prefix cap (2) for '<prefix2>_*'`
5. **Shared Suffix Collapse / Cap**
   - Applies to fields with `>= 2` tokens whose final token (`suffix1`) is **not** in `exempt_suffixes` (`id`, `name`, `type`, `types`, `status`, `level`, `score`, `range`, `date`, `time`, `plan`, `notes`, `summary`, `description`, `details`, `count`, `rate`, `list`, `items`, `category`, `mode`, `source`):
   - **Collapse (`>= 3` fields sharing a non-exempt suffix)**: Collapses all fields ending in `_*<suffix1>` into a single `Optional[List[str]]` field named `<suffix1>s`.
     - **Log reason**: `collapsed N fields sharing suffix '*_<suffix1>'`
   - **Cap (`max_shared_suffix = 2`)**: Caps non-exempt shared suffixes at 2 fields per class.
     - **Log reason**: `exceeded shared suffix cap (2) for '*_<suffix1>'`
6. **Max BaseModel Field Cap**
   - Enforces `MAX_BASEMODEL_FIELDS_PER_CLASS` (**15** fields) per `BaseModel` class.
   - When a class has `> 15` fields after Pass 1, fields are prioritized using the sort key:
     1. Non-boolean fields before boolean fields (`1 if is_bool else 0`)
     2. Fields without repeated 2-token prefixes (`1 if len(parts) >= 3 and raw_prefix2_counts[prefix2] > 1 else 0`)
     3. Concise 1–2 token core fields before `3+` token compound fields (`1 if len(parts) >= 3 else 0`)
     4. Original declaration order (`idx`)
   - Keeps the top 15 fields (restored to their original declaration order) and strips the overflow.
   - **Log reason**: `exceeded MAX_BASEMODEL_FIELDS_PER_CLASS (15)`
7. **Guaranteed Triple-Quoted `BaseModel` Docstring (`"""..."""`)**
   - Ensures every `BaseModel` class starts with a non-empty triple-quoted docstring (`"""..."""`). If the LLM omitted a class docstring, synthesizes a concise docstring from the class name and its top kept fields.
8. **Canonical Topological Module Ordering & Cyclic Forward-Reference Quoting**
   - Orders top-level declarations as: `__future__` imports (if any) $\rightarrow$ standard imports $\rightarrow$ constants/aliases $\rightarrow$ `Enum` classes $\rightarrow$ topologically sorted `BaseModel` classes (child models before parent models), and quotes any self-referential or cyclic model references so Pydantic v2 evaluates all classes eagerly without `NameError`.
9. **Empty Class Fallback**
   - If a class has `0` fields remaining, inserts `pass` so the class definition remains valid Python.

---

## 4. Consolidated Ontology AST Merger (`deduplicate_consolidated_classes`)

Applied in Phase 2 to `master_ontology.py` and `comprehensive_ontology.py`:

1. **Duplicate Class Merging**
   - Merges multiple `Enum` classes with the same name by unioning their unique members in order of appearance.
   - Merges multiple `BaseModel` classes with the same name by unioning their unique fields in order of appearance.
2. **Enum vs. BaseModel Name Collision Disambiguation**
   - If both an `Enum` and a `BaseModel` share the same class name `<Name>`, renames the `BaseModel` to `<Name>Detail`.
3. **Post-Merge Re-Sanitization**
   - Re-runs `_sanitize_ast_enums` and `_sanitize_ast_models` (with context label `"master_ontology (merged)"`) on the merged AST so merged classes strictly respect `MAX_ENUM_MEMBERS_PER_CLASS` and `MAX_BASEMODEL_FIELDS_PER_CLASS`.

