"""
SemanticPrism Stage 1: Pydantic Schemas
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
import json
import re
from pydantic import BaseModel, Field, field_validator, model_validator


class Theme(BaseModel):
    """Represents an isolated, high-level structural theme."""

    title: str = Field(description="The formal title of the extracted theme.")
    description: Optional[str] = Field(default="", description="A brief description of the theme.")
    reasoning: Optional[str] = Field(default="", description="Reasoning for classifying this as a critical theme.")

    @model_validator(mode="before")
    @classmethod
    def remap_and_clean(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"title": data.strip(), "description": "", "reasoning": ""}
        if isinstance(data, dict):
            title = data.get("title") or data.get("name") or data.get("theme") or data.get("theme_title") or ""
            desc = data.get("description") or data.get("desc") or data.get("summary") or data.get("scope") or ""
            reason = data.get("reasoning") or data.get("rationale") or data.get("justification") or data.get("reason") or ""
            return {
                "title": str(title).strip(),
                "description": str(desc).strip(),
                "reasoning": str(reason).strip(),
            }
        return data

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        if not v or len(v.strip()) == 0:
            raise ValueError("Theme title cannot be empty")
        return v.strip()


class ThemeDiscoveryResult(BaseModel):
    """The aggregate output of the Theme Discovery phase."""

    themes: List[Theme] = Field(default_factory=list, description="List of themes discovered in the text.")

    @model_validator(mode="before")
    @classmethod
    def handle_raw_payloads(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"themes": data}
        if isinstance(data, dict):
            if "themes" not in data:
                for alt_key in ["discovered_themes", "theme_list", "raw_themes", "categories", "items", "results"]:
                    if alt_key in data and isinstance(data[alt_key], list):
                        data["themes"] = data[alt_key]
                        break
                else:
                    if any(k in data for k in ("title", "name", "theme", "theme_title")):
                        return {"themes": [data]}
        return data

    @field_validator("themes", mode="before")
    @classmethod
    def filter_and_salvage_themes(cls, v: Any) -> List[Any]:
        if not isinstance(v, list):
            return []
        valid_themes = []
        for item in v:
            try:
                if isinstance(item, str) and item.strip():
                    valid_themes.append(Theme(title=item.strip()))
                elif isinstance(item, dict):
                    valid_themes.append(Theme.model_validate(item))
                elif isinstance(item, Theme):
                    valid_themes.append(item)
            except Exception:
                pass
        return valid_themes


class ClusterCandidateSynthesis(BaseModel):
    """Intermediate candidate synthesis for an individual cluster produced during the Map phase."""

    cluster_id: int = Field(default=0, description="The numeric ID of the cluster.")
    candidate_title: str = Field(description="A concise, authoritative candidate title (3 to 6 words).")
    scope_definition: str = Field(default="", description="1-2 sentences defining what this cluster strictly covers.")
    key_inclusions: List[str] = Field(default_factory=list, description="Key concepts or mechanisms that strictly belong to this theme.")
    key_exclusions: List[str] = Field(default_factory=list, description="Concepts or mechanisms that might seem related but belong elsewhere.")
    confidence: Optional[str] = Field(default="high", description="Assessment of cluster coherence: high, medium, or low.")


class MasterThemeSynthesisResult(BaseModel):
    """The normalized global output consolidating multiple document themes."""

    master_domain: str = Field(description="The singular, overarching domain describing the entire corpus.")
    master_themes: List[str] = Field(description="The consolidated, deduplicated list of formal theme names.")


class RawTriple(BaseModel):
    """Extracts a high-fidelity subject-predicate-object relationship."""

    subject: str = Field(
        ...,
        description=(
            "The primary entity, actor, or concept (concise and explicit; resolve bare pronouns "
            "like 'he', 'she', 'it', or 'they' to their named referent in the text)."
        ),
    )
    predicate: str = Field(..., description="The active, specific relationship or action connecting subject and object.")
    object: str = Field(
        ...,
        description="The target entity, concept, outcome, or measured value (concise and explicit; avoid bare pronouns).",
    )
    source_quote: str = Field(default="", description="The EXACT, verbatim substring from the text confirming this relationship.")
    theme_association: Optional[str] = Field(default="Other", description="The master theme this triple most closely aligns with.")

    @model_validator(mode="before")
    @classmethod
    def align_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "subject" not in data:
                for k in ("subj", "head", "entity", "source"):
                    if k in data:
                        data["subject"] = data[k]
                        break
            if "predicate" not in data:
                for k in ("pred", "relation", "relationship", "edge", "action"):
                    if k in data:
                        data["predicate"] = data[k]
                        break
            if "object" not in data:
                for k in ("obj", "tail", "target", "value"):
                    if k in data:
                        data["object"] = data[k]
                        break
            if "source_quote" not in data:
                for k in ("quote", "evidence", "sentence", "context"):
                    if k in data:
                        data["source_quote"] = data[k]
                        break
            if "theme_association" not in data:
                for k in ("theme", "master_theme", "category"):
                    if k in data:
                        data["theme_association"] = data[k]
                        break
        return data

    @field_validator("subject", "predicate", "object")
    @classmethod
    def prevent_empty(cls, v: Any) -> str:
        cleaned = re.sub(r"\s+", " ", str(v or "")).strip()
        if not cleaned:
            raise ValueError("Fields cannot be empty strings")
        return cleaned

    @field_validator("source_quote", mode="before")
    @classmethod
    def clean_source_quote(cls, v: Any) -> str:
        if v is None:
            return ""
        return str(v).strip()


def salvage_partial_triples(raw_text: str) -> List[RawTriple]:
    """
    Recover complete RawTriple JSON objects from a truncated or partially malformed
    LLM JSON output stream (e.g., when a smaller model hits max_new_tokens mid-array).
    """
    if not raw_text or not isinstance(raw_text, str):
        return []

    brace_stack: List[int] = []
    candidate_spans: List[Tuple[int, str]] = []
    in_string = False
    escape = False

    for idx, ch in enumerate(raw_text):
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            brace_stack.append(idx)
        elif ch == "}" and brace_stack:
            start_idx = brace_stack.pop()
            candidate_spans.append((start_idx, raw_text[start_idx : idx + 1]))

    candidate_spans.sort(key=lambda pair: pair[0])
    salvaged: List[RawTriple] = []
    wrapper_keys = {"triples", "triplets", "relationships", "items", "results", "extractions"}

    for _, snippet in candidate_spans:
        cleaned_snippet = re.sub(r",\s*}", "}", snippet)
        try:
            obj = json.loads(cleaned_snippet)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if any(wk in obj and isinstance(obj[wk], list) for wk in wrapper_keys):
            continue
        try:
            salvaged.append(RawTriple.model_validate(obj))
        except Exception:
            continue

    return salvaged


class TripleExtractionResult(BaseModel):
    """The aggregate result grouping multiple triples from a single text block."""

    triples: List[RawTriple] = Field(default_factory=list, description="The complete list of extracted node-edge relationships.")

    @model_validator(mode="before")
    @classmethod
    def align_triples(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"triples": salvage_partial_triples(data)}
        if isinstance(data, list):
            return {"triples": data}
        if isinstance(data, dict):
            if "triples" not in data:
                for alt_key in ("triplets", "relationships", "items", "results", "extractions"):
                    if alt_key in data and isinstance(data[alt_key], list):
                        data["triples"] = data[alt_key]
                        break
                else:
                    if any(k in data for k in ("subject", "subj", "head")):
                        return {"triples": [data]}
        return data

    @field_validator("triples", mode="before")
    @classmethod
    def drop_invalid_triples(cls, v: Any) -> List[Any]:
        if not isinstance(v, list):
            return []
        valid_chunk = []
        for item in v:
            try:
                if isinstance(item, dict):
                    valid_chunk.append(RawTriple.model_validate(item))
                elif isinstance(item, RawTriple):
                    valid_chunk.append(item)
            except Exception:
                pass
        return valid_chunk
