"""
SemanticPrism Stage 2: Pydantic Schemas
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class RawTriple(BaseModel):
    """Represents a single Subject-Predicate-Object triple from Stage 1."""

    subject: str = Field(description="The clear, concise entity acting as the subject.")
    predicate: str = Field(description="The active verb or relationship connecting the subject to the object.")
    object: str = Field(description="The target entity, concept, or condition affected by the predicate.")
    theme_association: str = Field(default="Other", description="Mapped Master Theme title or 'Other'.")
    source_quote: str = Field(default="", description="Verbatim sentence from the source text.")
    source_document: Optional[str] = Field(default=None, description="Identifier of the source document.")

    @model_validator(mode="before")
    @classmethod
    def normalize_triple_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            subj = data.get("subject") or data.get("head") or data.get("source") or data.get("entity_1") or ""
            pred = data.get("predicate") or data.get("relation") or data.get("relationship") or data.get("verb") or ""
            obj = data.get("object") or data.get("tail") or data.get("target") or data.get("entity_2") or ""
            theme = data.get("theme_association") or data.get("theme") or data.get("master_theme") or "Other"
            quote = data.get("source_quote") or data.get("quote") or data.get("evidence") or data.get("sentence") or ""
            out = dict(data)
            out.update({
                "subject": str(subj).strip(),
                "predicate": str(pred).strip(),
                "object": str(obj).strip(),
                "theme_association": str(theme).strip() or "Other",
                "source_quote": str(quote).strip(),
                "source_document": data.get("source_document"),
            })
            return out
        return data


class TaxonomicVerification(BaseModel):
    """
    Validates whether a cluster of terms shares a formal hypernym, forcing the creation of a
    hypernym regardless of heterogeneity, and provides a confidence score of accuracy.
    """

    formal_hypernym: str = Field(
        description="The formal taxonomic hypernym encompassing all terms in the cluster."
    )
    excluded_opposite: str = Field(
        description="A category this hypernym strictly IS NOT, providing axiomatic negative entailment."
    )
    confidence_score: float = Field(
        default=0.8,
        description="A score from 0.0 to 1.0 evaluating how accurately this label represents all members of the cluster.",
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="before")
    @classmethod
    def remap_hypernym_fields(cls, data: Any) -> Any:
        if isinstance(data, str) and data.strip():
            return {
                "formal_hypernym": data.strip(),
                "excluded_opposite": "Unrelated Entity",
                "confidence_score": 0.8,
            }
        if isinstance(data, dict):
            hyp = (
                data.get("formal_hypernym")
                or data.get("hypernym")
                or data.get("canonical_hypernym")
                or data.get("parent_category")
                or data.get("category")
                or data.get("label")
                or data.get("name")
                or ""
            )
            opp = (
                data.get("excluded_opposite")
                or data.get("opposite")
                or data.get("negative_category")
                or data.get("excluded")
                or data.get("not_category")
                or "Unrelated Category"
            )
            conf_raw = (
                data.get("confidence_score")
                if data.get("confidence_score") is not None
                else data.get("confidence", data.get("score", 0.8))
            )
            try:
                conf = float(conf_raw)
                if conf > 1.0 and conf <= 100.0:
                    conf = conf / 100.0
                conf = max(0.0, min(1.0, conf))
            except (ValueError, TypeError):
                conf = 0.8
            return {
                "formal_hypernym": str(hyp).strip(),
                "excluded_opposite": str(opp).strip() or "Unrelated Category",
                "confidence_score": conf,
            }
        return data

    @field_validator("formal_hypernym")
    @classmethod
    def validate_non_empty_hypernym(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("formal_hypernym cannot be empty")
        return v.strip()
