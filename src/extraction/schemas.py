"""
SemanticPrism Stage 1: Extraction Schemas
This module contains strictly Pydantic data models used to enforce structured JSON output from Pydantic AI Agents.
"""

import json
import re
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Any

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
                "reasoning": str(reason).strip()
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

    model_config = {
        "json_schema_extra": {
            "example": {
                "themes": [
                    {
                        "title": "Historical Philosophical Traditions",
                        "description": "Systemic analysis of classical thinkers and theoretical lineages.",
                        "reasoning": "Serves as the foundational intellectual framework across the text."
                    },
                    {
                        "title": "Epistemological Methodology",
                        "description": "Principles and criteria used to assess knowledge, truth, and logic.",
                        "reasoning": "Underpins the analytical arguments made throughout the discourse."
                    }
                ]
            }
        }
    }

class ClusterCandidateSynthesis(BaseModel):
    """Intermediate candidate synthesis for an individual cluster produced during the Map phase."""
    cluster_id: int = Field(description="The numeric ID of the cluster.")
    candidate_title: str = Field(description="A concise, authoritative candidate title (3 to 6 words).")
    scope_definition: str = Field(description="1-2 sentences defining what this cluster strictly covers.")
    key_inclusions: List[str] = Field(description="Key concepts or mechanisms that strictly belong to this theme.")
    key_exclusions: List[str] = Field(description="Concepts or mechanisms that might seem related but belong elsewhere.")
    confidence: Optional[str] = Field(default="high", description="Assessment of cluster coherence: high, medium, or low.")

class MasterThemeSynthesisResult(BaseModel):
    """The normalized global output consolidating multiple document themes."""
    master_domain: str = Field(description="The singular, overarching domain describing the entire corpus.")
    master_themes: List[str] = Field(description="The consolidated, deduplicated list of formal theme names.")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "master_domain": "Ontology & Knowledge Systems",
                "master_themes": [
                    "System Architecture",
                    "Data Modeling",
                    "Semantic Graph Infrastructure"
                ]
            }
        }
    }

class RawTriple(BaseModel):
    """Subject-predicate-object relationship extracted from the text."""
    subject: str = Field(description="Primary entity, concept, or person.")
    predicate: str = Field(description="Relationship, action, or finding connecting subject and object.")
    object: str = Field(description="Target entity, diagnosis, score, or concept.")
    source_quote: str = Field(default="", description="Verbatim quote from the text confirming this relationship.")
    theme_association: str = Field(default="Other", description="Matching theme name from allowed themes, or 'Other'.")

    @model_validator(mode='before')
    @classmethod
    def remap_and_clean(cls, data: Any) -> Any:
        if isinstance(data, dict):
            subj = (data.get("subject") or data.get("subj") or data.get("source_entity") 
                    or data.get("head") or data.get("source") or data.get("node1") or data.get("entity1") or "")
            pred = (data.get("predicate") or data.get("pred") or data.get("relation") 
                    or data.get("relationship") or data.get("edge") or data.get("action") or "")
            obj = (data.get("object") or data.get("obj") or data.get("target_entity") 
                   or data.get("tail") or data.get("target") or data.get("node2") or data.get("entity2") or "")
            quote = (data.get("source_quote") or data.get("quote") or data.get("evidence") 
                     or data.get("source") or data.get("source_text") or data.get("context") or "")
            theme = (data.get("theme_association") or data.get("theme") or data.get("theme_name") 
                     or data.get("category") or "Other")
            return {
                **data,
                "subject": str(subj).strip(),
                "predicate": str(pred).strip(),
                "object": str(obj).strip(),
                "source_quote": str(quote).strip() if quote is not None else "",
                "theme_association": str(theme).strip() if (theme and str(theme).strip()) else "Other"
            }
        return data

    @field_validator('subject', 'predicate', 'object')
    @classmethod
    def prevent_empty(cls, v):
        if not v or len(str(v).strip()) == 0:
            raise ValueError("Fields cannot be empty strings")
        return str(v).strip()

class TripleExtractionResult(BaseModel):
    """The aggregate result grouping multiple triples from a single text block."""
    triples: List[RawTriple] = Field(default_factory=list, description="The complete list of extracted node-edge relationships.")

    @model_validator(mode='before')
    @classmethod
    def handle_raw_payloads(cls, data: Any) -> Any:
        # If payload is a JSON string or markdown block, parse it first
        if isinstance(data, str):
            clean = data.strip()
            if "```" in clean:
                clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.MULTILINE)
                clean = re.sub(r"\s*```$", "", clean, flags=re.MULTILINE).strip()
            try:
                data = json.loads(clean)
            except Exception:
                pass

        # Auto-unwrap bare lists: [{"subject": ...}] -> {"triples": [...]}
        if isinstance(data, list):
            return {"triples": data}
        if isinstance(data, dict):
            # If payload is a single triple dict, wrap into list
            if any(k in data for k in ["subject", "subj", "source_entity", "head", "node1", "entity1"]) and any(k in data for k in ["predicate", "pred", "relation", "relationship", "edge"]):
                return {"triples": [data]}
            if "triples" not in data:
                for alt_key in ["triplets", "relationships", "extracted_triples", "triples_list", "data", "results", "items", "response", "output"]:
                    if alt_key in data:
                        data["triples"] = data[alt_key]
                        break
        return data

    @field_validator('triples', mode='before')
    @classmethod
    def drop_invalid_triples(cls, v):
        """Intercepts raw JSON dictionaries or strings before strict Pydantic crash enforcement."""
        if isinstance(v, str):
            clean = v.strip()
            if "```" in clean:
                clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.MULTILINE)
                clean = re.sub(r"\s*```$", "", clean, flags=re.MULTILINE).strip()
            try:
                parsed = json.loads(clean)
                if isinstance(parsed, list):
                    v = parsed
                elif isinstance(parsed, dict):
                    v = [parsed]
            except Exception:
                pass

        if not isinstance(v, list):
            return []
            
        valid_chunk = []
        for item in v:
            try:
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except Exception:
                        pass
                if isinstance(item, dict):
                    valid_chunk.append(RawTriple.model_validate(item))
                elif isinstance(item, RawTriple):
                    valid_chunk.append(item)
            except Exception:
                # Silently discard corrupted/incomplete individual entries
                pass
                
        # If candidate triples were present but all failed, raise to trigger salvage
        if len(v) > 0 and len(valid_chunk) == 0:
            raise ValueError(f"All {len(v)} candidate triples failed schema validation bounds.")
            
        return valid_chunk

    model_config = {
        "json_schema_extra": {
            "example": {
                "triples": [
                    {
                        "subject": "patient",
                        "predicate": "was diagnosed with",
                        "object": "type 2 diabetes",
                        "source_quote": "The patient was diagnosed with type 2 diabetes following routine lab work.",
                        "theme_association": "Endocrine Metabolic Conditions"
                    }
                ]
            }
        }
    }

