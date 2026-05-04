from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
class Theme(BaseModel):
    """Represents an isolated, high-level structural theme."""
    title: str = Field(description="The formal title of the extracted theme.")
    description: str = Field(description="A brief description of the theme.")
    reasoning: str = Field(description="Reasoning for classifying this as a critical theme.")

class ThemeDiscoveryResult(BaseModel):
    """The aggregate output of the Theme Discovery phase."""
    themes: List[Theme] = Field(description="List of themes discovered in the text.")

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
    """
    Extracts a high-fidelity subject-predicate-object relationship.
    Focuses on atomic facts to ensure graph density and accuracy.
    """
    subject: str = Field(..., description="The exact source entity exactly as it appears in the source text. Do not modify the casing or format.")
    predicate: str = Field(..., description="The exact relationship verb or linking phrase exactly as it appears in the source text. Do not use snake_case.")
    object: str = Field(..., description="The exact target entity or attribute value exactly as it appears in the source text. Do not modify the casing or format.")
    source_quote: str = Field(..., description="The exact snippet from the text that proves this relationship exists.")
    certainty_score: float = Field(ge=0, le=1, description="Confidence score: 1.0 for explicit facts, 0.5 for inferred.")
    theme_association: Optional[str] = Field(default="Other", description="The theme this triple most closely aligns with, if any.")

    @field_validator('subject', 'predicate', 'object')
    @classmethod
    def prevent_empty(cls, v):
        if not v or len(str(v).strip()) == 0:
            raise ValueError("Fields cannot be empty strings")
        return str(v).strip()

class TripleExtractionResult(BaseModel):
    """The aggregate result grouping multiple triples from a single text block."""
    triples: List[RawTriple] = Field(description="The complete list of extracted node-edge relationships.")

    @field_validator('triples', mode='before')
    @classmethod
    def drop_invalid_triples(cls, v):
        """Intercepts the raw JSON dictionaries before strict Pydantic crash enforcement."""
        if not isinstance(v, list):
            return v
            
        valid_chunk = []
        for item in v:
            try:
                if isinstance(item, dict):
                    # Test explicit validation against the strict RawTriple bounds natively
                    valid_chunk.append(RawTriple.model_validate(item))
                elif isinstance(item, RawTriple):
                    valid_chunk.append(item)
            except Exception:
                # Silently discard the corrupted/incomplete hallucination structurally 
                pass
        return valid_chunk

class NormalizedToken(BaseModel):
    """A mapping from the original raw string to its normalized form."""
    original: str = Field(description="The exact original string provided.")
    normalized: str = Field(description="The finalized, normalized version of the string.")

class NormalizedStrings(BaseModel):
    """The normalized array mapping original strings to their transformed states."""
    tokens: List[NormalizedToken] = Field(description="The explicit mappings of original strings to normalized strings.")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "tokens": [
                    {
                        "original": "data warehouses",
                        "normalized": "data warehouse"
                    },
                    {
                        "original": "is a part of",
                        "normalized": "part of"
                    }
                ]
            }
        }
    }
