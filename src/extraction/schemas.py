"""
SemanticPrism Stage 1: Extraction Schemas
This module contains strictly Pydantic data models used to enforce structured JSON output from Pydantic AI Agents.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any

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
    subject: str = Field(..., description="The primary entity or concept exactly as it appears in the source text. Do not modify the casing or format.")
    predicate: str = Field(..., description="The relationship or action connecting subject and object exactly as it appears in the source text. Do not modify the casing or format.")
    object: str = Field(..., description="The target entity, concept, or value. Do not modify the casing or format.")
    source_quote: str = Field(..., description="The EXACT, verbatim substring from the text confirming this relationship.")
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
                
        if len(v) > 0 and len(valid_chunk) == 0:
            raise ValueError("All extracted triples in the list failed schema validation constraints.")
            
        return valid_chunk

