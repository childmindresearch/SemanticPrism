from typing import List, Literal, Optional
from pydantic import BaseModel, Field


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
        description="A score from 0.0 to 1.0 evaluating how accurately this label represents all members of the cluster.",
        ge=0.0,
        le=1.0
    )
