from pydantic import BaseModel, Field

class TaxonomicVerification(BaseModel):
    """Evaluates whether an artificial centroid grouping is functionally accurate."""
    formal_hypernym: str = Field(description="The objective categorical Hypernym that strictly bounds all grouped concepts.")
    hypernym_meaning: str = Field(description="A clear, concise definition of the formal hypernym's categorical meaning.")
    excluded_opposite: str = Field(description="An explicitly defined concept that this hypernym is STRICTLY NOT. Forces negative boundary testing.")
    members_verified: bool = Field(description="True if all grouped members are logically valid subtypes of this hypernym. False if heterogenous.")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "formal_hypernym": "Automobile",
                "hypernym_meaning": "A four-wheeled motorized vehicle designed for fast passenger transportation on roads.",
                "excluded_opposite": "Bicycle",
                "members_verified": True
            }
        }
    }

class ClusterContextualValidation(BaseModel):
    """Determines if a mathematical distance grouping destroyed semantic meaning."""
    accuracy_destroyed: bool = Field(description="Set to true if grouping these terms destroys critical functional distinctions.")
    condition_detected: str = Field(description="Which specific critical failure conditions were violated, or which valid merge applied (e.g. 'Lexical Variation').")

    model_config = {
        "json_schema_extra": {
            "example": {
                "accuracy_destroyed": False,
                "condition_detected": "Lexical Variation"
            }
        }
    }
