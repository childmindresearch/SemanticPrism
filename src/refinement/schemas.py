from pydantic import BaseModel, Field



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
