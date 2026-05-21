from pydantic import BaseModel, Field
from typing import List, Optional

class GeneratedSchema(BaseModel):
    """Represents the final Python schema synthesized from the topology map."""
    title: str = Field(description="The functional title abstracting the core semantic nodes.")
    summary: str = Field(description="A concise summary of the interlocked hierarchy map.")
    core_theme: str = Field(description="The foundational underlying concept binding the nodes.")
    key_learnings: List[str] = Field(description="The primary factual insights derived.")
    isolated_facts: List[str] = Field(description="Key unique properties abstracted from the topological array.")
    protocols_code: str = Field(default="", description="Strictly typed Protocol/Mixin interface blocks. Output empty string if none apply.")
    concrete_models_code: str = Field(default="", description="Strictly typed Pydantic code blocks for concrete representation. Output empty string if none apply.")
