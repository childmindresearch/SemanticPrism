"""
SemanticPrism Stage 1: Extraction Package
"""

from src.extraction.clustering import ThemeClusteringEngine, format_theme_embedding_text
from src.extraction.pipeline import PipelineRunContext, SemanticPrismStage1Pipeline
from src.extraction.prompts import (
    CLUSTER_MAP_SYSTEM_PROMPT,
    CLUSTER_MAP_USER_PROMPT,
    CLUSTER_REDUCE_SYSTEM_PROMPT,
    CLUSTER_REDUCE_USER_PROMPT,
    THEME_DISCOVERY_SYSTEM_PROMPT,
    THEME_DISCOVERY_USER_PROMPT,
    TRIPLE_EXTRACTION_SYSTEM_PROMPT,
    TRIPLE_EXTRACTION_USER_PROMPT,
)
from src.extraction.schemas import (
    ClusterCandidateSynthesis,
    MasterThemeSynthesisResult,
    RawTriple,
    Theme,
    ThemeDiscoveryResult,
    TripleExtractionResult,
)

__all__ = [
    "Theme",
    "ThemeDiscoveryResult",
    "ClusterCandidateSynthesis",
    "MasterThemeSynthesisResult",
    "RawTriple",
    "TripleExtractionResult",
    "THEME_DISCOVERY_SYSTEM_PROMPT",
    "THEME_DISCOVERY_USER_PROMPT",
    "CLUSTER_MAP_SYSTEM_PROMPT",
    "CLUSTER_MAP_USER_PROMPT",
    "CLUSTER_REDUCE_SYSTEM_PROMPT",
    "CLUSTER_REDUCE_USER_PROMPT",
    "TRIPLE_EXTRACTION_SYSTEM_PROMPT",
    "TRIPLE_EXTRACTION_USER_PROMPT",
    "ThemeClusteringEngine",
    "format_theme_embedding_text",
    "PipelineRunContext",
    "SemanticPrismStage1Pipeline",
]
