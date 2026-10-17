"""
SemanticPrism Extraction Package
Contains tools for raw theme discovery, empirical theme clustering, master theme synthesis, and triple extraction.
"""

from .schemas import (
    Theme,
    ThemeDiscoveryResult,
    ClusterCandidateSynthesis,
    MasterThemeSynthesisResult,
    RawTriple,
    TripleExtractionResult,
)
from .theme_clusterer import ThemeClusteringEngine, format_theme_embedding_text

def __getattr__(name: str):
    if name in ("ExtractionPipeline", "PipelineRunContext"):
        import importlib
        ext_mod = importlib.import_module(".extractor", __name__)
        return getattr(ext_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "Theme",
    "ThemeDiscoveryResult",
    "ClusterCandidateSynthesis",
    "MasterThemeSynthesisResult",
    "RawTriple",
    "TripleExtractionResult",
    "ThemeClusteringEngine",
    "format_theme_embedding_text",
    "ExtractionPipeline",
    "PipelineRunContext",
]
