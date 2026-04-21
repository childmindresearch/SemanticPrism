"""
SemanticPrism: Naming Resolution and Consolidation Pipeline

A deterministic, 100% offline bridging operation replacing raw localized 
triple node instances with their globally verified formal semantic taxonomic classes.
"""

from typing import List, Dict
from src.extraction.schemas import RawTriple
from src.core.logger import get_logger

logger = get_logger("NamingResolution")

class NamingResolutionPipeline:
    def __init__(self):
        logger.info("Initializing Taxonomic Resolution & Consolidation Pipeline.")

    def resolve_names(self, raw_triples: List[RawTriple], mapping_dict: Dict[str, Dict[str, str]]) -> List[RawTriple]:
        """
        Takes raw extracted triples and computationally substitutes any verified hypernym.
        Leaves unmapped text elements intact and untampered natively to avoid data loss.
        """
        logger.info(f"Resolving taxonomy natively for {len(raw_triples)} semantic triples.")
        resolved_count = 0
        
        subject_map = mapping_dict.get("subject", {})
        predicate_map = mapping_dict.get("predicate", {})
        object_map = mapping_dict.get("object", {})

        for t in raw_triples:
            modified = False
            
            # Sub safely via Dict mapping exclusively
            if t.subject and t.subject in subject_map:
                t.subject = subject_map[t.subject]
                modified = True
                
            if t.predicate and t.predicate in predicate_map:
                t.predicate = predicate_map[t.predicate]
                modified = True
                
            if t.object and t.object in object_map:
                t.object = object_map[t.object]
                modified = True
                
            if modified:
                resolved_count += 1
                
        logger.info(f"Resolution cleanly completed safely. Triples formally updated: {resolved_count}")
        return raw_triples
