"""
SemanticPrism: Text Normalization Extractor

Handles the structural lexical alignment processing safely without congesting the core pipeline logic organically securely natively.
"""
import re
from typing import List, Callable, Tuple
from src.extraction.schemas import RawTriple
from src.extraction.extractor import ExtractionPipeline
from src.core.logger import get_logger

logger = get_logger("NormalizationPhase")

def nlp_preprocess(text: str) -> str:
    """Preprocesses raw strings to standard format natively before LLM batching."""
    if not text:
        return text
    # Convert snake_case to regular spaces
    text = text.replace('_', ' ')
    # Remove unnecessary formatting (<, >, /, \, |, [, ], {, })
    text = re.sub(r'[<>/\\|\[\]{}]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Normalize multiple spaces to single space and strip
    text = re.sub(r'\s+', ' ', text).strip()
    return text

async def execute_normalization_phase(
    extractor: ExtractionPipeline, 
    raw_triples: List[RawTriple], 
    master_domain: str, 
    save_state_fn: Callable
) -> Tuple[set, set, set]:
    """
    Executes the LLM text normalization sequentially, saving state output JSONs
    and mutating the raw_triples array in-place.
    
    Returns:
        A tuple of unique normalized string logic sets: (norm_subjs, norm_preds, norm_objs)
    """
    if not extractor.config.get('pipeline', {}).get('normalize_text', True):
        # Default bypass logic to properly count unaltered parameters natively effortlessly safely cleanly
        norm_subjs = {t.subject for t in raw_triples}
        norm_preds = {t.predicate for t in raw_triples}
        norm_objs = {t.object for t in raw_triples}
        normalized_triples = [t.model_copy() for t in raw_triples]
        return normalized_triples, norm_subjs, norm_preds, norm_objs
        
    logger.info("==================================================")
    logger.info("STAGE 2.5: LLM TEXT NORMALIZATION")
    logger.info("==================================================")
    
    # Deep copy the array to prevent in-place mutation of the true raw extraction footprint
    normalized_triples = [t.model_copy() for t in raw_triples]
    
    # NLP Preprocessing Pass
    logger.info("Executing NLP Preprocessing standardizations natively...")
    for t in normalized_triples:
        t.subject = nlp_preprocess(t.subject)
        t.predicate = nlp_preprocess(t.predicate)
        t.object = nlp_preprocess(t.object)
        
    subject_tokens = set()
    predicate_tokens = set()
    object_tokens = set()
    
    for t in normalized_triples:
        subject_tokens.add(t.subject)
        predicate_tokens.add(t.predicate)
        object_tokens.add(t.object)
        
    subject_list = list(subject_tokens)
    predicate_list = list(predicate_tokens)
    object_list = list(object_tokens)
    
    subject_map = {}
    predicate_map = {}
    object_map = {}
    
    async def process_batches(tokens_list, label, out_map):
        batch_size = 50
        total_batches = (len(tokens_list) - 1) // batch_size + 1 if tokens_list else 0
        for i in range(0, len(tokens_list), batch_size):
            batch = tokens_list[i:i+batch_size]
            logger.info(f"Normalizing {label} String Batch {i//batch_size + 1}/{total_batches}")
            norm_batch_dict = await extractor.normalize_triples_strings(batch, master_domain)
            
            for orig in batch:
                out_map[orig] = norm_batch_dict.get(orig, orig)

    if subject_list:
        await process_batches(subject_list, "Subject", subject_map)
    if predicate_list:
        await process_batches(predicate_list, "Predicate", predicate_map)
    if object_list:
        await process_batches(object_list, "Object", object_map)
        
    normalization_details = []
    seen_mappings = set()
    
    changed_text_log = {
        "subject": [],
        "predicate": [],
        "object": []
    }
    
    seen_changed = {
        "subject": set(),
        "predicate": set(),
        "object": set()
    }
    
    for raw_t, t in zip(raw_triples, normalized_triples):
        # Capture subject
        subj_true_orig = raw_t.subject
        subj_preprocessed = t.subject
        subj_norm = subject_map.get(subj_preprocessed, subj_preprocessed)
        subj_key = (subj_true_orig, subj_norm, "subject")
        if subj_key not in seen_mappings:
            seen_mappings.add(subj_key)
            was_changed = subj_true_orig != subj_norm
            normalization_details.append({
                "original_text": subj_true_orig,
                "normalized_text": subj_norm,
                "entity_type": "subject",
                "was_changed": was_changed
            })
            if was_changed and subj_true_orig not in seen_changed["subject"]:
                seen_changed["subject"].add(subj_true_orig)
                changed_text_log["subject"].append({subj_true_orig: subj_norm})
            
        # Capture predicate
        pred_true_orig = raw_t.predicate
        pred_preprocessed = t.predicate
        pred_norm = predicate_map.get(pred_preprocessed, pred_preprocessed)
        pred_key = (pred_true_orig, pred_norm, "predicate")
        if pred_key not in seen_mappings:
            seen_mappings.add(pred_key)
            was_changed = pred_true_orig != pred_norm
            normalization_details.append({
                "original_text": pred_true_orig,
                "normalized_text": pred_norm,
                "entity_type": "predicate",
                "was_changed": was_changed
            })
            if was_changed and pred_true_orig not in seen_changed["predicate"]:
                seen_changed["predicate"].add(pred_true_orig)
                changed_text_log["predicate"].append({pred_true_orig: pred_norm})
            
        # Capture object
        obj_true_orig = raw_t.object
        obj_preprocessed = t.object
        obj_norm = object_map.get(obj_preprocessed, obj_preprocessed)
        obj_key = (obj_true_orig, obj_norm, "object")
        if obj_key not in seen_mappings:
            seen_mappings.add(obj_key)
            was_changed = obj_true_orig != obj_norm
            normalization_details.append({
                "original_text": obj_true_orig,
                "normalized_text": obj_norm,
                "entity_type": "object",
                "was_changed": was_changed
            })
            if was_changed and obj_true_orig not in seen_changed["object"]:
                seen_changed["object"].add(obj_true_orig)
                changed_text_log["object"].append({obj_true_orig: obj_norm})
            
        # Apply changes
        t.subject = subj_norm
        t.predicate = pred_norm
        t.object = obj_norm
        
    save_state_fn(normalization_details, "outputs/01_extraction/normalization_mapping_details.json")
    save_state_fn(changed_text_log, "outputs/01_extraction/normalized_text.json")
    save_state_fn(normalized_triples, "outputs/01_extraction/normalized_triplets.json")
    
    norm_subjs = {t.subject for t in normalized_triples}
    norm_preds = {t.predicate for t in normalized_triples}
    norm_objs = {t.object for t in normalized_triples}
    
    return normalized_triples, norm_subjs, norm_preds, norm_objs
