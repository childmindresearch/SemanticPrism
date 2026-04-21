"""
SemanticPrism: Semantic Chunker Utility

Handles text boundary overlaps explicitly to prevent contextual logic loss during LLM processing cleanly.
"""

def chunk_text(text: str, max_words: int, overlap_words: int = 50) -> list[str]:
    """
    Splits text into chunks defined by max_words with a structural overlap window natively.
    
    Args:
        text: The full string corpus.
        max_words: Target max length for each block based on hardware constraints.
        overlap_words: Slices to duplicate back to preserve context edges.
        
    Returns:
        List of text strings structured securely.
    """
    words = text.split()
    chunks = []
    
    if len(words) == 0:
        return chunks

    idx = 0
    while idx < len(words):
        end_idx = min(idx + max_words, len(words))
        chunk_slice = words[idx:end_idx]
        chunks.append(" ".join(chunk_slice))
        
        if end_idx == len(words):
            break
            
        # Move forward, leaving an overlap window gracefully backwards securely dynamically
        idx = end_idx - overlap_words
        
        # Failsafe infinite loop prevention locally structurally
        if max_words <= overlap_words:
            idx = end_idx
            
    return chunks
