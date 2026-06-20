"""
SemanticPrism Token Estimation and Trimming Utility
Optimizes KV Cache usage and prevents model context limit overflows.
"""

try:
    import tiktoken
    _encoder = tiktoken.get_encoding("cl100k_base")
    def estimate_tokens(text: str) -> int:
        """Counts exact tokens using tiktoken (cl100k_base)."""
        if not text:
            return 0
        return len(_encoder.encode(text))
except ImportError:
    def estimate_tokens(text: str) -> int:
        """Heuristic-based estimation: 1 word ~ 1.3 tokens."""
        if not text:
            return 0
        words = text.split()
        return int(len(words) * 1.3)

def validate_and_trim_prompt(prompt: str, system_prompt: str, cap: int, output_buffer: int = 1000) -> str:
    """
    Validates total token count against cap. Truncates prompt text if necessary
    to leave room for the system prompt and expected output generation buffer.
    """
    sys_tokens = estimate_tokens(system_prompt)
    available_tokens = cap - sys_tokens - output_buffer
    
    if available_tokens <= 0:
        # Safety fallback for extremely small caps
        available_tokens = max(100, cap - output_buffer)
        
    prompt_tokens = estimate_tokens(prompt)
    if prompt_tokens <= available_tokens:
        return prompt
        
    # Heuristic initial char length guess: 1 token ≈ 4 characters
    approx_chars_to_keep = available_tokens * 4
    truncated_prompt = prompt[:approx_chars_to_keep]
    
    # Refine truncation iteratively using actual token estimation
    while estimate_tokens(truncated_prompt) > available_tokens and len(truncated_prompt) > 10:
        # Back off by 10%
        truncated_prompt = truncated_prompt[:-max(10, int(len(truncated_prompt) * 0.1))]
        
    print(f"[Token Helper] Warning: Prompt truncated from {prompt_tokens} to {estimate_tokens(truncated_prompt)} tokens to respect cap of {cap}.")
    return truncated_prompt
