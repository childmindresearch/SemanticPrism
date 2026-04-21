"""
SemanticPrism: Hardware Profiler & Context Manager

Dynamically profiles local Linux OS constraints (System RAM, GPU VRAM)
without requiring external unapproved dependencies (e.g., psutil).
Provides safe semantic token configurations dynamically.
"""
import os
import subprocess
import logging
from src.core.logger import get_logger

logger = get_logger("ContextManager", logging.INFO)

class ContextManager:
    """
    Evaluates current system parameters natively to supply safe text block bounds,
    preventing context overflows natively securely.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.vram_free_mb = self._get_native_vram()
        self.ram_free_mb = self._get_native_ram()
        logger.info(f"Initialized ContextManager - VRAM Free: {self.vram_free_mb} MB, RAM Free: {self.ram_free_mb} MB")

    def _get_native_vram(self) -> int:
        """
        Uses standard library subprocess to natively execute nvidia-smi.
        Defaults to 0 if nvidia-smi is unavailable or errors out organically.
        """
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            # Take the first GPU dynamically
            lines = result.stdout.strip().split('\n')
            if lines and lines[0].isdigit():
                return int(lines[0])
            return 0
        except Exception as e:
            logger.warning(f"Native VRAM check failed securely: {e}. Defaulting VRAM to 0 MB.")
            return 0

    def _get_native_ram(self) -> int:
        """
        Reads native Linux /proc/meminfo securely to calculate free memory.
        """
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
            
            mem_free = 0
            mem_available = 0
            
            for line in lines:
                if line.startswith('MemFree:'):
                    mem_free = int(line.split()[1]) // 1024  # Convert kB to MB
                elif line.startswith('MemAvailable:'):
                    mem_available = int(line.split()[1]) // 1024
            
            # Prefer MemAvailable as it is the most functionally accurate metric natively
            if mem_available > 0:
                return mem_available
            return mem_free
        except Exception as e:
            logger.warning(f"Native RAM check securely failed: {e}. Defaulting RAM to 0 MB.")
            return 0

    def calculate_safe_bounds(self, default_chunk_words: int = 6000) -> int:
        """
        Determines the safe max_chunk_words dynamically based on VRAM rules.
        A strict 8K context token envelope requires approx max ~6000 english words.
        
        Args:
            default_chunk_words: Fallback logic bound statically mapped.
            
        Returns:
            The constrained exact max word bounds logically safely.
        """
        # If we have extreme constraints, reduce organically
        if self.vram_free_mb > 0 and self.vram_free_mb < 4000:
            logger.warning("VRAM critically low natively. Compressing context boundaries safely.")
            return min(default_chunk_words, 2000)
            
        return default_chunk_words
