"""
SemanticPrism: VRAM Manager
Centralized utility to manage model VRAM across all pipeline stages.
"""

import json
import threading
import urllib.request
from src.config import settings

def purge_vram():
    """Asynchronously purges the Ollama model from VRAM."""
    if not settings.get('llm', {}).get('manage_vram', False):
        return
        
    def purge_task():
        try:
            base_url = settings['llm'].get('base_url', 'http://localhost:11434/v1')
            model = settings['llm'].get('model_name', 'mistral-nemo:12b-instruct-2407-q4_K_M')
            
            # Ollama's API for keeping alive / purging is /api/generate
            if base_url.endswith('/v1'):
                endpoint = base_url[:-3] + '/api/generate'
            else:
                endpoint = base_url.rstrip('/') + '/api/generate'
                
            data = json.dumps({"model": model, "keep_alive": 0}).encode('utf-8')
            req = urllib.request.Request(endpoint, data=data, headers={'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(req, timeout=5) as response:
                pass
        except Exception as e:
            print(f"\n[VRAM Manager] Failed to purge model from VRAM: {e}")
            
    # Run in a background thread to keep it lightweight and non-blocking
    threading.Thread(target=purge_task, daemon=True).start()
