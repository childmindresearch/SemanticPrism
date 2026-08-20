"""
SemanticPrism: VRAM Manager
Centralized utility to manage model VRAM across all pipeline stages.
"""

import json
import threading
import urllib.request
from src.config import settings

def purge_vram(target_model: str = None, target_base_url: str = None):
    """Asynchronously purges Ollama models from VRAM."""
    if not settings.get('llm', {}).get('manage_vram', False):
        return
        
    def purge_task():
        try:
            models_to_purge = set()
            if target_model:
                models_to_purge.add((target_model, target_base_url))
            else:
                global_url = settings.get('llm', {}).get('base_url', 'http://localhost:11434/v1')
                global_model = settings.get('llm', {}).get('model_name')
                if global_model:
                    models_to_purge.add((global_model, global_url))
                
                # Check stage overrides
                for stage in ['extraction', 'refinement', 'synthesis']:
                    stage_cfg = settings.get(stage, {})
                    if isinstance(stage_cfg, dict):
                        for key in ['llm', 'llm_theme', 'llm_triple']:
                            sub_cfg = stage_cfg.get(key)
                            if isinstance(sub_cfg, dict) and sub_cfg.get('model_name'):
                                url = sub_cfg.get('base_url', global_url)
                                models_to_purge.add((sub_cfg['model_name'], url))

            for model, base_url in models_to_purge:
                base_url = base_url or 'http://localhost:11434/v1'
                if base_url.endswith('/v1'):
                    endpoint = base_url[:-3] + '/api/generate'
                else:
                    endpoint = base_url.rstrip('/') + '/api/generate'
                    
                data = json.dumps({"model": model, "keep_alive": 0}).encode('utf-8')
                req = urllib.request.Request(endpoint, data=data, headers={'Content-Type': 'application/json'}, method='POST')
                try:
                    with urllib.request.urlopen(req, timeout=5) as response:
                        pass
                except Exception:
                    pass
        except Exception as e:
            print(f"\n[VRAM Manager] Failed to purge model from VRAM: {e}")
            
    # Run in a background thread to keep it lightweight and non-blocking
    threading.Thread(target=purge_task, daemon=True).start()

