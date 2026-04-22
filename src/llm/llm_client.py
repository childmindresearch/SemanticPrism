import yaml
import logging
import asyncio
import warnings

# Suppress harmless Python 3.10 deprecation probes triggered natively by google.api_core/instructor scanning
warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")

from src.core.logger import get_logger
from src.llm.local_llm import LocalLLMProvider
from src.llm.public_llm import PublicLLMProvider

logger = get_logger("SemanticLLMClient")

class SemanticLLMClient:
    def __init__(self, config_path: str = "config.yaml"):
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logger.critical(f"Config load failed: {e}")
            raise e
            
        self.backend = self.config['llm'].get('api_backend', 'ollama').lower()
        self.connection_protocol = self.config['llm'].get('connection_protocol', 'sdk')
        self.verbose = self.config['llm'].get('verbose', True)
        
        if self.backend == 'vertexai':
            self.provider = PublicLLMProvider(self.config)
        else:
            self.provider = LocalLLMProvider(self.config)
            
        self.context_history = []
        self.error_history = []

    @property
    def model_name(self):
        return self.provider.model_name

    async def safe_api_call_async(self, system_prompt: str, user_prompt: str, response_model, num_ctx: int = 8192):
        actual_ctx = num_ctx
        if isinstance(self.provider, LocalLLMProvider):
            actual_ctx = self.provider.get_context_size(num_ctx)
        self.context_history.append(actual_ctx)
        
        if self.verbose:
            provider_type = "public" if self.backend == 'vertexai' else "local"
            print(f"[LLM Diagnostic] Type: {provider_type} | Model: {self.provider.model_name} | Context: {actual_ctx} | Protocol: {self.connection_protocol} ({self.backend})")
        
        try:
            if self.connection_protocol == "http" and self.backend != 'vertexai':
                result = await asyncio.to_thread(self.provider.execute_http_raw, system_prompt, user_prompt, response_model, actual_ctx)
            elif self.connection_protocol == "http" and self.backend == 'vertexai':
                result = await asyncio.to_thread(self.provider.execute_http_raw, system_prompt, user_prompt, response_model)
            else:
                client = self.provider.get_async_client()
                
                kwargs = {"max_retries": 3,
                    "model": self.provider.model_name,
                    "response_model": response_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }
                
                if self.backend == 'ollama':
                    kwargs['extra_body'] = {"options": {"num_ctx": actual_ctx}}
                    
                result = await client.chat.completions.create(**kwargs)
                
            if hasattr(self.provider, 'release_vram'):
                self.provider.release_vram()
            return result
        except Exception as e:
            self.error_history.append(str(e))
            logger.error(f"Instructor API Execution failed: {e}")
            if hasattr(self.provider, 'release_vram'):
                self.provider.release_vram()
            return None

    def safe_api_call_sync(self, system_prompt: str, user_prompt: str, response_model, num_ctx: int = 8192):
        actual_ctx = num_ctx
        if isinstance(self.provider, LocalLLMProvider):
            actual_ctx = self.provider.get_context_size(num_ctx)
        self.context_history.append(actual_ctx)
        
        if self.verbose:
            provider_type = "public" if self.backend == 'vertexai' else "local"
            print(f"[LLM Diagnostic] Type: {provider_type} | Model: {self.provider.model_name} | Context: {actual_ctx} | Protocol: {self.connection_protocol} ({self.backend})")
        
        try:
            if self.connection_protocol == "http" and self.backend != 'vertexai':
                result = self.provider.execute_http_raw(system_prompt, user_prompt, response_model, actual_ctx)
            elif self.connection_protocol == "http" and self.backend == 'vertexai':
                result = self.provider.execute_http_raw(system_prompt, user_prompt, response_model)
            else:
                client = self.provider.get_sync_client()
                
                kwargs = {"max_retries": 3,
                    "model": self.provider.model_name,
                    "response_model": response_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }
                
                if self.backend == 'ollama':
                    kwargs['extra_body'] = {"options": {"num_ctx": actual_ctx}}
                    
                result = client.chat.completions.create(**kwargs)
                
            if hasattr(self.provider, 'release_vram'):
                self.provider.release_vram()
            return result
        except Exception as e:
            self.error_history.append(str(e))
            logger.error(f"Instructor API Execution failed: {e}")
            if hasattr(self.provider, 'release_vram'):
                self.provider.release_vram()
            return None
