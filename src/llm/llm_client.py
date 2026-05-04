import yaml
import logging
import asyncio
import warnings

# Suppress harmless Python 3.10 deprecation probes triggered natively by google.api_core/instructor scanning
warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")

from src.core.logger import get_logger
from src.llm.local_llm import LocalLLMProvider
from src.llm.public_llm import PublicLLMProvider
from pydantic_ai import Agent

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
            print(f"[LLM Diagnostic] Type: {provider_type} | Model: {self.provider.model_name} | Context: {actual_ctx} | Native Pydantic AI")
        
        try:
            model = self.provider.get_model()
            agent = Agent(model, system_prompt=system_prompt, output_type=response_model, retries=3)
            
            result = await agent.run(user_prompt)
                
            if hasattr(self.provider, 'release_vram'):
                self.provider.release_vram()
            return result.output
        except Exception as e:
            self.error_history.append(str(e))
            logger.error(f"Pydantic AI Execution failed: {e}")
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
            print(f"[LLM Diagnostic] Type: {provider_type} | Model: {self.provider.model_name} | Context: {actual_ctx} | Native Pydantic AI")
        
        try:
            model = self.provider.get_model()
            agent = Agent(model, system_prompt=system_prompt, output_type=response_model, retries=3)
            
            result = agent.run_sync(user_prompt)
                
            if hasattr(self.provider, 'release_vram'):
                self.provider.release_vram()
            return result.output
        except Exception as e:
            self.error_history.append(str(e))
            logger.error(f"Pydantic AI Execution failed: {e}")
            if hasattr(self.provider, 'release_vram'):
                self.provider.release_vram()
            return None
