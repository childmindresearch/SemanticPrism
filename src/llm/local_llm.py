import json
import logging
import urllib.request
import instructor
from openai import AsyncOpenAI, OpenAI

from src.core.logger import get_logger
from src.helpers.context_manager import ContextManager

logger = get_logger("LocalLLMProvider")

class LocalLLMProvider:
    def __init__(self, config):
        self.config = config
        self.backend = self.config['llm'].get('api_backend', 'ollama').lower()
        self.base_url = self.config['llm'].get('base_url', 'http://localhost:11434/v1')
        self.api_key = self.config['llm'].get('api_key', 'ollama')
        self.model_name = self.config['llm']['model_name']
        self.manage_vram = self.config['llm'].get('manage_vram', False)
        
        if self.backend == 'ollama':
            self.instructor_mode = instructor.Mode.JSON
        else:
            self.instructor_mode = instructor.Mode.TOOLS

        self.context_source = self.config['llm'].get('context_source', 'dynamic')
        self.fixed_num_ctx = self.config['llm'].get('fixed_num_ctx', 8192)
        
        if self.context_source == "dynamic" and self.backend == "ollama":
            self.context_manager = ContextManager(self.model_name)
        else:
            self.context_manager = None

    def get_context_size(self, target_tokens: int = 8192) -> int:
        if self.context_source == "dynamic" and self.context_manager:
            return self.context_manager.calculate_safe_bounds(target_tokens)
        return self.fixed_num_ctx

    def release_vram(self):
        if not self.manage_vram or self.backend != 'ollama':
            return
            
        logger.debug("Executing manual GPU memory flush.")
        
        endpoint = self.base_url.replace('/v1', '/api/generate') if '/v1' in self.base_url else "http://localhost:11434/api/generate"
        payload = {"model": self.model_name, "keep_alive": 0}
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(endpoint, data=data, headers={'Content-Type': 'application/json'})
        
        try:
            with urllib.request.urlopen(req) as response:
                pass 
        except Exception as e:
            logger.warning(f"VRAM clear structurally bypassed: {e}")

    def get_sync_client(self):
        logger.debug(f"Initializing Synchronous Local SDK Client ({self.backend}).")
        return instructor.from_openai(
            OpenAI(base_url=self.base_url, api_key=self.api_key),
            mode=self.instructor_mode
        )

    def get_async_client(self):
        logger.debug(f"Initializing Asynchronous Local SDK Client ({self.backend}).")
        return instructor.from_openai(
            AsyncOpenAI(base_url=self.base_url, api_key=self.api_key),
            mode=self.instructor_mode
        )

    def execute_http_raw(self, system_prompt: str, user_prompt: str, response_model, num_ctx: int):
        logger.debug("Executing raw native HTTP sequence.")
        
        schema_format = response_model.model_json_schema()
        custom_system = f"{system_prompt}\n\nCRITICAL: Your output MUST be a valid, populated JSON INSTANCE that strictly conforms to the following JSON Schema. Do NOT output the schema definition itself (no $defs, type declarations, etc.). Only output the factual JSON data:\n{json.dumps(schema_format)}"
        
        endpoint = self.base_url.replace('/v1', '/api/generate') if '/v1' in self.base_url else "http://localhost:11434/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": f"{custom_system}\n\n{user_prompt}",
            "options": {"num_ctx": num_ctx},
            "format": "json",
            "stream": False
        }
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(endpoint, data=data, headers={'Content-Type': 'application/json'})
        
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                if "response" in result:
                    return response_model.model_validate_json(result["response"])
        except Exception as e:
            logger.error(f"HTTP Raw Execution failed securely returning None map: {e}")
            return None
