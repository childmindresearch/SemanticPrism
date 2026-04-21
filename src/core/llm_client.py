"""
SemanticPrism: LLM Client Abstraction Layer
Centralized dynamic factory cleanly routing configurable logic.
Supports both SDK (Instructor/OpenAI) and Raw HTTP mappings.
"""

import os
import json
import logging
import asyncio
import urllib.request
import urllib.error
import warnings

# Suppress harmless Python 3.10 deprecation probes triggered natively by google.api_core/instructor scanning
warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")

import yaml
import httpx
from openai import AsyncOpenAI, OpenAI
import instructor

from src.core.logger import get_logger
from src.helpers.context_manager import ContextManager

logger = get_logger("SemanticLLMClient")

def _transform_vertex_request(request: httpx.Request, token: str) -> httpx.Request:
    import json
    body = json.loads(request.content)
    system_prompt = ""
    user_prompt = ""
    for msg in body.get("messages", []):
        if msg["role"] == "system":
            system_prompt += msg["content"] + "\n"
        else:
            user_prompt += msg["content"] + "\n"
            
    if "response_format" in body:
        fmt = body["response_format"]
        if fmt.get("type") == "json_schema":
            schema_dump = json.dumps(fmt.get("json_schema", {}).get("schema", {}), indent=2)
            system_prompt += f"\nCRITICAL: Your output MUST be a valid, populated JSON INSTANCE that strictly conforms to the following JSON Schema. Do NOT output the schema definition itself (no $defs, type declarations, etc.). Only output the factual JSON data:\n{schema_dump}\n"
        elif fmt.get("type") == "json_object":
            system_prompt += "\nYou MUST return a valid JSON object.\n"
            
    full_prompt = f"{system_prompt}\n\n{user_prompt}".strip()
    payload = {
        "instances": [{
            "prompt": full_prompt,
            "max_tokens": body.get("max_tokens", 8192)
        }],
        "parameters": {
            "temperature": body.get("temperature", 0.1),
            "maxOutputTokens": body.get("max_tokens", 8192)
        }
    }
    
    # Clean the URL to match the :predict endpoint instead of OpenAI's /chat/completions suffix
    url_str = str(request.url).replace("/chat/completions", "")
    return httpx.Request(
        "POST",
        url_str,
        json=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    )

def _transform_vertex_response(res: httpx.Response, request: httpx.Request) -> httpx.Response:
    import json
    import re
    if res.status_code == 200:
        vertex_data = res.json()
        content = ""
        if "predictions" in vertex_data and len(vertex_data["predictions"]) > 0:
            content = vertex_data["predictions"][0]
            if not isinstance(content, str):
                content = str(content)
        
        # Clean Gemma Output format
        if "Output:\n" in content:
            content = content.split("Output:\n")[-1]
            
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if match:
            content = match.group(1)
        else:
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                content = content[start_idx:end_idx+1]
                
        openai_res = {
            "id": "chatcmpl-vertex",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "vertex",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        return httpx.Response(status_code=200, json=openai_res, request=request)
    return res

class VertexTransport(httpx.BaseTransport):
    def __init__(self, token: str):
        self.token = token
        self.client = httpx.Client(timeout=120.0)
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        v_req = _transform_vertex_request(request, self.token)
        v_res = self.client.send(v_req)
        return _transform_vertex_response(v_res, request)

class AsyncVertexTransport(httpx.AsyncBaseTransport):
    def __init__(self, token: str):
        self.token = token
        self.client = httpx.AsyncClient(timeout=120.0)
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        v_req = _transform_vertex_request(request, self.token)
        v_res = await self.client.send(v_req)
        return _transform_vertex_response(v_res, request)

class SemanticLLMClient:
    def __init__(self, config_path: str = "config.yaml"):
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            logger.critical(f"Config load failed: {e}")
            raise e
            
        self.backend = self.config['llm'].get('api_backend', 'ollama').lower()
        self.base_url = self.config['llm'].get('base_url', 'http://localhost:11434/v1')
        self.api_key = self.config['llm'].get('api_key', 'ollama')
        self.model_name = self.config['llm']['model_name']
        
        self.connection_protocol = self.config['llm'].get('connection_protocol', 'sdk')
        self.context_source = self.config['llm'].get('context_source', 'dynamic')
        self.fixed_num_ctx = self.config['llm'].get('fixed_num_ctx', 8192)
        self.manage_vram = self.config['llm'].get('manage_vram', False)
        
        if self.backend == 'vertexai':
            google_credentials = self.config['llm'].get('google_credentials_path')
            if google_credentials:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(google_credentials)
            
            vertex_project = self.config['llm'].get('vertex_project')
            if vertex_project:
                os.environ["VERTEX_PROJECT"] = vertex_project
                
            vertex_location = self.config['llm'].get('vertex_location')
            if vertex_location:
                os.environ["VERTEX_LOCATION"] = vertex_location
                
            vertex_endpoint = self.config['llm'].get('vertex_endpoint')
            if vertex_endpoint and vertex_project and vertex_location:
                # IsoMorph Approach: Dynamically construct Vertex base_url to dedicated DNS for OpenAI compatibility
                dedicated_dns = None
                if self.base_url and '.goog' in self.base_url:
                    dedicated_dns = self.base_url.replace("https://", "").replace("http://", "").split("/")[0]
                else:
                    try:
                        from google.cloud import aiplatform
                        endpoint = aiplatform.Endpoint(f"projects/{vertex_project}/locations/{vertex_location}/endpoints/{vertex_endpoint}")
                        if endpoint.dedicated_endpoint_dns:
                            dedicated_dns = endpoint.dedicated_endpoint_dns
                    except Exception as e:
                        logger.warning(f"Could not resolve dedicated endpoint DNS: {e}")
                
                if dedicated_dns:
                    self.base_url = f"https://{dedicated_dns}/v1/projects/{vertex_project}/locations/{vertex_location}/endpoints/{vertex_endpoint}:predict"
                else:
                    self.base_url = f"https://{vertex_location}-aiplatform.googleapis.com/v1/projects/{vertex_project}/locations/{vertex_location}/endpoints/{vertex_endpoint}:predict"
                    
                self.model_name = "google-vertex-model"
                
                # Retrieve the active bearer token for OpenAI client
                self.api_key = self._get_vertex_token()
        
        # Instructor JSON/Tool bindings depend critically on the underlying Provider SDK
        if self.backend == 'ollama':
            self.instructor_mode = instructor.Mode.JSON
        elif self.backend == 'vertexai':
            self.instructor_mode = instructor.Mode.JSON_SCHEMA
        else:
            self.instructor_mode = instructor.Mode.TOOLS

        # Initialize Hardware Context Manager statically
        # Only allocate local hardware constraints if running a local backend
        if self.context_source == "dynamic" and self.backend == "ollama":
            self.context_manager = ContextManager(self.model_name)
        else:
            self.context_manager = None
            
        self.context_history = []
        self.error_history = []

    def get_context_size(self, target_tokens: int = 8192) -> int:
        """Determines the appropriate VRAM threshold based on configuration."""
        if self.context_source == "dynamic" and self.context_manager:
            return self.context_manager.calculate_safe_bounds(target_tokens)
        return self.fixed_num_ctx

    def release_vram(self):
        """
        Actively forces Ollama to drop the model from GPU to prevent thrashing.
        """
        if not self.manage_vram or self.backend != 'ollama':
            return
            
        logger.debug("Executing manual GPU memory flush.")
        
        # Parse the raw API path from the OpenAI-compatible v1 base url
        endpoint = self.base_url.replace('/v1', '/api/generate') if '/v1' in self.base_url else "http://localhost:11434/api/generate"
        payload = {"model": self.model_name, "keep_alive": 0}
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(endpoint, data=data, headers={'Content-Type': 'application/json'})
        
        try:
            with urllib.request.urlopen(req) as response:
                pass 
        except Exception as e:
            logger.warning(f"VRAM clear structurally bypassed: {e}")

    def _get_vertex_token(self) -> str:
        try:
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request
            
            cred_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if not cred_file:
                logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set. Vertex auth may fail.")
                return "dummy_key"
            
            SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
            credentials = service_account.Credentials.from_service_account_file(
                cred_file, 
                scopes=SCOPES
            )
            credentials.refresh(Request())
            return credentials.token
        except Exception as e:
            logger.error(f"Could not fetch Google Auth token: {e}")
            return "dummy_key"

    # -------- SDK ROUTING -------- #

    def get_sync_client(self):
        logger.debug(f"Initializing Synchronous SDK Client ({self.backend}).")
        
        http_client = None
        if self.backend == 'vertexai':
            http_client = httpx.Client(transport=VertexTransport(self.api_key))
            
        return instructor.from_openai(
            OpenAI(base_url=self.base_url, api_key=self.api_key, http_client=http_client),
            mode=self.instructor_mode
        )

    def get_async_client(self):
        logger.debug(f"Initializing Asynchronous SDK Client ({self.backend}).")
        
        http_client = None
        if self.backend == 'vertexai':
            http_client = httpx.AsyncClient(transport=AsyncVertexTransport(self.api_key))
            
        return instructor.from_openai(
            AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, http_client=http_client),
            mode=self.instructor_mode
        )

    # -------- HTTP RAW ROUTING -------- #

    def _execute_http_raw(self, system_prompt: str, user_prompt: str, response_model, num_ctx: int) -> dict:
        logger.debug("Executing raw native HTTP sequence.")
        
        # Inject the Pydantic JSON schema constraints directly into the prompt model
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
                    # Validate output using standard Pydantic logic natively mapping dictionary arrays
                    return response_model.model_validate_json(result["response"])
        except Exception as e:
            logger.error(f"HTTP Raw Execution failed securely returning None map: {e}")

    def _execute_vertex_raw(self, system_prompt: str, user_prompt: str, response_model, num_ctx: int):
        logger.debug("Executing raw native Vertex AI sequence via OpenAI-compatible REST tunnel.")
        
        schema_format = response_model.model_json_schema()
        custom_system = f"{system_prompt}\n\nCRITICAL: Your output MUST be a valid, populated JSON INSTANCE that strictly conforms to the following JSON Schema. Do NOT output the schema definition itself (no $defs, type declarations, etc.). Only output the factual JSON data:\n{json.dumps(schema_format)}"
        
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": custom_system},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
            "stream": False
        }
        
        data = json.dumps(payload).encode('utf-8')
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        req = urllib.request.Request(endpoint, data=data, headers=headers)
        
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    
                    import re
                    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
                    if match:
                        clean_json = match.group(1)
                    else:
                        content = content.strip()
                        start_idx = content.find('{')
                        end_idx = content.rfind('}')
                        if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                            clean_json = content[start_idx:end_idx+1]
                        else:
                            clean_json = content
                            
                    return response_model.model_validate_json(clean_json)
        except Exception as e:
            logger.error(f"Vertex HTTP Raw Execution failed securely: {e}")
            return None

    # -------- PUBLIC INTERFACES -------- #

    async def safe_api_call_async(self, system_prompt: str, user_prompt: str, response_model, num_ctx: int = 8192):
        actual_ctx = self.get_context_size(num_ctx)
        self.context_history.append(actual_ctx)
        
        try:
            if self.connection_protocol == "http" and self.backend != 'vertexai':
                result = await asyncio.to_thread(self._execute_http_raw, system_prompt, user_prompt, response_model, actual_ctx)
            else:
                client = self.get_async_client()
                
                kwargs = {"max_retries": 3,
                    "model": self.model_name,
                    "response_model": response_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }
                
                if self.backend == 'ollama':
                    kwargs['extra_body'] = {"options": {"num_ctx": actual_ctx}}
                    
                result = await client.chat.completions.create(**kwargs)
                
            self.release_vram()
            return result
        except Exception as e:
            self.error_history.append(str(e))
            logger.error(f"Instructor API Execution failed: {e}")
            self.release_vram()
            return None

    def safe_api_call_sync(self, system_prompt: str, user_prompt: str, response_model, num_ctx: int = 8192):
        actual_ctx = self.get_context_size(num_ctx)
        self.context_history.append(actual_ctx)
        
        try:
            if self.connection_protocol == "http" and self.backend != 'vertexai':
                result = self._execute_http_raw(system_prompt, user_prompt, response_model, actual_ctx)
            else:
                client = self.get_sync_client()
                
                kwargs = {"max_retries": 3,
                    "model": self.model_name,
                    "response_model": response_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }
                
                if self.backend == 'ollama':
                    kwargs['extra_body'] = {"options": {"num_ctx": actual_ctx}}
                    
                result = client.chat.completions.create(**kwargs)
                
            self.release_vram()
            return result
        except Exception as e:
            self.error_history.append(str(e))
            logger.error(f"Instructor API Execution failed: {e}")
            self.release_vram()
            return None
