import os
import json
import logging
import httpx
from openai import AsyncOpenAI, OpenAI
import instructor

from src.core.logger import get_logger

logger = get_logger("PublicLLMProvider")

def _transform_vertex_request(request: httpx.Request, token: str) -> httpx.Request:
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
    
    url_str = str(request.url).replace("/chat/completions", "")
    return httpx.Request(
        "POST",
        url_str,
        json=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    )

def _transform_vertex_response(res: httpx.Response, request: httpx.Request) -> httpx.Response:
    import re
    if res.status_code == 200:
        vertex_data = res.json()
        content = ""
        if "predictions" in vertex_data and len(vertex_data["predictions"]) > 0:
            content = vertex_data["predictions"][0]
            if not isinstance(content, str):
                content = str(content)
        
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

class PublicLLMProvider:
    def __init__(self, config):
        self.config = config
        self.model_name = "google-vertex-model"
        self.base_url = None
        self.api_key = "dummy_key"
        self.instructor_mode = instructor.Mode.JSON_SCHEMA
        self._initialize_vertex()

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

    def _initialize_vertex(self):
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
            dedicated_dns = None
            cfg_base_url = self.config['llm'].get('base_url', '')
            if cfg_base_url and '.goog' in cfg_base_url:
                dedicated_dns = cfg_base_url.replace("https://", "").replace("http://", "").split("/")[0]
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
                
            self.api_key = self._get_vertex_token()

    def get_sync_client(self):
        logger.debug("Initializing Synchronous Public SDK Client (VertexAI).")
        http_client = httpx.Client(transport=VertexTransport(self.api_key))
        return instructor.from_openai(
            OpenAI(base_url=self.base_url, api_key=self.api_key, http_client=http_client),
            mode=self.instructor_mode
        )

    def get_async_client(self):
        logger.debug("Initializing Asynchronous Public SDK Client (VertexAI).")
        http_client = httpx.AsyncClient(transport=AsyncVertexTransport(self.api_key))
        return instructor.from_openai(
            AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, http_client=http_client),
            mode=self.instructor_mode
        )

    def execute_http_raw(self, system_prompt: str, user_prompt: str, response_model):
        import urllib.request
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
