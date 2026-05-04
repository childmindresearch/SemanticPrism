import json
import logging
import urllib.request
import httpx
from src.core.logger import get_logger
from src.helpers.context_manager import ContextManager
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from src.llm.model_profiles import get_model_profile

logger = get_logger("LocalLLMProvider")

class AsyncOllamaTransport(httpx.AsyncBaseTransport):
    """
    HTTP Interceptor (Transport Hook)
    ---------------------------------
    Pydantic AI generates OpenAI-compliant payloads where an AI tool call is accompanied
    by a `content: null` block. Ollama's underlying Go parser strictly rejects `null` (or `<nil>`) 
    in favor of an empty string `""`. 
    
    This interceptor catches the outbound HTTP payload nanoseconds before it leaves the application,
    scans the message array, and natively patches any `null` contents into `""` if tool calls are present.
    This provides total framework immunity without modifying brittle site-packages.
    """
    def __init__(self, fallback_transport: httpx.AsyncBaseTransport = None):
        self._fallback_transport = fallback_transport or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # Only intercept JSON payloads heading out to the LLM
        if request.content and request.headers.get("content-type") == "application/json":
            try:
                body = json.loads(request.content)
                modified = False
                
                # Iterate through the history/messages looking for the `<nil>` bug trigger
                if "messages" in body:
                    for msg in body["messages"]:
                        # If any message natively passes `content: null` or omits it entirely, force it to `""`
                        if msg.get("content") is None:
                            msg["content"] = ""
                            modified = True
                            
                # If we mutated the payload, we must rebuild the HTTP Request object natively
                if modified:
                    new_content = json.dumps(body).encode("utf-8")
                    request = httpx.Request(
                        method=request.method,
                        url=request.url,
                        headers=request.headers,
                        content=new_content
                    )
                    # Reset the Content-Length to match our new payload size securely
                    request.headers["content-length"] = str(len(new_content))
                    
            except Exception as e:
                # If JSON parsing fails, fail open and just send the raw request
                logger.debug(f"AsyncOllamaTransport parsing bypass: {e}")
                
        return await self._fallback_transport.handle_async_request(request)

class LocalLLMProvider:
    def __init__(self, config):
        self.config = config
        self.backend = self.config['llm'].get('api_backend', 'ollama').lower()
        self.base_url = self.config['llm'].get('base_url', 'http://localhost:11434/v1')
        self.api_key = self.config['llm'].get('api_key', 'ollama')
        self.model_name = self.config['llm']['model_name']
        self.manage_vram = self.config['llm'].get('manage_vram', False)
        
        # Legacy instructor mode tracking removed

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
        # 1. First, surgically close the Pydantic AI connection pool that is locking Ollama natively
        if hasattr(self, 'active_client') and self.active_client:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.active_client.aclose())
                else:
                    loop.run_until_complete(self.active_client.aclose())
            except Exception as e:
                logger.debug(f"Bypassed active client teardown natively: {e}")
            finally:
                self.active_client = None

        # 2. If config has manage_vram disabled, we stop here.
        if not self.manage_vram or self.backend != 'ollama':
            return
            
        logger.debug("Executing manual GPU memory flush natively.")
        
        endpoint = self.base_url.replace('/v1', '/api/generate') if '/v1' in self.base_url else "http://localhost:11434/api/generate"
        payload = {"model": self.model_name, "keep_alive": 0}
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(endpoint, data=data, headers={'Content-Type': 'application/json'})
        
        try:
            with urllib.request.urlopen(req) as response:
                pass 
        except Exception as e:
            logger.warning(f"VRAM clear structurally bypassed: {e}")

    def get_model(self):
        logger.debug(f"Initializing Local Pydantic AI Model ({self.backend}).")
        
        # Instantiate the custom interceptor to protect against Ollama <nil> bugs
        # Store it on `self` so we can securely close it inside `release_vram`
        self.active_client = httpx.AsyncClient(transport=AsyncOllamaTransport(), timeout=httpx.Timeout(200.0))
        provider = OllamaProvider(base_url=self.base_url, http_client=self.active_client)
        
        profile = get_model_profile(self.model_name)
        if profile:
            return OllamaModel(self.model_name, provider=provider, profile=profile)
        return OllamaModel(self.model_name, provider=provider)
