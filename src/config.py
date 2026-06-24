"""
Global Configuration Loader
Acts as the single source of truth for all application settings.

WARNING: This file should NOT be edited directly by the user!
This module automatically loads its definitions and imports strictly from 
the central `config.yaml` file located at the project root. 
To modify system behavior, please edit `config.yaml` instead.
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

class ConfigLoader:
    _config = None

    @classmethod
    def get_config(cls):
        """Loads and caches the configuration from config.yaml."""
        if cls._config is None:
            # Dynamically calculate the root directory relative to this file
            # src/config.py -> parent is src/ -> parent is SemanticPrism/
            root_dir = Path(__file__).parent.parent
            config_path = root_dir / 'config.yaml'
            env_path = root_dir / '.env'
            
            # Load environment variables
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)
            
            with open(config_path, 'r', encoding='utf-8') as f:
                cls._config = yaml.safe_load(f)
                
            # Automatically override the API key from the environment
            env_key = os.getenv("GEMINI_API_KEY")
            if env_key:
                if "llm" in cls._config:
                    cls._config["llm"]["api_key"] = env_key
                for stage in ["extraction", "refinement", "synthesis"]:
                    if (
                        stage in cls._config 
                        and isinstance(cls._config[stage], dict) 
                        and "llm" in cls._config[stage]
                        and isinstance(cls._config[stage]["llm"], dict)
                    ):
                        cls._config[stage]["llm"]["api_key"] = env_key
                
        return cls._config

# Expose a global, read-only dictionary that any file can import
settings = ConfigLoader.get_config()

# Monkey patch pydantic_ai to fix Ollama Nil-Content Bug.
# Ollama's OpenAI compatibility layer fails with a 400 Bad Request error if the content of any message is null/nil.
# By forcing content to be an empty string if it is None, we keep Ollama happy.
try:
    from pydantic_ai.models.openai import OpenAIChatModel
    _original_map_messages = OpenAIChatModel._map_messages
    async def _patched_map_messages(self, *args, **kwargs):
        messages = await _original_map_messages(self, *args, **kwargs)
        if messages:
            for msg in messages:
                if msg.get("content") is None:
                    msg["content"] = ""
        return messages
    OpenAIChatModel._map_messages = _patched_map_messages
except Exception:
    pass
