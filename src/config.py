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
            if env_key and "llm" in cls._config:
                cls._config["llm"]["api_key"] = env_key
                
        return cls._config

# Expose a global, read-only dictionary that any file can import
settings = ConfigLoader.get_config()
