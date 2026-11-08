"""
Global Configuration Loader for Stage 3 Topology
Loads `configs/topology.yaml` (if present) or defaults to an empty dictionary.
"""

from pathlib import Path
import yaml


def deep_merge(dict_a: dict, dict_b: dict) -> dict:
    """Recursively merges dict_b into dict_a."""
    for key, value in dict_b.items():
        if key in dict_a and isinstance(dict_a[key], dict) and isinstance(value, dict):
            deep_merge(dict_a[key], value)
        else:
            dict_a[key] = value
    return dict_a


class ConfigLoader:
    _config = None

    @classmethod
    def get_config(cls) -> dict:
        """Loads, merges, and caches configuration from configs/*.yaml."""
        if cls._config is None:
            root_dir = Path(__file__).parent.parent
            configs_dir = root_dir / "configs"
            merged_config = {}

            if configs_dir.exists() and configs_dir.is_dir():
                for yf in sorted(configs_dir.glob("*.yaml")):
                    with open(yf, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                        deep_merge(merged_config, data)

            try:
                from model_config import EMBEDDING_MODEL
                merged_config.setdefault("refinement", {})
                merged_config["refinement"].setdefault("embedding_model", EMBEDDING_MODEL)
            except Exception:
                pass

            cls._config = merged_config

        return cls._config


settings = ConfigLoader.get_config()
