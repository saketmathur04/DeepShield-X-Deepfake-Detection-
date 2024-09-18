"""
DeepShield-X — Configuration System

YAML-based config with dot-access, defaults merging, and CLI override support.
"""

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


class Config:
    """
    Hierarchical configuration with dot-access.
    
    Usage:
        cfg = Config.from_yaml("configs/default.yaml")
        cfg.model.name          # "resnet50"
        cfg.training.epochs     # 20
        cfg.merge(overrides)    # merge CLI overrides
    """

    def __init__(self, data: Optional[Dict] = None):
        self._data = data or {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        if name not in self._data:
            raise AttributeError(f"Config has no attribute '{name}'")
        value = self._data[name]
        if isinstance(value, dict):
            return Config(value)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"Config({self._data})"

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value with a default fallback."""
        keys = key.split(".")
        data = self._data
        for k in keys:
            if isinstance(data, dict) and k in data:
                data = data[k]
            else:
                return default
        if isinstance(data, dict):
            return Config(data)
        return data

    def to_dict(self) -> Dict:
        """Convert config to a plain dictionary."""
        result = {}
        for key, value in self._data.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            elif isinstance(value, dict):
                result[key] = Config(value).to_dict()
            else:
                result[key] = value
        return result

    def merge(self, overrides: Dict) -> "Config":
        """
        Deep-merge overrides into this config. Returns self for chaining.
        """
        self._data = _deep_merge(self._data, overrides)
        return self

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load config from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(data)

    @classmethod
    def from_dict(cls, data: Dict) -> "Config":
        """Create config from a dictionary."""
        return cls(copy.deepcopy(data))

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Save config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def _deep_merge(base: Dict, overrides: Dict) -> Dict:
    """Recursively merge overrides into base dict."""
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    config_path: Union[str, Path] = "configs/default.yaml",
    overrides: Optional[Dict] = None,
    smoke_test: bool = False,
) -> Config:
    """
    Load configuration with optional overrides and smoke-test mode.

    Args:
        config_path: Path to the base YAML config.
        overrides: Dict of overrides to merge.
        smoke_test: If True, apply smoke-test settings.

    Returns:
        Merged Config object.
    """
    cfg = Config.from_yaml(config_path)

    if smoke_test and "smoke_test" in cfg:
        smoke_cfg = cfg.smoke_test.to_dict() if isinstance(cfg.smoke_test, Config) else cfg.smoke_test
        # Apply smoke test overrides
        training_overrides = {
            "training": {
                "epochs": smoke_cfg.get("epochs", 2),
                "batch_size": smoke_cfg.get("batch_size", 8),
                "num_workers": smoke_cfg.get("num_workers", 0),
            },
            "dataset": {
                "image_size": smoke_cfg.get("image_size", 64),
                "num_workers": smoke_cfg.get("num_workers", 0),
            },
            "evaluation": {
                "batch_size": smoke_cfg.get("batch_size", 8),
            },
            "explainability": {
                "num_samples": 5,
            },
        }
        cfg.merge(training_overrides)

    if overrides:
        cfg.merge(overrides)

    return cfg


def parse_cli_overrides(args: List[str]) -> Dict:
    """
    Parse CLI arguments like --model.name=resnet50 into a nested dict.

    Args:
        args: List of strings in "key=value" format.

    Returns:
        Nested dictionary of overrides.
    """
    overrides = {}
    for arg in args:
        if "=" not in arg:
            continue
        key, value = arg.lstrip("-").split("=", 1)
        # Try to parse value as Python literal
        try:
            import ast
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass  # Keep as string

        # Build nested dict from dotted key
        keys = key.split(".")
        current = overrides
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value

    return overrides
