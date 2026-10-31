"""
Persistent settings management for ovchat.
Coordinates defaults, environment variables (OVCHAT_*), and user_config.json.
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .config import (
    CHAT_HISTORY_DIR,
    DEFAULT_CONTEXT,
    DEFAULT_ENABLE_MOE_OFFLOAD,
    DEFAULT_ENABLE_REASONING,
    DEFAULT_HETERO_PRIORITY,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_DISTRIBUTION_POLICY,
    DEFAULT_MOE_OFFLOAD_RATIO,
    MODEL_ROOT,
)

CONFIG_FILE = Path(__file__).resolve().parent.parent / "user_config.json"


@dataclass
class Settings:
    """Strongly typed application configuration with dict compatibility."""

    model_root: str = str(MODEL_ROOT)
    history_dir: str = str(CHAT_HISTORY_DIR)
    selected_model: str = ""
    selected_device: str = "GPU"
    hetero_priorities: str = DEFAULT_HETERO_PRIORITY
    model_distribution_policy: str = DEFAULT_MODEL_DISTRIBUTION_POLICY
    enable_moe_offload: bool = DEFAULT_ENABLE_MOE_OFFLOAD
    moe_offload_ratio: int = DEFAULT_MOE_OFFLOAD_RATIO
    context_length: int = DEFAULT_CONTEXT
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = 0.7
    top_p: float = 0.95
    enable_reasoning: bool = DEFAULT_ENABLE_REASONING

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # Dictionary compatibility helpers
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def update(self, other: Dict[str, Any]) -> None:
        for k, v in other.items():
            if hasattr(self, k):
                # Type coercion
                target_type = type(getattr(self, k))
                try:
                    if target_type == bool and isinstance(v, str):
                        setattr(self, k, v.lower() in ("1", "true", "yes"))
                    else:
                        setattr(self, k, target_type(v))
                except Exception:
                    setattr(self, k, v)


DEFAULT_SETTINGS: Dict[str, Any] = Settings().to_dict()


def load_settings() -> Settings:
    """Loads persistent user configuration, overlaying file & environment variables."""
    cfg = Settings()

    # 1. Overlay from user_config.json if present
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, dict):
                    cfg.update(saved)
        except Exception:
            pass

    # 2. Overlay explicit environment variable overrides
    if "OVCHAT_MODEL_ROOT" in os.environ:
        cfg.model_root = os.environ["OVCHAT_MODEL_ROOT"]
    if "OVCHAT_HISTORY_DIR" in os.environ:
        cfg.history_dir = os.environ["OVCHAT_HISTORY_DIR"]
    if "OVCHAT_DEVICE" in os.environ:
        cfg.selected_device = os.environ["OVCHAT_DEVICE"]
    if "OVCHAT_ENABLE_REASONING" in os.environ:
        cfg.enable_reasoning = os.environ["OVCHAT_ENABLE_REASONING"].lower() in ("1", "true", "yes")
    if "OVCHAT_ENABLE_MOE_OFFLOAD" in os.environ:
        cfg.enable_moe_offload = os.environ["OVCHAT_ENABLE_MOE_OFFLOAD"].lower() not in ("0", "false", "no")
    if "OVCHAT_MOE_OFFLOAD_RATIO" in os.environ:
        try:
            cfg.moe_offload_ratio = int(os.environ["OVCHAT_MOE_OFFLOAD_RATIO"])
        except ValueError:
            pass

    return cfg


def save_settings(new_settings: Any) -> Dict[str, Any]:
    """Updates and saves persistent user configuration to user_config.json."""
    current = load_settings()
    if isinstance(new_settings, dict):
        current.update(new_settings)
    elif isinstance(new_settings, Settings):
        current = new_settings

    payload = current.to_dict()
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass
    return payload
