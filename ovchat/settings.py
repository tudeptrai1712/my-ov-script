import json
from pathlib import Path
from typing import Any, Dict

from .config import (
    CHAT_HISTORY_DIR,
    DEFAULT_CONTEXT,
    DEFAULT_ENABLE_REASONING,
    DEFAULT_MAX_NEW_TOKENS,
    MODEL_ROOT,
)

CONFIG_FILE = Path(__file__).resolve().parent.parent / "user_config.json"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "model_root": str(MODEL_ROOT),
    "history_dir": str(CHAT_HISTORY_DIR),
    "selected_model": "",
    "selected_device": "GPU",
    "context_length": DEFAULT_CONTEXT,
    "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
    "temperature": 0.7,
    "top_p": 0.95,
    "enable_reasoning": DEFAULT_ENABLE_REASONING,
}


def load_settings() -> Dict[str, Any]:
    """Loads persistent user configuration, falling back to defaults."""
    settings = dict(DEFAULT_SETTINGS)
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if isinstance(saved, dict):
                    settings.update(saved)
        except Exception:
            pass
    return settings


def save_settings(new_settings: Dict[str, Any]) -> Dict[str, Any]:
    """Updates and saves persistent user configuration to user_config.json."""
    current = load_settings()
    current.update(new_settings)
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
    except Exception:
        pass
    return current
