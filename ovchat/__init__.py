"""OpenVINO GenAI Chat Application."""

# Run dependency verifier on every launch before any submodules load
from .verifier import verify_dependencies
verify_dependencies(auto_install=True, show_banner=True)

from .chat import Chat
from .cli import main
from .config import CHAT_HISTORY_DIR, MODEL_ROOT
from .devices import get_devices
from .history import ChatHistorySaver
from .metadata import ModelMetadata, read_model_metadata
from .models import find_models

__version__ = "0.1.0"

__all__ = [
    "Chat",
    "main",
    "find_models",
    "get_devices",
    "ModelMetadata",
    "read_model_metadata",
    "ChatHistorySaver",
    "MODEL_ROOT",
    "CHAT_HISTORY_DIR",
]
