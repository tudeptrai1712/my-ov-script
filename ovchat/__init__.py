"""OpenVINO GenAI Chat Application."""

from .chat import Chat
from .cli import main
from .config import MODEL_ROOT
from .devices import get_devices
from .models import find_models

__version__ = "0.1.0"

__all__ = [
    "Chat",
    "main",
    "find_models",
    "get_devices",
    "MODEL_ROOT",
]

