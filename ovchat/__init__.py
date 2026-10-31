"""OpenVINO GenAI Chat Application."""

from .chat import Chat
from .cli import main
from .commands import CommandRouter
from .config import CHAT_HISTORY_DIR, MODEL_ROOT
from .devices import HardwareSpecs, get_devices, get_hardware_specs, render_hardware_and_models
from .engine import GenerationResult, InferenceEngine
from .history import ChatHistorySaver
from .metadata import ModelMetadata, read_model_metadata
from .models import find_models
from .verifier import verify_dependencies

__version__ = "0.2.0"

__all__ = [
    "Chat",
    "main",
    "InferenceEngine",
    "GenerationResult",
    "CommandRouter",
    "find_models",
    "get_devices",
    "get_hardware_specs",
    "render_hardware_and_models",
    "HardwareSpecs",
    "ModelMetadata",
    "read_model_metadata",
    "ChatHistorySaver",
    "MODEL_ROOT",
    "CHAT_HISTORY_DIR",
    "verify_dependencies",
]
