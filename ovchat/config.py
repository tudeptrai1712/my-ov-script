import os
from pathlib import Path

# Base directory where OpenVINO IR models are stored
MODEL_ROOT = Path(
    os.environ.get("OVCHAT_MODEL_ROOT", r"D:\AI models\openvino-genai")
)

# Default generation parameters
DEFAULT_CONTEXT: int = 32768
DEFAULT_MAX_NEW_TOKENS: int = 8192

# Rough fallback estimate for KV-cache size per token (in bytes)
DEFAULT_KV_BYTES_PER_TOKEN: int = 4096

# Estimated runtime framework overhead (in GB)
RUNTIME_OVERHEAD_GB: float = 0.75

# Memory polling frequency (in seconds)
MEMORY_POLL_INTERVAL: float = 0.25

# Default reasoning / thinking mode state
DEFAULT_ENABLE_REASONING: bool = os.environ.get(
    "OVCHAT_ENABLE_REASONING", ""
).strip().lower() in ("1", "true", "yes")

# Directory where chat history transcripts are always saved
CHAT_HISTORY_DIR: Path = Path(
    os.environ.get(
        "OVCHAT_HISTORY_DIR",
        r"D:\AI models\openvino-genai\chat history",
    )
)

