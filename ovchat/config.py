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

# Heterogeneous execution defaults (OpenVINO Pipeline Parallelism Preview)
DEFAULT_HETERO_PRIORITY: str = os.environ.get("OVCHAT_HETERO_PRIORITY", "GPU,CPU")
DEFAULT_HETERO_DEVICE: str = f"HETERO:{DEFAULT_HETERO_PRIORITY}"
DEFAULT_MODEL_DISTRIBUTION_POLICY: str = os.environ.get(
    "OVCHAT_MODEL_DISTRIBUTION_POLICY", "PIPELINE_PARALLEL"
)

# Mixture of Experts (MoE) offloading defaults (OpenVINO 2026 GPU plugin)
DEFAULT_ENABLE_MOE_OFFLOAD: bool = os.environ.get(
    "OVCHAT_ENABLE_MOE_OFFLOAD", "1"
).strip().lower() not in ("0", "false", "no")
DEFAULT_MOE_OFFLOAD_RATIO: int = int(os.environ.get("OVCHAT_MOE_OFFLOAD_RATIO", "50"))

# Directory where chat history transcripts are always saved
CHAT_HISTORY_DIR: Path = Path(
    os.environ.get(
        "OVCHAT_HISTORY_DIR",
        r"D:\AI models\openvino-genai\chat history",
    )
)
