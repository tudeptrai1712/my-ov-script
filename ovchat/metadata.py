import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openvino as ov

from .models import model_size


@dataclass
class ModelMetadata:
    """Parsed metadata for an OpenVINO model."""

    path: Path
    name: str
    model_type: str = "unknown"
    architecture: str = "unknown"
    is_vlm: bool = False
    precision: str = "unknown"
    max_position_embeddings: Optional[int] = None
    supports_reasoning: bool = False
    weight_size_bytes: int = 0

    @property
    def display_type(self) -> str:
        return "Vision-Language Model (VLM)" if self.is_vlm else "Language Model (LLM)"


def read_model_metadata(model_path: Path) -> ModelMetadata:
    """Reads metadata files in the model directory to determine model capabilities."""
    name = model_path.name
    config_path = model_path / "config.json"
    ov_config_path = model_path / "openvino_config.json"
    tokenizer_config_path = model_path / "tokenizer_config.json"
    chat_template_path = model_path / "chat_template.jinja"

    config: Dict = {}
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            pass

    ov_config: Dict = {}
    if ov_config_path.exists():
        try:
            with open(ov_config_path, encoding="utf-8") as f:
                ov_config = json.load(f)
        except Exception:
            pass

    # 1. Architecture and model type
    model_type = config.get("model_type", "unknown")
    architectures = config.get("architectures", [])
    architecture = architectures[0] if architectures else model_type

    # 2. VLM vs LLM detection
    is_vlm = (
        (model_path / "openvino_vision_embeddings_model.xml").exists()
        or (model_path / "processor_config.json").exists()
        or (model_path / "preprocessor_config.json").exists()
        or "vision_config" in config
        or "audio_config" in config
        or any(
            keyword in architecture.lower()
            for keyword in ("conditionalgeneration", "vlm", "vision", "gemma4")
        )
    )

    # 3. Precision / Quantization dtype
    precision = (
        ov_config.get("dtype")
        or config.get("dtype")
        or config.get("torch_dtype", "unknown")
    )
    if isinstance(precision, str):
        precision = precision.lower()

    # 4. Context length / max position embeddings
    text_config = config.get("text_config", {})
    max_position = (
        text_config.get("max_position_embeddings")
        or config.get("max_position_embeddings")
        or config.get("max_sequence_length")
    )

    # 5. Reasoning / thinking capability detection
    supports_reasoning = False
    if chat_template_path.exists():
        try:
            template_text = chat_template_path.read_text(encoding="utf-8")
            if any(
                tag in template_text
                for tag in ("enable_thinking", "<|think|>", "<think>", "thought")
            ):
                supports_reasoning = True
        except Exception:
            pass

    if not supports_reasoning and tokenizer_config_path.exists():
        try:
            with open(tokenizer_config_path, encoding="utf-8") as f:
                tok_config = json.load(f)
            if (
                "think_token" in tok_config.get("model_specific_special_tokens", {})
                or "thinking" in str(tok_config.get("response_schema", ""))
            ):
                supports_reasoning = True
        except Exception:
            pass

    # 6. Weight size
    weight_size = model_size(model_path)

    return ModelMetadata(
        path=model_path,
        name=name,
        model_type=str(model_type),
        architecture=str(architecture),
        is_vlm=is_vlm,
        precision=str(precision),
        max_position_embeddings=max_position,
        supports_reasoning=supports_reasoning,
        weight_size_bytes=weight_size,
    )


def check_device_compatibility(
    device: str,
    metadata: ModelMetadata,
    core: Optional[ov.Core] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Checks if a given hardware device can run the specified model.
    Returns (is_compatible, reason_if_incompatible).
    """
    device_upper = device.upper()

    # CPU is the reference baseline and always supported
    if device_upper.startswith("CPU"):
        return True, None

    # GPU supports LLMs and VLMs across INT4/INT8/FP16
    if device_upper.startswith("GPU"):
        return True, None

    # NPU has specific restrictions in OpenVINO GenAI
    if device_upper.startswith("NPU"):
        # 1. VLMPipeline does not support NPU
        if metadata.is_vlm:
            return (
                False,
                "Vision-Language Models (VLM) are not supported on NPU in OpenVINO GenAI",
            )

        # 2. Check NPU hardware optimization capabilities
        if core is None:
            core = ov.Core()

        try:
            capabilities = core.get_property("NPU", "OPTIMIZATION_CAPABILITIES")
        except Exception:
            capabilities = []

        # If model is INT4 and NPU doesn't list INT4 capability
        if "int4" in metadata.precision and "INT4" not in capabilities:
            return (
                False,
                f"INT4 precision is not supported by NPU compiler (supports: {', '.join(capabilities) or 'FP16/INT8 only'})",
            )

        # 3. Check NPU platform compiler support
        try:
            platform = core.get_property("NPU", "NPU_PLATFORM")
            if platform == "AUTO_DETECT":
                arch = core.get_property("NPU", "DEVICE_ARCHITECTURE")
                if str(arch) in ("5010",):
                    return (
                        False,
                        f"NPU compiler does not support platform 'AUTO_DETECT' for architecture {arch}",
                    )
        except Exception:
            pass

        # 4. Whitelisted architectures for NPU in OpenVINO GenAI
        npu_supported_architectures = ("llama", "qwen2", "mistral", "phi3")
        model_type_lower = metadata.model_type.lower()
        if not any(arch in model_type_lower for arch in npu_supported_architectures):
            return (
                False,
                f"Model type '{metadata.model_type}' is not supported on NPU (NPU requires Llama/Qwen2/Mistral/Phi3)",
            )

        return True, None

    return True, None

