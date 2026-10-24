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
    is_moe: bool = False
    num_experts: Optional[int] = None
    top_k_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None

    @property
    def display_type(self) -> str:
        base = "Vision-Language Model (VLM)" if self.is_vlm else "Language Model (LLM)"
        if self.is_moe:
            summary = self.moe_summary
            return f"{base} [{summary}]" if summary else f"{base} [MoE]"
        return base

    @property
    def moe_summary(self) -> str:
        """Formatted MoE architecture summary."""
        if not self.is_moe:
            return ""
        if self.num_experts and self.top_k_experts:
            return f"MoE {self.top_k_experts}/{self.num_experts} active"
        if self.num_experts:
            return f"MoE {self.num_experts} experts"
        return "MoE"


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
    if not isinstance(text_config, dict):
        text_config = {}

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

    # 6. Mixture of Experts (MoE) detection
    enable_moe = (
        config.get("enable_moe_block")
        or text_config.get("enable_moe_block")
        or ("num_experts" in config and config.get("num_experts") is not None)
        or ("num_experts" in text_config and text_config.get("num_experts") is not None)
        or "moe" in str(architecture).lower()
        or "moe" in str(model_type).lower()
        or "-a4b" in name.lower()
        or "-a3b" in name.lower()
        or "-a1b" in name.lower()
    )
    num_experts = config.get("num_experts") or text_config.get("num_experts")
    top_k_experts = (
        config.get("top_k_experts")
        or text_config.get("top_k_experts")
        or config.get("num_experts_per_tok")
        or text_config.get("num_experts_per_tok")
    )
    moe_intermediate_size = config.get("moe_intermediate_size") or text_config.get("moe_intermediate_size")

    is_moe = bool(enable_moe or (num_experts and num_experts > 1))

    # 7. Weight size
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
        is_moe=is_moe,
        num_experts=int(num_experts) if num_experts else None,
        top_k_experts=int(top_k_experts) if top_k_experts else None,
        moe_intermediate_size=int(moe_intermediate_size) if moe_intermediate_size else None,
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

    # Heterogeneous execution: orchestrates across multiple devices (e.g. HETERO:GPU,CPU)
    if device_upper.startswith("HETERO"):
        return True, None

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

        # Whitelisted architectures for NPU in OpenVINO GenAI
        npu_supported_architectures = ("llama", "qwen", "mistral", "phi", "gemma", "deepseek")
        model_type_lower = metadata.model_type.lower()
        if not any(arch in model_type_lower for arch in npu_supported_architectures):
            return (
                False,
                f"Model type '{metadata.model_type}' is not supported on NPU (requires Llama/Qwen/Mistral/Phi/Gemma/DeepSeek)",
            )

        return True, None

    return True, None
