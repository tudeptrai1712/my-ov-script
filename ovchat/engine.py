import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import openvino as ov
import openvino_genai as ov_genai

from .config import (
    DEFAULT_CONTEXT,
    DEFAULT_ENABLE_MOE_OFFLOAD,
    DEFAULT_ENABLE_REASONING,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_DISTRIBUTION_POLICY,
    DEFAULT_MOE_OFFLOAD_RATIO,
)
from .metadata import ModelMetadata, read_model_metadata


@dataclass
class GenerationResult:
    """Detailed metrics and generated text from an inference run."""

    text: str
    tokens_generated: int
    ttft_sec: float
    total_time_sec: float
    tokens_per_second: float
    prompt_tokens: int


def count_tokens(tokenizer: Any, text: str) -> int:
    """Counts token length of arbitrary text using the model tokenizer."""
    try:
        encoded = tokenizer.encode(text)
        shape = encoded.input_ids.get_shape()
        return int(shape[1]) if len(shape) >= 2 else int(shape[0])
    except Exception:
        # Fallback estimation if encoding fails: ~4 characters per token
        return max(1, len(text) // 4)


class InferenceEngine:
    """
    Pure OpenVINO GenAI runtime wrapper for dual LLM & VLM pipelines.
    Decoupled from terminal presentation, input loops, and command parsing.
    Supports Heterogeneous Pipeline Parallelism and MoE offloading to disk.
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "GPU",
        context_length: int = DEFAULT_CONTEXT,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.7,
        top_p: float = 0.95,
        enable_reasoning: bool = DEFAULT_ENABLE_REASONING,
        metadata: Optional[ModelMetadata] = None,
        enable_moe_offload: bool = DEFAULT_ENABLE_MOE_OFFLOAD,
        moe_offload_ratio: int = DEFAULT_MOE_OFFLOAD_RATIO,
        model_distribution_policy: str = DEFAULT_MODEL_DISTRIBUTION_POLICY,
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.context_length = context_length
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.metadata = metadata or read_model_metadata(self.model_path)

        self.reasoning_enabled = enable_reasoning and self.metadata.supports_reasoning
        self.enable_moe_offload = enable_moe_offload
        self.moe_offload_ratio = moe_offload_ratio
        self.model_distribution_policy = model_distribution_policy
        self.applied_properties: Dict[str, Any] = {}

        self.pipe: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.generation_config: Optional[Any] = None

        self.chat_history = ov_genai.ChatHistory()
        self.history_messages: List[Dict[str, str]] = []

        self.pending_images: List[ov.Tensor] = []
        self.load_time: float = 0.0

    @property
    def is_loaded(self) -> bool:
        return self.pipe is not None

    def load(self, warmup: bool = True) -> float:
        """
        Initializes the OpenVINO GenAI pipeline (VLMPipeline or LLMPipeline).
        Configures Heterogeneous execution (Pipeline Parallelism) and MoE offload properties.
        Returns total load time in seconds.
        """
        start_time = time.perf_counter()

        pipeline_cls = ov_genai.VLMPipeline if self.metadata.is_vlm else ov_genai.LLMPipeline

        # Build pipeline device configuration properties
        properties: Dict[str, Any] = {}

        # 1. Heterogeneous Execution with Pipeline Parallelism (Preview)
        # Reference: https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/hetero-execution.html#pipeline-parallelism-preview
        if self.device.upper().startswith("HETERO"):
            properties["MODEL_DISTRIBUTION_POLICY"] = self.model_distribution_policy

        # 2. Mixture of Experts (MoE) Offloading to Disk (OpenVINO 2026 GPU plugin)
        # Reduces VRAM requirement for large MoE models through partial expert weight offloading
        if self.metadata.is_moe and self.enable_moe_offload:
            if "GPU" in self.device.upper():
                properties["OFFLOAD_RATIO"] = int(self.moe_offload_ratio)
                properties["ENABLE_MMAP"] = "YES"

        self.applied_properties = dict(properties)

        # Load pipeline on requested device with properties
        try:
            if properties:
                self.pipe = pipeline_cls(str(self.model_path), self.device, **properties)
            else:
                self.pipe = pipeline_cls(str(self.model_path), self.device)
        except TypeError:
            try:
                self.pipe = pipeline_cls(str(self.model_path), self.device, properties)
            except Exception:
                self.pipe = pipeline_cls(str(self.model_path), self.device)

        self.tokenizer = self.pipe.get_tokenizer()
        self.generation_config = self.pipe.get_generation_config()
        self.generation_config.max_new_tokens = self.max_new_tokens
        self.generation_config.temperature = self.temperature
        self.generation_config.top_p = self.top_p

        # Re-initialize history
        self.chat_history = ov_genai.ChatHistory()
        self.history_messages = []

        # Optional warmup to initialize OpenVINO runtime buffers
        if warmup:
            try:
                warmup_cfg = self.pipe.get_generation_config()
                warmup_cfg.max_new_tokens = 1
                if self.metadata.is_vlm:
                    # Warmup with dummy 1x1 image tensor
                    import numpy as np
                    dummy_img = ov.Tensor(np.zeros((64, 64, 3), dtype=np.uint8))
                    self.pipe.generate("hi", images=[dummy_img], generation_config=warmup_cfg)
                else:
                    self.pipe.generate("hi", generation_config=warmup_cfg)
            except Exception:
                pass

        self.load_time = time.perf_counter() - start_time
        return self.load_time

    def unload(self) -> None:
        """Ejects active model and frees VRAM and system memory."""
        self.pipe = None
        self.tokenizer = None
        self.generation_config = None
        self.pending_images = []
        gc.collect()

    def rebuild_history(self) -> None:
        """Reconstructs native ov_genai.ChatHistory from internal history message list."""
        self.chat_history = ov_genai.ChatHistory()
        for msg in self.history_messages:
            self.chat_history.append({
                "role": msg["role"],
                "content": msg["content"],
            })

    def context_usage(self) -> int:
        """Calculates current context token consumption."""
        if not self.history_messages or not self.tokenizer:
            return 0

        try:
            prompt = self.tokenizer.apply_chat_template(
                self.chat_history,
                True,
                extra_context={"enable_thinking": self.reasoning_enabled},
            )
            encoded = self.tokenizer.encode(prompt)
            shape = encoded.input_ids.get_shape()
            return int(shape[1]) if len(shape) >= 2 else int(shape[0])
        except Exception:
            text = "\n".join(m["content"] for m in self.history_messages)
            return count_tokens(self.tokenizer, text)

    def trim_context(self) -> int:
        """Trims oldest user/assistant turns if approaching context length limit."""
        while len(self.history_messages) > 2:
            used = self.context_usage()
            required = used + self.max_new_tokens
            if required <= self.context_length:
                break
            # Remove oldest turn (user + assistant)
            del self.history_messages[:2]
            self.rebuild_history()

        return self.context_usage()

    def clear_history(self) -> None:
        """Clears conversation context."""
        self.history_messages.clear()
        self.chat_history = ov_genai.ChatHistory()

    def set_reasoning(self, enable: bool) -> bool:
        """Enables or disables reasoning mode if supported by model."""
        if not self.metadata.supports_reasoning:
            self.reasoning_enabled = False
            return False
        self.reasoning_enabled = bool(enable)
        return self.reasoning_enabled

    def set_moe_offload(self, enable: bool, ratio: Optional[int] = None) -> None:
        """Updates MoE offload settings."""
        self.enable_moe_offload = bool(enable)
        if ratio is not None:
            self.moe_offload_ratio = max(1, min(99, int(ratio)))

    def attach_images(self, tensors: List[ov.Tensor]) -> None:
        """Stages visual image tensors for the upcoming generation turn."""
        self.pending_images = list(tensors)

    def generate(
        self,
        prompt: str,
        streaming_callback: Optional[Callable[[str], bool]] = None,
    ) -> GenerationResult:
        """
        Executes streaming inference for the given user prompt.
        Accepts an optional streaming_callback(token_str) -> bool (return True to abort).
        Returns GenerationResult containing generated text and performance metrics.
        """
        if self.pipe is None:
            raise RuntimeError("Cannot generate: No OpenVINO model is currently loaded.")

        prompt_tokens_est = count_tokens(self.tokenizer, prompt) if self.tokenizer else 0

        # Append turn to history
        self.history_messages.append({"role": "user", "content": prompt})
        self.rebuild_history()
        self.trim_context()

        generated_chunks: List[str] = []
        ttft: Optional[float] = None
        gen_start = time.perf_counter()

        def internal_streamer(text: str) -> bool:
            nonlocal ttft
            if ttft is None:
                ttft = time.perf_counter() - gen_start

            generated_chunks.append(text)
            if streaming_callback:
                should_abort = streaming_callback(text)
                if should_abort:
                    return True
            return False

        kwargs: Dict[str, Any] = {
            "generation_config": self.generation_config,
            "streamer": internal_streamer,
            "extra_context": {"enable_thinking": self.reasoning_enabled},
        }

        # Inject multimodal image tensors if VLM
        if self.metadata.is_vlm and self.pending_images:
            kwargs["images"] = self.pending_images
            self.pending_images = []

        # Run inference
        self.pipe.generate(self.chat_history, **kwargs)

        total_time = time.perf_counter() - gen_start
        full_text = "".join(generated_chunks)

        # Count generated tokens
        tokens_gen = count_tokens(self.tokenizer, full_text) if self.tokenizer else max(1, len(full_text) // 4)
        tokens_per_sec = tokens_gen / total_time if total_time > 0 else 0.0
        ttft_val = ttft if ttft is not None else 0.0

        # Save assistant response in internal history
        self.history_messages.append({"role": "assistant", "content": full_text})
        self.rebuild_history()

        return GenerationResult(
            text=full_text,
            tokens_generated=tokens_gen,
            ttft_sec=ttft_val,
            total_time_sec=total_time,
            tokens_per_second=tokens_per_sec,
            prompt_tokens=prompt_tokens_est,
        )
