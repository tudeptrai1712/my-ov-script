import gc
import threading
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
from .reasoning import StreamTokenParser


@dataclass
class GenerationResult:
    """Detailed metrics and generated text from an inference run."""

    text: str
    tokens_generated: int
    ttft_sec: float
    total_time_sec: float
    tokens_per_second: float
    prompt_tokens: int
    reasoning: Optional[str] = None
    raw_text: str = ""


def count_tokens(tokenizer: Any, text: str) -> int:
    """Counts token length of arbitrary text using the model tokenizer."""
    if not tokenizer:
        return max(1, len(text) // 4)
    try:
        encoded = tokenizer.encode(text)
        shape = encoded.input_ids.get_shape()
        return int(shape[1]) if len(shape) >= 2 else int(shape[0])
    except Exception:
        return max(1, len(text) // 4)


class InferenceEngine:
    """
    Unified OpenVINO GenAI runtime wrapper for dual LLM & VLM pipelines.
    Single source of truth used by both terminal CLI and FastAPI OpenAI server.
    Thread-safe, supporting Heterogeneous Pipeline Parallelism and MoE offloading.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
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
        self.lock = threading.Lock()
        self.model_path: Optional[Path] = Path(model_path) if model_path else None
        self.device: str = device
        self.context_length: int = context_length
        self.max_new_tokens: int = max_new_tokens
        self.temperature: float = temperature
        self.top_p: float = top_p
        self.enable_reasoning_default: bool = enable_reasoning
        self.metadata: Optional[ModelMetadata] = metadata
        if self.model_path and not self.metadata:
            self.metadata = read_model_metadata(self.model_path)

        self.reasoning_enabled: bool = bool(
            enable_reasoning and (not self.metadata or self.metadata.supports_reasoning)
        )
        self.reasoning_effort: str = "medium" if self.reasoning_enabled else "none"
        self.enable_moe_offload: bool = enable_moe_offload
        self.moe_offload_ratio: int = moe_offload_ratio
        self.model_distribution_policy: str = model_distribution_policy
        self.applied_properties: Dict[str, Any] = {}

        self.pipe: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.generation_config: Optional[Any] = None

        # Interactive CLI session history
        self.chat_history = ov_genai.ChatHistory()
        self.history_messages: List[Dict[str, str]] = []

        self.pending_images: List[ov.Tensor] = []
        self.load_time: float = 0.0

    @property
    def is_loaded(self) -> bool:
        return self.pipe is not None

    @property
    def model_name(self) -> str:
        if self.metadata:
            return self.metadata.name
        if self.model_path:
            return self.model_path.name
        return ""

    def load_model(
        self,
        model_path: Path,
        device: str = "GPU",
        context_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        enable_reasoning: Optional[bool] = None,
        enable_moe_offload: Optional[bool] = None,
        moe_offload_ratio: Optional[int] = None,
        model_distribution_policy: Optional[str] = None,
        warmup: bool = True,
    ) -> float:
        """Configures parameters and compiles the pipeline."""
        with self.lock:
            self.unload()
            self.model_path = Path(model_path)
            self.device = device
            if context_length is not None:
                self.context_length = context_length
            if max_new_tokens is not None:
                self.max_new_tokens = max_new_tokens
            if temperature is not None:
                self.temperature = temperature
            if top_p is not None:
                self.top_p = top_p
            if enable_moe_offload is not None:
                self.enable_moe_offload = enable_moe_offload
            if moe_offload_ratio is not None:
                self.moe_offload_ratio = moe_offload_ratio
            if model_distribution_policy is not None:
                self.model_distribution_policy = model_distribution_policy

            self.metadata = read_model_metadata(self.model_path)
            if enable_reasoning is not None:
                self.reasoning_enabled = bool(enable_reasoning)
            else:
                self.reasoning_enabled = bool(
                    self.enable_reasoning_default and (not self.metadata or self.metadata.supports_reasoning)
                )
            self.reasoning_effort = "medium" if self.reasoning_enabled else "none"

            return self._compile_pipeline(warmup=warmup)

    def load(self, warmup: bool = True) -> float:
        """Loads and compiles pipeline using current instance configuration."""
        with self.lock:
            return self._compile_pipeline(warmup=warmup)

    def _compile_pipeline(self, warmup: bool = True) -> float:
        """Internal compilation logic. Must be called under self.lock."""
        if not self.model_path or not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        start_time = time.perf_counter()
        if not self.metadata:
            self.metadata = read_model_metadata(self.model_path)

        pipeline_cls = ov_genai.VLMPipeline if self.metadata.is_vlm else ov_genai.LLMPipeline

        # Device configuration properties
        properties: Dict[str, Any] = {}

        # 1. Heterogeneous Pipeline Parallelism
        if self.device.upper().startswith("HETERO"):
            properties["MODEL_DISTRIBUTION_POLICY"] = self.model_distribution_policy

        # 2. Mixture of Experts (MoE) Offloading to Disk
        if self.metadata.is_moe and self.enable_moe_offload:
            if "GPU" in self.device.upper():
                properties["OFFLOAD_RATIO"] = int(self.moe_offload_ratio)
                properties["ENABLE_MMAP"] = "YES"

        self.applied_properties = dict(properties)

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
        if self.model_path:
            chat_template_file = self.model_path / "chat_template.jinja"
            if chat_template_file.exists():
                try:
                    tmpl_text = chat_template_file.read_text(encoding="utf-8")
                    self.tokenizer.set_chat_template(tmpl_text)
                except Exception:
                    pass

        self.generation_config = self.pipe.get_generation_config()
        self.generation_config.max_new_tokens = self.max_new_tokens
        if hasattr(self.generation_config, "temperature"):
            self.generation_config.temperature = self.temperature
        if hasattr(self.generation_config, "top_p"):
            self.generation_config.top_p = self.top_p

        # Reset interactive history
        self.chat_history = ov_genai.ChatHistory()
        self.history_messages = []

        if warmup:
            try:
                warmup_cfg = self.pipe.get_generation_config()
                warmup_cfg.max_new_tokens = 1
                if self.metadata.is_vlm:
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
        """Ejects active model and frees memory buffers."""
        self.pipe = None
        self.tokenizer = None
        self.generation_config = None
        self.pending_images = []
        self.chat_history = ov_genai.ChatHistory()
        self.history_messages = []
        self.applied_properties = {}
        gc.collect()

    def rebuild_history(self) -> None:
        """Reconstructs native ov_genai.ChatHistory from internal history message list."""
        self.chat_history = ov_genai.ChatHistory()
        for msg in self.history_messages:
            self.chat_history.append({
                "role": msg["role"],
                "content": msg["content"],
            })

    def _build_extra_context(
        self, user_extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, int]:
        """
        Builds the extra context dict passed to OpenVINO GenAI pipeline generation.
        OpenVINO GenAI C++ pybind requires std::map<std::string, int64_t> for extra_context.
        All values MUST be integers or booleans (int64_t) to prevent C++ type-cast errors.
        """
        extra: Dict[str, int] = {
            "enable_thinking": 1 if self.reasoning_enabled else 0,
            "thinking": 1 if self.reasoning_enabled else 0,
        }
        if user_extra:
            for k, v in user_extra.items():
                if isinstance(v, bool):
                    extra[k] = 1 if v else 0
                elif isinstance(v, int):
                    extra[k] = v
                elif isinstance(v, str):
                    v_lower = v.lower()
                    if v_lower in ("none", "false", "off", "0"):
                        extra[k] = 0
                    elif v_lower in ("true", "on", "1", "low", "medium", "high"):
                        extra[k] = 1
        return extra

    def context_usage(self) -> int:
        """Calculates current context token consumption for interactive session."""
        if not self.history_messages or not self.tokenizer:
            return 0

        try:
            prompt = self.tokenizer.apply_chat_template(
                self.chat_history,
                True,
                extra_context=self._build_extra_context(),
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
            del self.history_messages[:2]
            self.rebuild_history()

        return self.context_usage()

    def clear_history(self) -> None:
        """Clears interactive conversation context."""
        self.history_messages.clear()
        self.chat_history = ov_genai.ChatHistory()

    def set_reasoning(self, enable: bool, effort: Optional[str] = None) -> bool:
        """Enables or disables reasoning mode."""
        self.reasoning_enabled = bool(enable)
        if effort:
            self.reasoning_effort = effort
        elif enable:
            if not getattr(self, "reasoning_effort", None) or self.reasoning_effort == "none":
                self.reasoning_effort = "medium"
        else:
            self.reasoning_effort = "none"
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
        reasoning_callback: Optional[Callable[[str], bool]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        """
        Executes streaming inference for the interactive CLI session.
        Appends user prompt and clean assistant response to session history.
        """
        if self.pipe is None:
            raise RuntimeError("Cannot generate: No OpenVINO model is currently loaded.")

        prompt_tokens_est = count_tokens(self.tokenizer, prompt) if self.tokenizer else 0

        self.history_messages.append({"role": "user", "content": prompt})
        self.rebuild_history()
        self.trim_context()

        raw_chunks: List[str] = []
        ttft: Optional[float] = None
        gen_start = time.perf_counter()

        def on_thought_token(token: str) -> None:
            if reasoning_callback:
                reasoning_callback(token)

        def on_content_token(token: str) -> None:
            if streaming_callback:
                streaming_callback(token)

        parser = StreamTokenParser(
            on_content=on_content_token,
            on_thought=on_thought_token,
        )

        def internal_streamer(text: str) -> bool:
            nonlocal ttft
            if ttft is None:
                ttft = time.perf_counter() - gen_start
            raw_chunks.append(text)
            parser.feed(text)
            return False

        kwargs: Dict[str, Any] = {
            "generation_config": self.generation_config,
            "streamer": internal_streamer,
            "extra_context": self._build_extra_context(extra_context),
        }

        if self.metadata and self.metadata.is_vlm and self.pending_images:
            kwargs["images"] = self.pending_images
            self.pending_images = []

        with self.lock:
            self.pipe.generate(self.chat_history, **kwargs)

        parser.finish()

        total_time = time.perf_counter() - gen_start
        raw_text = "".join(raw_chunks)
        clean_thought = "".join(parser.thought_chunks).strip() if parser.thought_chunks else None
        clean_content = "".join(parser.content_chunks).strip()

        if not clean_content and not clean_thought:
            clean_content = raw_text.strip()

        tokens_gen = count_tokens(self.tokenizer, raw_text) if self.tokenizer else max(1, len(raw_text) // 4)
        tokens_per_sec = tokens_gen / total_time if total_time > 0 else 0.0
        ttft_val = ttft if ttft is not None else 0.0

        # Save clean content into conversation history so next turn prompt is clean
        self.history_messages.append({"role": "assistant", "content": clean_content})
        self.rebuild_history()

        return GenerationResult(
            text=clean_content,
            tokens_generated=tokens_gen,
            ttft_sec=ttft_val,
            total_time_sec=total_time,
            tokens_per_second=tokens_per_sec,
            prompt_tokens=prompt_tokens_est,
            reasoning=clean_thought,
            raw_text=raw_text,
        )

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        images: Optional[List[ov.Tensor]] = None,
        streaming_callback: Optional[Callable[[str], bool]] = None,
        reasoning_callback: Optional[Callable[[str], bool]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> GenerationResult:
        """
        Executes inference for an arbitrary list of messages (e.g. OpenAI /v1/chat/completions).
        Stateless with respect to the interactive CLI session. Thread-safe.
        """
        if self.pipe is None:
            raise RuntimeError("Cannot generate: No OpenVINO model is currently loaded.")

        chat_hist = ov_genai.ChatHistory()
        for msg in messages:
            chat_hist.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        # Calculate prompt tokens
        prompt_text = "\n".join(m.get("content", "") for m in messages)
        prompt_tokens_est = count_tokens(self.tokenizer, prompt_text) if self.tokenizer else 0

        # Clone and configure generation parameters
        gen_cfg = self.pipe.get_generation_config()
        if max_new_tokens:
            gen_cfg.max_new_tokens = max_new_tokens
        elif self.max_new_tokens:
            gen_cfg.max_new_tokens = self.max_new_tokens

        if temperature is not None and hasattr(gen_cfg, "temperature"):
            gen_cfg.temperature = temperature
        if top_p is not None and hasattr(gen_cfg, "top_p"):
            gen_cfg.top_p = top_p

        raw_chunks: List[str] = []
        ttft: Optional[float] = None
        gen_start = time.perf_counter()

        def on_thought_token(token: str) -> None:
            if reasoning_callback:
                reasoning_callback(token)

        def on_content_token(token: str) -> None:
            if streaming_callback:
                streaming_callback(token)

        parser = StreamTokenParser(
            on_content=on_content_token,
            on_thought=on_thought_token,
        )

        def internal_streamer(text: str) -> bool:
            nonlocal ttft
            if ttft is None:
                ttft = time.perf_counter() - gen_start
            raw_chunks.append(text)
            parser.feed(text)
            return False

        kwargs: Dict[str, Any] = {
            "generation_config": gen_cfg,
            "streamer": internal_streamer,
            "extra_context": self._build_extra_context(extra_context),
        }

        if self.metadata and self.metadata.is_vlm and images:
            kwargs["images"] = images

        with self.lock:
            self.pipe.generate(chat_hist, **kwargs)

        parser.finish()

        total_time = time.perf_counter() - gen_start
        raw_text = "".join(raw_chunks)
        clean_thought = "".join(parser.thought_chunks).strip() if parser.thought_chunks else None
        clean_content = "".join(parser.content_chunks).strip()

        if not clean_content and not clean_thought:
            clean_content = raw_text.strip()

        tokens_gen = count_tokens(self.tokenizer, raw_text) if self.tokenizer else max(1, len(raw_text) // 4)
        tokens_per_sec = tokens_gen / total_time if total_time > 0 else 0.0
        ttft_val = ttft if ttft is not None else 0.0

        return GenerationResult(
            text=clean_content,
            tokens_generated=tokens_gen,
            ttft_sec=ttft_val,
            total_time_sec=total_time,
            tokens_per_second=tokens_per_sec,
            prompt_tokens=prompt_tokens_est,
            reasoning=clean_thought,
            raw_text=raw_text,
        )
