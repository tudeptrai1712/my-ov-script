import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openvino as ov

from .commands import CommandContext, CommandRouter
from .config import (
    DEFAULT_CONTEXT,
    DEFAULT_ENABLE_MOE_OFFLOAD,
    DEFAULT_ENABLE_REASONING,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_DISTRIBUTION_POLICY,
    DEFAULT_MOE_OFFLOAD_RATIO,
)
from .engine import GenerationResult, InferenceEngine
from .hardware import MemoryMonitor, powershell_gpu_memory
from .history import ChatHistorySaver
from .loading import LoadingScreen
from .metadata import ModelMetadata, read_model_metadata
from .parser import extract_inline_file, extract_inline_image, sanitize_file_path
from .ui import UI, human_bytes


class Chat:
    """
    Interactive multi-turn chat session controller.
    Coordinates the InferenceEngine, CommandRouter, LoadingScreen, and ChatHistorySaver.
    """

    def __init__(
        self,
        model_path: Path,
        device: str,
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
        self.enable_moe_offload = enable_moe_offload
        self.moe_offload_ratio = moe_offload_ratio
        self.model_distribution_policy = model_distribution_policy

        # 1. Background memory monitor
        self.memory_monitor = MemoryMonitor()
        self.memory_monitor.current = powershell_gpu_memory()
        self.memory_monitor.start()

        # 2. History saver
        self.history_saver = ChatHistorySaver(
            model_name=self.metadata.name,
            model_type=self.metadata.display_type,
            device=self.device,
            precision=self.metadata.precision,
            context_length=self.context_length,
            max_new_tokens=self.max_new_tokens,
            reasoning_enabled=enable_reasoning and self.metadata.supports_reasoning,
            supports_reasoning=self.metadata.supports_reasoning,
        )

        # 3. Inference engine
        self.engine = InferenceEngine(
            model_path=self.model_path,
            device=self.device,
            context_length=self.context_length,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            enable_reasoning=enable_reasoning,
            metadata=self.metadata,
            enable_moe_offload=self.enable_moe_offload,
            moe_offload_ratio=self.moe_offload_ratio,
            model_distribution_policy=self.model_distribution_policy,
        )

        # 4. Command Router & Context
        self.command_router = CommandRouter()
        self.cmd_context = CommandContext(
            engine=self.engine,
            history_saver=self.history_saver,
            chat_session=self,
        )

        # 5. Model Loading UI
        loading = LoadingScreen(
            self.model_path,
            self.device,
            self.context_length,
            self.memory_monitor,
            metadata=self.metadata,
            enable_moe_offload=self.enable_moe_offload,
            moe_offload_ratio=self.moe_offload_ratio,
            model_distribution_policy=self.model_distribution_policy,
        )
        loading.start()

        try:
            self.load_time = self.engine.load(warmup=True)
        finally:
            time.sleep(0.3)
            loading.stop()

    @property
    def pipe(self):
        return self.engine.pipe

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    @property
    def reasoning_enabled(self) -> bool:
        return self.engine.reasoning_enabled

    @property
    def pending_images(self) -> List[ov.Tensor]:
        return self.engine.pending_images

    @pending_images.setter
    def pending_images(self, val: List[ov.Tensor]) -> None:
        self.engine.pending_images = list(val)

    def context_usage(self) -> int:
        return self.engine.context_usage()

    def clear(self) -> None:
        self.engine.clear_history()
        self.history_saver.reset()
        self.cmd_context.last_assistant_answer = None

    def info(self) -> None:
        from .commands.session import handle_info

        handle_info("", self.cmd_context)

    def toggle_reasoning(self, enable: Optional[bool] = None) -> bool:
        from .commands.session import handle_think

        arg = "" if enable is None else ("on" if enable else "off")
        handle_think(arg, self.cmd_context)
        return self.engine.reasoning_enabled

    def load_model(
        self,
        model_path: Path,
        device: str,
        enable_moe_offload: Optional[bool] = None,
        moe_offload_ratio: Optional[int] = None,
    ) -> None:
        """Dynamically loads and switches active model within current session."""
        print(f"\n{UI.CYAN}Switching model to: {model_path.name} on {device}...{UI.RESET}")
        self.engine.unload()

        meta = read_model_metadata(model_path)
        self.model_path = model_path
        self.device = device
        self.metadata = meta

        if enable_moe_offload is not None:
            self.enable_moe_offload = enable_moe_offload
        if moe_offload_ratio is not None:
            self.moe_offload_ratio = moe_offload_ratio

        loading = LoadingScreen(
            self.model_path,
            self.device,
            self.context_length,
            self.memory_monitor,
            metadata=self.metadata,
            enable_moe_offload=self.enable_moe_offload,
            moe_offload_ratio=self.moe_offload_ratio,
            model_distribution_policy=self.model_distribution_policy,
        )
        loading.start()

        try:
            self.engine = InferenceEngine(
                model_path=self.model_path,
                device=self.device,
                context_length=self.context_length,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                enable_reasoning=self.reasoning_enabled,
                metadata=self.metadata,
                enable_moe_offload=self.enable_moe_offload,
                moe_offload_ratio=self.moe_offload_ratio,
                model_distribution_policy=self.model_distribution_policy,
            )
            self.load_time = self.engine.load(warmup=True)
            self.cmd_context.engine = self.engine
        finally:
            time.sleep(0.3)
            loading.stop()

        self.history_saver.model_name = self.metadata.name
        self.history_saver.model_type = self.metadata.display_type
        self.history_saver.precision = self.metadata.precision
        self.history_saver.device = self.device
        self.history_saver.reset()
        print(f"\n{UI.GREEN}[OK] Successfully loaded '{self.metadata.name}' on {self.device} in {self.load_time:.2f}s.{UI.RESET}\n")

    def generate(self, prompt: str, export_file: Optional[Path] = None) -> None:
        """Executes streaming inference turn, tracks metrics, and updates transcripts."""
        if not self.engine.is_loaded:
            print(f"\n{UI.YELLOW}No model is loaded. Type /load to load a model.{UI.RESET}\n")
            return

        # Inject pending file context if staged
        if self.cmd_context.pending_file_context:
            user_content = (
                f"=== File: {self.cmd_context.pending_file_name} ===\n"
                f"{self.cmd_context.pending_file_context}\n"
                f"=== End of File ===\n\n{prompt}"
            )
            self.cmd_context.pending_file_context = None
            self.cmd_context.pending_file_name = None
        else:
            user_content = prompt

        # Visual prompt banner
        if self.engine.reasoning_enabled:
            print(f"\n{UI.BOLD}{UI.GREEN}AI {UI.MAGENTA}[thinking]{UI.RESET}: ", end="", flush=True)
        else:
            print(f"\n{UI.BOLD}{UI.GREEN}AI:{UI.RESET} ", end="", flush=True)

        def streamer(chunk: str) -> bool:
            print(chunk, end="", flush=True)
            return False

        try:
            result = self.engine.generate(user_content, streaming_callback=streamer)
        except Exception as ex:
            print(f"\n\n{UI.RED}Generation error: {ex}{UI.RESET}\n")
            return

        print()  # Final newline

        # Print throughput metrics
        print(
            f"{UI.DIM}[TTFT: {result.ttft_sec:.2f}s | "
            f"Speed: {result.tokens_per_second:.1f} tok/s | "
            f"Total: {result.tokens_generated} tokens in {result.total_time_sec:.2f}s]{UI.RESET}\n"
        )

        # Store for /export
        self.cmd_context.last_assistant_answer = result.text

        # Record turn in markdown session history
        self.history_saver.record_turn(
            user_prompt=user_content,
            assistant_response=result.text,
            ttft_sec=result.ttft_sec,
            generation_time_sec=result.total_time_sec,
            tokens_generated=result.tokens_generated,
            tokens_per_sec=result.tokens_per_second,
            context_tokens_used=self.engine.context_usage(),
            attached_file=None,
            attached_image=self.cmd_context.pending_image_info,
            reasoning_used=self.engine.reasoning_enabled,
        )
        self.cmd_context.pending_image_info = None

        # Write to export file if requested
        if export_file:
            try:
                export_file.write_text(result.text, encoding="utf-8")
                print(
                    f"{UI.BOLD}{UI.GREEN}[Exported]{UI.RESET} "
                    f"Answer saved directly to: {export_file.resolve()}\n"
                )
            except Exception as ex:
                print(f"{UI.RED}Failed to write export file: {ex}{UI.RESET}\n")

    def run(self) -> None:
        """Main interactive terminal chat REPL loop."""
        while True:
            try:
                prompt = input(f"{UI.BOLD}{UI.CYAN}You: {UI.RESET}").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not prompt:
                continue

            # Support inline /file embedded directly in prompt
            if "/file " in prompt.lower():
                prompt, inline_file = extract_inline_file(prompt)
                if inline_file:
                    if inline_file.exists():
                        try:
                            from .files import extract_text_from_path

                            self.cmd_context.pending_file_context = extract_text_from_path(inline_file)
                            self.cmd_context.pending_file_name = inline_file.name
                            print(
                                f"\n{UI.GREEN}Attached file: {inline_file.name} "
                                f"({len(self.cmd_context.pending_file_context):,} chars).{UI.RESET}"
                            )
                        except Exception as ex:
                            print(f"\n{UI.RED}Failed to read attached file: {ex}{UI.RESET}\n")
                    else:
                        print(f"\n{UI.RED}Attached file not found: {inline_file}{UI.RESET}\n")

            # Support inline /image embedded directly in prompt
            if any(k in prompt.lower() for k in ("/image ", "/img ", "/pic ", "/pdf-image ", "/pdfimg ")):
                prompt, inline_img, page_spec = extract_inline_image(prompt)
                if inline_img:
                    from .commands.attachments import load_image_or_pdf_into_context

                    load_image_or_pdf_into_context(inline_img, page_spec, self.cmd_context)

            # Check if input is a slash command
            cmd_result = self.command_router.dispatch(prompt, self.cmd_context)
            if cmd_result:
                if cmd_result.should_exit:
                    break
                if cmd_result.prompt_to_generate:
                    self.generate(cmd_result.prompt_to_generate, export_file=cmd_result.export_file)
                continue

            # Regular prompt turn
            self.generate(prompt)

        print(f"\n{UI.DIM}Chat transcript saved: {self.history_saver.session_file}{UI.RESET}")
