import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import openvino_genai as ov_genai

from .history import ChatHistorySaver
from .loading import LoadingScreen
from .memory import MemoryMonitor, powershell_gpu_memory
from .metadata import ModelMetadata, read_model_metadata
from .ui import UI, human_bytes, progress_bar


def count_tokens(tokenizer: Any, text: str) -> int:
    """Best-effort token count using the pipeline's tokenizer."""
    if not text:
        return 0

    try:
        encoded = tokenizer.encode(text)
        shape = encoded.input_ids.get_shape()

        if len(shape) >= 2:
            return int(shape[1])

        return int(shape[0])

    except Exception:
        return len(text.split())


class Chat:
    """Manages an interactive OpenVINO GenAI chat session."""

    def __init__(
        self,
        model_path: Path,
        device: str,
        context_length: int,
        max_new_tokens: int,
        enable_reasoning: bool = False,
        metadata: Optional[ModelMetadata] = None,
    ):
        self.model_path = model_path
        self.device = device
        self.context_length = context_length
        self.max_new_tokens = max_new_tokens
        self.metadata = metadata or read_model_metadata(model_path)
        self.reasoning_enabled = enable_reasoning and self.metadata.supports_reasoning

        # ----------------------------------------------------
        # Persistent chat history saving
        # ----------------------------------------------------
        self.history_saver = ChatHistorySaver(
            model_name=self.model_path.name,
            device=self.device,
            context_length=self.context_length,
            reasoning_enabled=self.reasoning_enabled,
        )

        # ----------------------------------------------------
        # Memory monitoring
        # ----------------------------------------------------
        self.memory_monitor = MemoryMonitor()

        # Take a baseline before loading.
        self.memory_monitor.current = powershell_gpu_memory()
        self.memory_monitor.start()

        # ----------------------------------------------------
        # Loading UI
        # ----------------------------------------------------
        loading = LoadingScreen(
            model_path,
            device,
            context_length,
            self.memory_monitor,
        )

        load_start = time.perf_counter()
        loading.start()

        try:
            # Select pipeline based on model metadata (VLM vs text-only LLM)
            if self.metadata.is_vlm:
                self.pipe = ov_genai.VLMPipeline(
                    str(model_path),
                    device,
                )
            else:
                self.pipe = ov_genai.LLMPipeline(
                    str(model_path),
                    device,
                )

        except Exception:
            loading.stop()
            self.memory_monitor.stop()
            raise

        self.load_time = time.perf_counter() - load_start

        # Let the memory monitor get a final sample.
        time.sleep(0.5)
        loading.stop()

        # Keep monitoring after model load for /info command.

        # ----------------------------------------------------
        # Tokenizer
        # ----------------------------------------------------
        self.tokenizer = self.pipe.get_tokenizer()

        # ----------------------------------------------------
        # Generation config
        # ----------------------------------------------------
        self.generation_config = self.pipe.get_generation_config()
        self.generation_config.max_new_tokens = self.max_new_tokens

        # ----------------------------------------------------
        # Chat history
        # ----------------------------------------------------
        self.chat_history = ov_genai.ChatHistory()

        # Python-side copy.
        # We keep our own list because it lets us trim old
        # conversation turns and then rebuild ChatHistory.
        self.history_messages: List[Dict[str, str]] = []

        # ----------------------------------------------------
        # Memory statistics
        # ----------------------------------------------------
        self.memory_before = loading.baseline_memory
        self.memory_after = self.memory_monitor.current

        if self.memory_before is not None and self.memory_after is not None:
            self.actual_memory_increase = max(
                0.0,
                self.memory_after - self.memory_before,
            )
        else:
            self.actual_memory_increase = None

        # Multimodal image & file attachments state
        self.pending_image: Optional[ov.Tensor] = None
        self.pending_image_path: Optional[str] = None
        self.pending_file_context: Optional[str] = None
        self.pending_file_name: Optional[str] = None

    def rebuild_history(self) -> None:
        """Reconstructs ov_genai.ChatHistory from our tracked messages."""
        self.chat_history = ov_genai.ChatHistory()
        for message in self.history_messages:
            self.chat_history.append({
                "role": message["role"],
                "content": message["content"],
            })

    def context_usage(self) -> int:
        """Calculates current context token consumption."""
        if not self.history_messages:
            return 0

        try:
            # Convert history to a string using the tokenizer's chat template when possible.
            prompt = self.tokenizer.apply_chat_template(
                self.chat_history,
                True,
                extra_context={"enable_thinking": self.reasoning_enabled},
            )
            encoded = self.tokenizer.encode(prompt)
            shape = encoded.input_ids.get_shape()

            if len(shape) >= 2:
                return int(shape[1])

            return int(shape[0])

        except Exception:
            text = "\n".join(
                message["content"] for message in self.history_messages
            )
            return count_tokens(self.tokenizer, text)

    def trim_context(self) -> int:
        """Trims oldest user/assistant turns if approaching context length."""
        while len(self.history_messages) > 2:
            used = self.context_usage()
            required = used + self.max_new_tokens

            if required <= self.context_length:
                break

            # Remove oldest user + assistant turn.
            del self.history_messages[:2]
            self.rebuild_history()

        return self.context_usage()

    def clear(self) -> None:
        """Clears conversation history and begins a new history log file."""
        self.history_messages.clear()
        self.chat_history = ov_genai.ChatHistory()
        self.history_saver.reset()
        print(f"\n{UI.GREEN}Conversation cleared.{UI.RESET}")
        print(f"{UI.DIM}New history log: {self.history_saver.session_file}{UI.RESET}\n")

    def toggle_reasoning(self, enable: Optional[bool] = None) -> bool:
        """Toggles or sets the reasoning/thinking mode state."""
        if not self.metadata.supports_reasoning:
            print(
                f"\n{UI.YELLOW}Reasoning mode is not supported by model "
                f"'{self.metadata.name}'.{UI.RESET}\n"
            )
            self.reasoning_enabled = False
            return False

        if enable is None:
            self.reasoning_enabled = not self.reasoning_enabled
        else:
            self.reasoning_enabled = bool(enable)

        state_str = "enabled" if self.reasoning_enabled else "disabled"
        color = UI.GREEN if self.reasoning_enabled else UI.YELLOW
        print(f"\n{color}Reasoning mode {state_str}.{UI.RESET}\n")
        return self.reasoning_enabled

    def info(self) -> None:
        """Prints current session and memory statistics."""
        used = self.context_usage()
        current_memory = self.memory_monitor.current
        if self.metadata.supports_reasoning:
            reason_status = (
                f"{UI.GREEN}Enabled{UI.RESET}"
                if self.reasoning_enabled
                else f"{UI.YELLOW}Disabled{UI.RESET}"
            )
        else:
            reason_status = f"{UI.DIM}Not supported by model{UI.RESET}"

        print()
        print(f"{UI.BOLD}Model:{UI.RESET} {self.model_path.name}")
        print(
            f"{UI.BOLD}Type:{UI.RESET} {self.metadata.display_type} "
            f"({self.metadata.precision.upper()})"
        )
        print(f"{UI.BOLD}Device:{UI.RESET} {self.device}")
        print(f"{UI.BOLD}Reasoning:{UI.RESET} {reason_status}")
        print(f"{UI.BOLD}Context:{UI.RESET} {used:,} / {self.context_length:,}")
        print(f"{UI.BOLD}Max output:{UI.RESET} {self.max_new_tokens:,}")
        print(f"{UI.BOLD}Load time:{UI.RESET} {self.load_time:.2f}s")
        print(f"{UI.BOLD}History file:{UI.RESET} {self.history_saver.session_file}")
        print(f"{UI.BOLD}GPU/shared memory:{UI.RESET} {human_bytes(current_memory)}")

        if self.actual_memory_increase is not None:
            print(
                f"{UI.BOLD}Observed increase:{UI.RESET} "
                f"{human_bytes(self.actual_memory_increase)}"
            )

        print()

    def generate(self, prompt: str) -> None:
        """Runs streaming generation for a single user turn."""
        if self.pending_file_context:
            user_content = (
                f"=== File: {self.pending_file_name} ===\n"
                f"{self.pending_file_context}\n"
                f"=== End of File ===\n\n{prompt}"
            )
            self.pending_file_context = None
            self.pending_file_name = None
        else:
            user_content = prompt

        self.history_messages.append({
            "role": "user",
            "content": user_content,
        })
        self.rebuild_history()

        # Make room for response
        self.trim_context()

        # Streaming setup
        if self.reasoning_enabled:
            print(f"\n{UI.BOLD}{UI.GREEN}AI {UI.MAGENTA}[thinking]{UI.RESET}: ", end="", flush=True)
        else:
            print(f"\n{UI.BOLD}{UI.GREEN}AI:{UI.RESET} ", end="", flush=True)

        generated_chunks: List[str] = []

        def streamer(text: str) -> bool:
            print(text, end="", flush=True)
            generated_chunks.append(text)
            # Return False to continue streaming
            return False

        try:
            kwargs = {
                "generation_config": self.generation_config,
                "streamer": streamer,
                "extra_context": {"enable_thinking": self.reasoning_enabled},
            }
            if self.metadata.is_vlm and self.pending_image is not None:
                kwargs["images"] = [self.pending_image]
                self.pending_image = None
                self.pending_image_path = None

            result = self.pipe.generate(
                self.chat_history,
                **kwargs
            )
        except Exception as e:
            print(f"\n\n{UI.RED}Generation error:{UI.RESET} {e}\n")
            # Roll back failed user message
            if self.history_messages:
                self.history_messages.pop()
            self.rebuild_history()
            return

        print("\n")

        # Performance metrics
        try:
            metrics = result.perf_metrics
            input_tokens = metrics.get_num_input_tokens()
            output_tokens = metrics.get_num_generated_tokens()
            throughput = metrics.get_throughput().mean
            ttft = metrics.get_ttft().mean

            try:
                tpot: Optional[float] = metrics.get_tpot().mean
            except Exception:
                tpot = None

        except Exception:
            input_tokens = 0
            output_tokens = 0
            throughput = 0.0
            ttft = 0.0
            tpot = None

        # Resolve generated text
        assistant_text = "".join(generated_chunks)
        if not assistant_text:
            try:
                assistant_text = result.texts[0]
            except Exception:
                assistant_text = ""

        # Save assistant response
        self.history_messages.append({
            "role": "assistant",
            "content": assistant_text,
        })
        self.rebuild_history()

        # Persist turn to chat history file immediately
        metrics_dict = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "throughput": throughput,
            "ttft": ttft,
        }
        if tpot is not None:
            metrics_dict["tpot"] = tpot

        self.history_saver.add_turn(
            user_prompt=prompt,
            assistant_response=assistant_text,
            metrics=metrics_dict,
            reasoning_used=self.reasoning_enabled,
        )

        # Context bar
        context_used = self.context_usage()
        context_percent = min((context_used / self.context_length) * 100, 100.0)
        bar_fraction = context_used / self.context_length
        bar = progress_bar(bar_fraction, width=30)

        print(
            f"{UI.DIM}"
            f"{bar} {context_used:,}/{self.context_length:,} "
            f"({context_percent:.1f}%)"
            f"{UI.RESET}"
        )

        # Performance stats
        stats = (
            f"Input: {input_tokens:,} tok  |  "
            f"Output: {output_tokens:,} tok  |  "
            f"Speed: {throughput:.2f} tok/s  |  "
            f"TTFT: {ttft:.0f} ms"
        )
        if tpot is not None:
            stats += f"  |  TPOT: {tpot:.2f} ms/tok"

        print(f"{UI.DIM}{stats}{UI.RESET}\n")

    def run(self) -> None:
        """Main chat read-eval-print loop."""
        while True:
            try:
                prompt = input(f"{UI.BOLD}{UI.CYAN}You:{UI.RESET} ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break

            if not prompt:
                continue

            command = prompt.lower()
            parts = command.split()
            root_cmd = parts[0]

            if root_cmd in ("/quit", "/exit", "/q"):
                break

            if root_cmd == "/clear":
                self.clear()
                continue

            if root_cmd == "/info":
                self.info()
                continue

            if root_cmd in ("/think", "/reason", "/reasoning"):
                if len(parts) > 1:
                    arg = parts[1]
                    if arg in ("on", "1", "true", "enable", "yes"):
                        self.toggle_reasoning(True)
                    elif arg in ("off", "0", "false", "disable", "no"):
                        self.toggle_reasoning(False)
                    else:
                        print(f"{UI.YELLOW}Usage: /think [on|off]{UI.RESET}")
                else:
                    self.toggle_reasoning()
                continue

            if root_cmd in ("/image", "/img", "/pic"):
                if not self.metadata.is_vlm:
                    print(
                        f"\n{UI.YELLOW}Current model '{self.metadata.name}' is an LLM (text-only). "
                        f"Images are only supported on VLMs.{UI.RESET}\n"
                    )
                    continue
                if len(parts) > 1:
                    raw_path = " ".join(parts[1:]).strip('\"\'')
                    img_path = Path(raw_path)
                    if not img_path.exists():
                        print(f"\n{UI.RED}Image file not found: {img_path}{UI.RESET}\n")
                        continue
                    try:
                        import numpy as np
                        from PIL import Image

                        pil_img = Image.open(img_path).convert("RGB")
                        self.pending_image = ov.Tensor(np.array(pil_img))
                        self.pending_image_path = str(img_path)
                        print(
                            f"\n{UI.GREEN}Image loaded: {img_path.name} "
                            f"({pil_img.width}x{pil_img.height}). Type your prompt next.{UI.RESET}\n"
                        )
                    except Exception as ex:
                        print(f"\n{UI.RED}Failed to load image: {ex}{UI.RESET}\n")
                else:
                    print(f"{UI.YELLOW}Usage: /image <path/to/image.jpg>{UI.RESET}")
                continue

            if root_cmd in ("/file", "/doc", "/attach"):
                if len(parts) > 1:
                    raw_doc_path = " ".join(parts[1:]).strip('\"\'')
                    doc_path = Path(raw_doc_path)
                    if not doc_path.exists():
                        print(f"\n{UI.RED}File not found: {doc_path}{UI.RESET}\n")
                        continue
                    try:
                        from .files import extract_text_from_path

                        text = extract_text_from_path(doc_path)
                        self.pending_file_context = text
                        self.pending_file_name = doc_path.name
                        print(
                            f"\n{UI.GREEN}Attached file: {doc_path.name} "
                            f"({len(text):,} chars extracted). Type your prompt next.{UI.RESET}\n"
                        )
                    except Exception as ex:
                        print(f"\n{UI.RED}Failed to extract file text: {ex}{UI.RESET}\n")
                else:
                    print(f"{UI.YELLOW}Usage: /file <path/to/document.pdf or file.txt>{UI.RESET}")
                continue

            if root_cmd in ("/unload", "/eject"):
                self.unload()
                continue

            if root_cmd == "/load":
                from .models import find_models
                from .devices import choose_device
                from .config import MODEL_ROOT
                all_models = find_models(MODEL_ROOT)
                if not all_models:
                    print(f"{UI.RED}No models found in {MODEL_ROOT}{UI.RESET}")
                    continue
                print("\nAvailable Models:")
                for idx, m in enumerate(all_models, 1):
                    print(f"  {idx}. {m.name}")
                sel = input("\nSelect model number: ").strip()
                if sel.isdigit() and 1 <= int(sel) <= len(all_models):
                    target_model = all_models[int(sel) - 1]
                    from .metadata import read_model_metadata
                    meta = read_model_metadata(target_model)
                    dev = choose_device(meta)
                    if dev:
                        self.load_model(target_model, dev)
                continue

            if self.pipe is None:
                print(f"\n{UI.YELLOW}No model is loaded. Type /load to load a model.{UI.RESET}\n")
                continue

            self.generate(prompt)

        print(f"{UI.DIM}Chat transcript saved: {self.history_saver.session_file}{UI.RESET}")

    def unload(self) -> None:
        """Unloads active model and frees VRAM/RAM."""
        if self.pipe is None:
            print(f"\n{UI.YELLOW}No model is currently loaded.{UI.RESET}\n")
            return
        import gc
        model_name = self.metadata.name if self.metadata else "model"
        self.pipe = None
        self.tokenizer = None
        self.generation_config = None
        gc.collect()
        print(f"\n{UI.GREEN}Model '{model_name}' unloaded from memory. VRAM freed.{UI.RESET}\n")

    def load_model(self, model_path: Path, device: str) -> None:
        """Loads a model dynamically into active session."""
        import gc
        self.pipe = None
        self.tokenizer = None
        self.generation_config = None
        gc.collect()

        from .metadata import read_model_metadata
        self.model_path = model_path
        self.metadata = read_model_metadata(model_path)
        self.device = device
        self.reasoning_enabled = self.metadata.supports_reasoning

        print(f"{UI.CYAN}Loading {self.metadata.display_type} on {device}...{UI.RESET}")
        t0 = time.perf_counter()
        if self.metadata.is_vlm:
            self.pipe = ov_genai.VLMPipeline(str(model_path), device)
        else:
            self.pipe = ov_genai.LLMPipeline(str(model_path), device)
        self.load_time = time.perf_counter() - t0
        self.tokenizer = self.pipe.get_tokenizer()
        self.generation_config = self.pipe.get_generation_config()
        self.generation_config.max_new_tokens = self.max_new_tokens

        self.history_saver = ChatHistorySaver(
            model_name=self.metadata.name,
            device=self.device,
            context_length=self.context_length,
            reasoning_enabled=self.reasoning_enabled,
        )
        print(f"{UI.GREEN}Loaded '{self.metadata.name}' successfully in {self.load_time:.2f}s.{UI.RESET}\n")

