import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import openvino as ov
import openvino_genai as ov_genai

from .history import ChatHistorySaver
from .loading import LoadingScreen
from .memory import MemoryMonitor, powershell_gpu_memory
from .metadata import ModelMetadata, read_model_metadata
from .ui import UI, human_bytes, progress_bar


def sanitize_file_path(raw: Union[str, Path]) -> Path:
    """
    Sanitizes file paths from CLI inputs, clipboard pastes, and tab completions.
    Handles trailing backslashes/quotes, PowerShell '& ' prefixes, etc.
    """
    if isinstance(raw, Path):
        s = str(raw)
    else:
        s = str(raw).strip()
    if s.startswith("&"):
        s = s[1:].strip()
    cleaned = s.replace('"', '').replace("'", '').strip()
    cleaned = re.sub(r'^[\\/]+([a-zA-Z]:)', r'\1', cleaned)
    cleaned = cleaned.rstrip('\\/').strip()
    return Path(cleaned).expanduser()


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

    @staticmethod
    def extract_inline_file(prompt: str) -> Tuple[str, Optional[Path]]:
        """
        Detects if /file <path> is embedded within the prompt string.
        Returns (cleaned_prompt, attached_file_path).
        """
        match = re.search(r'/file\s+("([^"]+)"|(\S+))', prompt, re.IGNORECASE)
        if not match:
            return prompt, None

        raw_file = match.group(2) or match.group(3)
        file_path = sanitize_file_path(raw_file)
        cleaned_prompt = prompt[:match.start()] + prompt[match.end():]
        cleaned_prompt = re.sub(r'\s+', ' ', cleaned_prompt).strip()
        return cleaned_prompt, file_path

    @staticmethod
    def extract_inline_image(prompt: str) -> Tuple[str, Optional[Path]]:
        """
        Detects if /image, /img, or /pic <path> is embedded within the prompt string.
        Returns (cleaned_prompt, attached_image_path).
        """
        match = re.search(r'/(?:image|img|pic)\s+("([^"]+)"|(\S+))', prompt, re.IGNORECASE)
        if not match:
            return prompt, None

        raw_img = match.group(2) or match.group(3)
        img_path = sanitize_file_path(raw_img)
        cleaned_prompt = prompt[:match.start()] + prompt[match.end():]
        cleaned_prompt = re.sub(r'\s+', ' ', cleaned_prompt).strip()
        return cleaned_prompt, img_path

    @staticmethod
    def parse_export_args(raw_args: str) -> Tuple[Optional[str], Optional[Path]]:
        """
        Parses arguments passed to /export.
        Supports:
          - /export                                     -> (None, None) (export last answer to default file)
          - /export output.txt                          -> (None, Path("output.txt")) (export last answer)
          - /export "prompt" "output.txt"               -> ("prompt", Path("output.txt"))
          - /export "prompt" output.txt                 -> ("prompt", Path("output.txt"))
          - /export output.txt prompt                   -> ("prompt", Path("output.txt"))
          - /export "output.txt" prompt                 -> ("prompt", Path("output.txt"))
          - /export prompt output.txt                   -> ("prompt", Path("output.txt"))
        """
        raw = raw_args.strip()
        if not raw:
            return None, None

        # Pattern 1: "prompt" "output.txt" or "prompt" output.txt
        match1 = re.match(r'^"(.+)"\s+("([^"]+)"|(\S+))$', raw)
        if match1:
            prompt = match1.group(1).strip()
            filename = match1.group(3) or match1.group(4)
            return prompt, sanitize_file_path(filename)

        # Pattern 2: "output.txt" prompt
        match2 = re.match(r'^"([^"]+\.(?:txt|md|log|json|csv|html))"\s+(.+)$', raw, re.IGNORECASE)
        if match2:
            filename = match2.group(1).strip()
            prompt = match2.group(2).strip()
            if prompt.startswith('"') and prompt.endswith('"') and len(prompt) >= 2:
                prompt = prompt[1:-1]
            return prompt, sanitize_file_path(filename)

        # Pattern 3: output.txt prompt ...
        parts = raw.split(maxsplit=1)
        first_token = sanitize_file_path(parts[0])
        if any(str(first_token).lower().endswith(ext) for ext in (".txt", ".md", ".log", ".json", ".csv", ".html")):
            if len(parts) > 1:
                prompt = parts[1].strip()
                if prompt.startswith('"') and prompt.endswith('"') and len(prompt) >= 2:
                    prompt = prompt[1:-1]
                return prompt, first_token
            else:
                return None, first_token

        # Pattern 4: prompt ... output.txt
        parts_r = raw.rsplit(maxsplit=1)
        last_token = sanitize_file_path(parts_r[-1])
        if any(str(last_token).lower().endswith(ext) for ext in (".txt", ".md", ".log", ".json", ".csv")):
            prompt = parts_r[0].strip()
            if prompt.startswith('"') and prompt.endswith('"') and len(prompt) >= 2:
                prompt = prompt[1:-1]
            return prompt, last_token

        # Single word without space ending with file extension
        sanitized_single = sanitize_file_path(raw)
        if any(str(sanitized_single).lower().endswith(ext) for ext in (".txt", ".md", ".log", ".json", ".csv")):
            return None, sanitized_single

        # Fallback: treat entire string as prompt to be generated
        return raw.strip('\"\''), None

    def generate(self, prompt: str, export_file: Optional[Path] = None) -> None:
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

        if export_file is not None:
            try:
                export_target = export_file.expanduser().resolve()
                export_target.parent.mkdir(parents=True, exist_ok=True)
                export_target.write_text(assistant_text, encoding="utf-8")
                print(f"{UI.GREEN}[OK] Model answer exported to:{UI.RESET} {export_target}\n")
            except Exception as ex:
                print(f"{UI.RED}[!] Failed to export answer to {export_file}: {ex}{UI.RESET}\n")

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

                raw_args = prompt[len(root_cmd):].strip()
                if not raw_args:
                    print(f"{UI.YELLOW}Usage: /image <path/to/image.jpg> [optional prompt]{UI.RESET}")
                    continue

                prompt_text = ""
                q_match = re.match(r'^"([^"]+)"\s*(.*)$', raw_args)
                if q_match:
                    img_path = sanitize_file_path(q_match.group(1))
                    remaining = q_match.group(2).strip()
                    if remaining and remaining not in ("\\", "/"):
                        prompt_text = remaining
                else:
                    parts_args = raw_args.split(maxsplit=1)
                    img_path = sanitize_file_path(parts_args[0])
                    if len(parts_args) > 1:
                        prompt_text = parts_args[1].strip()

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
                        f"({pil_img.width}x{pil_img.height}).{UI.RESET}"
                    )
                    if prompt_text:
                        print(f"{UI.BOLD}{UI.CYAN}Prompt:{UI.RESET} {prompt_text}\n")
                        self.generate(prompt_text)
                    else:
                        print(f"{UI.DIM}Type your prompt next.{UI.RESET}\n")
                except Exception as ex:
                    print(f"\n{UI.RED}Failed to load image: {ex}{UI.RESET}\n")
                continue

            if root_cmd in ("/file", "/doc", "/attach"):
                raw_args = prompt[len(root_cmd):].strip()
                if not raw_args:
                    print(f"{UI.YELLOW}Usage: /file <path/to/document.pdf or file.txt> [optional prompt]{UI.RESET}")
                    continue

                prompt_text = ""
                q_match = re.match(r'^"([^"]+)"\s*(.*)$', raw_args)
                if q_match:
                    doc_path = sanitize_file_path(q_match.group(1))
                    remaining = q_match.group(2).strip()
                    if remaining and remaining not in ("\\", "/"):
                        prompt_text = remaining
                else:
                    parts_args = raw_args.split(maxsplit=1)
                    doc_path = sanitize_file_path(parts_args[0])
                    if len(parts_args) > 1:
                        prompt_text = parts_args[1].strip()

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
                        f"({len(text):,} chars extracted).{UI.RESET}"
                    )
                    if prompt_text:
                        print(f"{UI.BOLD}{UI.CYAN}Prompt:{UI.RESET} {prompt_text}\n")
                        self.generate(prompt_text)
                    else:
                        print(f"{UI.DIM}Type your prompt next.{UI.RESET}\n")
                except Exception as ex:
                    print(f"\n{UI.RED}Failed to extract file text: {ex}{UI.RESET}\n")
                continue

            if root_cmd in ("/export", "/save", "/export-answer"):
                raw_args = prompt[len(root_cmd):].strip()
                prompt_text, export_path = self.parse_export_args(raw_args)

                if prompt_text:
                    # 1. User specified a prompt: generate answer and export directly
                    cleaned_prompt, inline_file = self.extract_inline_file(prompt_text)
                    if inline_file:
                        if not inline_file.exists():
                            print(f"\n{UI.RED}Attached file not found: {inline_file}{UI.RESET}\n")
                            continue
                        try:
                            from .files import extract_text_from_path
                            self.pending_file_context = extract_text_from_path(inline_file)
                            self.pending_file_name = inline_file.name
                            print(f"\n{UI.GREEN}Attached file: {inline_file.name} ({len(self.pending_file_context):,} chars).{UI.RESET}")
                        except Exception as ex:
                            print(f"\n{UI.RED}Failed to read attached file: {ex}{UI.RESET}\n")
                            continue

                    cleaned_prompt, inline_img = self.extract_inline_image(cleaned_prompt)
                    if inline_img:
                        if not self.metadata.is_vlm:
                            print(f"\n{UI.YELLOW}Current model '{self.metadata.name}' is an LLM (text-only). Images are only supported on VLMs.{UI.RESET}\n")
                        elif not inline_img.exists():
                            print(f"\n{UI.RED}Attached image not found: {inline_img}{UI.RESET}\n")
                            continue
                        else:
                            try:
                                import numpy as np
                                from PIL import Image

                                pil_img = Image.open(inline_img).convert("RGB")
                                self.pending_image = ov.Tensor(np.array(pil_img))
                                self.pending_image_path = str(inline_img)
                                print(f"\n{UI.GREEN}Attached image: {inline_img.name} ({pil_img.width}x{pil_img.height}).{UI.RESET}")
                            except Exception as ex:
                                print(f"\n{UI.RED}Failed to load attached image: {ex}{UI.RESET}\n")
                                continue

                    if not export_path:
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        export_path = Path(f"answer_{ts}.txt")

                    if self.pipe is None:
                        print(f"\n{UI.YELLOW}No model is loaded. Type /load to load a model.{UI.RESET}\n")
                        continue

                    self.generate(cleaned_prompt, export_file=export_path)
                    continue

                else:
                    # 2. No prompt specified: export the LAST generated model answer
                    last_reply = None
                    for m in reversed(self.history_messages):
                        if m["role"] == "assistant":
                            last_reply = m["content"]
                            break

                    if not last_reply:
                        print(f"\n{UI.YELLOW}No model answer to export yet. Please chat with the model first.{UI.RESET}\n")
                        continue

                    if not export_path:
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        export_path = Path(f"answer_{ts}.txt")

                    target_file = export_path.expanduser().resolve()
                    try:
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        target_file.write_text(last_reply, encoding="utf-8")
                        print(f"\n{UI.GREEN}[OK] Last model answer exported to:{UI.RESET} {target_file}\n")
                    except Exception as ex:
                        print(f"\n{UI.RED}Failed to export file: {ex}{UI.RESET}\n")
                    continue

            if root_cmd in ("/export-chat", "/save-chat", "/export-all"):
                if not self.history_messages:
                    print(f"\n{UI.YELLOW}No chat history to export yet.{UI.RESET}\n")
                    continue

                if len(parts) > 1:
                    raw_target = " ".join(parts[1:]).strip('\"\'')
                    if not raw_target.lower().endswith((".txt", ".md")):
                        raw_target += ".txt"
                else:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    raw_target = f"chat_{ts}.txt"

                target_file = Path(raw_target).expanduser().resolve()
                try:
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    lines = []
                    for m in self.history_messages:
                        speaker = "You" if m["role"] == "user" else "Assistant"
                        lines.append(f"[{speaker}]\n{m['content']}\n")
                    target_file.write_text("\n".join(lines), encoding="utf-8")
                    print(f"\n{UI.GREEN}[OK] Entire chat conversation exported to:{UI.RESET} {target_file}\n")
                except Exception as ex:
                    print(f"\n{UI.RED}Failed to export file: {ex}{UI.RESET}\n")
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

            # Support inline /file embedded directly in regular prompt
            if "/file " in prompt.lower():
                prompt, inline_file = self.extract_inline_file(prompt)
                if inline_file:
                    if inline_file.exists():
                        try:
                            from .files import extract_text_from_path
                            self.pending_file_context = extract_text_from_path(inline_file)
                            self.pending_file_name = inline_file.name
                            print(f"\n{UI.GREEN}Attached file: {inline_file.name} ({len(self.pending_file_context):,} chars).{UI.RESET}")
                        except Exception as ex:
                            print(f"\n{UI.RED}Failed to read attached file: {ex}{UI.RESET}\n")
                    else:
                        print(f"\n{UI.RED}Attached file not found: {inline_file}{UI.RESET}\n")

            # Support inline /image embedded directly in regular prompt
            if any(k in prompt.lower() for k in ("/image ", "/img ", "/pic ")):
                if not self.metadata.is_vlm:
                    print(f"\n{UI.YELLOW}Current model '{self.metadata.name}' is an LLM (text-only). Images are only supported on VLMs.{UI.RESET}\n")
                else:
                    prompt, inline_img = self.extract_inline_image(prompt)
                    if inline_img:
                        if inline_img.exists():
                            try:
                                import numpy as np
                                from PIL import Image

                                pil_img = Image.open(inline_img).convert("RGB")
                                self.pending_image = ov.Tensor(np.array(pil_img))
                                self.pending_image_path = str(inline_img)
                                print(f"\n{UI.GREEN}Attached image: {inline_img.name} ({pil_img.width}x{pil_img.height}).{UI.RESET}")
                            except Exception as ex:
                                print(f"\n{UI.RED}Failed to load attached image: {ex}{UI.RESET}\n")
                        else:
                            print(f"\n{UI.RED}Attached image not found: {inline_img}{UI.RESET}\n")

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

