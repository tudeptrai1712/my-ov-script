import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import openvino_genai as ov_genai

from .loading import LoadingScreen
from .memory import MemoryMonitor, powershell_gpu_memory
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
    ):
        self.model_path = model_path
        self.device = device
        self.context_length = context_length
        self.max_new_tokens = max_new_tokens

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
            # =================================================
            # IMPORTANT:
            #
            # Use VLMPipeline for Gemma 4 / VLM-capable models.
            #
            # This avoids the LLMPipeline 3/4-input restriction
            # that caused the original "you have 5 inputs" error.
            # =================================================
            self.pipe = ov_genai.VLMPipeline(
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
        """Clears conversation history."""
        self.history_messages.clear()
        self.chat_history = ov_genai.ChatHistory()
        print(f"\n{UI.GREEN}Conversation cleared.{UI.RESET}\n")

    def info(self) -> None:
        """Prints current session and memory statistics."""
        used = self.context_usage()
        current_memory = self.memory_monitor.current

        print()
        print(f"{UI.BOLD}Model:{UI.RESET} {self.model_path.name}")
        print(f"{UI.BOLD}Device:{UI.RESET} {self.device}")
        print(f"{UI.BOLD}Context:{UI.RESET} {used:,} / {self.context_length:,}")
        print(f"{UI.BOLD}Max output:{UI.RESET} {self.max_new_tokens:,}")
        print(f"{UI.BOLD}Load time:{UI.RESET} {self.load_time:.2f}s")
        print(f"{UI.BOLD}GPU/shared memory:{UI.RESET} {human_bytes(current_memory)}")

        if self.actual_memory_increase is not None:
            print(
                f"{UI.BOLD}Observed increase:{UI.RESET} "
                f"{human_bytes(self.actual_memory_increase)}"
            )

        print()

    def generate(self, prompt: str) -> None:
        """Streams response generation for the user prompt."""
        # Add user message
        self.history_messages.append({
            "role": "user",
            "content": prompt,
        })
        self.rebuild_history()

        # Make room for response
        self.trim_context()

        # Streaming setup
        print(f"\n{UI.BOLD}{UI.GREEN}AI:{UI.RESET} ", end="", flush=True)
        generated_chunks: List[str] = []

        def streamer(text: str) -> bool:
            print(text, end="", flush=True)
            generated_chunks.append(text)
            # Return False to continue streaming
            return False

        try:
            # IMPORTANT: ChatHistory is passed directly as the input.
            result = self.pipe.generate(
                self.chat_history,
                generation_config=self.generation_config,
                streamer=streamer,
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
            if command in ("/quit", "/exit", "/q"):
                break

            if command == "/clear":
                self.clear()
                continue

            if command == "/info":
                self.info()
                continue

            self.generate(prompt)

