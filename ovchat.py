import os
import sys
import time
import threading
import subprocess
from pathlib import Path

import openvino as ov
import openvino_genai as ov_genai


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ROOT = Path(r"D:\AI models\openvino-genai")

DEFAULT_CONTEXT = 32768
DEFAULT_MAX_NEW_TOKENS = 8192

# This is ONLY a rough fallback estimate.
# It is NOT the actual KV-cache size of Gemma.
DEFAULT_KV_BYTES_PER_TOKEN = 4096

RUNTIME_OVERHEAD_GB = 0.75

MEMORY_POLL_INTERVAL = 0.25


# ============================================================
# ANSI UI
# ============================================================

class UI:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def human_bytes(value):
    if value is None:
        return "N/A"

    value = float(value)

    units = [
        "B",
        "KB",
        "MB",
        "GB",
        "TB"
    ]

    for unit in units:
        if value < 1024:
            return f"{value:.2f} {unit}"

        value /= 1024

    return f"{value:.2f} PB"


def format_gb(value):
    if value is None:
        return "N/A"

    return f"{value / (1024 ** 3):.2f} GB"


def clamp(value, low, high):
    return max(low, min(high, value))


def progress_bar(
    fraction,
    width=40
):
    fraction = clamp(
        fraction,
        0.0,
        1.0
    )

    filled = int(
        fraction * width
    )

    empty = width - filled

    return (
        "["
        + "█" * filled
        + "░" * empty
        + "]"
    )


# ============================================================
# MODEL DISCOVERY
# ============================================================

def find_models():
    """
    Find directories containing OpenVINO IR models.

    A model directory normally contains one or more .xml/.bin
    files.
    """

    if not MODEL_ROOT.exists():
        return []

    models = []

    for directory in MODEL_ROOT.iterdir():

        if not directory.is_dir():
            continue

        xml_files = list(
            directory.glob("*.xml")
        )

        bin_files = list(
            directory.glob("*.bin")
        )

        if xml_files and bin_files:
            models.append(directory)

    return sorted(
        models,
        key=lambda x: x.name.lower()
    )


def model_size(model_path):
    """
    Return total .bin weight size.
    """

    total = 0

    for file in model_path.glob("*.bin"):
        try:
            total += file.stat().st_size
        except OSError:
            pass

    return total


# ============================================================
# OPENVINO DEVICE DETECTION
# ============================================================

def get_devices():
    core = ov.Core()

    try:
        devices = list(
            core.available_devices
        )
    except Exception:
        devices = []

    return devices


def choose_device():
    devices = get_devices()

    if not devices:
        print(
            f"{UI.RED}"
            "No OpenVINO devices detected."
            f"{UI.RESET}"
        )

        input("Press Enter...")
        return None

    print()
    print(
        f"{UI.BOLD}"
        "Available OpenVINO devices"
        f"{UI.RESET}"
    )

    for i, device in enumerate(
        devices,
        start=1
    ):
        print(
            f"  {i}. {device}"
        )

    print()

    while True:

        value = input(
            f"{UI.CYAN}"
            "Select device [1]: "
            f"{UI.RESET}"
        ).strip()

        if not value:
            return devices[0]

        try:
            index = int(value) - 1

            if 0 <= index < len(devices):
                return devices[index]

        except ValueError:
            pass

        print(
            f"{UI.YELLOW}"
            "Invalid selection."
            f"{UI.RESET}"
        )


# ============================================================
# MEMORY MONITOR
# ============================================================

def powershell_gpu_memory():
    """
    Reads Windows GPU adapter memory counters.

    IMPORTANT:
    For an integrated Arc GPU, this represents shared GPU
    memory and is NOT equivalent to dedicated VRAM usage.

    The counters are system-level, not process-specific.
    """

    if os.name != "nt":
        return None

    ps_script = r"""
$items = Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage',
                    '\GPU Adapter Memory(*)\Shared Usage' `
                    -ErrorAction SilentlyContinue

if ($null -eq $items) {
    exit
}

$total = 0

foreach ($sample in $items.CounterSamples) {
    if ($sample.Path -match 'Dedicated Usage|Shared Usage') {
        $total += [double]$sample.CookedValue
    }
}

[math]::Round($total)
"""

    try:

        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                ps_script
            ],
            capture_output=True,
            text=True,
            timeout=3
        )

        output = result.stdout.strip()

        if not output:
            return None

        return float(output)

    except Exception:
        return None


class MemoryMonitor:

    def __init__(self):

        self.current = None
        self.running = False

        self.thread = None

    def poll(self):

        while self.running:

            value = powershell_gpu_memory()

            if value is not None:
                self.current = value

            time.sleep(
                MEMORY_POLL_INTERVAL
            )

    def start(self):

        if self.running:
            return

        self.running = True

        self.thread = threading.Thread(
            target=self.poll,
            daemon=True
        )

        self.thread.start()

    def stop(self):

        self.running = False

        if self.thread is not None:
            self.thread.join(
                timeout=1
            )


# ============================================================
# MEMORY ESTIMATION
# ============================================================

def estimate_memory(
    model_path,
    context_length
):
    """
    Rough memory estimate.

    Model weights:
        actual .bin size

    KV:
        intentionally approximate

    Runtime:
        generic overhead estimate

    This is NOT a measurement of OpenVINO's actual memory
    allocation.
    """

    weights = model_size(
        model_path
    )

    kv = (
        context_length
        * DEFAULT_KV_BYTES_PER_TOKEN
    )

    runtime = int(
        RUNTIME_OVERHEAD_GB
        * 1024 ** 3
    )

    total = (
        weights
        + kv
        + runtime
    )

    return {
        "weights": weights,
        "kv": kv,
        "runtime": runtime,
        "total": total
    }


# ============================================================
# LOADING SCREEN
# ============================================================

class LoadingScreen:

    def __init__(
        self,
        model_path,
        device,
        context_length,
        memory_monitor
    ):

        self.model_path = model_path
        self.device = device
        self.context_length = context_length
        self.memory_monitor = memory_monitor

        self.baseline_memory = (
            memory_monitor.current
        )

        self.running = False
        self.thread = None

        self.estimate = estimate_memory(
            model_path,
            context_length
        )

    def render(self):

        current = (
            self.memory_monitor.current
        )

        baseline = (
            self.baseline_memory
        )

        if (
            current is not None
            and baseline is not None
        ):
            increase = max(
                0,
                current - baseline
            )

            estimated_total = (
                self.estimate["total"]
            )

            fraction = (
                increase /
                estimated_total
            )

            fraction = clamp(
                fraction,
                0,
                0.99
            )

        else:
            increase = None
            fraction = 0

        print(
            "\033[2J\033[H",
            end=""
        )

        print(
            f"{UI.BOLD}{UI.CYAN}"
            "Loading OpenVINO model"
            f"{UI.RESET}"
        )

        print()

        print(
            f"Model       : "
            f"{self.model_path.name}"
        )

        print(
            f"Device      : "
            f"{self.device}"
        )

        print(
            f"Context     : "
            f"{self.context_length:,}"
        )

        print()

        print(
            f"{UI.BOLD}"
            "Estimated memory"
            f"{UI.RESET}"
        )

        print(
            f"  Weights   : "
            f"{human_bytes(self.estimate['weights'])}"
        )

        print(
            f"  KV cache  : "
            f"{human_bytes(self.estimate['kv'])}"
        )

        print(
            f"  Runtime   : "
            f"{human_bytes(self.estimate['runtime'])}"
        )

        print(
            f"  Total     : "
            f"{human_bytes(self.estimate['total'])}"
        )

        print()

        print(
            f"{UI.BOLD}"
            "GPU/shared memory"
            f"{UI.RESET}"
        )

        print(
            f"  Before    : "
            f"{human_bytes(baseline)}"
        )

        print(
            f"  Current   : "
            f"{human_bytes(current)}"
        )

        print(
            f"  Increase  : "
            f"{human_bytes(increase)}"
        )

        print()

        print(
            progress_bar(
                fraction,
                width=45
            ),
            f"{fraction * 100:.1f}%"
        )

        print()

        print(
            f"{UI.YELLOW}"
            "Loading progress is an estimate based on observed "
            "GPU/shared-memory growth."
            f"{UI.RESET}"
        )

        print(
            f"{UI.DIM}"
            "OpenVINO does not expose a byte-by-byte model "
            "loading progress callback."
            f"{UI.RESET}"
        )

    def loop(self):

        while self.running:

            self.render()

            time.sleep(
                0.25
            )

    def start(self):

        self.running = True

        self.thread = threading.Thread(
            target=self.loop,
            daemon=True
        )

        self.thread.start()

    def stop(self):

        self.running = False

        if self.thread:

            self.thread.join(
                timeout=1
            )

        clear_screen()


# ============================================================
# TOKENIZER
# ============================================================

def count_tokens(
    tokenizer,
    text
):
    """
    Best-effort token count.
    """

    if not text:
        return 0

    try:

        encoded = tokenizer.encode(
            text
        )

        shape = encoded.input_ids.get_shape()

        if len(shape) >= 2:
            return int(
                shape[1]
            )

        return int(
            shape[0]
        )

    except Exception:

        return len(
            text.split()
        )


# ============================================================
# CHAT
# ============================================================

class Chat:

    def __init__(
        self,
        model_path,
        device,
        context_length,
        max_new_tokens
    ):

        self.model_path = model_path
        self.device = device

        self.context_length = (
            context_length
        )

        self.max_new_tokens = (
            max_new_tokens
        )

        # ----------------------------------------------------
        # Memory monitoring
        # ----------------------------------------------------

        self.memory_monitor = (
            MemoryMonitor()
        )

        # Take a baseline before loading.
        self.memory_monitor.current = (
            powershell_gpu_memory()
        )

        self.memory_monitor.start()

        # ----------------------------------------------------
        # Loading UI
        # ----------------------------------------------------

        loading = LoadingScreen(
            model_path,
            device,
            context_length,
            self.memory_monitor
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
            # that caused your original:
            #
            # "you have 5 inputs"
            #
            # error.
            # =================================================

            self.pipe = (
                ov_genai.VLMPipeline(
                    str(model_path),
                    device
                )
            )

        except Exception:

            loading.stop()
            self.memory_monitor.stop()

            raise

        self.load_time = (
            time.perf_counter()
            - load_start
        )

        # Let the memory monitor get a final sample.
        time.sleep(0.5)

        loading.stop()

        # Keep monitoring after model load.
        # This allows /info to report current memory.
        #
        # self.memory_monitor remains running.

        # ----------------------------------------------------
        # Tokenizer
        # ----------------------------------------------------

        self.tokenizer = (
            self.pipe.get_tokenizer()
        )

        # ----------------------------------------------------
        # Generation config
        # ----------------------------------------------------

        self.generation_config = (
            self.pipe.get_generation_config()
        )

        self.generation_config.max_new_tokens = (
            self.max_new_tokens
        )

        # ----------------------------------------------------
        # Chat history
        # ----------------------------------------------------

        self.chat_history = (
            ov_genai.ChatHistory()
        )

        # Python-side copy.
        #
        # We keep our own list because it lets us trim old
        # conversation turns and then rebuild ChatHistory.
        self.history_messages = []

        # ----------------------------------------------------
        # Memory statistics
        # ----------------------------------------------------

        self.memory_before = (
            loading.baseline_memory
        )

        self.memory_after = (
            self.memory_monitor.current
        )

        if (
            self.memory_before is not None
            and self.memory_after is not None
        ):

            self.actual_memory_increase = max(
                0,
                self.memory_after
                - self.memory_before
            )

        else:

            self.actual_memory_increase = None

    # ========================================================
    # Rebuild ChatHistory
    # ========================================================

    def rebuild_history(self):

        self.chat_history = (
            ov_genai.ChatHistory()
        )

        for message in (
            self.history_messages
        ):

            self.chat_history.append({
                "role": message["role"],
                "content": message["content"]
            })

    # ========================================================
    # Context estimation
    # ========================================================

    def context_usage(self):

        if not self.history_messages:
            return 0

        try:

            # Convert history to a string using the tokenizer's
            # chat template when possible.
            prompt = (
                self.tokenizer.apply_chat_template(
                    self.chat_history,
                    True
                )
            )

            encoded = (
                self.tokenizer.encode(
                    prompt
                )
            )

            shape = (
                encoded.input_ids.get_shape()
            )

            if len(shape) >= 2:
                return int(shape[1])

            return int(shape[0])

        except Exception:

            text = "\n".join(
                message["content"]
                for message
                in self.history_messages
            )

            return count_tokens(
                self.tokenizer,
                text
            )

    # ========================================================
    # Context trimming
    # ========================================================

    def trim_context(self):

        while (
            len(self.history_messages)
            > 2
        ):

            used = (
                self.context_usage()
            )

            required = (
                used
                + self.max_new_tokens
            )

            if required <= self.context_length:
                break

            # Remove oldest user + assistant turn.
            del self.history_messages[:2]

            self.rebuild_history()

        return self.context_usage()

    # ========================================================
    # Clear
    # ========================================================

    def clear(self):

        self.history_messages.clear()

        self.chat_history = (
            ov_genai.ChatHistory()
        )

        print()

        print(
            f"{UI.GREEN}"
            "Conversation cleared."
            f"{UI.RESET}"
        )

        print()

    # ========================================================
    # Info
    # ========================================================

    def info(self):

        used = (
            self.context_usage()
        )

        current_memory = (
            self.memory_monitor.current
        )

        print()

        print(
            f"{UI.BOLD}"
            "Model:"
            f"{UI.RESET} "
            f"{self.model_path.name}"
        )

        print(
            f"{UI.BOLD}"
            "Device:"
            f"{UI.RESET} "
            f"{self.device}"
        )

        print(
            f"{UI.BOLD}"
            "Context:"
            f"{UI.RESET} "
            f"{used:,} / "
            f"{self.context_length:,}"
        )

        print(
            f"{UI.BOLD}"
            "Max output:"
            f"{UI.RESET} "
            f"{self.max_new_tokens:,}"
        )

        print(
            f"{UI.BOLD}"
            "Load time:"
            f"{UI.RESET} "
            f"{self.load_time:.2f}s"
        )

        print(
            f"{UI.BOLD}"
            "GPU/shared memory:"
            f"{UI.RESET} "
            f"{human_bytes(current_memory)}"
        )

        if (
            self.actual_memory_increase
            is not None
        ):

            print(
                f"{UI.BOLD}"
                "Observed increase:"
                f"{UI.RESET} "
                f"{human_bytes(self.actual_memory_increase)}"
            )

        print()

    # ========================================================
    # Generate
    # ========================================================

    def generate(
        self,
        prompt
    ):

        # ----------------------------------------------------
        # Add user message
        # ----------------------------------------------------

        self.history_messages.append({
            "role": "user",
            "content": prompt
        })

        self.rebuild_history()

        # ----------------------------------------------------
        # Make room for response
        # ----------------------------------------------------

        self.trim_context()

        # ----------------------------------------------------
        # Streaming
        # ----------------------------------------------------

        print(
            f"\n"
            f"{UI.BOLD}{UI.GREEN}"
            "AI:"
            f"{UI.RESET} ",
            end="",
            flush=True
        )

        generated_chunks = []

        def streamer(text):

            print(
                text,
                end="",
                flush=True
            )

            generated_chunks.append(
                text
            )

            # False means continue.
            return False

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        try:

            # IMPORTANT:
            #
            # ChatHistory is passed as the input.
            #
            # Do NOT use:
            #
            #     history=self.chat_history
            #
            # The current VLM API supports ChatHistory directly.
            #
            result = self.pipe.generate(
                self.chat_history,
                generation_config=(
                    self.generation_config
                ),
                streamer=streamer
            )

        except Exception as e:

            print()

            print(
                f"\n{UI.RED}"
                "Generation error:"
                f"{UI.RESET} "
                f"{e}"
            )

            print()

            # Roll back failed user message.
            if self.history_messages:
                self.history_messages.pop()

            self.rebuild_history()

            return

        print()
        print()

        # ----------------------------------------------------
        # Performance metrics
        # ----------------------------------------------------

        try:

            metrics = (
                result.perf_metrics
            )

            input_tokens = (
                metrics.get_num_input_tokens()
            )

            output_tokens = (
                metrics.get_num_generated_tokens()
            )

            throughput = (
                metrics.get_throughput().mean
            )

            ttft = (
                metrics.get_ttft().mean
            )

            try:

                tpot = (
                    metrics.get_tpot().mean
                )

            except Exception:

                tpot = None

        except Exception:

            input_tokens = 0
            output_tokens = 0
            throughput = 0
            ttft = 0
            tpot = None

        # ----------------------------------------------------
        # Get generated text
        # ----------------------------------------------------

        assistant_text = "".join(
            generated_chunks
        )

        if not assistant_text:

            try:
                assistant_text = (
                    result.texts[0]
                )

            except Exception:
                assistant_text = ""

        # ----------------------------------------------------
        # Save assistant response
        # ----------------------------------------------------

        self.history_messages.append({
            "role": "assistant",
            "content": assistant_text
        })

        self.rebuild_history()

        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------

        context_used = (
            self.context_usage()
        )

        context_percent = (
            context_used
            / self.context_length
            * 100
        )

        context_percent = min(
            context_percent,
            100
        )

        # ----------------------------------------------------
        # Context bar
        # ----------------------------------------------------

        bar_fraction = (
            context_used
            / self.context_length
        )

        bar = progress_bar(
            bar_fraction,
            width=30
        )

        print(
            f"{UI.DIM}"
            f"{bar} "
            f"{context_used:,}/"
            f"{self.context_length:,} "
            f"({context_percent:.1f}%)"
            f"{UI.RESET}"
        )

        # ----------------------------------------------------
        # Performance
        # ----------------------------------------------------

        stats = (
            f"Input: {input_tokens:,} tok"
            f"  |  "
            f"Output: {output_tokens:,} tok"
            f"  |  "
            f"Speed: {throughput:.2f} tok/s"
            f"  |  "
            f"TTFT: {ttft:.0f} ms"
        )

        if tpot is not None:

            stats += (
                f"  |  "
                f"TPOT: {tpot:.2f} ms/tok"
            )

        print(
            f"{UI.DIM}"
            f"{stats}"
            f"{UI.RESET}"
        )

        print()

    # ========================================================
    # Chat loop
    # ========================================================

    def run(self):

        while True:

            try:

                prompt = input(
                    f"{UI.BOLD}{UI.CYAN}"
                    "You:"
                    f"{UI.RESET} "
                ).strip()

            except (
                KeyboardInterrupt,
                EOFError
            ):

                print()
                break

            if not prompt:
                continue

            command = (
                prompt.lower()
            )

            if command in (
                "/quit",
                "/exit",
                "/q"
            ):

                break

            if command == "/clear":

                self.clear()
                continue

            if command == "/info":

                self.info()
                continue

            self.generate(
                prompt
            )


# ============================================================
# MENU
# ============================================================

def choose_model():

    models = find_models()

    if not models:

        print(
            f"{UI.RED}"
            "No OpenVINO models found."
            f"{UI.RESET}"
        )

        print()

        print(
            "Expected model directory:"
        )

        print(
            MODEL_ROOT
        )

        print()

        print(
            "Each model should contain .xml and .bin files."
        )

        input(
            "\nPress Enter..."
        )

        return None

    print()

    print(
        f"{UI.BOLD}{UI.CYAN}"
        "OpenVINO Models"
        f"{UI.RESET}"
    )

    print()

    for i, model in enumerate(
        models,
        start=1
    ):

        size = model_size(
            model
        )

        print(
            f"  {i}. "
            f"{model.name}"
            f" "
            f"{UI.DIM}"
            f"({human_bytes(size)})"
            f"{UI.RESET}"
        )

    print()

    while True:

        value = input(
            f"{UI.CYAN}"
            "Select model [1]: "
            f"{UI.RESET}"
        ).strip()

        if not value:
            return models[0]

        try:

            index = (
                int(value) - 1
            )

            if (
                0 <= index < len(models)
            ):
                return models[index]

        except ValueError:
            pass

        print(
            f"{UI.YELLOW}"
            "Invalid selection."
            f"{UI.RESET}"
        )


def choose_integer(
    label,
    default,
    minimum=1
):

    while True:

        value = input(
            f"{UI.CYAN}"
            f"{label} [{default}]: "
            f"{UI.RESET}"
        ).strip()

        if not value:
            return default

        try:

            number = int(value)

            if number >= minimum:
                return number

        except ValueError:
            pass

        print(
            f"{UI.YELLOW}"
            f"Enter an integer >= {minimum}."
            f"{UI.RESET}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    clear_screen()

    print(
        f"{UI.BOLD}{UI.CYAN}"
        "OpenVINO GenAI Chat"
        f"{UI.RESET}"
    )

    print()

    print(
        f"{UI.DIM}"
        f"Model root: {MODEL_ROOT}"
        f"{UI.RESET}"
    )

    print()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model_path = choose_model()

    if model_path is None:
        return

    print()

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    context_length = choose_integer(
        "Context length",
        DEFAULT_CONTEXT,
        minimum=256
    )

    print()

    # --------------------------------------------------------
    # Maximum output
    # --------------------------------------------------------

    max_new_tokens = choose_integer(
        "Maximum output tokens",
        DEFAULT_MAX_NEW_TOKENS,
        minimum=1
    )

    print()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = choose_device()

    if device is None:
        return

    print()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        f"{UI.BOLD}"
        "Configuration"
        f"{UI.RESET}"
    )

    print(
        f"  Model   : "
        f"{model_path.name}"
    )

    print(
        f"  Context : "
        f"{context_length:,}"
    )

    print(
        f"  Output  : "
        f"{max_new_tokens:,}"
    )

    print(
        f"  Device  : "
        f"{device}"
    )

    print()

    input(
        "Press Enter to load..."
    )

    # --------------------------------------------------------
    # Start chat
    # --------------------------------------------------------

    try:

        chat = Chat(
            model_path,
            device,
            context_length,
            max_new_tokens
        )

    except Exception as e:

        print()

        print(
            f"{UI.RED}"
            "Failed to load model:"
            f"{UI.RESET}"
        )

        print()

        print(e)

        print()

        input(
            "Press Enter..."
        )

        return

    # --------------------------------------------------------
    # Chat
    # --------------------------------------------------------

    clear_screen()

    print(
        f"{UI.BOLD}{UI.GREEN}"
        "OpenVINO model loaded."
        f"{UI.RESET}"
    )

    print()

    print(
        f"Model : "
        f"{model_path.name}"
    )

    print(
        f"Device: "
        f"{device}"
    )

    print(
        f"Context: "
        f"{context_length:,}"
    )

    print(
        f"Max output: "
        f"{max_new_tokens:,}"
    )

    print()

    print(
        f"{UI.DIM}"
        "Commands:"
        f"{UI.RESET}"
    )

    print(
        "  /clear  - clear conversation"
    )

    print(
        "  /info   - show model/memory info"
    )

    print(
        "  /quit   - exit"
    )

    print()

    chat.run()

    # --------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------

    chat.memory_monitor.stop()

    print()

    print(
        f"{UI.GREEN}"
        "Goodbye."
        f"{UI.RESET}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()

        print(
            f"{UI.YELLOW}"
            "Interrupted."
            f"{UI.RESET}"
        )

    except Exception as e:

        print()

        print(
            f"{UI.RED}"
            "Fatal error:"
            f"{UI.RESET} "
            f"{e}"
        )

        print()

        input(
            "Press Enter..."
        )