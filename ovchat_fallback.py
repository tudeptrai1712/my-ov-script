"""
OpenVINO GenAI Chat - Standalone Fallback Script.

Self-contained single-file version supporting:
- Both LLM (pure text) and VLM (vision-language multimodal) models
- Document/file attachment support via /file <path> (PDF, code, text, CSV, markdown)
- Multimodal image support via /image <path> (for VLMs)
- Model metadata inspection and incompatible device filtering (e.g. hiding NPU for VLM)
- Reasoning / thinking mode toggle on launch and live during chat (/think)
- Live GPU/shared memory monitoring via Windows counters
- Automatic persistent chat history logging to disk
- Context management and streaming generation
"""

import datetime
import io
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ------------------------------------------------------------
# DEPENDENCY VERIFIER (Self-healing on launch)
# ------------------------------------------------------------

def _verify_dependencies() -> None:
    required = [
        ("openvino", "openvino"),
        ("openvino_genai", "openvino-genai"),
        ("numpy", "numpy"),
        ("PIL", "Pillow"),
        ("pypdf", "pypdf"),
    ]
    missing = []
    for mod, pkg in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("\n" + "=" * 60)
        print("  [Dependency Verifier] Installing missing packages...")
        print(f"  Packages: {', '.join(missing)}")
        print("=" * 60 + "\n")
        for pkg in missing:
            print(f"Installing {pkg}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
                print(f"  [OK] {pkg} installed.")
            except Exception as e:
                print(f"  [!] Failed to install {pkg}: {e}")
        print("\n[Dependency Verifier] Verification complete.\n")
    else:
        print("[Dependency Verifier] Checking required packages... All dependencies satisfied [OK]")

_verify_dependencies()

import openvino as ov
import openvino_genai as ov_genai


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ROOT = Path(
    os.environ.get("OVCHAT_MODEL_ROOT", r"D:\AI models\openvino-genai")
)

CHAT_HISTORY_DIR = Path(
    os.environ.get(
        "OVCHAT_HISTORY_DIR",
        r"D:\AI models\openvino-genai\chat history",
    )
)

DEFAULT_CONTEXT: int = 32768
DEFAULT_MAX_NEW_TOKENS: int = 8192
DEFAULT_KV_BYTES_PER_TOKEN: int = 4096
RUNTIME_OVERHEAD_GB: float = 0.75
MEMORY_POLL_INTERVAL: float = 0.25
DEFAULT_ENABLE_REASONING: bool = False


# ============================================================
# ANSI UI HELPERS
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


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def human_bytes(value: Optional[Union[int, float]]) -> str:
    if value is None:
        return "N/A"
    val = float(value)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if val < 1024:
            return f"{val:.2f} {unit}"
        val /= 1024
    return f"{val:.2f} PB"


def clamp(value: Union[int, float], low: Union[int, float], high: Union[int, float]) -> Union[int, float]:
    return max(low, min(high, value))


def progress_bar(fraction: float, width: int = 40) -> str:
    clamped_fraction = clamp(fraction, 0.0, 1.0)
    filled = int(clamped_fraction * width)
    empty = width - filled
    return f"[{'█' * filled}{'░' * empty}]"


# ============================================================
# DOCUMENT & FILE EXTRACTION
# ============================================================

def extract_file_text(path: Path, max_chars: int = 50000) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    raw_bytes = path.read_bytes()

    if ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
            pages = []
            chars = 0
            for i, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                pages.append(f"--- Page {i + 1} ---\n{txt}")
                chars += len(txt)
                if chars > max_chars:
                    pages.append(f"\n[... Truncated after {max_chars:,} characters ...]")
                    break
            return "\n\n".join(pages).strip()
        except Exception as e:
            return f"[Error extracting PDF text: {e}]"

    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            txt = raw_bytes.decode(enc)
            if len(txt) > max_chars:
                txt = txt[:max_chars] + f"\n[... Truncated after {max_chars:,} characters ...]"
            return txt
        except UnicodeDecodeError:
            continue

    return "[Binary or unsupported file encoding]"


# ============================================================
# MODEL DISCOVERY & METADATA
# ============================================================

def model_size(model_path: Path) -> int:
    total = 0
    for file in model_path.glob("*.bin"):
        try:
            total += file.stat().st_size
        except OSError:
            pass
    return total


def find_models(root: Optional[Path] = None) -> List[Path]:
    target_root = root or MODEL_ROOT
    if not target_root.exists():
        return []

    models = []
    for directory in target_root.iterdir():
        if not directory.is_dir():
            continue
        xml_files = list(directory.glob("*.xml"))
        bin_files = list(directory.glob("*.bin"))
        if xml_files and bin_files:
            models.append(directory)

    return sorted(models, key=lambda x: x.name.lower())


class ModelMetadata:
    def __init__(self, path: Path):
        self.path = path
        self.name = path.name

        config_path = path / "config.json"
        ov_config_path = path / "openvino_config.json"
        tokenizer_config_path = path / "tokenizer_config.json"
        chat_template_path = path / "chat_template.jinja"

        config = {}
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    config = json.load(f)
            except Exception:
                pass

        ov_config = {}
        if ov_config_path.exists():
            try:
                with open(ov_config_path, encoding="utf-8") as f:
                    ov_config = json.load(f)
            except Exception:
                pass

        self.model_type = config.get("model_type", "unknown")
        architectures = config.get("architectures", [])
        self.architecture = architectures[0] if architectures else self.model_type

        # VLM vs LLM detection
        self.is_vlm = (
            (path / "openvino_vision_embeddings_model.xml").exists()
            or (path / "processor_config.json").exists()
            or (path / "preprocessor_config.json").exists()
            or "vision_config" in config
            or "audio_config" in config
            or any(
                k in self.architecture.lower()
                for k in ("conditionalgeneration", "vlm", "vision", "gemma4")
            )
        )

        self.precision = str(
            ov_config.get("dtype")
            or config.get("dtype")
            or config.get("torch_dtype", "unknown")
        ).lower()

        text_config = config.get("text_config", {})
        self.max_position_embeddings = (
            text_config.get("max_position_embeddings")
            or config.get("max_position_embeddings")
            or config.get("max_sequence_length")
        )

        self.supports_reasoning = False
        if chat_template_path.exists():
            try:
                txt = chat_template_path.read_text(encoding="utf-8")
                if any(tag in txt for tag in ("enable_thinking", "<|think|>", "<think>", "thought")):
                    self.supports_reasoning = True
            except Exception:
                pass

        if not self.supports_reasoning and tokenizer_config_path.exists():
            try:
                with open(tokenizer_config_path, encoding="utf-8") as f:
                    tok_cfg = json.load(f)
                if (
                    "think_token" in tok_cfg.get("model_specific_special_tokens", {})
                    or "thinking" in str(tok_cfg.get("response_schema", ""))
                ):
                    self.supports_reasoning = True
            except Exception:
                pass

        self.weight_size_bytes = model_size(path)

    @property
    def display_type(self) -> str:
        return "Vision-Language Model (VLM)" if self.is_vlm else "Language Model (LLM)"


# ============================================================
# DEVICE COMPATIBILITY & SELECTION
# ============================================================

def check_device_compatibility(
    device: str,
    metadata: ModelMetadata,
    core: Optional[ov.Core] = None,
) -> Tuple[bool, Optional[str]]:
    dev_upper = device.upper()

    if dev_upper.startswith("CPU") or dev_upper.startswith("GPU"):
        return True, None

    if dev_upper.startswith("NPU"):
        if metadata.is_vlm:
            return False, "Vision-Language Models (VLM) are not supported on NPU in OpenVINO GenAI"

        if core is None:
            core = ov.Core()

        try:
            caps = core.get_property("NPU", "OPTIMIZATION_CAPABILITIES")
        except Exception:
            caps = []

        if "int4" in metadata.precision and "INT4" not in caps:
            return False, "INT4 precision is not supported by NPU compiler"

        npu_supported = ("llama", "qwen2", "mistral", "phi3")
        if not any(arch in metadata.model_type.lower() for arch in npu_supported):
            return False, f"Model architecture '{metadata.model_type}' is not supported on NPU"

        return True, None

    return True, None


def choose_device(metadata: Optional[ModelMetadata] = None) -> Optional[str]:
    core = ov.Core()
    try:
        raw_devices = list(core.available_devices)
    except Exception:
        raw_devices = []

    if not raw_devices:
        print(f"{UI.RED}No OpenVINO devices detected.{UI.RESET}")
        input("Press Enter...")
        return None

    compatible_devices = []
    hidden_devices = []

    for dev in raw_devices:
        if metadata is not None:
            is_ok, reason = check_device_compatibility(dev, metadata, core)
            if is_ok:
                compatible_devices.append(dev)
            else:
                hidden_devices.append((dev, reason or "Incompatible with model"))
        else:
            compatible_devices.append(dev)

    if not compatible_devices:
        print(f"{UI.RED}No compatible OpenVINO devices found for model.{UI.RESET}")
        input("\nPress Enter...")
        return None

    print()
    if metadata is not None:
        print(f"{UI.BOLD}Compatible OpenVINO devices for {metadata.name}{UI.RESET}")
    else:
        print(f"{UI.BOLD}Available OpenVINO devices{UI.RESET}")

    for i, dev in enumerate(compatible_devices, start=1):
        print(f"  {i}. {dev}")

    if hidden_devices:
        print()
        for dev, reason in hidden_devices:
            print(f"  {UI.DIM}[Hidden unavailable: {dev} - {reason}]{UI.RESET}")

    print()

    while True:
        value = input(f"{UI.CYAN}Select device [1]: {UI.RESET}").strip()
        if not value:
            return compatible_devices[0]

        try:
            idx = int(value) - 1
            if 0 <= idx < len(compatible_devices):
                return compatible_devices[idx]
        except ValueError:
            pass

        print(f"{UI.YELLOW}Invalid selection.{UI.RESET}")


# ============================================================
# MEMORY MONITOR
# ============================================================

def powershell_gpu_memory() -> Optional[float]:
    if os.name != "nt":
        return None

    ps_script = r"""
$items = Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage',
                    '\GPU Adapter Memory(*)\Shared Usage' `
                    -ErrorAction SilentlyContinue
if ($null -eq $items) { exit }
$total = 0
foreach ($sample in $items.CounterSamples) {
    if ($sample.Path -match 'Dedicated Usage|Shared Usage') {
        $total += [double]$sample.CookedValue
    }
}
[math]::Round($total)
"""
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=3,
        )
        out = res.stdout.strip()
        return float(out) if out else None
    except Exception:
        return None


class MemoryMonitor:
    def __init__(self, interval: float = MEMORY_POLL_INTERVAL):
        self.interval = interval
        self.current: Optional[float] = None
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None

    def poll(self) -> None:
        while self.running:
            val = powershell_gpu_memory()
            if val is not None:
                self.current = val
            time.sleep(self.interval)

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self.poll, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1)


# ============================================================
# PERSISTENT CHAT HISTORY SAVER
# ============================================================

class ChatHistorySaver:
    def __init__(
        self,
        model_name: str,
        device: str,
        context_length: int,
        reasoning_enabled: bool,
        history_dir: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.context_length = context_length
        self.reasoning_enabled = reasoning_enabled
        self.history_dir = history_dir or CHAT_HISTORY_DIR
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.start_time = datetime.datetime.now()
        self.session_file = self._create_session_file()
        self.turns: List[Dict[str, Any]] = []
        self.save()

    def _create_session_file(self) -> Path:
        safe_model = re.sub(r"[^\w\-.]", "_", self.model_name)
        timestamp = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        base = f"{timestamp}_{safe_model}"
        file_path = self.history_dir / f"{base}.md"
        counter = 1
        while file_path.exists():
            file_path = self.history_dir / f"{base}_{counter}.md"
            counter += 1
        return file_path

    def add_turn(
        self,
        user_prompt: str,
        assistant_response: str,
        metrics: Optional[Dict[str, Any]] = None,
        reasoning_used: bool = False,
    ) -> None:
        turn_time = datetime.datetime.now()
        self.turns.append({
            "timestamp": turn_time.strftime("%H:%M:%S"),
            "user": user_prompt,
            "assistant": assistant_response,
            "metrics": metrics or {},
            "reasoning_used": reasoning_used,
        })
        self.save()

    def save(self) -> None:
        lines: List[str] = [
            "---",
            f"started_at: '{self.start_time.isoformat()}'",
            f"model: '{self.model_name}'",
            f"device: '{self.device}'",
            f"context_length: {self.context_length}",
            f"reasoning_mode: {str(self.reasoning_enabled).lower()}",
            f"total_turns: {len(self.turns)}",
            "---\n",
            f"# Chat Session: {self.model_name}\n",
            f"- **Date**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **Device**: {self.device}",
            f"- **Context Limit**: {self.context_length:,} tokens",
            f"- **Reasoning Mode**: {'Enabled' if self.reasoning_enabled else 'Disabled'}\n",
            "---\n",
        ]

        if not self.turns:
            lines.append("*(Session started - waiting for conversation...)*\n")
        else:
            for idx, turn in enumerate(self.turns, start=1):
                ts = turn.get("timestamp", "")
                reason_tag = " `[Thinking Mode]`" if turn.get("reasoning_used") else ""
                lines.append(f"### Turn {idx} ({ts})\n")
                lines.append(f"**User:**\n\n{turn['user']}\n")
                lines.append(f"**AI{reason_tag}:**\n\n{turn['assistant']}\n")

                m = turn.get("metrics", {})
                if m:
                    parts = []
                    if m.get("input_tokens"):
                        parts.append(f"Input: {m['input_tokens']:,} tok")
                    if m.get("output_tokens"):
                        parts.append(f"Output: {m['output_tokens']:,} tok")
                    if m.get("throughput", 0) > 0:
                        parts.append(f"Speed: {m['throughput']:.2f} tok/s")
                    if m.get("ttft", 0) > 0:
                        parts.append(f"TTFT: {m['ttft']:.0f} ms")
                    if parts:
                        lines.append(f"> *Metrics: {' | '.join(parts)}*\n")
                lines.append("---\n")

        try:
            self.session_file.write_text("\n".join(lines), encoding="utf-8")
        except Exception:
            pass

    def reset(self) -> None:
        self.start_time = datetime.datetime.now()
        self.session_file = self._create_session_file()
        self.turns = []
        self.save()


# ============================================================
# CHAT SESSION
# ============================================================

class Chat:
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
        self.metadata = metadata or ModelMetadata(model_path)
        self.reasoning_enabled = enable_reasoning and self.metadata.supports_reasoning

        self.pending_image: Optional[ov.Tensor] = None
        self.pending_image_path: Optional[str] = None
        self.pending_file_context: Optional[str] = None
        self.pending_file_name: Optional[str] = None

        self.history_saver = ChatHistorySaver(
            model_name=self.model_path.name,
            device=self.device,
            context_length=self.context_length,
            reasoning_enabled=self.reasoning_enabled,
        )

        self.memory_monitor = MemoryMonitor()
        self.memory_monitor.current = powershell_gpu_memory()
        self.memory_monitor.start()

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

        self.chat_history = ov_genai.ChatHistory()
        self.history_messages: List[Dict[str, str]] = []

    def rebuild_history(self) -> None:
        self.chat_history = ov_genai.ChatHistory()
        for msg in self.history_messages:
            self.chat_history.append({"role": msg["role"], "content": msg["content"]})

    def clear(self) -> None:
        self.history_messages.clear()
        self.chat_history = ov_genai.ChatHistory()
        self.pending_image = None
        self.pending_image_path = None
        self.pending_file_context = None
        self.pending_file_name = None
        self.history_saver.reset()
        print(f"\n{UI.GREEN}Conversation cleared.{UI.RESET}")
        print(f"{UI.DIM}New history log: {self.history_saver.session_file}{UI.RESET}\n")

    def toggle_reasoning(self, enable: Optional[bool] = None) -> bool:
        if not self.metadata.supports_reasoning:
            print(f"\n{UI.YELLOW}Reasoning mode is not supported by model '{self.metadata.name}'.{UI.RESET}\n")
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
        current_memory = self.memory_monitor.current
        reason_status = (
            f"{UI.GREEN}Enabled{UI.RESET}" if self.reasoning_enabled else f"{UI.YELLOW}Disabled{UI.RESET}"
        ) if self.metadata.supports_reasoning else f"{UI.DIM}Not supported{UI.RESET}"

        print()
        print(f"{UI.BOLD}Model:{UI.RESET} {self.model_path.name}")
        print(f"{UI.BOLD}Type:{UI.RESET} {self.metadata.display_type} ({self.metadata.precision.upper()})")
        print(f"{UI.BOLD}Device:{UI.RESET} {self.device}")
        print(f"{UI.BOLD}Reasoning:{UI.RESET} {reason_status}")
        print(f"{UI.BOLD}Load time:{UI.RESET} {self.load_time:.2f}s")
        print(f"{UI.BOLD}History file:{UI.RESET} {self.history_saver.session_file}")
        print(f"{UI.BOLD}GPU/shared memory:{UI.RESET} {human_bytes(current_memory)}\n")

    def generate(self, prompt: str) -> None:
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

        self.history_messages.append({"role": "user", "content": user_content})
        self.rebuild_history()

        if self.reasoning_enabled:
            print(f"\n{UI.BOLD}{UI.GREEN}AI {UI.MAGENTA}[thinking]{UI.RESET}: ", end="", flush=True)
        else:
            print(f"\n{UI.BOLD}{UI.GREEN}AI:{UI.RESET} ", end="", flush=True)

        generated_chunks: List[str] = []

        def streamer(text: str) -> bool:
            print(text, end="", flush=True)
            generated_chunks.append(text)
            return False

        generate_kwargs = {
            "generation_config": self.generation_config,
            "streamer": streamer,
            "extra_context": {"enable_thinking": self.reasoning_enabled},
        }
        if self.metadata.is_vlm and self.pending_image is not None:
            generate_kwargs["images"] = [self.pending_image]
            self.pending_image = None
            self.pending_image_path = None

        try:
            result = self.pipe.generate(self.chat_history, **generate_kwargs)
        except Exception as e:
            print(f"\n\n{UI.RED}Generation error:{UI.RESET} {e}\n")
            if self.history_messages:
                self.history_messages.pop()
            self.rebuild_history()
            return

        print("\n")

        assistant_text = "".join(generated_chunks)
        if not assistant_text:
            try:
                assistant_text = result.texts[0]
            except Exception:
                assistant_text = ""

        self.history_messages.append({"role": "assistant", "content": assistant_text})
        self.rebuild_history()

        metrics_dict = {}
        try:
            m = result.perf_metrics
            metrics_dict = {
                "input_tokens": m.get_num_input_tokens(),
                "output_tokens": m.get_num_generated_tokens(),
                "throughput": round(m.get_throughput().mean, 2),
                "ttft": round(m.get_ttft().mean, 1),
            }
            print(f"{UI.DIM}Input: {metrics_dict['input_tokens']} tok | Output: {metrics_dict['output_tokens']} tok | Speed: {metrics_dict['throughput']} tok/s | TTFT: {metrics_dict['ttft']} ms{UI.RESET}\n")
        except Exception:
            pass

        self.history_saver.add_turn(
            user_prompt=user_content,
            assistant_response=assistant_text,
            metrics=metrics_dict,
            reasoning_used=self.reasoning_enabled,
        )

    def run(self) -> None:
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

            if root_cmd in ("/file", "/doc", "/attach"):
                if len(parts) > 1:
                    raw_path = " ".join(parts[1:]).strip('\"\'')
                    doc_path = Path(raw_path)
                    if not doc_path.exists():
                        print(f"\n{UI.RED}File not found: {doc_path}{UI.RESET}\n")
                        continue
                    try:
                        text = extract_file_text(doc_path)
                        self.pending_file_context = text
                        self.pending_file_name = doc_path.name
                        print(f"\n{UI.GREEN}Attached file: {doc_path.name} ({len(text):,} chars). Type your prompt next.{UI.RESET}\n")
                    except Exception as ex:
                        print(f"\n{UI.RED}Failed to extract file: {ex}{UI.RESET}\n")
                else:
                    print(f"{UI.YELLOW}Usage: /file <path/to/document.pdf or file.txt>{UI.RESET}")
                continue

            if root_cmd in ("/image", "/img", "/pic"):
                if not self.metadata.is_vlm:
                    print(f"\n{UI.YELLOW}Current model '{self.metadata.name}' is an LLM (text-only). Images are only supported on VLMs.{UI.RESET}\n")
                    continue
                if len(parts) > 1:
                    raw_path = " ".join(parts[1:]).strip('\"\'')
                    img_path = Path(raw_path)
                    if not img_path.exists():
                        print(f"\n{UI.RED}File not found: {img_path}{UI.RESET}\n")
                        continue
                    try:
                        import numpy as np
                        from PIL import Image
                        pil_img = Image.open(img_path).convert("RGB")
                        self.pending_image = ov.Tensor(np.array(pil_img))
                        self.pending_image_path = str(img_path)
                        print(f"\n{UI.GREEN}Image loaded: {img_path.name} ({pil_img.width}x{pil_img.height}). Type your prompt next.{UI.RESET}\n")
                    except Exception as ex:
                        print(f"\n{UI.RED}Failed to load image: {ex}{UI.RESET}\n")
                else:
                    print(f"{UI.YELLOW}Usage: /image <path/to/image.jpg>{UI.RESET}")
                continue

            if root_cmd in ("/unload", "/eject"):
                self.unload()
                continue

            if root_cmd == "/load":
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
                    meta = ModelMetadata(target_model)
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

        self.model_path = model_path
        self.metadata = ModelMetadata(model_path)
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


# ============================================================
# CLI MAIN
# ============================================================

def main() -> None:
    clear_screen()
    print(f"{UI.BOLD}{UI.CYAN}OpenVINO GenAI Chat (Fallback Standalone){UI.RESET}\n")

    models = find_models(MODEL_ROOT)
    if not models:
        print(f"{UI.RED}No models found in {MODEL_ROOT}{UI.RESET}")
        return

    print("Available Models:")
    for i, m in enumerate(models, 1):
        meta = ModelMetadata(m)
        print(f"  {i}. {m.name} [{meta.display_type}, {meta.precision.upper()}]")
    print()

    sel = input("Select model [1]: ").strip()
    idx = int(sel) - 1 if sel.isdigit() and 1 <= int(sel) <= len(models) else 0
    model_path = models[idx]

    meta = ModelMetadata(model_path)
    device = choose_device(meta)
    if not device:
        return

    enable_reasoning = False
    if meta.supports_reasoning:
        val = input("Enable reasoning / thinking mode? [y/N]: ").strip().lower()
        enable_reasoning = val in ("y", "yes", "1", "true")

    chat = Chat(
        model_path=model_path,
        device=device,
        context_length=DEFAULT_CONTEXT,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        enable_reasoning=enable_reasoning,
        metadata=meta,
    )

    clear_screen()
    print(f"{UI.BOLD}{UI.GREEN}Model loaded.{UI.RESET}")
    print(f"Model: {model_path.name} ({meta.display_type})")
    print(f"Device: {device}\n")
    print(f"{UI.DIM}Commands: /clear, /info, /file <path>, {('/image <path>, ' if meta.is_vlm else '')}{('/think, ' if meta.supports_reasoning else '')}/quit{UI.RESET}\n")

    chat.run()
    chat.memory_monitor.stop()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print(f"\n{UI.YELLOW}Exited.{UI.RESET}")
        sys.exit(0)
