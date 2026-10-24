import threading
import time
from pathlib import Path
from typing import Optional

from .hardware.monitor import MemoryMonitor, estimate_memory
from .metadata import ModelMetadata, read_model_metadata
from .ui import UI, clamp, clear_screen, human_bytes, progress_bar


class LoadingScreen:
    """Renders an animated loading terminal UI while OpenVINO model compiles."""

    def __init__(
        self,
        model_path: Path,
        device: str,
        context_length: int,
        memory_monitor: MemoryMonitor,
        metadata: Optional[ModelMetadata] = None,
        enable_moe_offload: bool = True,
        moe_offload_ratio: int = 50,
        model_distribution_policy: str = "PIPELINE_PARALLEL",
    ):
        self.model_path = model_path
        self.device = device
        self.context_length = context_length
        self.memory_monitor = memory_monitor
        self.metadata = metadata or read_model_metadata(model_path)
        self.enable_moe_offload = enable_moe_offload
        self.moe_offload_ratio = moe_offload_ratio
        self.model_distribution_policy = model_distribution_policy

        self.baseline_memory = memory_monitor.current
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.estimate = estimate_memory(model_path, context_length)

    def render(self) -> None:
        current = self.memory_monitor.current
        baseline = self.baseline_memory

        if current is not None and baseline is not None:
            increase = max(0.0, current - baseline)
            estimated_total = self.estimate["total"]
            fraction = increase / estimated_total if estimated_total > 0 else 0.0
            fraction = float(clamp(fraction, 0.0, 0.99))
        else:
            increase = None
            fraction = 0.0

        print("\033[2J\033[H", end="")
        print(f"{UI.BOLD}{UI.CYAN}Compiling OpenVINO GenAI Pipeline{UI.RESET}\n")

        print(f"Model       : {self.model_path.name}")
        if self.device.upper().startswith("HETERO"):
            print(f"Device      : {UI.GREEN}{self.device}{UI.RESET} {UI.DIM}(Pipeline Parallelism Preview){UI.RESET}")
            print(f"Policy      : {self.model_distribution_policy}")
        else:
            print(f"Device      : {self.device}")

        if self.metadata.is_moe:
            moe_desc = self.metadata.moe_summary or "Mixture of Experts"
            if self.enable_moe_offload and "GPU" in self.device.upper():
                offload_str = f"{UI.GREEN}Enabled ({self.moe_offload_ratio}% offload to disk){UI.RESET}"
            else:
                offload_str = f"{UI.YELLOW}Disabled (full weights resident){UI.RESET}"
            print(f"MoE Status  : {moe_desc}")
            print(f"MoE Offload : {offload_str}")

        print(f"Context     : {self.context_length:,}\n")

        print(f"{UI.BOLD}Estimated memory{UI.RESET}")
        print(f"  Weights   : {human_bytes(self.estimate['weights'])}")
        print(f"  KV cache  : {human_bytes(self.estimate['kv'])}")
        print(f"  Runtime   : {human_bytes(self.estimate['runtime'])}")
        print(f"  Total     : {human_bytes(self.estimate['total'])}\n")

        print(f"{UI.BOLD}GPU/shared memory{UI.RESET}")
        print(f"  Before    : {human_bytes(baseline)}")
        print(f"  Current   : {human_bytes(current)}")
        print(f"  Increase  : {human_bytes(increase)}\n")

        print(
            progress_bar(fraction, width=45),
            f"{fraction * 100:.1f}%\n",
        )

        print(
            f"{UI.YELLOW}"
            "Loading progress is an estimate based on observed "
            "GPU/shared-memory growth."
            f"{UI.RESET}"
        )
        print(
            f"{UI.DIM}"
            "OpenVINO compiles execution graphs and initializes device buffers."
            f"{UI.RESET}"
        )

    def loop(self) -> None:
        while self.running:
            self.render()
            time.sleep(0.25)

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        clear_screen()
