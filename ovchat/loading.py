import threading
import time
from pathlib import Path
from typing import Optional

from .memory import MemoryMonitor, estimate_memory
from .ui import UI, clamp, clear_screen, human_bytes, progress_bar


class LoadingScreen:
    """Renders an animated loading terminal UI while OpenVINO model compiles."""

    def __init__(
        self,
        model_path: Path,
        device: str,
        context_length: int,
        memory_monitor: MemoryMonitor,
    ):
        self.model_path = model_path
        self.device = device
        self.context_length = context_length
        self.memory_monitor = memory_monitor

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
        print(f"{UI.BOLD}{UI.CYAN}Loading OpenVINO model{UI.RESET}\n")

        print(f"Model       : {self.model_path.name}")
        print(f"Device      : {self.device}")
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
            "OpenVINO does not expose a byte-by-byte model "
            "loading progress callback."
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

