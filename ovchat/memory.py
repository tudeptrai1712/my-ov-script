import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from .config import (
    DEFAULT_KV_BYTES_PER_TOKEN,
    MEMORY_POLL_INTERVAL,
    RUNTIME_OVERHEAD_GB,
)
from .models import model_size


def powershell_gpu_memory() -> Optional[float]:
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
                ps_script,
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )

        output = result.stdout.strip()
        if not output:
            return None

        return float(output)

    except Exception:
        return None


class MemoryMonitor:
    """Background thread that polls GPU / shared memory at regular intervals."""

    def __init__(self, interval: float = MEMORY_POLL_INTERVAL):
        self.interval = interval
        self.current: Optional[float] = None
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None

    def poll(self) -> None:
        while self.running:
            value = powershell_gpu_memory()
            if value is not None:
                self.current = value
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


def estimate_memory(model_path: Path, context_length: int) -> Dict[str, int]:
    """
    Rough memory estimate breakdown (weights, KV cache, runtime overhead, total).

    Model weights:
        actual .bin size
    KV:
        intentionally approximate
    Runtime:
        generic overhead estimate

    This is NOT a measurement of OpenVINO's actual memory allocation.
    """
    weights = model_size(model_path)
    kv = context_length * DEFAULT_KV_BYTES_PER_TOKEN
    runtime = int(RUNTIME_OVERHEAD_GB * 1024 ** 3)
    total = weights + kv + runtime

    return {
        "weights": weights,
        "kv": kv,
        "runtime": runtime,
        "total": total,
    }

