"""
Backward-compatibility bridge for ovchat.hardware.monitor.
Direct all new imports to ovchat.hardware.
"""

from .hardware.monitor import MemoryMonitor, estimate_memory, powershell_gpu_memory

__all__ = [
    "MemoryMonitor",
    "powershell_gpu_memory",
    "estimate_memory",
]
