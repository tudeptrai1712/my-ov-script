"""
Hardware telemetry, device discovery, and static system specifications module.
"""

from .monitor import MemoryMonitor, estimate_memory, powershell_gpu_memory
from .specs import (
    GPUSpec,
    HardwareSpecs,
    NPUSpec,
    choose_device,
    format_hardware_specs_card,
    get_devices,
    get_hardware_specs,
    get_model_hardware_fit,
    print_hardware_specs,
    render_hardware_and_models,
)

__all__ = [
    "GPUSpec",
    "NPUSpec",
    "HardwareSpecs",
    "get_hardware_specs",
    "get_devices",
    "choose_device",
    "get_model_hardware_fit",
    "format_hardware_specs_card",
    "print_hardware_specs",
    "render_hardware_and_models",
    "MemoryMonitor",
    "powershell_gpu_memory",
    "estimate_memory",
]

