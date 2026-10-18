"""
Backward-compatibility bridge for ovchat.hardware.specs.
Direct all new imports to ovchat.hardware.
"""

from .hardware.specs import (
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
]
