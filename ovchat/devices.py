from typing import List, Optional, Tuple

import openvino as ov

from .metadata import ModelMetadata, check_device_compatibility
from .ui import UI


def get_devices(metadata: Optional[ModelMetadata] = None) -> List[str]:
    """
    Retrieve list of available OpenVINO hardware device names.
    If metadata is provided, filters out any devices incompatible with the model.
    """
    core = ov.Core()
    try:
        raw_devices = list(core.available_devices)
    except Exception:
        return []

    if metadata is None:
        return raw_devices

    compatible: List[str] = []
    for dev in raw_devices:
        is_ok, _ = check_device_compatibility(dev, metadata, core)
        if is_ok:
            compatible.append(dev)

    return compatible


def choose_device(metadata: Optional[ModelMetadata] = None) -> Optional[str]:
    """
    Interactively prompts the user to select an OpenVINO execution device.
    Hides any hardware devices that cannot run the selected model based on model metadata.
    """
    core = ov.Core()
    try:
        raw_devices = list(core.available_devices)
    except Exception:
        raw_devices = []

    if not raw_devices:
        print(f"{UI.RED}No OpenVINO devices detected.{UI.RESET}")
        input("Press Enter...")
        return None

    # Evaluate compatibility for each detected device
    compatible_devices: List[str] = []
    hidden_devices: List[Tuple[str, str]] = []

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
        print(
            f"{UI.RED}No compatible OpenVINO devices found for {metadata.name if metadata else 'model'}.{UI.RESET}"
        )
        if hidden_devices:
            for dev, reason in hidden_devices:
                print(f"  - {dev}: {reason}")
        input("\nPress Enter...")
        return None

    print()
    if metadata is not None:
        print(f"{UI.BOLD}Compatible OpenVINO devices for {metadata.name}{UI.RESET}")
    else:
        print(f"{UI.BOLD}Available OpenVINO devices{UI.RESET}")

    for i, device in enumerate(compatible_devices, start=1):
        print(f"  {i}. {device}")

    # Show hidden options informatively so user knows why a device was omitted
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
            index = int(value) - 1
            if 0 <= index < len(compatible_devices):
                return compatible_devices[index]
        except ValueError:
            pass

        print(f"{UI.YELLOW}Invalid selection.{UI.RESET}")
