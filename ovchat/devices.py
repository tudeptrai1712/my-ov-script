from typing import List, Optional

import openvino as ov

from .ui import UI


def get_devices() -> List[str]:
    """Retrieve list of available OpenVINO hardware device names."""
    core = ov.Core()
    try:
        return list(core.available_devices)
    except Exception:
        return []


def choose_device() -> Optional[str]:
    """Interactively prompts the user to select an OpenVINO execution device."""
    devices = get_devices()

    if not devices:
        print(f"{UI.RED}No OpenVINO devices detected.{UI.RESET}")
        input("Press Enter...")
        return None

    print()
    print(f"{UI.BOLD}Available OpenVINO devices{UI.RESET}")

    for i, device in enumerate(devices, start=1):
        print(f"  {i}. {device}")

    print()

    while True:
        value = input(f"{UI.CYAN}Select device [1]: {UI.RESET}").strip()

        if not value:
            return devices[0]

        try:
            index = int(value) - 1
            if 0 <= index < len(devices):
                return devices[index]
        except ValueError:
            pass

        print(f"{UI.YELLOW}Invalid selection.{UI.RESET}")

