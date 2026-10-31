"""
Interactive terminal prompt helpers for OpenVINO GenAI Chat.
Decouples terminal user interaction from low-level hardware probing and model discovery.
"""

from pathlib import Path
from typing import List, Optional

from .config import (
    DEFAULT_CONTEXT,
    DEFAULT_ENABLE_MOE_OFFLOAD,
    DEFAULT_ENABLE_REASONING,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_DISTRIBUTION_POLICY,
    DEFAULT_MOE_OFFLOAD_RATIO,
    MODEL_ROOT,
)
from .hardware.specs import (
    get_devices,
    get_hardware_specs,
    print_hardware_specs,
    render_hardware_and_models,
)
from .metadata import ModelMetadata, read_model_metadata
from .models import find_models
from .ui import UI, human_bytes


def choose_integer(
    label: str,
    default: int,
    minimum: int = 1,
    maximum: Optional[int] = None,
) -> int:
    """Interactively prompts for an integer value with validation against min and optional max."""
    while True:
        value = input(f"{UI.CYAN}{label} [{default}]: {UI.RESET}").strip()
        if not value:
            return default
        try:
            number = int(value)
            if number < minimum:
                print(f"{UI.YELLOW}Enter an integer >= {minimum}.{UI.RESET}")
                continue
            if maximum is not None and number > maximum:
                print(f"{UI.YELLOW}Enter an integer <= {maximum:,}.{UI.RESET}")
                continue
            return number
        except ValueError:
            pass
        print(f"{UI.YELLOW}Invalid number.{UI.RESET}")


def choose_boolean(label: str, default: bool = False) -> bool:
    """Interactively prompts for a boolean yes/no choice."""
    prompt_suffix = "[Y/n]" if default else "[y/N]"
    while True:
        value = input(f"{UI.CYAN}{label} {prompt_suffix}: {UI.RESET}").strip().lower()
        if not value:
            return default
        if value in ("y", "yes", "true", "1", "on"):
            return True
        if value in ("n", "no", "false", "0", "off"):
            return False
        print(f"{UI.YELLOW}Please enter 'y' or 'n'.{UI.RESET}")


def choose_model(model_root: Optional[Path] = None) -> Optional[Path]:
    """
    Interactively prompts the user to select an OpenVINO model.
    Displays detailed system hardware specs alongside the model list and their hardware fit.
    """
    root = model_root or MODEL_ROOT
    models = find_models(root)

    if not models:
        print(f"{UI.RED}No OpenVINO models found.{UI.RESET}\n")
        print("Expected model directory:")
        print(root)
        print("\nEach model should contain .xml and .bin files.")
        input("\nPress Enter...")
        return None

    print()
    render_hardware_and_models(models, root)

    while True:
        value = input(f"{UI.CYAN}Select model [1]: {UI.RESET}").strip()
        if not value:
            return models[0]
        try:
            index = int(value) - 1
            if 0 <= index < len(models):
                return models[index]
        except ValueError:
            pass
        print(f"{UI.YELLOW}Invalid selection.{UI.RESET}")


def choose_device(metadata: Optional[ModelMetadata] = None) -> Optional[str]:
    """
    Interactively prompts the user to select an OpenVINO execution device.
    Displays detailed hardware specs next to each compatible device.
    Includes Heterogeneous Pipeline Parallelism (GPU + CPU) and Intel Arc GPU acceleration.
    """
    specs = get_hardware_specs()
    compatible_devices = get_devices(metadata)

    if not compatible_devices:
        print(
            f"{UI.RED}No compatible OpenVINO devices found for {metadata.name if metadata else 'model'}.{UI.RESET}"
        )
        input("\nPress Enter...")
        return None

    # Priority sorting: GPU -> HETERO -> CPU -> NPU
    def dev_priority(d: str) -> int:
        u = d.upper()
        if u == "GPU" or u.startswith("GPU."):
            return 0
        if u.startswith("HETERO"):
            return 1
        if u.startswith("CPU"):
            return 2
        return 99

    compatible_devices.sort(key=dev_priority)

    print()
    if metadata is not None:
        moe_note = f" [{metadata.moe_summary}]" if metadata.is_moe else ""
        print(f"{UI.BOLD}Execution devices for {metadata.name}{moe_note}:{UI.RESET}")
    else:
        print(f"{UI.BOLD}Available OpenVINO execution devices:{UI.RESET}")

    for i, device in enumerate(compatible_devices, start=1):
        dev_upper = device.upper()
        detail = ""
        if dev_upper.startswith("HETERO"):
            detail = f" (Pipeline Parallelism Preview: stages split across {device.split(':', 1)[-1]})"
        elif dev_upper.startswith("GPU") and specs.gpus:
            g = specs.gpus[0]
            eu_s = f", {g.execution_units} EUs" if g.execution_units else ""
            moe_hint = " - MoE Disk Offload enabled" if metadata and metadata.is_moe else ""
            detail = f" ({g.name} - {human_bytes(g.total_memory)} memory{eu_s}{moe_hint})"
        elif dev_upper.startswith("CPU"):
            detail = f" ({specs.cpu_name} - {specs.cpu_cores_physical}C/{specs.cpu_cores_logical}T)"
        elif dev_upper.startswith("NPU") and specs.npus:
            n = specs.npus[0]
            arch_s = f" - Arch {n.architecture}" if n.architecture else ""
            detail = f" ({n.name}{arch_s})"

        print(f"  {i}. {device}{UI.DIM}{detail}{UI.RESET}")

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


def prompt_change_parameters() -> None:
    """Interactively prompts user to inspect or modify model load parameters before server launch."""
    from .settings import load_settings, save_settings

    settings = load_settings()

    dev = settings.get("selected_device", "GPU")
    ctx = settings.get("context_length", DEFAULT_CONTEXT)
    max_tok = settings.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
    temp = settings.get("temperature", 0.7)
    top_p = settings.get("top_p", 0.95)
    reasoning = settings.get("enable_reasoning", DEFAULT_ENABLE_REASONING)
    moe_offload = settings.get("enable_moe_offload", DEFAULT_ENABLE_MOE_OFFLOAD)
    moe_ratio = settings.get("moe_offload_ratio", DEFAULT_MOE_OFFLOAD_RATIO)
    policy = settings.get("model_distribution_policy", DEFAULT_MODEL_DISTRIBUTION_POLICY)

    try:
        print()
        print_hardware_specs()
    except Exception:
        pass

    print("\n" + "=" * 70)
    print("  Active Server Model Load Parameters:")
    print(f"  [1] Target Device       : {dev}")
    if dev.upper().startswith("HETERO"):
        print(f"      Distribution Policy : {policy} (Pipeline Parallelism Preview)")
    print(f"  [2] Context Length      : {ctx}")
    print(f"  [3] Max Output Tokens   : {max_tok}")
    print(f"  [4] Temperature         : {temp}")
    print(f"  [5] Top-P               : {top_p}")
    print(f"  [6] Reasoning / Thinking: {'Enabled' if reasoning else 'Disabled'}")
    print(f"  [7] MoE Disk Offload    : {'Enabled (' + str(moe_ratio) + '% to disk)' if moe_offload else 'Disabled'}")
    print("=" * 70)

    try:
        choice = input("\nDo you want to change any load parameters before launching? (y/N): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return

    if choice == "y":
        print("\n--- Change Parameters (Press Enter on any setting to keep current value) ---")
        print(
            f"1. Target Device [1=GPU, 2=HETERO:GPU,CPU (Pipeline Parallelism), 3=CPU, 4=AUTO] (Current: {dev}): ",
            end="",
            flush=True,
        )
        try:
            dev_in = input().strip()
            if dev_in == "1":
                settings["selected_device"] = "GPU"
            elif dev_in == "2":
                settings["selected_device"] = "HETERO:GPU,CPU"
            elif dev_in == "3":
                settings["selected_device"] = "CPU"
            elif dev_in == "4":
                settings["selected_device"] = "AUTO"
            elif dev_in:
                settings["selected_device"] = dev_in

            ctx_in = input(f"2. Context Length (Current: {ctx}): ").strip()
            if ctx_in.isdigit() and int(ctx_in) > 0:
                settings["context_length"] = int(ctx_in)

            max_in = input(f"3. Max Output Tokens (Current: {max_tok}): ").strip()
            if max_in.isdigit() and int(max_in) > 0:
                settings["max_new_tokens"] = int(max_in)

            temp_in = input(f"4. Temperature (Current: {temp}): ").strip()
            if temp_in:
                try:
                    settings["temperature"] = float(temp_in)
                except ValueError:
                    pass

            top_p_in = input(f"5. Top-P (Current: {top_p}): ").strip()
            if top_p_in:
                try:
                    settings["top_p"] = float(top_p_in)
                except ValueError:
                    pass

            reason_in = input(f"6. Reasoning Mode [y/N] (Current: {'y' if reasoning else 'n'}): ").strip().lower()
            if reason_in in ("y", "yes", "true", "1"):
                settings["enable_reasoning"] = True
            elif reason_in in ("n", "no", "false", "0"):
                settings["enable_reasoning"] = False

            moe_in = input(f"7. MoE Disk Offload [Y/n] (Current: {'y' if moe_offload else 'n'}): ").strip().lower()
            if moe_in in ("y", "yes", "true", "1"):
                settings["enable_moe_offload"] = True
                ratio_in = input(f"   MoE Offload Ratio % [1-99] (Current: {moe_ratio}%): ").strip()
                if ratio_in.isdigit():
                    settings["moe_offload_ratio"] = max(1, min(99, int(ratio_in)))
            elif moe_in in ("n", "no", "false", "0"):
                settings["enable_moe_offload"] = False

            save_settings(settings)
            print(f"\n{UI.GREEN}[OK] Updated settings saved to user_config.json{UI.RESET}\n")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted parameter changes.")

