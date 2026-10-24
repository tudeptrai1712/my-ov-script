import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openvino as ov
import psutil

from ..config import DEFAULT_HETERO_PRIORITY, DEFAULT_MOE_OFFLOAD_RATIO
from ..metadata import ModelMetadata, check_device_compatibility, read_model_metadata
from ..ui import UI, human_bytes, pad_ansi, visible_len


@dataclass
class GPUSpec:
    """Detailed specifications for an OpenVINO GPU device."""

    device_id: str
    name: str
    total_memory: int = 0
    execution_units: Optional[int] = None
    architecture: Optional[str] = None
    device_type: str = "GPU"


@dataclass
class NPUSpec:
    """Detailed specifications for an OpenVINO NPU device."""

    device_id: str
    name: str
    architecture: Optional[str] = None
    driver_version: Optional[str] = None
    is_available: bool = False
    status_note: str = "Compiler temporarily disabled in current GenAI release"


@dataclass
class HardwareSpecs:
    """Complete system hardware specification profile for local OpenVINO inference."""

    cpu_name: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_total_bytes: int
    ram_available_bytes: int
    ram_used_pct: float
    gpus: List[GPUSpec]
    npus: List[NPUSpec]
    openvino_version: str


_CACHED_HARDWARE_SPECS: Optional[HardwareSpecs] = None


def get_hardware_specs(cached: bool = True) -> HardwareSpecs:
    """
    Retrieves and caches detailed system hardware specifications
    including CPU, RAM, Intel Arc GPU(s), NPU, and OpenVINO runtime version.
    """
    global _CACHED_HARDWARE_SPECS
    if cached and _CACHED_HARDWARE_SPECS is not None:
        return _CACHED_HARDWARE_SPECS

    core = ov.Core()

    # 1. CPU name & cores
    try:
        cpu_name = core.get_property("CPU", "FULL_DEVICE_NAME")
    except Exception:
        cpu_name = platform.processor() or "x86_64 CPU"

    try:
        cpu_p = psutil.cpu_count(logical=False) or 1
        cpu_l = psutil.cpu_count(logical=True) or 1
    except Exception:
        cpu_p = os.cpu_count() or 1
        cpu_l = os.cpu_count() or 1

    # 2. System RAM
    try:
        mem = psutil.virtual_memory()
        ram_total = mem.total
        ram_avail = mem.available
        ram_pct = mem.percent
    except Exception:
        ram_total = 0
        ram_avail = 0
        ram_pct = 0.0

    # 3. OpenVINO available devices
    try:
        raw_devices = list(core.available_devices)
    except Exception:
        raw_devices = []

    # 4. GPUs
    gpus: List[GPUSpec] = []
    for dev in raw_devices:
        if dev.upper().startswith("GPU"):
            try:
                name = core.get_property(dev, "FULL_DEVICE_NAME")
            except Exception:
                name = f"Intel Graphics ({dev})"
            try:
                tot = core.get_property(dev, "GPU_DEVICE_TOTAL_MEM_SIZE")
            except Exception:
                tot = 0
            try:
                eus = core.get_property(dev, "GPU_EXECUTION_UNITS_COUNT")
            except Exception:
                eus = None
            try:
                arch = str(core.get_property(dev, "DEVICE_ARCHITECTURE"))
            except Exception:
                arch = None
            gpus.append(
                GPUSpec(
                    device_id=dev,
                    name=name,
                    total_memory=tot,
                    execution_units=eus,
                    architecture=arch,
                )
            )

    # 5. NPUs
    npus: List[NPUSpec] = []
    for dev in raw_devices:
        if dev.upper().startswith("NPU"):
            try:
                name = core.get_property(dev, "FULL_DEVICE_NAME")
            except Exception:
                name = f"Intel NPU ({dev})"
            try:
                arch = str(core.get_property(dev, "DEVICE_ARCHITECTURE"))
            except Exception:
                arch = None
            try:
                driver = str(core.get_property(dev, "NPU_DRIVER_VERSION"))
            except Exception:
                driver = None
            npus.append(
                NPUSpec(
                    device_id=dev,
                    name=name,
                    architecture=arch,
                    driver_version=driver,
                    is_available=False,
                )
            )

    # 6. OpenVINO runtime version
    try:
        ov_ver = ov.__version__.split("-")[0]
    except Exception:
        ov_ver = "Unknown"

    specs = HardwareSpecs(
        cpu_name=cpu_name,
        cpu_cores_physical=cpu_p,
        cpu_cores_logical=cpu_l,
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_avail,
        ram_used_pct=ram_pct,
        gpus=gpus,
        npus=npus,
        openvino_version=ov_ver,
    )

    _CACHED_HARDWARE_SPECS = specs
    return specs


def get_model_hardware_fit(
    metadata: ModelMetadata,
    specs: Optional[HardwareSpecs] = None,
    device: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Evaluates whether a model fits in GPU memory (with KV cache headroom,
    taking MoE offloading and Heterogeneous pipeline execution into account).
    Returns (fit_status_colored, compatible_devices_str).
    """
    if specs is None:
        specs = get_hardware_specs()

    has_gpu = bool(specs.gpus)
    comp_devices = "GPU, HETERO:GPU,CPU, CPU" if has_gpu else "CPU"
    model_bytes = metadata.weight_size_bytes
    primary_gpu_mem = specs.gpus[0].total_memory if specs.gpus else 0
    ram_avail = specs.ram_available_bytes
    combined_mem = primary_gpu_mem + ram_avail

    # Specific device check if passed
    dev_check = (device or "").upper()
    if dev_check.startswith("HETERO"):
        if model_bytes <= combined_mem:
            fit_status = f"{UI.GREEN}Fits in HETERO Pipeline (GPU + CPU RAM){UI.RESET}"
        else:
            fit_status = f"{UI.YELLOW}Exceeds combined HETERO capacity{UI.RESET}"
        return fit_status, comp_devices

    if primary_gpu_mem > 0:
        if model_bytes <= primary_gpu_mem:
            headroom = primary_gpu_mem - model_bytes
            fit_status = f"{UI.GREEN}Fits in GPU (+{human_bytes(headroom)} KV){UI.RESET}"
        elif metadata.is_moe:
            # MoE offloading partial weights to disk/RAM
            # For MoE models, active parameter weights represent ~25-40% of total weights
            offload_ratio = DEFAULT_MOE_OFFLOAD_RATIO
            est_resident = int(model_bytes * (1.0 - (offload_ratio / 100.0) * 0.70))
            if est_resident <= primary_gpu_mem:
                fit_status = f"{UI.GREEN}Fits in GPU (MoE Offload {offload_ratio}% to disk){UI.RESET}"
            elif model_bytes <= combined_mem:
                fit_status = f"{UI.CYAN}Fits via HETERO:GPU,CPU (Pipeline Parallelism){UI.RESET}"
            else:
                overflow = model_bytes - primary_gpu_mem
                fit_status = f"{UI.YELLOW}Exceeds GPU by {human_bytes(overflow)} (spills to RAM){UI.RESET}"
        else:
            overflow = model_bytes - primary_gpu_mem
            if model_bytes <= combined_mem:
                fit_status = f"{UI.CYAN}Fits via HETERO:GPU,CPU (Pipeline Parallelism){UI.RESET}"
            else:
                fit_status = f"{UI.YELLOW}Exceeds GPU by {human_bytes(overflow)} (spills to RAM){UI.RESET}"
    else:
        if ram_avail > 0 and model_bytes <= ram_avail:
            fit_status = f"{UI.GREEN}Fits in RAM ({human_bytes(ram_avail)} free){UI.RESET}"
        else:
            fit_status = f"{UI.YELLOW}High RAM usage{UI.RESET}"

    return fit_status, comp_devices


def format_hardware_specs_card(
    specs: Optional[HardwareSpecs] = None,
    width: int = 78,
) -> List[str]:
    """Formats full hardware specifications into an ASCII bordered card."""
    if specs is None:
        specs = get_hardware_specs()

    lines = []
    top_title = " System Hardware Specifications "
    dash_count = max(0, width - 2 - len(top_title))
    d1 = dash_count // 2
    d2 = dash_count - d1
    lines.append(f"{UI.BOLD}+{'-' * d1}{top_title}{'-' * d2}+{UI.RESET}")

    lines.append(
        f"{UI.BOLD}|{UI.RESET}  {UI.CYAN}CPU{UI.RESET} : {specs.cpu_name} "
        f"({specs.cpu_cores_physical} physical / {specs.cpu_cores_logical} logical cores)"
    )
    lines.append(
        f"{UI.BOLD}|{UI.RESET}  {UI.CYAN}RAM{UI.RESET} : {human_bytes(specs.ram_total_bytes)} total | "
        f"{UI.GREEN}{human_bytes(specs.ram_available_bytes)} available{UI.RESET} "
        f"({specs.ram_used_pct:.1f}% used)"
    )

    if specs.gpus:
        for g in specs.gpus:
            eu_s = f" [{g.execution_units} EUs]" if g.execution_units else ""
            lines.append(
                f"{UI.BOLD}|{UI.RESET}  {UI.CYAN}GPU{UI.RESET} : {g.name}{eu_s} | "
                f"{human_bytes(g.total_memory)} memory"
            )
    else:
        lines.append(f"{UI.BOLD}|{UI.RESET}  {UI.CYAN}GPU{UI.RESET} : No OpenVINO-compatible GPU detected")

    if specs.npus:
        for n in specs.npus:
            arch_s = f" [Arch {n.architecture}]" if n.architecture else ""
            lines.append(
                f"{UI.BOLD}|{UI.RESET}  {UI.CYAN}NPU{UI.RESET} : {n.name}{arch_s} "
                f"{UI.DIM}(Temporarily disabled in GenAI release){UI.RESET}"
            )
    else:
        lines.append(f"{UI.BOLD}|{UI.RESET}  {UI.CYAN}NPU{UI.RESET} : No OpenVINO-compatible NPU detected")

    # Hardware acceleration modes note
    modes = "GPU (Arc)"
    if specs.gpus:
        modes += " | HETERO:GPU,CPU (Pipeline Parallelism) | MoE Disk Offload"
    lines.append(f"{UI.BOLD}|{UI.RESET}  {UI.GREEN}Acc : {modes}{UI.RESET}")
    lines.append(f"{UI.BOLD}|{UI.RESET}  {UI.DIM}OV  : OpenVINO {specs.openvino_version}{UI.RESET}")
    lines.append(f"{UI.BOLD}+{'-' * (width - 2)}+{UI.RESET}")
    return lines


def format_hardware_specs_box(specs: HardwareSpecs, width: int = 40) -> List[str]:
    """Formats a compact hardware specifications box for side-by-side terminal views."""
    inner_w = width - 4
    lines = []
    title = " Hardware Specs "
    dashes = max(0, width - 2 - len(title))
    d1 = dashes // 2
    d2 = dashes - d1
    lines.append(f"+{'-' * d1}{title}{'-' * d2}+")

    cpu_short = specs.cpu_name
    if visible_len(cpu_short) > inner_w - 6:
        cpu_short = cpu_short[: inner_w - 8] + ".."
    lines.append(f"| {pad_ansi(f'{UI.CYAN}CPU{UI.RESET}: {cpu_short}', inner_w)} |")
    core_info = f"     {specs.cpu_cores_physical} phys / {specs.cpu_cores_logical} log cores"
    lines.append(f"| {pad_ansi(f'{UI.DIM}{core_info}{UI.RESET}', inner_w)} |")

    ram_str = f"{human_bytes(specs.ram_total_bytes)} ({human_bytes(specs.ram_available_bytes)} free)"
    lines.append(f"| {pad_ansi(f'{UI.CYAN}RAM{UI.RESET}: {ram_str}', inner_w)} |")

    if specs.gpus:
        for g in specs.gpus:
            g_name = g.name
            if visible_len(g_name) > inner_w - 6:
                g_name = g_name[: inner_w - 8] + ".."
            lines.append(f"| {pad_ansi(f'{UI.CYAN}GPU{UI.RESET}: {g_name}', inner_w)} |")
            eu_part = f"{g.execution_units} EUs | " if g.execution_units else ""
            g_detail = f"     {eu_part}{human_bytes(g.total_memory)} Mem"
            lines.append(f"| {pad_ansi(f'{UI.DIM}{g_detail}{UI.RESET}', inner_w)} |")

    if specs.npus:
        for n in specs.npus:
            arch_s = f" [{n.architecture}]" if n.architecture else ""
            n_str = f"{n.name}{arch_s}"
            if visible_len(n_str) > inner_w - 6:
                n_str = n_str[: inner_w - 8] + ".."
            lines.append(f"| {pad_ansi(f'{UI.CYAN}NPU{UI.RESET}: {n_str}', inner_w)} |")
            lines.append(f"| {pad_ansi(f'{UI.DIM}     (Disabled in GenAI){UI.RESET}', inner_w)} |")

    lines.append(f"| {pad_ansi(f'{UI.DIM}HETERO: Pipeline Parallelism Active{UI.RESET}', inner_w)} |")
    lines.append(f"| {pad_ansi(f'{UI.DIM}OV: OpenVINO {specs.openvino_version}{UI.RESET}', inner_w)} |")
    lines.append(f"+{'-' * (width - 2)}+")
    return lines


def format_models_box(
    model_items: List[Tuple[Path, ModelMetadata, str, str]],
    root_str: str,
    width: int = 65,
) -> List[str]:
    """Formats available models into a bordered box for side-by-side terminal views."""
    inner_w = width - 4
    lines = []
    title = " Available Models "
    dashes = max(0, width - 2 - len(title))
    d1 = dashes // 2
    d2 = dashes - d1
    lines.append(f"+{'-' * d1}{title}{'-' * d2}+")

    for i, (m, meta, fit_status, comp_devices) in enumerate(model_items, start=1):
        m_name = m.name
        size_s = human_bytes(meta.weight_size_bytes)
        moe_tag = f" {UI.MAGENTA}[{meta.moe_summary}]{UI.RESET}" if meta.is_moe else ""
        line1 = f"{UI.BOLD}{i}. {m_name}{UI.RESET}{moe_tag} [{size_s}]"
        lines.append(f"| {pad_ansi(line1, inner_w)} |")

        t_short = "VLM" if meta.is_vlm else "LLM"
        line2 = f"   {UI.DIM}Type:{UI.RESET} {t_short} ({meta.precision.upper()}) | {UI.DIM}Fit:{UI.RESET} {fit_status}"
        lines.append(f"| {pad_ansi(line2, inner_w)} |")

    lines.append(f"+{'-' * (width - 2)}+")
    return lines


def print_hardware_specs(
    specs: Optional[HardwareSpecs] = None,
    width: int = 78,
) -> None:
    """Prints the system hardware specifications card to the console."""
    card_lines = format_hardware_specs_card(specs, width=width)
    for line in card_lines:
        print(line)


def render_hardware_and_models(
    models: List[Path],
    model_root: Optional[Path] = None,
    specs: Optional[HardwareSpecs] = None,
) -> None:
    """
    Renders system hardware specifications alongside available models.
    Automatically uses a side-by-side two-column layout on wide terminals (>= 108 columns),
    or a stacked layout on standard terminals.
    """
    if specs is None:
        specs = get_hardware_specs()

    cols, _ = shutil.get_terminal_size((80, 24))
    root_str = str(model_root) if model_root else "OpenVINO Model Directory"

    model_items = []
    for m in models:
        meta = read_model_metadata(m)
        fit_status, comp_devices = get_model_hardware_fit(meta, specs)
        model_items.append((m, meta, fit_status, comp_devices))

    can_side_by_side = cols >= 108 and len(models) <= 8

    if can_side_by_side:
        col1_w = 40
        col2_w = min(cols - col1_w - 4, 75)
        left_lines = format_hardware_specs_box(specs, width=col1_w)
        right_lines = format_models_box(model_items, root_str, width=col2_w)
        max_rows = max(len(left_lines), len(right_lines))
        for r in range(max_rows):
            left_txt = left_lines[r] if r < len(left_lines) else " " * col1_w
            right_txt = right_lines[r] if r < len(right_lines) else ""
            print(f"{left_txt}  {right_txt}")
        print()
    else:
        box_width = min(cols - 2, 78)
        print_hardware_specs(specs, width=box_width)
        print()
        print(f"{UI.BOLD}{UI.CYAN}Available Models:{UI.RESET} {UI.DIM}({root_str}){UI.RESET}\n")
        for i, (m, meta, fit_status, comp_devices) in enumerate(model_items, start=1):
            size_str = human_bytes(meta.weight_size_bytes)
            moe_tag = f" {UI.MAGENTA}[{meta.moe_summary}]{UI.RESET}" if meta.is_moe else ""
            print(f"  {UI.BOLD}{i}. {m.name}{UI.RESET}{moe_tag}  {UI.YELLOW}[{size_str}]{UI.RESET}")
            print(
                f"     {UI.DIM}Type:{UI.RESET} {meta.display_type} | "
                f"{UI.DIM}Precision:{UI.RESET} {meta.precision.upper()}"
            )
            print(
                f"     {UI.DIM}Fit :{UI.RESET} {fit_status} | "
                f"{UI.DIM}Compatible:{UI.RESET} {comp_devices}\n"
            )


def get_devices(metadata: Optional[ModelMetadata] = None) -> List[str]:
    """
    Retrieve list of available OpenVINO hardware device names.
    Supports individual devices (GPU, CPU) and Heterogeneous Pipeline Parallelism targets.
    """
    core = ov.Core()
    try:
        raw_devices = list(core.available_devices)
    except Exception:
        raw_devices = []

    devices: List[str] = []

    # 1. Primary accelerators (GPU, CPU, etc.)
    has_gpu = any(d.upper().startswith("GPU") for d in raw_devices)
    has_cpu = "CPU" in [d.upper() for d in raw_devices]

    if has_gpu:
        devices.append("GPU")
    if has_gpu and has_cpu:
        devices.append(f"HETERO:{DEFAULT_HETERO_PRIORITY}")
    if has_cpu:
        devices.append("CPU")

    # Multi-GPU setups (e.g. GPU.0, GPU.1)
    gpus = [d for d in raw_devices if d.upper().startswith("GPU.")]
    if len(gpus) > 1:
        devices.append(f"HETERO:{','.join(gpus)},CPU")

    # Include any remaining raw devices not already represented
    for d in raw_devices:
        if d.upper() not in ("GPU", "CPU") and d not in devices:
            devices.append(d)

    if metadata is None:
        return devices

    compatible: List[str] = []
    for dev in devices:
        is_ok, _ = check_device_compatibility(dev, metadata, core)
        if is_ok:
            compatible.append(dev)

    return compatible


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
