import os
from typing import Optional, Union


class UI:
    """ANSI terminal styling and color codes."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


def clear_screen() -> None:
    """Clears the console screen."""
    os.system("cls" if os.name == "nt" else "clear")


def human_bytes(value: Optional[Union[int, float]]) -> str:
    """Formats a byte count into a human-readable string (B, KB, MB, GB, TB, PB)."""
    if value is None:
        return "N/A"

    val = float(value)
    units = ["B", "KB", "MB", "GB", "TB"]

    for unit in units:
        if val < 1024:
            return f"{val:.2f} {unit}"
        val /= 1024

    return f"{val:.2f} PB"


def format_gb(value: Optional[Union[int, float]]) -> str:
    """Formats a byte count into gigabytes (GB)."""
    if value is None:
        return "N/A"

    return f"{value / (1024 ** 3):.2f} GB"


def clamp(value: Union[int, float], low: Union[int, float], high: Union[int, float]) -> Union[int, float]:
    """Clamps a numeric value between low and high bounds."""
    return max(low, min(high, value))


def progress_bar(fraction: float, width: int = 40) -> str:
    """Generates an ASCII/Unicode progress bar string."""
    clamped_fraction = clamp(fraction, 0.0, 1.0)
    filled = int(clamped_fraction * width)
    empty = width - filled
    return f"[{'█' * filled}{'░' * empty}]"

