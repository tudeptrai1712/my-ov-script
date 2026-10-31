"""
Model discovery and file inspections for OpenVINO IR models.
Pure filesystem operations without terminal UI dependencies.
"""

from pathlib import Path
from typing import List, Optional

from .config import MODEL_ROOT


def find_models(model_root: Optional[Path] = None) -> List[Path]:
    """
    Find directories containing OpenVINO IR models.

    A model directory normally contains one or more .xml/.bin files.
    """
    root = model_root or MODEL_ROOT
    if not root.exists():
        return []

    models = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue

        xml_files = list(directory.glob("*.xml"))
        bin_files = list(directory.glob("*.bin"))

        if xml_files and bin_files:
            models.append(directory)

    return sorted(models, key=lambda x: x.name.lower())


def model_size(model_path: Path) -> int:
    """Return total .bin weight size for a model directory in bytes."""
    total = 0
    for file in model_path.glob("*.bin"):
        try:
            total += file.stat().st_size
        except OSError:
            pass
    return total


def choose_model(model_root: Optional[Path] = None) -> Optional[Path]:
    """
    Backward-compatibility alias.
    Directs to ovchat.prompts.choose_model.
    """
    from .prompts import choose_model as _choose_model
    return _choose_model(model_root)


__all__ = [
    "find_models",
    "model_size",
    "choose_model",
]
