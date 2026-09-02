from pathlib import Path
from typing import List, Optional

from .config import MODEL_ROOT
from .ui import UI, human_bytes


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
    """Interactively prompts the user to select an OpenVINO model."""
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
    print(f"{UI.BOLD}{UI.CYAN}OpenVINO Models{UI.RESET}\n")

    for i, model in enumerate(models, start=1):
        size = model_size(model)
        print(f"  {i}. {model.name} {UI.DIM}({human_bytes(size)}){UI.RESET}")

    print()

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

