import sys
from typing import Optional

from .chat import Chat
from .config import (
    DEFAULT_CONTEXT,
    DEFAULT_MAX_NEW_TOKENS,
    MODEL_ROOT,
)
from .devices import choose_device
from .models import choose_model
from .ui import UI, clear_screen


def choose_integer(label: str, default: int, minimum: int = 1) -> int:
    """Interactively prompts for an integer value with validation."""
    while True:
        value = input(f"{UI.CYAN}{label} [{default}]: {UI.RESET}").strip()

        if not value:
            return default

        try:
            number = int(value)
            if number >= minimum:
                return number
        except ValueError:
            pass

        print(f"{UI.YELLOW}Enter an integer >= {minimum}.{UI.RESET}")


def main() -> None:
    """Top-level entry point for interactive OpenVINO chat."""
    clear_screen()

    print(f"{UI.BOLD}{UI.CYAN}OpenVINO GenAI Chat{UI.RESET}\n")
    print(f"{UI.DIM}Model root: {MODEL_ROOT}{UI.RESET}\n")

    # 1. Model selection
    model_path = choose_model(MODEL_ROOT)
    if model_path is None:
        return

    print()

    # 2. Context length
    context_length = choose_integer(
        "Context length",
        DEFAULT_CONTEXT,
        minimum=256,
    )

    print()

    # 3. Maximum output tokens
    max_new_tokens = choose_integer(
        "Maximum output tokens",
        DEFAULT_MAX_NEW_TOKENS,
        minimum=1,
    )

    print()

    # 4. Hardware device selection
    device = choose_device()
    if device is None:
        return

    print()

    # 5. Configuration summary
    print(f"{UI.BOLD}Configuration{UI.RESET}")
    print(f"  Model   : {model_path.name}")
    print(f"  Context : {context_length:,}")
    print(f"  Output  : {max_new_tokens:,}")
    print(f"  Device  : {device}\n")

    input("Press Enter to load...")

    # 6. Initialize pipeline & chat session
    try:
        chat = Chat(
            model_path=model_path,
            device=device,
            context_length=context_length,
            max_new_tokens=max_new_tokens,
        )
    except Exception as e:
        print(f"\n{UI.RED}Failed to load model:{UI.RESET}\n")
        print(e)
        input("\nPress Enter...")
        return

    # 7. Chat loop
    clear_screen()
    print(f"{UI.BOLD}{UI.GREEN}OpenVINO model loaded.{UI.RESET}\n")
    print(f"Model : {model_path.name}")
    print(f"Device: {device}")
    print(f"Context: {context_length:,}")
    print(f"Max output: {max_new_tokens:,}\n")

    print(f"{UI.DIM}Commands:{UI.RESET}")
    print("  /clear  - clear conversation")
    print("  /info   - show model/memory info")
    print("  /quit   - exit\n")

    chat.run()

    # 8. Clean shutdown
    chat.memory_monitor.stop()
    print(f"\n{UI.GREEN}Goodbye.{UI.RESET}")

