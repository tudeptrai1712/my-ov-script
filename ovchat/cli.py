import sys
from typing import Optional

from .chat import Chat
from .config import (
    DEFAULT_CONTEXT,
    DEFAULT_ENABLE_REASONING,
    DEFAULT_MAX_NEW_TOKENS,
    MODEL_ROOT,
)
from .devices import choose_device
from .metadata import read_model_metadata
from .models import choose_model
from .ui import UI, clear_screen


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

        print(f"{UI.YELLOW}Enter 'y' for yes or 'n' for no.{UI.RESET}")


def main() -> None:
    """Top-level entry point for interactive OpenVINO chat."""
    clear_screen()

    print(f"{UI.BOLD}{UI.CYAN}OpenVINO GenAI Chat{UI.RESET}\n")
    print(f"{UI.DIM}Model root: {MODEL_ROOT}{UI.RESET}\n")

    # 1. Model selection
    model_path = choose_model(MODEL_ROOT)
    if model_path is None:
        return

    # Read model metadata
    metadata = read_model_metadata(model_path)
    print(
        f"{UI.DIM}Model type: {metadata.display_type} "
        f"| Precision: {metadata.precision.upper()}{UI.RESET}\n"
    )

    # 2. Context length (bounded by model metadata if available)
    if metadata.max_position_embeddings:
        default_ctx = min(DEFAULT_CONTEXT, metadata.max_position_embeddings)
        max_ctx = metadata.max_position_embeddings
        ctx_label = f"Context length (max: {max_ctx:,})"
    else:
        default_ctx = DEFAULT_CONTEXT
        max_ctx = None
        ctx_label = "Context length"

    context_length = choose_integer(
        ctx_label,
        default=default_ctx,
        minimum=256,
        maximum=max_ctx,
    )

    print()

    # 3. Maximum output tokens
    max_new_tokens = choose_integer(
        "Maximum output tokens",
        DEFAULT_MAX_NEW_TOKENS,
        minimum=1,
    )

    print()

    # 4. Hardware device selection (incompatible devices are hidden based on metadata)
    device = choose_device(metadata)
    if device is None:
        return

    print()

    # 5. Reasoning mode (hidden if model does not support reasoning)
    if metadata.supports_reasoning:
        enable_reasoning = choose_boolean(
            "Enable reasoning / thinking mode?",
            default=DEFAULT_ENABLE_REASONING,
        )
        print()
    else:
        enable_reasoning = False

    # 6. Configuration summary
    print(f"{UI.BOLD}Configuration{UI.RESET}")
    print(f"  Model    : {model_path.name}")
    print(f"  Type     : {metadata.display_type} ({metadata.precision.upper()})")
    print(f"  Context  : {context_length:,}")
    print(f"  Output   : {max_new_tokens:,}")
    print(f"  Device   : {device}")

    if metadata.supports_reasoning:
        reason_label = "Enabled" if enable_reasoning else "Disabled"
    else:
        reason_label = "Not supported by model"
    print(f"  Reasoning: {reason_label}\n")

    input("Press Enter to load...")

    # 7. Initialize pipeline & chat session
    try:
        chat = Chat(
            model_path=model_path,
            device=device,
            context_length=context_length,
            max_new_tokens=max_new_tokens,
            enable_reasoning=enable_reasoning,
            metadata=metadata,
        )
    except Exception as e:
        print(f"\n{UI.RED}Failed to load model:{UI.RESET}\n")
        print(e)
        input("\nPress Enter...")
        return

    # 8. Chat loop
    clear_screen()
    print(f"{UI.BOLD}{UI.GREEN}OpenVINO model loaded.{UI.RESET}\n")
    print(f"Model    : {model_path.name}")
    print(f"Type     : {metadata.display_type}")
    print(f"Device   : {device}")
    print(f"Reasoning: {reason_label}")
    print(f"Context  : {context_length:,}")
    print(f"Max output: {max_new_tokens:,}")
    print(f"History  : {chat.history_saver.session_file}\n")

    print(f"{UI.DIM}Commands:{UI.RESET}")
    print("  /clear  - clear conversation")
    print("  /info   - show model/memory info")
    print("  /file   - attach file or PDF for context (/file <path>)")
    if metadata.is_vlm:
        print("  /image  - attach image for vision understanding (/image <path>)")
    if metadata.supports_reasoning:
        print("  /think  - toggle reasoning mode (or /think on, /think off)")
    print("  /unload - eject/unload model from memory (free VRAM)")
    print("  /load   - switch or load a model into session")
    print("  /quit   - exit\n")

    chat.run()

    # 9. Clean shutdown
    chat.memory_monitor.stop()
    print(f"\n{UI.GREEN}Goodbye.{UI.RESET}")
