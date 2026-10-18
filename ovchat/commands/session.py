from pathlib import Path

from ..config import MODEL_ROOT
from ..hardware import get_hardware_specs, render_hardware_and_models
from ..metadata import read_model_metadata
from ..models import find_models
from ..ui import UI, human_bytes
from .base import CommandContext, CommandResult


def handle_clear(args_str: str, context: CommandContext) -> CommandResult:
    """Clears conversation history and creates a fresh history transcript file."""
    context.engine.clear_history()
    context.history_saver.reset()
    context.last_assistant_answer = None
    context.pending_file_context = None
    context.pending_file_name = None
    context.pending_image_info = None
    print(f"\n{UI.GREEN}Conversation cleared.{UI.RESET}")
    print(f"{UI.DIM}New history log: {context.history_saver.session_file}{UI.RESET}\n")
    return CommandResult()


def handle_info(args_str: str, context: CommandContext) -> CommandResult:
    """Displays active model parameters, context consumption, and system hardware specs."""
    engine = context.engine
    used = engine.context_usage()

    if engine.metadata.supports_reasoning:
        reason_status = (
            f"{UI.GREEN}Enabled{UI.RESET}"
            if engine.reasoning_enabled
            else f"{UI.YELLOW}Disabled{UI.RESET}"
        )
    else:
        reason_status = f"{UI.DIM}Not supported by model{UI.RESET}"

    # Query live GPU/shared memory if monitor is active
    current_memory = getattr(context.chat_session, "memory_monitor", None)
    mem_str = human_bytes(current_memory.current) if current_memory and current_memory.current else "N/A"

    print()
    print(f"{UI.BOLD}Model:{UI.RESET} {engine.model_path.name}")
    print(
        f"{UI.BOLD}Type:{UI.RESET} {engine.metadata.display_type} "
        f"({engine.metadata.precision.upper()})"
    )
    print(f"{UI.BOLD}Device:{UI.RESET} {engine.device}")
    print(f"{UI.BOLD}Reasoning:{UI.RESET} {reason_status}")
    print(f"{UI.BOLD}Context:{UI.RESET} {used:,} / {engine.context_length:,}")
    print(f"{UI.BOLD}Max output:{UI.RESET} {engine.max_new_tokens:,}")
    print(f"{UI.BOLD}Load time:{UI.RESET} {engine.load_time:.2f}s")
    print(f"{UI.BOLD}History file:{UI.RESET} {context.history_saver.session_file}")
    print(f"{UI.BOLD}GPU/shared memory:{UI.RESET} {mem_str}")

    try:
        specs = get_hardware_specs()
        print(f"\n{UI.BOLD}System Hardware:{UI.RESET}")
        print(f"  CPU: {specs.cpu_name} ({specs.cpu_cores_physical}C/{specs.cpu_cores_logical}T)")
        print(f"  RAM: {human_bytes(specs.ram_total_bytes)} total | {human_bytes(specs.ram_available_bytes)} available")
        if specs.gpus:
            g = specs.gpus[0]
            eu_s = f", {g.execution_units} EUs" if g.execution_units else ""
            print(f"  GPU: {g.name} ({human_bytes(g.total_memory)}{eu_s})")
    except Exception:
        pass

    print()
    return CommandResult()


def handle_think(args_str: str, context: CommandContext) -> CommandResult:
    """Toggles or explicitly configures reasoning mode (/think on, /think off)."""
    arg = args_str.strip().lower()
    if arg in ("on", "1", "true", "enable", "yes"):
        enabled = context.engine.set_reasoning(True)
    elif arg in ("off", "0", "false", "disable", "no"):
        enabled = context.engine.set_reasoning(False)
    else:
        # Toggle current state
        enabled = context.engine.set_reasoning(not context.engine.reasoning_enabled)

    state_str = "enabled" if enabled else "disabled"
    color = UI.GREEN if enabled else UI.YELLOW
    print(f"\n{color}Reasoning mode {state_str}.{UI.RESET}\n")
    return CommandResult()


def handle_unload(args_str: str, context: CommandContext) -> CommandResult:
    """Ejects active model from memory and frees GPU VRAM."""
    if not context.engine.is_loaded:
        print(f"\n{UI.YELLOW}No model is currently loaded.{UI.RESET}\n")
        return CommandResult()

    model_name = context.engine.metadata.name
    context.engine.unload()
    print(f"\n{UI.GREEN}Model '{model_name}' unloaded from memory. VRAM freed.{UI.RESET}\n")
    return CommandResult()


def handle_load(args_str: str, context: CommandContext) -> CommandResult:
    """Prompts available models alongside hardware specs to load without restarting."""
    from ..hardware import choose_device

    all_models = find_models(MODEL_ROOT)
    if not all_models:
        print(f"{UI.RED}No models found in {MODEL_ROOT}{UI.RESET}")
        return CommandResult()

    print()
    render_hardware_and_models(all_models, MODEL_ROOT)
    sel = input(f"{UI.CYAN}Select model number: {UI.RESET}").strip()
    if sel.isdigit() and 1 <= int(sel) <= len(all_models):
        target_model = all_models[int(sel) - 1]
        meta = read_model_metadata(target_model)
        dev = choose_device(meta)
        if dev and hasattr(context.chat_session, "load_model"):
            context.chat_session.load_model(target_model, dev)

    return CommandResult()


def handle_quit(args_str: str, context: CommandContext) -> CommandResult:
    """Exits the interactive chat session."""
    return CommandResult(should_exit=True)

