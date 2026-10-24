from pathlib import Path

from ..config import DEFAULT_HETERO_PRIORITY, MODEL_ROOT
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

    # Device & Heterogeneous status
    dev_str = engine.device
    if dev_str.upper().startswith("HETERO"):
        dev_str = f"{dev_str} {UI.GREEN}[Pipeline Parallelism Preview]{UI.RESET}"
    print(f"{UI.BOLD}Device:{UI.RESET} {dev_str}")

    # MoE Status
    if engine.metadata.is_moe:
        moe_str = engine.metadata.moe_summary or "Mixture of Experts"
        offload_str = (
            f"{UI.GREEN}Enabled ({engine.moe_offload_ratio}% to disk via ENABLE_MMAP){UI.RESET}"
            if engine.enable_moe_offload
            else f"{UI.YELLOW}Disabled{UI.RESET}"
        )
        print(f"{UI.BOLD}MoE Architecture:{UI.RESET} {moe_str}")
        print(f"{UI.BOLD}MoE Offload:{UI.RESET} {offload_str}")

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
    """Toggles or explicitly configures reasoning mode (/think on, /think off, /think status)."""
    if not context.engine.metadata.supports_reasoning:
        print(f"\n{UI.YELLOW}Reasoning mode is not supported by model: {context.engine.metadata.name}.{UI.RESET}\n")
        return CommandResult()

    arg = args_str.strip().lower()

    if arg in ("status", "info", "state"):
        curr = "enabled" if context.engine.reasoning_enabled else "disabled"
        col = UI.GREEN if context.engine.reasoning_enabled else UI.YELLOW
        print(f"\n{col}Reasoning mode is currently {curr}.{UI.RESET}\n")
        return CommandResult()

    if arg in ("on", "1", "true", "enable", "yes"):
        new_state = True
    elif arg in ("off", "0", "false", "disable", "no"):
        new_state = False
    elif not arg:
        new_state = not context.engine.reasoning_enabled
    else:
        print(f"\n{UI.YELLOW}Usage: /think [on|off|status]{UI.RESET}\n")
        return CommandResult()

    enabled = context.engine.set_reasoning(new_state)
    if context.history_saver:
        context.history_saver.reasoning_enabled = enabled

    state_str = "enabled" if enabled else "disabled"
    color = UI.GREEN if enabled else UI.YELLOW
    print(f"\n{color}Reasoning mode {state_str}.{UI.RESET}\n")
    return CommandResult()


def handle_moe(args_str: str, context: CommandContext) -> CommandResult:
    """Configures MoE expert offload to disk (/moe on, /moe off, /moe 50)."""
    if not context.engine.metadata.is_moe:
        print(f"\n{UI.YELLOW}Current model '{context.engine.metadata.name}' is not an MoE model.{UI.RESET}\n")
        return CommandResult()

    arg = args_str.strip().lower()

    if not arg or arg in ("status", "info"):
        status = "enabled" if context.engine.enable_moe_offload else "disabled"
        col = UI.GREEN if context.engine.enable_moe_offload else UI.YELLOW
        print(f"\n{col}MoE Offload is currently {status} ({context.engine.moe_offload_ratio}% to disk).{UI.RESET}\n")
        return CommandResult()

    if arg in ("off", "0", "false", "disable"):
        context.engine.set_moe_offload(False)
        print(f"\n{UI.YELLOW}MoE offload disabled. (Reload model with /load or restart session to apply).{UI.RESET}\n")
        return CommandResult()

    ratio = context.engine.moe_offload_ratio
    if arg.isdigit():
        ratio = max(1, min(99, int(arg)))
        context.engine.set_moe_offload(True, ratio=ratio)
    elif arg in ("on", "1", "true", "enable"):
        context.engine.set_moe_offload(True)
    else:
        print(f"\n{UI.YELLOW}Usage: /moe [on|off|<ratio 1-99>|status]{UI.RESET}\n")
        return CommandResult()

    print(
        f"\n{UI.GREEN}MoE offload set to {ratio}% to disk. "
        f"Run /load to recompile with new offload ratio.{UI.RESET}\n"
    )
    return CommandResult()


def handle_hetero(args_str: str, context: CommandContext) -> CommandResult:
    """Toggles or configures Heterogeneous Execution with Pipeline Parallelism (/hetero [priorities])."""
    arg = args_str.strip()
    target_priorities = arg if arg else DEFAULT_HETERO_PRIORITY
    target_device = f"HETERO:{target_priorities}"

    if hasattr(context.chat_session, "load_model"):
        print(f"\n{UI.CYAN}Switching to Heterogeneous Pipeline Parallelism on {target_device}...{UI.RESET}")
        context.chat_session.load_model(context.engine.model_path, target_device)
    else:
        print(f"\n{UI.GREEN}Target Hetero Device set to: {target_device}{UI.RESET}\n")

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
