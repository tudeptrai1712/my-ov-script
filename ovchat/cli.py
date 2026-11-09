import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

from .chat import Chat
from .config import (
    DEFAULT_CONTEXT,
    DEFAULT_ENABLE_MOE_OFFLOAD,
    DEFAULT_ENABLE_REASONING,
    DEFAULT_HETERO_DEVICE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_DISTRIBUTION_POLICY,
    DEFAULT_MOE_OFFLOAD_RATIO,
    MODEL_ROOT,
)
from .hardware import print_hardware_specs, render_hardware_and_models
from .metadata import read_model_metadata
from .models import find_models
from .process import find_pids_on_port, kill_server_on_port
from .prompts import (
    choose_boolean,
    choose_device,
    choose_integer,
    choose_model,
    prompt_change_parameters,
)
from .ui import UI, clear_screen
from .verifier import print_dependency_status, verify_dependencies


def console_listener(server) -> None:
    """Listens for console input commands like 'q' or 'kill' to stop the server."""
    while not server.should_exit:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            cmd = line.strip().lower()
            if cmd in ("q", "quit", "k", "kill", "stop", "exit"):
                print("\n[API Server] Shutdown command received in console. Stopping server...\n")
                server.should_exit = True
                break
        except Exception:
            break


# ============================================================
# COMMAND ACTIONS
# ============================================================

def chat_command(enable_reasoning_override: Optional[bool] = None) -> None:
    """Runs interactive OpenVINO terminal chat."""
    clear_screen()
    print(f"{UI.BOLD}{UI.CYAN}OpenVINO GenAI Chat{UI.RESET}\n")
    print(f"{UI.DIM}Model root: {MODEL_ROOT}{UI.RESET}\n")

    # 1. Model selection
    model_path = choose_model(MODEL_ROOT)
    if model_path is None:
        return

    metadata = read_model_metadata(model_path)
    moe_note = f" | {metadata.moe_summary}" if metadata.is_moe else ""
    print(
        f"{UI.DIM}Model type: {metadata.display_type} "
        f"| Precision: {metadata.precision.upper()}{moe_note}{UI.RESET}\n"
    )

    # 2. Context length
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

    # 4. Hardware device selection (GPU, HETERO:GPU,CPU, CPU)
    device = choose_device(metadata)
    if device is None:
        return
    print()

    # 5. MoE Offload settings for MoE models
    enable_moe_offload = DEFAULT_ENABLE_MOE_OFFLOAD
    moe_offload_ratio = DEFAULT_MOE_OFFLOAD_RATIO
    if metadata.is_moe and "GPU" in device.upper():
        enable_moe_offload = choose_boolean(
            f"Enable MoE Offload to disk? ({metadata.moe_summary})",
            default=DEFAULT_ENABLE_MOE_OFFLOAD,
        )
        if enable_moe_offload:
            moe_offload_ratio = choose_integer(
                "MoE offload ratio percentage (1-99%)",
                default=DEFAULT_MOE_OFFLOAD_RATIO,
                minimum=1,
                maximum=99,
            )
        print()

    # 6. Reasoning mode
    if enable_reasoning_override is not None:
        enable_reasoning = enable_reasoning_override
    elif metadata.supports_reasoning:
        enable_reasoning = choose_boolean(
            "Enable reasoning / thinking mode?",
            default=DEFAULT_ENABLE_REASONING,
        )
        print()
    else:
        enable_reasoning = False

    # 7. Configuration summary
    print(f"{UI.BOLD}Configuration{UI.RESET}")
    print(f"  Model    : {model_path.name}")
    print(f"  Type     : {metadata.display_type} ({metadata.precision.upper()})")
    print(f"  Context  : {context_length:,}")
    print(f"  Output   : {max_new_tokens:,}")
    print(f"  Device   : {device}")
    if device.upper().startswith("HETERO"):
        print(f"  Policy   : Pipeline Parallelism Preview ({DEFAULT_MODEL_DISTRIBUTION_POLICY})")
    if metadata.is_moe:
        moe_off_s = f"Enabled ({moe_offload_ratio}% to disk)" if enable_moe_offload else "Disabled"
        print(f"  MoE      : {moe_off_s}")

    reason_label = "Enabled" if enable_reasoning else ("Disabled" if metadata.supports_reasoning else "Not supported by model")
    print(f"  Reasoning: {reason_label}\n")

    input("Press Enter to load...")

    # 8. Initialize pipeline & chat session
    try:
        chat = Chat(
            model_path=model_path,
            device=device,
            context_length=context_length,
            max_new_tokens=max_new_tokens,
            enable_reasoning=enable_reasoning,
            metadata=metadata,
            enable_moe_offload=enable_moe_offload,
            moe_offload_ratio=moe_offload_ratio,
        )
    except Exception as e:
        print(f"\n{UI.RED}Failed to load model:{UI.RESET}\n")
        print(e)
        input("\nPress Enter...")
        return

    # 9. Chat loop
    clear_screen()
    print(f"{UI.BOLD}{UI.GREEN}OpenVINO model loaded.{UI.RESET}\n")
    print(f"Model    : {model_path.name}")
    print(f"Type     : {metadata.display_type}")
    print(f"Device   : {device}")
    if metadata.is_moe:
        moe_off_s = f"Enabled ({moe_offload_ratio}% to disk)" if enable_moe_offload else "Disabled"
        print(f"MoE      : {moe_off_s}")
    print(f"Reasoning: {reason_label}")
    print(f"Context  : {context_length:,}")
    print(f"Max output: {max_new_tokens:,}")
    print(f"History  : {chat.history_saver.session_file}\n")

    print(f"{UI.DIM}Commands:{UI.RESET}")
    print("  /clear   - clear conversation")
    print("  /info    - show model & system hardware info")
    print("  /file    - attach file or PDF for context (/file <path>)")
    if metadata.is_vlm:
        print("  /image   - attach image or visual PDF pages (/image <path> [page(s)])")
    print("  /think   - toggle reasoning mode (/think on, /think off)")
    if metadata.is_moe:
        print("  /moe     - configure MoE expert offload to disk (/moe <ratio%>, /moe off)")
    print("  /hetero  - switch to Hetero Pipeline Parallelism (/hetero GPU,CPU)")
    print("  /export  - export last answer or generate to txt (/export [output.txt])")
    print("  /unload  - eject/unload model from memory (free VRAM)")
    print("  /load    - switch or load a model into session")
    print("  /quit    - exit\n")

    chat.run()

    # 10. Clean shutdown
    chat.memory_monitor.stop()
    print(f"\n{UI.GREEN}Goodbye.{UI.RESET}")


def serve_command(
    host: str = "0.0.0.0",
    port: int = 8000,
    device: Optional[str] = None,
    no_prompt: bool = False,
    enable_moe_offload: Optional[bool] = None,
    moe_offload_ratio: Optional[int] = None,
) -> None:
    """Starts the OpenAI-compatible FastAPI server for Open WebUI."""
    import uvicorn
    from .settings import load_settings, save_settings
    from .web.app import app

    # Check for port collisions
    existing_pids = find_pids_on_port(port)
    if existing_pids:
        print(f"\n[Notice] Port {port} is currently in use by process PID(s): {existing_pids}")
        try:
            choice = input(f"Do you want to terminate existing process on port {port}? (Y/n): ").strip().lower()
            if choice not in ("n", "no"):
                kill_server_on_port(port)
                time.sleep(0.5)
            else:
                print(f"[Notice] Exiting server launch to avoid port conflict.\n")
                return
        except (EOFError, KeyboardInterrupt):
            return

    settings = load_settings()
    if device:
        settings["selected_device"] = device
    if enable_moe_offload is not None:
        settings["enable_moe_offload"] = enable_moe_offload
    if moe_offload_ratio is not None:
        settings["moe_offload_ratio"] = moe_offload_ratio
    save_settings(settings)

    if not no_prompt and sys.stdin.isatty():
        prompt_change_parameters()

    settings = load_settings()
    active_device = settings.get("selected_device", "GPU")
    moe_off_active = settings.get("enable_moe_offload", True)
    moe_ratio_active = settings.get("moe_offload_ratio", 50)

    print("\n" + "=" * 70)
    print("  OpenVINO GenAI OpenAI-Compatible API Server")
    print("=" * 70)
    print(f"  API Base URL          : http://127.0.0.1:{port}/v1")
    print(f"  Models Endpoint       : http://127.0.0.1:{port}/v1/models")
    print(f"  Chat Completions      : http://127.0.0.1:{port}/v1/chat/completions")
    print(f"  Target Hardware Device: {active_device}")
    if active_device.upper().startswith("HETERO"):
        print(f"  Pipeline Parallelism  : Enabled (Stages distributed across {active_device})")
    print(f"  MoE Disk Offload      : {'Enabled (' + str(moe_ratio_active) + '% to disk)' if moe_off_active else 'Disabled'}")
    print(f"  Open WebUI URL        : http://localhost:3000")
    print("=" * 70)
    print("  [Tip] Connect in Open WebUI: Settings > Connections > OpenAI API")
    print(f"        Set URL: http://127.0.0.1:{port}/v1 | Key: openvino")
    print("  [Controls] Press 'q' or 'k' + Enter, or run 'ovchat stop' to shutdown.")
    print("=" * 70 + "\n")

    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)

    listener_thread = threading.Thread(target=console_listener, args=(server,), daemon=True)
    listener_thread.start()

    try:
        server.run()
    except (KeyboardInterrupt, SystemExit):
        print("\n[API Server] Shutting down...")
    finally:
        print("[API Server] Server successfully stopped.")


def stop_command(port: int = 8000) -> None:
    """Terminates any active server on the specified port."""
    print("\n" + "=" * 62)
    print("  OpenVINO GenAI Server Killer")
    print("=" * 62)
    print(f"\nChecking port {port}...")
    killed = kill_server_on_port(port)
    if not killed:
        print(f"[Result] No active servers found running on port {port}.\n")
    else:
        print(f"[Result] Done. Stopped server on port {port}.\n")


def check_command() -> None:
    """Displays dependency health and system hardware specifications."""
    print_dependency_status()
    print()
    print_hardware_specs()
    print()


def models_command() -> None:
    """Lists available models with hardware fit and precision details."""
    models = find_models(MODEL_ROOT)
    if not models:
        print(f"\n{UI.RED}No OpenVINO models found in: {MODEL_ROOT}{UI.RESET}\n")
        return
    print()
    render_hardware_and_models(models, MODEL_ROOT)


# ============================================================
# UNIFIED CLI ENTRYPOINT
# ============================================================

def entrypoint(argv: Optional[List[str]] = None) -> None:
    """Main CLI entrypoint supporting subcommands and legacy flags."""
    if argv is None:
        argv = sys.argv[1:]

    # 1. Quick diagnostic and help flags (no auto-install side-effects)
    if any(k in argv for k in ("--deps", "--check-deps", "--status", "check")):
        check_command()
        return

    # 2. Legacy flag translation
    if any(k in argv for k in ("--kill", "--stop", "--kill-server", "--stop-server", "--kill-web")):
        stop_command()
        return

    if any(k in argv for k in ("--server", "--api", "--web")):
        clean_argv = [a for a in argv if a not in ("--server", "--api", "--web")]
        port = 8000
        for i, a in enumerate(clean_argv):
            if a in ("--port", "-p") and i + 1 < len(clean_argv) and clean_argv[i + 1].isdigit():
                port = int(clean_argv[i + 1])
        no_prompt = "-y" in clean_argv or "--no-prompt" in clean_argv
        dev = DEFAULT_HETERO_DEVICE if "--hetero" in clean_argv else None
        serve_command(port=port, device=dev, no_prompt=no_prompt)
        return

    # 3. Argument parser for subcommands
    parser = argparse.ArgumentParser(
        prog="ovchat",
        description="OpenVINO GenAI Chat & Serving Suite with Heterogeneous Pipeline Parallelism and MoE Offloading",
    )
    parser.add_argument("--reasoning", action="store_true", default=None, help="Enable reasoning / thinking mode")
    parser.add_argument("--no-reasoning", action="store_true", help="Disable reasoning / thinking mode")
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # subcommand: chat (default)
    chat_parser = subparsers.add_parser("chat", help="Start interactive terminal chat session (default)")
    chat_parser.add_argument("--reasoning", action="store_true", default=None, help="Enable reasoning / thinking mode")
    chat_parser.add_argument("--no-reasoning", action="store_true", help="Disable reasoning / thinking mode")

    # subcommand: serve
    serve_parser = subparsers.add_parser("serve", help="Start OpenAI-compatible FastAPI server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    serve_parser.add_argument("--device", default=None, help="Inference device (GPU, HETERO:GPU,CPU, CPU, AUTO)")
    serve_parser.add_argument("--hetero", action="store_true", help="Shortcut for --device HETERO:GPU,CPU (Pipeline Parallelism Preview)")
    serve_parser.add_argument("--moe-offload", action="store_true", default=None, help="Enable MoE weight offloading to disk")
    serve_parser.add_argument("--no-moe-offload", action="store_true", help="Disable MoE weight offloading to disk")
    serve_parser.add_argument("--moe-ratio", type=int, default=None, help="MoE offload ratio percentage (1-99, default 50)")
    serve_parser.add_argument("-y", "--no-prompt", action="store_true", help="Launch immediately without parameter prompt")

    # subcommand: stop
    stop_parser = subparsers.add_parser("stop", help="Stop running OpenAI API server")
    stop_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")

    # subcommand: check
    subparsers.add_parser("check", help="Run hardware and dependency health diagnostics")

    # subcommand: models
    subparsers.add_parser("models", help="List local OpenVINO models with sizes and hardware fit")

    # If help requested, show immediately
    if any(k in argv for k in ("-h", "--help")):
        parser.print_help()
        return

    args, remaining = parser.parse_known_args(argv)

    if args.subcommand == "stop":
        stop_command(port=args.port)
        return
    elif args.subcommand == "check":
        check_command()
        return
    elif args.subcommand == "models":
        models_command()
        return

    # Verify dependencies before launching chat or serve
    ok, missing = verify_dependencies(auto_install=True)
    if not ok:
        print(f"\n[Dependency Verifier] Could not resolve: {missing}. Please install manually.\n")
        sys.exit(1)

    if args.subcommand == "serve":
        target_device = args.device
        if args.hetero:
            target_device = DEFAULT_HETERO_DEVICE

        moe_offload = None
        if args.no_moe_offload:
            moe_offload = False
        elif args.moe_offload:
            moe_offload = True

        serve_command(
            host=args.host,
            port=args.port,
            device=target_device,
            no_prompt=args.no_prompt,
            enable_moe_offload=moe_offload,
            moe_offload_ratio=args.moe_ratio,
        )
    elif args.subcommand == "stop":
        stop_command(port=args.port)
    elif args.subcommand == "check":
        check_command()
    elif args.subcommand == "models":
        models_command()
    else:
        # Default action: chat
        reasoning_override: Optional[bool] = None
        if getattr(args, "no_reasoning", False):
            reasoning_override = False
        elif getattr(args, "reasoning", None):
            reasoning_override = True

        try:
            chat_command(enable_reasoning_override=reasoning_override)
        except (KeyboardInterrupt, EOFError):
            print(f"\n{UI.YELLOW}Exited.{UI.RESET}")
        except Exception as e:
            print(f"\n{UI.RED}Fatal error:{UI.RESET} {e}\n")
            sys.exit(1)


def main() -> None:
    """Backward-compatible function signature for existing callers."""
    chat_command()


if __name__ == "__main__":
    entrypoint()
