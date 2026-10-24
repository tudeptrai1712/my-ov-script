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
from .hardware import choose_device, print_hardware_specs, render_hardware_and_models
from .metadata import read_model_metadata
from .models import choose_model, find_models
from .ui import UI, clear_screen
from .verifier import print_dependency_status, verify_dependencies


# ============================================================
# INTERACTIVE PROMPT HELPERS
# ============================================================

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


# ============================================================
# SERVER MANAGEMENT HELPERS
# ============================================================

def find_pids_on_port(port: int) -> List[int]:
    """Finds all process IDs listening on the given port."""
    if os.name == "nt":
        try:
            res = subprocess.run(
                f"netstat -ano | findstr :{port}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=3,
            )
            pids = set()
            for line in res.stdout.splitlines():
                parts = line.strip().split()
                if len(parts) >= 4 and parts[1].endswith(f":{port}"):
                    last_col = parts[-1]
                    if last_col.isdigit() and int(last_col) > 0:
                        pids.add(int(last_col))
            if pids:
                return list(pids)
        except Exception:
            pass

        cmd = (
            f"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | "
            f"Select-Object -ExpandProperty OwningProcess"
        )
        try:
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=5,
            )
            pids = [
                int(p.strip())
                for p in res.stdout.splitlines()
                if p.strip().isdigit() and int(p.strip()) > 0
            ]
            return list(set(pids))
        except Exception:
            return []
    else:
        try:
            res = subprocess.run(
                ["lsof", "-t", f"-i:{port}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return [int(p) for p in res.stdout.split() if p.isdigit()]
        except Exception:
            return []


def kill_server_on_port(port: int = 8000) -> bool:
    """Finds and terminates any process listening on the specified port."""
    pids = find_pids_on_port(port)
    if not pids:
        return False

    killed_any = False
    for pid in pids:
        if pid == os.getpid():
            continue
        try:
            print(f"[API Server] Killing process PID {pid} listening on port {port}...")
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True,
                )
            else:
                os.kill(pid, signal.SIGTERM)
            killed_any = True
        except Exception as e:
            print(f"[API Server] Warning: Could not terminate PID {pid}: {e}")

    return killed_any


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
# INTERACTIVE PARAMETER CONFIGURATION
# ============================================================

def prompt_change_parameters() -> None:
    """Interactively prompts user to inspect or modify model load parameters before server launch."""
    from .settings import load_settings, save_settings

    settings = load_settings()

    dev = settings.get("selected_device", "GPU")
    ctx = settings.get("context_length", 32768)
    max_tok = settings.get("max_new_tokens", 8192)
    temp = settings.get("temperature", 0.7)
    top_p = settings.get("top_p", 0.95)
    reasoning = settings.get("enable_reasoning", False)
    moe_offload = settings.get("enable_moe_offload", True)
    moe_ratio = settings.get("moe_offload_ratio", 50)
    policy = settings.get("model_distribution_policy", "PIPELINE_PARALLEL")

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

            reas_in = input(f"6. Enable Reasoning/Thinking [y/n] (Current: {'y' if reasoning else 'n'}): ").strip().lower()
            if reas_in == "y":
                settings["enable_reasoning"] = True
            elif reas_in == "n":
                settings["enable_reasoning"] = False

            moe_in = input(f"7. Enable MoE Offload to Disk [y/n] (Current: {'y' if moe_offload else 'n'}): ").strip().lower()
            if moe_in == "y":
                settings["enable_moe_offload"] = True
            elif moe_in == "n":
                settings["enable_moe_offload"] = False

            moe_ratio_in = input(f"8. MoE Offload Ratio % (1-99) (Current: {moe_ratio}): ").strip()
            if moe_ratio_in.isdigit() and 1 <= int(moe_ratio_in) <= 99:
                settings["moe_offload_ratio"] = int(moe_ratio_in)

            save_settings(settings)
            print("\n[OK] Configuration updated and saved to user_config.json!\n")
        except (EOFError, KeyboardInterrupt):
            print("\nParameter change skipped.")


# ============================================================
# COMMAND ACTIONS
# ============================================================

def chat_command() -> None:
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
    if metadata.supports_reasoning:
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
    if metadata.supports_reasoning:
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

    # 1. Dependency verification
    if any(k in argv for k in ("--deps", "--check-deps", "--status", "check")):
        check_command()
        return

    ok, missing = verify_dependencies(auto_install=True)
    if not ok:
        print(f"\n[Dependency Verifier] Could not resolve: {missing}. Please install manually.\n")
        sys.exit(1)

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
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # subcommand: chat (default)
    subparsers.add_parser("chat", help="Start interactive terminal chat session (default)")

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

    args, remaining = parser.parse_known_args(argv)

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
        try:
            chat_command()
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
