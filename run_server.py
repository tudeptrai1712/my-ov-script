"""
OpenVINO GenAI OpenAI-Compatible API Server.
Dedicated backend for Open WebUI and other OpenAI API clients.

Reference implementation & guide:
https://docs.openvino.ai/2025/model-server/ovms_demos_integration_with_open_webui.html

Usage:
    python run_server.py [--port 8000] [--host 0.0.0.0]
    python run_server.py --kill [--port 8000]
"""

import argparse
import os
import signal
import subprocess
import sys
import threading

# Verify dependencies before importing uvicorn or web modules
from ovchat.verifier import print_dependency_status, verify_dependencies


import time


def find_pids_on_port(port: int) -> list:
    """Finds all process IDs listening on the given port."""
    if os.name == "nt":
        # Fast Windows netstat scan
        try:
            res = subprocess.run(f"netstat -ano | findstr :{port}", shell=True, capture_output=True, text=True, timeout=3)
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

        # Fallback to PowerShell Get-NetTCPConnection
        cmd = f"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess"
        try:
            res = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=5)
            pids = [int(p.strip()) for p in res.stdout.splitlines() if p.strip().isdigit() and int(p.strip()) > 0]
            return list(set(pids))
        except Exception:
            return []
    else:
        try:
            res = subprocess.run(["lsof", "-t", f"-i:{port}"], capture_output=True, text=True, timeout=5)
            return [int(p) for p in res.stdout.split() if p.isdigit()]
        except Exception:
            return []


def kill_server_on_port(port: int = 8000) -> bool:
    """Finds and terminates any running process on the specified API port."""
    pids = find_pids_on_port(port)
    if not pids:
        print(f"\n[API Server] No active server found running on port {port}.\n")
        return False

    killed_any = False
    for pid in pids:
        if pid == os.getpid():
            continue
        try:
            print(f"[API Server] Killing process PID {pid} listening on port {port}...")
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
            else:
                os.kill(pid, signal.SIGTERM)
            killed_any = True
        except Exception as e:
            print(f"[API Server] Failed to kill PID {pid}: {e}")

    if killed_any:
        print(f"[API Server] Successfully stopped server on port {port}.\n")
    return killed_any


def console_listener(server):
    """Listens for keyboard input (q, kill, exit, Ctrl+C) to cleanly stop the server."""
    if os.name == "nt":
        try:
            import msvcrt
            while not server.should_exit:
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch in (b'q', b'Q', b'k', b'K', b'x', b'X', b'\x03'):
                        print(f"\n[API Server] Stop key '{ch.decode('latin-1', errors='ignore')}' pressed. Shutting down server...\n")
                        server.should_exit = True
                        break
                time.sleep(0.1)
            return
        except Exception:
            pass

    # Standard stream fallback
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


def prompt_change_parameters():
    """Interactively prompts user to inspect or modify model load parameters before server launch."""
    from ovchat.settings import load_settings, save_settings
    settings = load_settings()

    dev = settings.get("selected_device", "GPU")
    ctx = settings.get("context_length", 32768)
    max_tok = settings.get("max_new_tokens", 8192)
    temp = settings.get("temperature", 0.7)
    top_p = settings.get("top_p", 0.95)
    reasoning = settings.get("enable_reasoning", False)

    print("\n" + "=" * 68)
    print("  Active Server Model Load Parameters:")
    print(f"  [1] Target Device       : {dev}")
    print(f"  [2] Context Length      : {ctx}")
    print(f"  [3] Max Output Tokens   : {max_tok}")
    print(f"  [4] Temperature         : {temp}")
    print(f"  [5] Top-P               : {top_p}")
    print(f"  [6] Reasoning / Thinking: {'Enabled' if reasoning else 'Disabled'}")
    print("=" * 68)

    try:
        choice = input("\nDo you want to change any load parameters before launching? (y/N): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return

    if choice == "y":
        print("\n--- Change Parameters (Press Enter on any setting to keep current value) ---")
        # 1. Device
        print(f"1. Target Device [1=GPU, 2=CPU, 3=AUTO] (Current: {dev}): ", end="", flush=True)
        try:
            dev_in = input().strip()
            if dev_in == "1":
                settings["selected_device"] = "GPU"
            elif dev_in == "2":
                settings["selected_device"] = "CPU"
            elif dev_in == "3":
                settings["selected_device"] = "AUTO"
            elif dev_in.upper() in ("GPU", "CPU", "AUTO"):
                settings["selected_device"] = dev_in.upper()

            # 2. Context Length
            ctx_in = input(f"2. Context Length (Current: {ctx}): ").strip()
            if ctx_in.isdigit() and int(ctx_in) > 0:
                settings["context_length"] = int(ctx_in)

            # 3. Max Tokens
            max_in = input(f"3. Max Output Tokens (Current: {max_tok}): ").strip()
            if max_in.isdigit() and int(max_in) > 0:
                settings["max_new_tokens"] = int(max_in)

            # 4. Temperature
            temp_in = input(f"4. Temperature (Current: {temp}): ").strip()
            if temp_in:
                try:
                    settings["temperature"] = float(temp_in)
                except ValueError:
                    pass

            # 5. Top-P
            top_p_in = input(f"5. Top-P (Current: {top_p}): ").strip()
            if top_p_in:
                try:
                    settings["top_p"] = float(top_p_in)
                except ValueError:
                    pass

            # 6. Reasoning
            reas_in = input(f"6. Enable Reasoning/Thinking [y/n] (Current: {'y' if reasoning else 'n'}): ").strip().lower()
            if reas_in == "y":
                settings["enable_reasoning"] = True
            elif reas_in == "n":
                settings["enable_reasoning"] = False

            save_settings(settings)
            print("\n[OK] Configuration updated and saved to user_config.json!\n")
        except (EOFError, KeyboardInterrupt):
            print("\nParameter change skipped.")


def main():
    parser = argparse.ArgumentParser(description="Launch OpenVINO GenAI OpenAI-Compatible API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0 to allow Open WebUI / Docker access)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--device", choices=["GPU", "CPU", "AUTO"], default=None, help="Inference device (GPU, CPU, or AUTO)")
    parser.add_argument("-y", "--no-prompt", action="store_true", help="Launch server immediately without prompting for parameter changes")
    parser.add_argument("--status", action="store_true", help="Display dependency check status")
    parser.add_argument("--kill", "--stop", action="store_true", help="Kill any existing API server on port and exit")

    args, _ = parser.parse_known_args()

    if args.status:
        print_dependency_status()
        sys.exit(0)

    if args.kill:
        kill_server_on_port(args.port)
        sys.exit(0)

    # Automatic dependency verification
    ok, missing = verify_dependencies(auto_install=True)
    if not ok:
        print(f"\n[Dependency Verifier] Could not resolve: {missing}. Please install manually.\n")
        sys.exit(1)

    from ovchat.settings import load_settings, save_settings
    settings = load_settings()

    # If --device was specified via CLI, update settings directly
    if args.device:
        settings["selected_device"] = args.device
        save_settings(settings)
    elif not args.no_prompt and sys.stdin.isatty():
        prompt_change_parameters()

    import uvicorn
    from ovchat.web.app import create_app

    app = create_app()
    active_settings = load_settings()
    active_dev = active_settings.get("selected_device", "GPU")
    api_url = f"http://127.0.0.1:{args.port}/v1"

    print("\n" + "=" * 68)
    print("  OpenVINO GenAI - OpenAI-Compatible API Server")
    print(f"  OpenAI API Base URL : {api_url}")
    print(f"  Default Device      : {active_dev}")
    print(f"  Models Endpoint     : {api_url}/models")
    print(f"  Chat Endpoint       : {api_url}/chat/completions")
    print("  Open WebUI Target   : Set OpenAI Base URL to " + api_url)
    print("  Console Control     : Type 'q' or 'kill' + Enter in console to stop")
    print("=" * 68 + "\n")

    config = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    # Start console input listener thread
    threading.Thread(target=console_listener, args=(server,), daemon=True).start()

    try:
        server.run()
    except (KeyboardInterrupt, SystemExit):
        print("\n[API Server] Server stopped.")


if __name__ == "__main__":
    main()

