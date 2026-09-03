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


def find_pids_on_port(port: int) -> list:
    """Finds all process IDs listening on the given port."""
    if os.name == "nt":
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


def main():
    parser = argparse.ArgumentParser(description="Launch OpenVINO GenAI OpenAI-Compatible API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0 to allow Open WebUI / Docker access)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--status", action="store_true", help="Display dependency check status")
    parser.add_argument("--kill", "--stop", action="store_true", help="Kill any existing API server on port and exit")

    args = parser.parse_args()

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

    import uvicorn
    from ovchat.web.app import create_app

    app = create_app()
    api_url = f"http://127.0.0.1:{args.port}/v1"

    print("\n" + "=" * 68)
    print("  OpenVINO GenAI - OpenAI-Compatible API Server")
    print(f"  OpenAI API Base URL : {api_url}")
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

