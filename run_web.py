"""
Run OpenVINO GenAI Web UI.

Usage:
    python run_web.py [--port 8080] [--host 127.0.0.1] [--no-browser] [--status]
    python run_web.py --kill [--port 8080]   # Kill any running Web UI on port
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser

# Verify dependencies before importing uvicorn or web modules
from ovchat.verifier import print_dependency_status, verify_dependencies


def open_browser(url: str):
    time.sleep(1.2)
    try:
        webbrowser.open_new_tab(url)
    except Exception:
        pass


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


def kill_web_server(port: int = 8080) -> bool:
    """Finds and terminates any running process on the specified web port."""
    pids = find_pids_on_port(port)
    if not pids:
        print(f"\n[Web UI] No active web server found running on port {port}.\n")
        return False

    killed_any = False
    for pid in pids:
        if pid == os.getpid():
            continue
        try:
            print(f"[Web UI] Killing process PID {pid} listening on port {port}...")
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
            else:
                os.kill(pid, signal.SIGTERM)
            killed_any = True
        except Exception as e:
            print(f"[Web UI] Failed to kill PID {pid}: {e}")

    if killed_any:
        print(f"[Web UI] Successfully stopped web server on port {port}.\n")
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
                print("\n[Web UI] Shutdown command received in console. Stopping server...\n")
                server.should_exit = True
                break
        except Exception:
            break


def main():
    parser = argparse.ArgumentParser(description="Launch OpenVINO GenAI Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    parser.add_argument("--status", action="store_true", help="Display dependency check status")
    parser.add_argument("--kill", "--stop", action="store_true", help="Kill any existing Web UI server on port and exit")

    args = parser.parse_args()

    if args.status:
        print_dependency_status()
        sys.exit(0)

    if args.kill:
        kill_web_server(args.port)
        sys.exit(0)

    # Automatic dependency verification
    ok, missing = verify_dependencies(auto_install=True)
    if not ok:
        print(f"\n[Dependency Verifier] Could not resolve: {missing}. Please install manually.\n")
        sys.exit(1)

    import uvicorn
    from ovchat.web.app import create_app

    app = create_app()
    url = f"http://{args.host}:{args.port}"

    print("\n" + "=" * 62)
    print("  OpenVINO GenAI Web UI")
    print(f"  URL: {url}")
    print("  Control: Type 'q' or 'kill' + Enter in console to stop")
    print("=" * 62 + "\n")

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(url,), daemon=True).start()

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
        print("\n[Web UI] Server stopped.")


if __name__ == "__main__":
    main()
