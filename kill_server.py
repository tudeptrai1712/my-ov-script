"""
Quick utility to terminate running OpenVINO API Server (Port 8000) or Web UI (Port 8080).

Usage:
    python kill_server.py
    python kill_server.py --port 8000
    py kill_server.py
"""

import argparse
import sys
from run_server import kill_server_on_port

def main(argv=None):
    parser = argparse.ArgumentParser(description="Kill running OpenVINO GenAI servers")
    parser.add_argument("--port", type=int, default=None, help="Specific port to terminate (default: checks 8000 and 8080)")
    parser.add_argument("--kill", "--stop", action="store_true", help="Kill active servers")
    parser.add_argument("--kill-server", "--stop-server", "--kill-web", action="store_true", help="Kill active servers")
    args, _ = parser.parse_known_args(argv)

    ports_to_check = [args.port] if args.port else [8000, 8080]
    total_killed = 0

    print("\n" + "=" * 62)
    print("  OpenVINO GenAI Server Killer")
    print("=" * 62)

    for p in ports_to_check:
        print(f"\nChecking port {p}...")
        if kill_server_on_port(p):
            total_killed += 1

    if total_killed == 0:
        print("\n[Result] No active servers found running on checked ports.\n")
    else:
        print(f"\n[Result] Done. Stopped {total_killed} server(s).\n")

if __name__ == "__main__":
    main()
