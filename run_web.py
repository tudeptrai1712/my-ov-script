"""
Run OpenVINO GenAI Web UI.

Usage:
    python run_web.py [--port 8080] [--host 127.0.0.1] [--no-browser] [--status]
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Launch OpenVINO GenAI Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    parser.add_argument("--status", action="store_true", help="Display dependency check status")

    args = parser.parse_args()

    if args.status:
        print_dependency_status()
        sys.exit(0)

    # Automatic dependency verification
    ok, missing = verify_dependencies(auto_install=True)
    if not ok:
        print(f"\n[Dependency Verifier] Could not resolve: {missing}. Please install manually.\n")
        sys.exit(1)

    import uvicorn

    url = f"http://{args.host}:{args.port}"

    print("\n" + "=" * 60)
    print("  OpenVINO GenAI Web UI")
    print(f"  URL: {url}")
    print("=" * 60 + "\n")

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    uvicorn.run(
        "ovchat.web.app:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
