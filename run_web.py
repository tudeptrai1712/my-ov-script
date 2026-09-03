"""
Run OpenVINO GenAI Web UI.

Usage:
    python run_web.py [--port 8080] [--host 127.0.0.1] [--no-browser]
"""

import argparse
import threading
import time
import webbrowser
import uvicorn


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

    args = parser.parse_args()
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
