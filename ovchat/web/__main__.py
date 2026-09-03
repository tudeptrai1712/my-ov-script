import argparse
import sys
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="OpenVINO GenAI Chat Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port number (default: 8080)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    args = parser.parse_args()

    print(f"\n========================================================")
    print(f"  Starting OpenVINO GenAI Web UI")
    print(f"  Access at: http://{args.host}:{args.port}")
    print(f"========================================================\n")

    uvicorn.run(
        "ovchat.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
