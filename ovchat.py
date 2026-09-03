"""
OpenVINO GenAI Chat runner.
Delegates execution to either the CLI interface or the Web UI.

Usage:
    python ovchat.py          # Start CLI
    python ovchat.py --web    # Start Web UI
"""

import sys

if __name__ == "__main__":
    if "--web" in sys.argv:
        import uvicorn
        port = 8080
        host = "127.0.0.1"
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
            elif arg == "--host" and i + 1 < len(sys.argv):
                host = sys.argv[i + 1]

        print(f"\nStarting OpenVINO GenAI Web UI at http://{host}:{port} ...\n")
        uvicorn.run("ovchat.web.app:app", host=host, port=port, log_level="info")
    else:
        from ovchat.cli import main
        from ovchat.ui import UI
        try:
            main()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{UI.YELLOW}Exited.{UI.RESET}")
            sys.exit(0)
        except Exception as e:
            print(f"\n{UI.RED}Fatal error:{UI.RESET} {e}\n")
            try:
                input("Press Enter...")
            except (KeyboardInterrupt, EOFError):
                pass
            sys.exit(1)