"""
OpenVINO GenAI Chat runner.
Delegates execution to either the CLI interface or the Web UI.

Usage:
    python ovchat.py          # Start CLI
    python ovchat.py --web    # Start Web UI
    python ovchat.py --deps   # Check dependency status
"""

import sys

# 1. Automatic dependency verification (installs missing packages on launch)
from ovchat.verifier import print_dependency_status, verify_dependencies

if __name__ == "__main__":
    if "--deps" in sys.argv or "--check-deps" in sys.argv or "--status" in sys.argv:
        print_dependency_status()
        sys.exit(0)

    # Ensure all required packages are present before running
    ok, missing = verify_dependencies(auto_install=True)
    if not ok:
        print(f"\n[Dependency Verifier] Could not resolve: {missing}. Please install manually.\n")
        sys.exit(1)

    if "--kill-server" in sys.argv or "--stop-server" in sys.argv:
        from run_server import kill_server_on_port
        port = 8000
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        kill_server_on_port(port)
        sys.exit(0)

    # 2. Launch requested interface
    if "--server" in sys.argv or "--api" in sys.argv:
        from run_server import main as run_server_main
        run_server_main()
    elif "--web" in sys.argv:
        print("\n[Notice] The custom built-in web UI on port 8080 is deprecated in favor of:")
        print("  1. Starting the OpenAI-compatible API server on port 8000:  python run_server.py")
        print("  2. Connecting via Open WebUI (http://localhost:3000) as documented in README.md.\n")
        print("Starting legacy web UI for fallback...\n")
        import uvicorn
        port = 8080
        host = "127.0.0.1"
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
            elif arg == "--host" and i + 1 < len(sys.argv):
                host = sys.argv[i + 1]

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