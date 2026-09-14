"""
OpenVINO GenAI Chat runner.
Delegates execution to either the interactive CLI or the OpenAI-compatible API Server.

Usage:
    python ovchat.py          # Start Interactive CLI
    python ovchat.py --server # Start OpenAI-Compatible API Server (Port 8000)
    python ovchat.py --kill   # Stop running API Server
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

    if any(k in sys.argv for k in ("--kill", "--stop", "--kill-server", "--stop-server", "--kill-web")):
        from kill_server import main as kill_main
        kill_main()
        sys.exit(0)

    # 2. Launch requested interface
    if "--server" in sys.argv or "--api" in sys.argv or "--web" in sys.argv:
        if "--web" in sys.argv:
            print("\n[Notice] The custom Web UI has been removed in favor of the OpenAI-compatible API server.")
            print("Starting the API server on port 8000 for Open WebUI (http://localhost:3000)...\n")
        sys.argv = [a for a in sys.argv if a not in ("--server", "--api", "--web")]
        from run_server import main as run_server_main
        run_server_main()
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