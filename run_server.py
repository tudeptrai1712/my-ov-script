"""
OpenVINO GenAI OpenAI-Compatible API Server Runner.
Dedicated entrypoint delegating to the unified ovchat CLI.

Usage:
    python run_server.py [--port 8000] [--host 0.0.0.0] [--device GPU]
    python run_server.py --hetero
    python run_server.py --stop / --kill
"""

import sys
from ovchat.cli import entrypoint


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    # Handle stop / kill flags
    if any(k in argv for k in ("--kill", "--stop", "-k")):
        stop_args = ["stop"] + [a for a in argv if a not in ("--kill", "--stop", "-k")]
        entrypoint(stop_args)
        return

    # Forward to serve subcommand
    entrypoint(["serve"] + argv)


if __name__ == "__main__":
    main()
