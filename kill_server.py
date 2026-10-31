"""
Quick utility to terminate running OpenVINO API Server (Port 8000).
Delegates to the unified ovchat CLI.

Usage:
    python kill_server.py [--port 8000]
"""

import sys
from ovchat.cli import entrypoint


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    entrypoint(["stop"] + argv)


if __name__ == "__main__":
    main()
