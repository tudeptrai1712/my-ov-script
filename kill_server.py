"""
Quick utility to terminate running OpenVINO API Server (Port 8000).

Usage:
    python kill_server.py [--port 8000]
"""

import sys
from ovchat.cli import kill_server_on_port, stop_command


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    port = 8000
    for i, a in enumerate(argv):
        if a in ("--port", "-p") and i + 1 < len(argv) and argv[i + 1].isdigit():
            port = int(argv[i + 1])

    stop_command(port=port)


if __name__ == "__main__":
    main()
