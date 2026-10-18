"""
OpenVINO GenAI Chat & Serving Suite Runner.
Delegates to the unified ovchat CLI.

Usage:
    python ovchat.py              # Interactive terminal chat
    python ovchat.py chat         # Interactive terminal chat
    python ovchat.py serve        # OpenAI-compatible API Server (Port 8000)
    python ovchat.py stop         # Stop running API Server
    python ovchat.py check        # Hardware & dependency health check
    python ovchat.py models       # List local models & hardware fit
"""

import sys
from ovchat.cli import entrypoint

if __name__ == "__main__":
    entrypoint()