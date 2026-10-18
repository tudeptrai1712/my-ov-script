"""
OpenVINO GenAI Chat - Standalone Launcher.
Provides an immediate runner for the ovchat suite with automatic dependency verification.
For generating a single-file bundled executable, see scripts/build_standalone.py.
"""

import os
import sys
from pathlib import Path

# Ensure the project root is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Verify dependencies and launch
from ovchat.cli import entrypoint

if __name__ == "__main__":
    entrypoint()
