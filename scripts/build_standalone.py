"""
Build script to package the ovchat suite into a single-file executable zipapp.

Usage:
    python scripts/build_standalone.py
Output:
    dist/ovchat.pyz (Runnable via `python dist/ovchat.pyz`)
"""

import os
import shutil
import sys
import zipapp
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
BUILD = ROOT / "build_temp"


def build():
    print("Building single-file ovchat.pyz executable...")
    DIST.mkdir(exist_ok=True)
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)

    # Copy ovchat package
    pkg_src = ROOT / "ovchat"
    pkg_dest = BUILD / "ovchat"
    shutil.copytree(pkg_src, pkg_dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # Create __main__.py inside the build directory
    main_file = BUILD / "__main__.py"
    main_file.write_text(
        "import sys\n"
        "from ovchat.cli import entrypoint\n\n"
        "if __name__ == '__main__':\n"
        "    entrypoint()\n",
        encoding="utf-8",
    )

    target_pyz = DIST / "ovchat.pyz"
    zipapp.create_archive(
        source=str(BUILD),
        target=str(target_pyz),
        interpreter="/usr/bin/env python3" if os.name != "nt" else None,
        main=None,  # __main__.py is used
    )

    shutil.rmtree(BUILD)
    size_kb = target_pyz.stat().st_size / 1024
    print(f"[Success] Built standalone archive: {target_pyz.resolve()} ({size_kb:.1f} KB)")
    print("Run with: python dist/ovchat.pyz")


if __name__ == "__main__":
    build()

