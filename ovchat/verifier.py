"""
Automatic Dependency Verifier for OpenVINO GenAI Chat.
Checks required pip dependencies and automatically installs them when necessary.
Uses only standard library modules (sys, subprocess, importlib.util) so it runs anywhere.
"""

import importlib.util
import subprocess
import sys
from typing import Dict, List, Tuple


REQUIRED_DEPENDENCIES: List[Dict[str, str]] = [
    {
        "name": "openvino",
        "package": "openvino>=2024.0.0",
        "module": "openvino",
        "description": "Intel OpenVINO Inference Engine",
    },
    {
        "name": "openvino-genai",
        "package": "openvino-genai",
        "module": "openvino_genai",
        "description": "OpenVINO GenAI LLM/VLM Pipeline",
    },
    {
        "name": "fastapi",
        "package": "fastapi>=0.110.0",
        "module": "fastapi",
        "description": "FastAPI Web Framework",
    },
    {
        "name": "uvicorn",
        "package": "uvicorn>=0.28.0",
        "module": "uvicorn",
        "description": "ASGI Web Server",
    },
    {
        "name": "pillow",
        "package": "Pillow>=10.0.0",
        "module": "PIL",
        "description": "Pillow Image Processing (VLM)",
    },
    {
        "name": "numpy",
        "package": "numpy",
        "module": "numpy",
        "description": "NumPy Array Processing",
    },
    {
        "name": "pypdf",
        "package": "pypdf>=4.0.0",
        "module": "pypdf",
        "description": "PDF Document Parser",
    },
    {
        "name": "httpx",
        "package": "httpx",
        "module": "httpx",
        "description": "HTTP Client for Web/API Streaming",
    },
    {
        "name": "psutil",
        "package": "psutil>=5.9.0",
        "module": "psutil",
        "description": "CPU & System Memory Telemetry",
    },
]


def check_module_installed(module_name: str) -> bool:
    """Checks if a module can be imported without loading it."""
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except (ImportError, ValueError, AttributeError):
        return False


def install_package(package_spec: str, description: str) -> bool:
    """Installs a pip package via the active python executable."""
    print(f"\n[Dependency Verifier] Installing missing package: {package_spec} ({description})...")
    cmd = [sys.executable, "-m", "pip", "install", package_spec]
    try:
        res = subprocess.run(cmd, check=True)
        return res.returncode == 0
    except Exception as e:
        print(f"[Dependency Verifier] ERROR: Failed to install {package_spec}: {e}")
        return False


_has_verified = False


def verify_dependencies(auto_install: bool = True, show_banner: bool = True) -> Tuple[bool, List[str]]:
    """
    Verifies all required pip packages on every application launch.
    If auto_install is True, installs any missing packages automatically.
    Returns (all_satisfied: bool, missing_list: List[str]).
    """
    global _has_verified
    if _has_verified:
        return True, []

    missing = []
    for dep in REQUIRED_DEPENDENCIES:
        if not check_module_installed(dep["module"]):
            missing.append(dep)

    if not missing:
        _has_verified = True
        if show_banner:
            print(f"[Dependency Verifier] Checking required packages... All dependencies satisfied ({len(REQUIRED_DEPENDENCIES)}/{len(REQUIRED_DEPENDENCIES)}) [OK]")
        return True, []

    if not auto_install:
        missing_names = [d["name"] for d in missing]
        return False, missing_names

    print("\n" + "=" * 65)
    print("  [Dependency Verifier] Checking required packages on launch...")
    print(f"  Missing {len(missing)} package(s): {', '.join(d['name'] for d in missing)}")
    print("  Auto-installing required packages via pip...")
    print("=" * 65 + "\n")

    failed = []
    for dep in missing:
        success = install_package(dep["package"], dep["description"])
        if not success or not check_module_installed(dep["module"]):
            failed.append(dep["name"])
        else:
            print(f"  [OK] Successfully installed {dep['name']}.")

    if failed:
        print(f"\n[Dependency Verifier] WARNING: Failed to satisfy dependencies: {', '.join(failed)}\n")
        return False, failed

    _has_verified = True
    print("\n[Dependency Verifier] All dependencies successfully verified and ready!\n")
    return True, []


def print_dependency_status() -> None:
    """Prints a formatted report of all dependencies and their installed status."""
    print("\n" + "=" * 65)
    print("  OpenVINO GenAI Chat - Dependency Status")
    print("=" * 65)
    for dep in REQUIRED_DEPENDENCIES:
        installed = check_module_installed(dep["module"])
        status = "INSTALLED" if installed else "MISSING"
        ver = ""
        if installed:
            try:
                mod = importlib.import_module(dep["module"])
                ver = f" (v{getattr(mod, '__version__', 'unknown')})"
            except Exception:
                pass
        print(f"  - {dep['name']:<18} [{status:<9}] {dep['description']}{ver}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    if "--check" in sys.argv or "--status" in sys.argv:
        print_dependency_status()
    else:
        verify_dependencies(auto_install=True, quiet=False)
        print_dependency_status()

