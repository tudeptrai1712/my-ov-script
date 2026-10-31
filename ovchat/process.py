"""
Process and port management utilities for ovchat servers.
Cross-platform support for Windows (PowerShell/taskkill) and Linux/macOS.
"""

import os
import signal
import subprocess
from typing import List


def find_pids_on_port(port: int) -> List[int]:
    """Finds all process IDs listening on the specified network port."""
    if os.name == "nt":
        cmd = f"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess"
        try:
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=5,
            )
            pids = [
                int(p.strip())
                for p in res.stdout.splitlines()
                if p.strip().isdigit() and int(p.strip()) > 0
            ]
            return list(set(pids))
        except Exception:
            return []
    else:
        try:
            res = subprocess.run(
                ["lsof", "-t", f"-i:{port}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return [int(p) for p in res.stdout.split() if p.isdigit()]
        except Exception:
            return []


def kill_server_on_port(port: int, service_name: str = "server") -> bool:
    """Terminates any processes listening on the given port."""
    pids = find_pids_on_port(port)
    if not pids:
        print(f"\n[{service_name.capitalize()}] No active server found listening on port {port}.\n")
        return False

    killed_any = False
    for pid in pids:
        if pid == os.getpid():
            continue
        try:
            print(f"[{service_name.capitalize()}] Terminating process PID {pid} on port {port}...")
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
            else:
                os.kill(pid, signal.SIGTERM)
            killed_any = True
        except Exception as e:
            print(f"[{service_name.capitalize()}] Failed to terminate PID {pid}: {e}")

    if killed_any:
        print(f"[{service_name.capitalize()}] Successfully terminated server on port {port}.\n")
    return killed_any

