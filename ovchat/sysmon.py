"""
Hardware & System Monitor for OpenVINO GenAI Chat.
Continuously samples CPU, RAM, GPU (Compute, 3D, Intel XMX / Neural Engine), and VRAM usage.
Uses psutil for CPU/RAM and Windows PDH (Performance Data Helper) via ctypes for ultra-low overhead GPU metrics.
"""

import ctypes
import os
import subprocess
import threading
import time
from ctypes import wintypes
from typing import Any, Dict, Optional


class PDH_FMT_COUNTERVALUE_ITEM(ctypes.Structure):
    _fields_ = [
        ("szName", wintypes.LPWSTR),
        ("CStatus", wintypes.DWORD),
        ("doubleValue", ctypes.c_double),
    ]


PDH_FMT_DOUBLE = 0x00000200


class SystemMonitor:
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        self.gpu_name = "Intel GPU"
        self.vram_dedicated_total_bytes = 0
        self._detect_gpu_info()

        self.latest_metrics: Dict[str, Any] = {
            "cpu_percent": 0.0,
            "ram_used_gb": 0.0,
            "ram_total_gb": 0.0,
            "ram_percent": 0.0,
            "gpu_name": self.gpu_name,
            "gpu_compute_percent": 0.0,
            "gpu_xmx_percent": 0.0,
            "gpu_3d_percent": 0.0,
            "gpu_total_percent": 0.0,
            "vram_dedicated_used_gb": 0.0,
            "vram_dedicated_total_gb": round(self.vram_dedicated_total_bytes / (1024**3), 2),
            "vram_dedicated_percent": 0.0,
            "vram_shared_used_gb": 0.0,
            "vram_shared_total_gb": 0.0,
            "vram_shared_percent": 0.0,
        }

    def _detect_gpu_info(self) -> None:
        """Detects GPU Adapter name and dedicated memory limit."""
        if os.name != "nt":
            return
        try:
            cmd = "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json"
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=4,
            )
            import json
            data = json.loads(res.stdout)
            if isinstance(data, list) and data:
                data = data[0]
            if isinstance(data, dict):
                self.gpu_name = data.get("Name", "Intel GPU")
                ram = data.get("AdapterRAM")
                if ram and isinstance(ram, (int, float)) and ram > 0:
                    self.vram_dedicated_total_bytes = int(ram)
        except Exception:
            pass

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)

    def _monitor_loop(self) -> None:
        # Initialize psutil
        try:
            import psutil
            psutil.cpu_percent(interval=None)
        except ImportError:
            psutil = None

        # Initialize Windows PDH
        pdh = None
        h_query = wintypes.HANDLE()
        h_eng = wintypes.HANDLE()
        h_mem = wintypes.HANDLE()
        h_shared = wintypes.HANDLE()

        if os.name == "nt":
            try:
                pdh = ctypes.windll.pdh
                status = pdh.PdhOpenQueryW(None, 0, ctypes.byref(h_query))
                if status == 0:
                    pdh.PdhAddEnglishCounterW(h_query, r"\GPU Engine(*)\Utilization Percentage", 0, ctypes.byref(h_eng))
                    pdh.PdhAddEnglishCounterW(h_query, r"\GPU Adapter Memory(*)\Dedicated Usage", 0, ctypes.byref(h_mem))
                    pdh.PdhAddEnglishCounterW(h_query, r"\GPU Adapter Memory(*)\Shared Usage", 0, ctypes.byref(h_shared))
                    pdh.PdhCollectQueryData(h_query)
            except Exception:
                pdh = None

        while self.running:
            try:
                # 1. CPU & RAM
                cpu_pct = 0.0
                ram_used_gb = 0.0
                ram_total_gb = 0.0
                ram_pct = 0.0

                if psutil is not None:
                    cpu_pct = round(psutil.cpu_percent(interval=None), 1)
                    vm = psutil.virtual_memory()
                    ram_used_gb = round(vm.used / (1024**3), 2)
                    ram_total_gb = round(vm.total / (1024**3), 2)
                    ram_pct = round(vm.percent, 1)

                # 2. GPU via PDH
                gpu_compute = 0.0
                gpu_xmx = 0.0
                gpu_3d = 0.0
                vram_ded_used = 0.0
                vram_shared_used = 0.0

                if pdh is not None and h_query.value:
                    status = pdh.PdhCollectQueryData(h_query)
                    if status == 0:
                        # Query GPU Engine utilization
                        dwBufferSize = wintypes.DWORD(0)
                        dwItemCount = wintypes.DWORD(0)
                        pdh.PdhGetFormattedCounterArrayW(
                            h_eng, PDH_FMT_DOUBLE, ctypes.byref(dwBufferSize), ctypes.byref(dwItemCount), None
                        )
                        if dwBufferSize.value > 0:
                            buf = (ctypes.c_byte * dwBufferSize.value)()
                            st = pdh.PdhGetFormattedCounterArrayW(
                                h_eng, PDH_FMT_DOUBLE, ctypes.byref(dwBufferSize), ctypes.byref(dwItemCount), ctypes.byref(buf)
                            )
                            if st == 0:
                                items = ctypes.cast(buf, ctypes.POINTER(PDH_FMT_COUNTERVALUE_ITEM))
                                for i in range(dwItemCount.value):
                                    name = items[i].szName or ""
                                    val = items[i].doubleValue
                                    if val <= 0:
                                        continue
                                    if "engtype_Compute" in name:
                                        gpu_compute += val
                                    elif "engtype_Neural" in name or "engtype_Tensor" in name or "engtype_Matrix" in name:
                                        gpu_xmx += val
                                    elif "engtype_3D" in name:
                                        gpu_3d += val

                        # Query Dedicated VRAM
                        dwBufferSize.value = 0
                        dwItemCount.value = 0
                        pdh.PdhGetFormattedCounterArrayW(
                            h_mem, PDH_FMT_DOUBLE, ctypes.byref(dwBufferSize), ctypes.byref(dwItemCount), None
                        )
                        if dwBufferSize.value > 0:
                            buf = (ctypes.c_byte * dwBufferSize.value)()
                            if pdh.PdhGetFormattedCounterArrayW(
                                h_mem, PDH_FMT_DOUBLE, ctypes.byref(dwBufferSize), ctypes.byref(dwItemCount), ctypes.byref(buf)
                            ) == 0:
                                items = ctypes.cast(buf, ctypes.POINTER(PDH_FMT_COUNTERVALUE_ITEM))
                                for i in range(dwItemCount.value):
                                    vram_ded_used += items[i].doubleValue

                        # Query Shared VRAM
                        dwBufferSize.value = 0
                        dwItemCount.value = 0
                        pdh.PdhGetFormattedCounterArrayW(
                            h_shared, PDH_FMT_DOUBLE, ctypes.byref(dwBufferSize), ctypes.byref(dwItemCount), None
                        )
                        if dwBufferSize.value > 0:
                            buf = (ctypes.c_byte * dwBufferSize.value)()
                            if pdh.PdhGetFormattedCounterArrayW(
                                h_shared, PDH_FMT_DOUBLE, ctypes.byref(dwBufferSize), ctypes.byref(dwItemCount), ctypes.byref(buf)
                            ) == 0:
                                items = ctypes.cast(buf, ctypes.POINTER(PDH_FMT_COUNTERVALUE_ITEM))
                                for i in range(dwItemCount.value):
                                    vram_shared_used += items[i].doubleValue

                # Scale and clamp metrics
                gpu_compute = min(100.0, round(gpu_compute, 1))
                gpu_xmx = min(100.0, round(gpu_xmx, 1))
                gpu_3d = min(100.0, round(gpu_3d, 1))
                gpu_total = min(100.0, round(max(gpu_compute, gpu_xmx, gpu_3d, (gpu_compute + gpu_xmx + gpu_3d) / 2), 1))

                ded_gb = round(vram_ded_used / (1024**3), 2)
                ded_total_gb = self.latest_metrics["vram_dedicated_total_gb"]
                if ded_total_gb > 0:
                    ded_pct = round(min(100.0, (ded_gb / ded_total_gb) * 100), 1)
                else:
                    ded_pct = 0.0

                shared_gb = round(vram_shared_used / (1024**3), 2)
                shared_total_gb = round(ram_total_gb * 0.5, 2)
                shared_pct = round(min(100.0, (shared_gb / shared_total_gb) * 100), 1) if shared_total_gb > 0 else 0.0

                with self.lock:
                    self.latest_metrics = {
                        "cpu_percent": cpu_pct,
                        "ram_used_gb": ram_used_gb,
                        "ram_total_gb": ram_total_gb,
                        "ram_percent": ram_pct,
                        "gpu_name": self.gpu_name,
                        "gpu_compute_percent": gpu_compute,
                        "gpu_xmx_percent": gpu_xmx,
                        "gpu_3d_percent": gpu_3d,
                        "gpu_total_percent": gpu_total,
                        "vram_dedicated_used_gb": ded_gb,
                        "vram_dedicated_total_gb": ded_total_gb,
                        "vram_dedicated_percent": ded_pct,
                        "vram_shared_used_gb": shared_gb,
                        "vram_shared_total_gb": shared_total_gb,
                        "vram_shared_percent": shared_pct,
                    }

            except Exception:
                pass

            time.sleep(self.interval)

        if pdh is not None and h_query.value:
            try:
                pdh.PdhCloseQuery(h_query)
            except Exception:
                pass

    def get_metrics(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.latest_metrics)


_global_system_monitor: Optional[SystemMonitor] = None


def get_system_monitor() -> SystemMonitor:
    global _global_system_monitor
    if _global_system_monitor is None:
        _global_system_monitor = SystemMonitor(interval=1.0)
        _global_system_monitor.start()
    return _global_system_monitor

