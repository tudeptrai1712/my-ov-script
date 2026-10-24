"""
OpenVINO GenAI OpenAI-Compatible API Server.
Dedicated backend for Open WebUI and standard OpenAI API clients.
Supports Heterogeneous Pipeline Parallelism and Mixture of Experts (MoE) Offloading.

Usage:
    python run_server.py [--port 8000] [--host 0.0.0.0]
    python run_server.py --hetero [--port 8000]
    python run_server.py --device HETERO:GPU,CPU
    python run_server.py --moe-ratio 50
    python run_server.py --kill [--port 8000]
"""

import sys
from ovchat.cli import serve_command, stop_command
from ovchat.config import DEFAULT_HETERO_DEVICE


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if any(k in argv for k in ("--kill", "--stop")):
        port = 8000
        for i, a in enumerate(argv):
            if a in ("--port", "-p") and i + 1 < len(argv) and argv[i + 1].isdigit():
                port = int(argv[i + 1])
        stop_command(port=port)
        return

    port = 8000
    host = "0.0.0.0"
    device = None
    no_prompt = False
    enable_moe_offload = None
    moe_offload_ratio = None

    if "--hetero" in argv:
        device = DEFAULT_HETERO_DEVICE
    if "--moe-offload" in argv:
        enable_moe_offload = True
    if "--no-moe-offload" in argv:
        enable_moe_offload = False

    for i, a in enumerate(argv):
        if a in ("--port", "-p") and i + 1 < len(argv) and argv[i + 1].isdigit():
            port = int(argv[i + 1])
        elif a in ("--host", "-h") and i + 1 < len(argv):
            host = argv[i + 1]
        elif a in ("--device", "-d") and i + 1 < len(argv):
            device = argv[i + 1]
        elif a in ("--moe-ratio", "--offload-ratio") and i + 1 < len(argv) and argv[i + 1].isdigit():
            moe_offload_ratio = int(argv[i + 1])

    if "-y" in argv or "--no-prompt" in argv:
        no_prompt = True

    serve_command(
        host=host,
        port=port,
        device=device,
        no_prompt=no_prompt,
        enable_moe_offload=enable_moe_offload,
        moe_offload_ratio=moe_offload_ratio,
    )


if __name__ == "__main__":
    main()
