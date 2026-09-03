"""
OpenVINO GenAI Model Manager.
Handles downloading from Hugging Face, automated conversion to OpenVINO IR,
model deletion/cleanup, and server status.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import HfApi, snapshot_download

from .config import MODEL_ROOT
from .metadata import check_device_compatibility, read_model_metadata
from .models import find_models, model_size
from .settings import load_settings, save_settings
from .ui import UI, human_bytes


RECOMMENDED_MODELS: List[Dict[str, str]] = [
    {
        "name": "Qwen 2.5 Coder 0.5B (Code & Chat)",
        "repo_id": "OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int8-ov",
        "type": "LLM",
        "size": "~0.6 GB",
        "description": "Ultra-lightweight coding and conversational model.",
    },
    {
        "name": "Qwen 2.5 1.5B Instruct",
        "repo_id": "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov",
        "type": "LLM",
        "size": "~1.1 GB",
        "description": "Fast, high quality small language model.",
    },
    {
        "name": "Llama 3.2 1B Instruct",
        "repo_id": "OpenVINO/Llama-3.2-1B-Instruct-int4-ov",
        "type": "LLM",
        "size": "~0.8 GB",
        "description": "Meta's lightweight on-device assistant.",
    },
    {
        "name": "Llama 3.2 3B Instruct",
        "repo_id": "OpenVINO/Llama-3.2-3B-Instruct-int4-ov",
        "type": "LLM",
        "size": "~2.1 GB",
        "description": "State-of-the-art compact conversational model.",
    },
    {
        "name": "DeepSeek R1 Distill Qwen 1.5B (Reasoning)",
        "repo_id": "OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-ov",
        "type": "LLM",
        "size": "~1.1 GB",
        "description": "Compact reasoning model with thinking capabilities.",
    },
    {
        "name": "Gemma 2 2B IT",
        "repo_id": "OpenVINO/gemma-2-2b-it-int4-ov",
        "type": "LLM",
        "size": "~1.6 GB",
        "description": "Google's high-efficiency lightweight model.",
    },
    {
        "name": "Llava 1.6 Mistral 7B (Vision & Language)",
        "repo_id": "OpenVINO/llava-v1.6-mistral-7b-int4-ov",
        "type": "VLM",
        "size": "~4.8 GB",
        "description": "Multimodal vision-language model for image understanding.",
    },
]


def clean_repo_id(repo_input: str) -> str:
    """Strips URL prefixes if the user pasted a full Hugging Face link."""
    repo = repo_input.strip()
    repo = re.sub(r"^https?://huggingface\.co/", "", repo)
    repo = repo.strip("/")
    return repo


def get_model_root() -> Path:
    """Returns the configured model root directory."""
    settings = load_settings()
    root_str = settings.get("model_root")
    if root_str:
        p = Path(root_str)
        p.mkdir(parents=True, exist_ok=True)
        return p
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    return MODEL_ROOT


def list_models_table() -> List[Path]:
    """Displays a formatted table of all local models and returns the list."""
    root = get_model_root()
    models = find_models(root)

    print("\n" + "=" * 78)
    print(f"  Local OpenVINO Models Directory: {root}")
    print("=" * 78)

    if not models:
        print(f"\n  {UI.YELLOW}No models found in {root}.{UI.RESET}")
        print("  Use 'Download' or 'Convert' to add models.\n")
        return []

    header = f"  {'#':<3} {'Model Name':<32} {'Type':<6} {'Precision':<10} {'Size':<10} {'Devices'}"
    print(header)
    print("  " + "-" * 74)

    for i, m in enumerate(models, start=1):
        meta = read_model_metadata(m)
        size_str = human_bytes(model_size(m))
        model_type = "VLM" if meta.is_vlm else "LLM"
        prec = meta.precision.upper() or "UNKNOWN"

        # Check supported devices
        devices = []
        for dev in ["GPU", "CPU", "NPU"]:
            ok, _ = check_device_compatibility(dev, meta)
            if ok:
                devices.append(dev)
        dev_str = "/".join(devices)

        name_display = m.name if len(m.name) <= 30 else m.name[:27] + "..."
        print(f"  {i:<3} {name_display:<32} {model_type:<6} {prec:<10} {size_str:<10} {dev_str}")

    print("=" * 78 + "\n")
    return models


def delete_model(model_identifier: str, force: bool = False) -> bool:
    """Deletes a local model by name or index."""
    root = get_model_root()
    models = find_models(root)

    target_path: Optional[Path] = None
    if model_identifier.isdigit():
        idx = int(model_identifier) - 1
        if 0 <= idx < len(models):
            target_path = models[idx]
    else:
        candidate = root / model_identifier
        if candidate.exists() and candidate.is_dir():
            target_path = candidate
        else:
            for m in models:
                if m.name.lower() == model_identifier.lower():
                    target_path = m
                    break

    if not target_path or not target_path.exists():
        print(f"{UI.RED}Error: Model '{model_identifier}' not found in {root}.{UI.RESET}")
        return False

    size_str = human_bytes(model_size(target_path))
    if not force:
        print(f"\n{UI.BOLD}Target Model:{UI.RESET} {target_path.name}")
        print(f"{UI.BOLD}Location:{UI.RESET}     {target_path}")
        print(f"{UI.BOLD}Disk Size:{UI.RESET}    {size_str}")
        confirm = input(f"\n{UI.YELLOW}Are you sure you want to permanently delete this model? (y/N): {UI.RESET}").strip().lower()
        if confirm != "y":
            print(f"{UI.CYAN}Deletion cancelled.{UI.RESET}\n")
            return False

    print(f"\nDeleting '{target_path.name}' ({size_str})...")
    try:
        shutil.rmtree(target_path)
        print(f"{UI.GREEN}[✓] Successfully deleted {target_path.name}. Freed {size_str}.{UI.RESET}\n")

        # Check if this model was default in user_config.json
        settings = load_settings()
        if settings.get("selected_model") == target_path.name:
            settings["selected_model"] = ""
            save_settings(settings)

        return True
    except Exception as e:
        print(f"{UI.RED}[!] Failed to delete {target_path.name}: {e}{UI.RESET}\n")
        return False


def check_hf_repo_is_openvino(repo_id: str, token: Optional[str] = None) -> Tuple[bool, List[str]]:
    """Checks whether a Hugging Face repository already has OpenVINO IR files."""
    api = HfApi(token=token)
    try:
        info = api.model_info(repo_id)
        files = [s.rfilename for s in info.siblings]
        has_xml = any(f.endswith(".xml") for f in files)
        return has_xml, files
    except Exception as e:
        return False, []


def find_alternative_ov_repo(base_model_id: str, token: Optional[str] = None) -> Optional[str]:
    """Attempts to find an official or community OpenVINO model for a given model ID."""
    api = HfApi(token=token)
    basename = base_model_id.split("/")[-1]

    # Search OpenVINO namespace
    try:
        candidates = list(api.list_models(search=basename, author="OpenVINO", limit=5))
        for c in candidates:
            if basename.lower() in c.id.lower() and ("-ov" in c.id.lower() or "openvino" in c.id.lower()):
                return c.id
    except Exception:
        pass
    return None


def convert_model_to_openvino(
    model_id_or_path: str,
    output_dir: Path,
    weight_format: str = "int4",
    trust_remote_code: bool = True,
) -> bool:
    """
    Converts a raw Hugging Face / PyTorch / SafeTensors model into OpenVINO IR format
    using optimum-intel and NNCF quantization.
    """
    # Check if optimum-intel is available
    optimum_installed = False
    try:
        import optimum.intel  # noqa: F401
        optimum_installed = True
    except ImportError:
        pass

    if not optimum_installed:
        print("\n" + "=" * 68)
        print("  [Dependency Required] optimum-intel")
        print("  Conversion of raw PyTorch/SafeTensors models to OpenVINO requires")
        print("  the 'optimum-intel' package.")
        print("=" * 68)
        choice = input("\nWould you like to install optimum-intel now via pip? (Y/n): ").strip().lower()
        if choice == "n":
            print(f"{UI.RED}Conversion aborted. Please install optimum-intel or download a pre-converted model.{UI.RESET}")
            return False

        print("\nInstalling optimum-intel via pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "optimum-intel[openvino]"])
            print(f"{UI.GREEN}[✓] optimum-intel installed successfully!{UI.RESET}\n")
        except Exception as e:
            print(f"{UI.RED}[!] Failed to install optimum-intel: {e}{UI.RESET}")
            return False

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  Starting OpenVINO Conversion: {model_id_or_path}")
    print(f"  Weight Quantization Format  : {weight_format.upper()}")
    print(f"  Target Destination Directory: {output_dir}")
    print("=" * 70 + "\n")

    cmd = [
        sys.executable,
        "-m",
        "optimum.commands.optimum_cli",
        "export",
        "openvino",
        "--model",
        model_id_or_path,
        "--weight-format",
        weight_format,
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd.append(str(output_dir))

    print(f"Executing: {' '.join(cmd)}\n")
    try:
        res = subprocess.run(cmd, check=True)
        if res.returncode == 0 and any(output_dir.glob("*.xml")):
            print(f"\n{UI.GREEN}[✓] Model successfully converted and saved to:{UI.RESET} {output_dir}\n")
            return True
        else:
            print(f"\n{UI.RED}[!] Export finished but no .xml files were found in {output_dir}.{UI.RESET}\n")
            return False
    except Exception as e:
        print(f"\n{UI.RED}[!] Conversion failed: {e}{UI.RESET}\n")
        return False


def download_model(
    repo_id: str,
    output_name: Optional[str] = None,
    weight_format: str = "int4",
    token: Optional[str] = None,
) -> bool:
    """
    Downloads a model from Hugging Face.
    - If already in OpenVINO format: downloads directly.
    - If not in OpenVINO format: offers pre-converted alternative or converts locally.
    """
    repo_id = clean_repo_id(repo_id)
    root = get_model_root()

    folder_name = output_name or repo_id.split("/")[-1]
    target_dir = root / folder_name

    if target_dir.exists() and any(target_dir.glob("*.xml")):
        print(f"{UI.YELLOW}Notice: Model directory '{folder_name}' already exists in {root}.{UI.RESET}")
        overwrite = input("Do you want to re-download / overwrite it? (y/N): ").strip().lower()
        if overwrite != "y":
            print("Download cancelled.\n")
            return False

    print(f"\nInspecting Hugging Face repository: {repo_id} ...")
    is_ov, files = check_hf_repo_is_openvino(repo_id, token=token)

    if is_ov:
        print(f"{UI.GREEN}[✓] Confirmed OpenVINO format repository!{UI.RESET}")
        print(f"Downloading files to: {target_dir} ...\n")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),
                token=token,
                local_dir_use_symlinks=False,
            )
            print(f"\n{UI.GREEN}[✓] Successfully downloaded {repo_id} to:{UI.RESET} {target_dir}\n")

            # Show metadata
            meta = read_model_metadata(target_dir)
            print(f"  Architecture : {'VLM' if meta.is_vlm else 'LLM'}")
            print(f"  Precision    : {meta.precision.upper()}")
            print(f"  Disk Size    : {human_bytes(model_size(target_dir))}\n")
            return True
        except Exception as e:
            print(f"{UI.RED}[!] Download failed: {e}{UI.RESET}\n")
            return False

    # Repo is not an OpenVINO model
    print(f"\n{UI.YELLOW}[!] Note: '{repo_id}' is a raw PyTorch/SafeTensors model (not OpenVINO IR).{UI.RESET}")

    # Check for an official pre-converted OpenVINO model
    alt_repo = find_alternative_ov_repo(repo_id, token=token)
    if alt_repo:
        print(f"\n{UI.CYAN}Found pre-converted OpenVINO model on Hugging Face:{UI.RESET} {UI.BOLD}{alt_repo}{UI.RESET}")
        choice = input(f"Would you prefer to download '{alt_repo}' directly? (Faster & pre-optimized) [Y/n]: ").strip().lower()
        if choice != "n":
            return download_model(alt_repo, output_name=output_name, weight_format=weight_format, token=token)

    # Convert locally
    print(f"\nConverting '{repo_id}' to OpenVINO format ({weight_format.upper()}) ...")
    return convert_model_to_openvino(repo_id, target_dir, weight_format=weight_format)


def interactive_menu() -> None:
    """Runs the interactive console menu for model management."""
    while True:
        print("\n" + "=" * 68)
        print("  OpenVINO GenAI - Model Management Console")
        print(f"  Root Directory: {get_model_root()}")
        print("=" * 68)
        print("  [1] List Local Models")
        print("  [2] Download Model from Hugging Face (OpenVINO IR or Auto-Convert)")
        print("  [3] Choose from Recommended OpenVINO Models")
        print("  [4] Convert Local or Remote Model to OpenVINO (optimum-intel)")
        print("  [5] Delete a Local Model")
        print("  [6] Stop / Kill Running API Server (Port 8000)")
        print("  [0] Exit")
        print("=" * 68)

        choice = input("\nEnter choice [0-6]: ").strip()

        if choice == "1":
            list_models_table()
            input("Press Enter to continue...")

        elif choice == "2":
            repo = input("\nEnter Hugging Face Repo ID (e.g. OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int8-ov): ").strip()
            if repo:
                custom_name = input("Custom local folder name (leave blank for default): ").strip() or None
                token = input("Hugging Face token (optional, leave blank if public): ").strip() or None
                download_model(repo, output_name=custom_name, token=token)
            input("Press Enter to continue...")

        elif choice == "3":
            print("\n" + "=" * 76)
            print("  Recommended Pre-Converted OpenVINO Models (Ready for Intel GPU / NPU)")
            print("=" * 76)
            for i, item in enumerate(RECOMMENDED_MODELS, start=1):
                print(f"  [{i}] {item['name']:<38} ({item['type']}, {item['size']})")
                print(f"      Repo: {item['repo_id']}")
                print(f"      Desc: {item['description']}\n")
            print("  [0] Cancel")
            print("=" * 76)

            sub = input("\nSelect model to download [0-7]: ").strip()
            if sub.isdigit() and 1 <= int(sub) <= len(RECOMMENDED_MODELS):
                rec = RECOMMENDED_MODELS[int(sub) - 1]
                download_model(rec["repo_id"])
            input("Press Enter to continue...")

        elif choice == "4":
            model_src = input("\nEnter Model ID on HF or Local Directory path to convert: ").strip()
            if model_src:
                fmt = input("Weight format (int4/int8/fp16, default: int4): ").strip().lower() or "int4"
                def_name = model_src.replace("\\", "/").split("/")[-1] + f"-{fmt}-ov"
                folder_name = input(f"Destination folder name (default: {def_name}): ").strip() or def_name
                out_path = get_model_root() / folder_name
                convert_model_to_openvino(model_src, out_path, weight_format=fmt)
            input("Press Enter to continue...")

        elif choice == "5":
            models = list_models_table()
            if models:
                del_choice = input("Enter model number or name to delete (or 0 to cancel): ").strip()
                if del_choice and del_choice != "0":
                    delete_model(del_choice)
            input("Press Enter to continue...")

        elif choice == "6":
            from kill_server import main as kill_main
            kill_main()
            input("Press Enter to continue...")

        elif choice == "0" or choice.lower() in ("q", "quit", "exit"):
            print("\nExiting Model Manager.")
            break
        else:
            print("Invalid selection.")


def main():
    parser = argparse.ArgumentParser(description="OpenVINO GenAI Model Management Tool")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # list
    subparsers.add_parser("list", help="List all local OpenVINO models")

    # delete
    del_p = subparsers.add_parser("delete", help="Delete a local model")
    del_p.add_argument("model", help="Model name or directory to delete")
    del_p.add_argument("-f", "--force", action="store_true", help="Skip confirmation prompt")

    # download
    dl_p = subparsers.add_parser("download", help="Download a model from Hugging Face")
    dl_p.add_argument("repo_id", help="Hugging Face repo ID (e.g. OpenVINO/Qwen2.5-Coder-0.5B-Instruct-int8-ov)")
    dl_p.add_argument("-o", "--output", help="Custom output folder name")
    dl_p.add_argument("--format", default="int4", choices=["int4", "int8", "fp16"], help="Weight format if conversion is needed (default: int4)")
    dl_p.add_argument("--token", help="Hugging Face access token (optional)")

    # convert
    conv_p = subparsers.add_parser("convert", help="Convert a model to OpenVINO format")
    conv_p.add_argument("model", help="Hugging Face model ID or local directory to convert")
    conv_p.add_argument("-o", "--output", help="Output directory path (default: model_root / <name>-<format>-ov)")
    conv_p.add_argument("--format", default="int4", choices=["int4", "int8", "fp16"], help="Weight quantization format (default: int4)")

    # kill
    subparsers.add_parser("kill", help="Stop running API server")

    args = parser.parse_args()

    if not args.command:
        interactive_menu()
    elif args.command == "list":
        list_models_table()
    elif args.command == "delete":
        delete_model(args.model, force=args.force)
    elif args.command == "download":
        download_model(args.repo_id, output_name=args.output, weight_format=args.format, token=args.token)
    elif args.command == "convert":
        root = get_model_root()
        if args.output:
            out_dir = Path(args.output)
        else:
            base_name = args.model.replace("\\", "/").split("/")[-1]
            out_dir = root / f"{base_name}-{args.format}-ov"
        convert_model_to_openvino(args.model, out_dir, weight_format=args.format)
    elif args.command == "kill":
        from kill_server import main as kill_main
        kill_main()


if __name__ == "__main__":
    main()

