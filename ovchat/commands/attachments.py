import re
from pathlib import Path
from typing import Optional, Tuple

import openvino as ov
from PIL import Image

from ..files import extract_text_from_path, render_pdf_to_images
from ..parser import parse_image_command_args, sanitize_file_path
from ..ui import UI
from .base import CommandContext, CommandResult


def load_image_or_pdf_into_context(
    path: Path,
    page_spec: Optional[str],
    context: CommandContext,
) -> bool:
    """
    Loads an image file (.jpg, .png, etc.) or renders pages of a .pdf into OpenVINO tensors.
    Attaches the tensors to context.engine.
    Returns True on success, False on failure.
    """
    if not context.engine.metadata.is_vlm:
        print(
            f"\n{UI.YELLOW}Current model '{context.engine.metadata.name}' is an LLM (text-only). "
            f"Images/PDF visual inputs are only supported on VLMs.{UI.RESET}\n"
        )
        return False

    if not path.exists():
        print(f"\n{UI.RED}File not found: {path}{UI.RESET}\n")
        return False

    ext = path.suffix.lower()
    if ext == ".pdf":
        try:
            print(f"{UI.DIM}Rendering PDF pages as visual images (pypdfium2)...{UI.RESET}")
            rendered = render_pdf_to_images(path, page_selection=page_spec)
            if not rendered:
                print(f"\n{UI.YELLOW}No pages could be rendered from: {path.name}{UI.RESET}\n")
                return False

            import numpy as np

            tensors = [ov.Tensor(np.array(img)) for _, img in rendered]
            context.engine.attach_images(tensors)

            pages_str = ", ".join(f"p.{p}" for p, _ in rendered)
            first_w, first_h = rendered[0][1].size
            context.pending_image_info = f"{path.name} ({len(rendered)} page(s): {pages_str})"
            print(
                f"\n{UI.GREEN}[OK] Loaded {len(rendered)} page(s) from PDF: {path.name} "
                f"({pages_str}, ~{first_w}x{first_h}px).{UI.RESET}"
            )
            return True
        except Exception as ex:
            print(f"\n{UI.RED}Failed to process PDF as images: {ex}{UI.RESET}\n")
            return False
    else:
        try:
            import numpy as np

            pil_img = Image.open(path).convert("RGB")
            tensor = ov.Tensor(np.array(pil_img))
            context.engine.attach_images([tensor])
            context.pending_image_info = f"{path.name} ({pil_img.width}x{pil_img.height})"
            print(
                f"\n{UI.GREEN}[OK] Image loaded: {path.name} "
                f"({pil_img.width}x{pil_img.height}).{UI.RESET}"
            )
            return True
        except Exception as ex:
            print(f"\n{UI.RED}Failed to load image: {ex}{UI.RESET}\n")
            return False


def handle_file(args_str: str, context: CommandContext) -> CommandResult:
    """Handles /file <path> [prompt] to attach document text context."""
    raw = args_str.strip()
    if not raw:
        print(f"{UI.YELLOW}Usage: /file <path/to/document.pdf or file.txt> [optional prompt]{UI.RESET}")
        return CommandResult()

    prompt_text = ""
    q_match = re.match(r'^"([^"]+)"\s*(.*)$', raw)
    if q_match:
        doc_path = sanitize_file_path(q_match.group(1))
        remaining = q_match.group(2).strip()
        if remaining and remaining not in ("\\", "/"):
            prompt_text = remaining
    else:
        parts_args = raw.split(maxsplit=1)
        doc_path = sanitize_file_path(parts_args[0])
        if len(parts_args) > 1:
            prompt_text = parts_args[1].strip()

    if not doc_path.exists():
        print(f"\n{UI.RED}File not found: {doc_path}{UI.RESET}\n")
        return CommandResult()

    # If user explicitly requests PDF visual rendering on a VLM via --image / -i
    if (
        doc_path.suffix.lower() == ".pdf"
        and context.engine.metadata.is_vlm
        and any(f in prompt_text.lower() for f in ("--image", "--as-image", "--as-images", "-i"))
    ):
        cleaned_p = re.sub(r'--(?:as-)?images?|-i', '', prompt_text, flags=re.IGNORECASE).strip()
        if load_image_or_pdf_into_context(doc_path, page_spec=None, context=context):
            if cleaned_p:
                print(f"{UI.BOLD}{UI.CYAN}Prompt:{UI.RESET} {cleaned_p}\n")
                return CommandResult(prompt_to_generate=cleaned_p)
            else:
                print(f"{UI.DIM}Type your prompt next.{UI.RESET}\n")
        return CommandResult()

    try:
        text = extract_text_from_path(doc_path)
        context.pending_file_context = text
        context.pending_file_name = doc_path.name
        print(
            f"\n{UI.GREEN}Attached file: {doc_path.name} "
            f"({len(text):,} chars extracted).{UI.RESET}"
        )
        if doc_path.suffix.lower() == ".pdf" and context.engine.metadata.is_vlm:
            print(
                f"{UI.DIM}(Tip: For math formulas/diagrams, use '/image {doc_path.name}' "
                f"to process PDF pages as visual images){UI.RESET}"
            )

        if prompt_text:
            print(f"{UI.BOLD}{UI.CYAN}Prompt:{UI.RESET} {prompt_text}\n")
            return CommandResult(prompt_to_generate=prompt_text)
        else:
            print(f"{UI.DIM}Type your prompt next.{UI.RESET}\n")
    except Exception as ex:
        print(f"\n{UI.RED}Failed to extract file text: {ex}{UI.RESET}\n")

    return CommandResult()


def handle_image(args_str: str, context: CommandContext) -> CommandResult:
    """Handles /image <path> [page] [prompt] to load images or PDF pages as visual tensors."""
    raw = args_str.strip()
    if not raw:
        print(f"{UI.YELLOW}Usage: /image <path/to/image.jpg or file.pdf> [page(s)] [optional prompt]{UI.RESET}")
        return CommandResult()

    path_str, page_spec, prompt_text = parse_image_command_args(raw)
    img_path = sanitize_file_path(path_str)

    if load_image_or_pdf_into_context(img_path, page_spec=page_spec, context=context):
        if prompt_text:
            print(f"{UI.BOLD}{UI.CYAN}Prompt:{UI.RESET} {prompt_text}\n")
            return CommandResult(prompt_to_generate=prompt_text)
        else:
            print(f"{UI.DIM}Type your prompt next.{UI.RESET}\n")

    return CommandResult()

