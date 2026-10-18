import re
from pathlib import Path
from typing import Optional, Tuple, Union


def sanitize_file_path(path_input: Union[str, Path]) -> Path:
    """
    Cleans up user-provided file paths:
      - Strips accidental enclosing or trailing quotes (e.g. from copy-pasting or PowerShell).
      - Strips PowerShell drag-and-drop prefix (& '...').
      - Strips illegal leading backslashes before Windows drive letters (e.g. \\C:\\... -> C:\\...).
      - Strips trailing backslashes or slashes from file paths.
    """
    if isinstance(path_input, Path):
        cleaned = str(path_input).strip()
    else:
        cleaned = str(path_input).strip()

    # Strip PowerShell drag & drop prefix
    if cleaned.startswith("&"):
        cleaned = cleaned[1:].strip()

    # Strip enclosing quotes repeatedly
    while (cleaned.startswith('"') and cleaned.endswith('"')) or (
        cleaned.startswith("'") and cleaned.endswith("'")
    ):
        if len(cleaned) <= 1:
            break
        cleaned = cleaned[1:-1].strip()

    # If the path has a trailing quote or escaped quote
    cleaned = re.sub(r'["\']+$', '', cleaned).strip()

    # If the path starts with leading quotes or escaped quotes
    cleaned = re.sub(r'^["\']+', '', cleaned).strip()

    # Fix escaped quote artifacts before drive letters (e.g. \\C:\... or \C:\...)
    cleaned = re.sub(r'^[\\/]+([a-zA-Z]:)', r'\1', cleaned)

    # Strip trailing backslash/slash from file paths
    if len(cleaned) > 3 and cleaned.endswith(("\\", "/")):
        cleaned = cleaned[:-1].strip()

    return Path(cleaned)


def extract_inline_file(prompt: str) -> Tuple[str, Optional[Path]]:
    """
    Detects if /file <path> is embedded within the prompt string.
    Returns (cleaned_prompt, attached_file_path).
    """
    match = re.search(r'/file\s+("([^"]+)"|(\S+))', prompt, re.IGNORECASE)
    if not match:
        return prompt, None

    raw_file = match.group(2) or match.group(3)
    file_path = sanitize_file_path(raw_file)
    cleaned_prompt = prompt[:match.start()] + prompt[match.end():]
    cleaned_prompt = re.sub(r'\s+', ' ', cleaned_prompt).strip()
    return cleaned_prompt, file_path


def parse_image_command_args(raw_args: str) -> Tuple[str, Optional[str], str]:
    """
    Parses arguments passed to /image or /pic:
      /image "doc.pdf" 1 prompt text
      /image doc.pdf 1-3 prompt text
      /image "doc.pdf:2" prompt text
      /image "image.png" prompt text

    Returns:
      (sanitized_path_str, optional_page_spec, remaining_prompt)
    """
    raw = raw_args.strip()
    path_str = ""
    remaining = ""

    q_match = re.match(r'^"([^"]+)"\s*(.*)$', raw)
    if q_match:
        path_str = q_match.group(1).strip()
        remaining = q_match.group(2).strip()
    else:
        parts = raw.split(maxsplit=1)
        path_str = parts[0].strip()
        if len(parts) > 1:
            remaining = parts[1].strip()

    # Check for colon page specifier, e.g. path/to/doc.pdf:1 or doc.pdf:1-3
    page_spec = None
    colon_match = re.search(r'\.pdf:([0-9,\-all*]+)$', path_str, re.IGNORECASE)
    if colon_match:
        page_spec = colon_match.group(1)
        path_str = path_str[: colon_match.start() + 4]

    # If it's a PDF and next token is a page specifier (e.g. 1, 1-3, all)
    if path_str.lower().endswith(".pdf") and not page_spec and remaining:
        rem_parts = remaining.split(maxsplit=1)
        token = rem_parts[0].lower()
        if re.match(r'^(?:all|\*|\d+(?:-\d+)?|(?:\d+,)+\d+)$', token):
            page_spec = token
            remaining = rem_parts[1].strip() if len(rem_parts) > 1 else ""

    if remaining in ("\\", "/"):
        remaining = ""

    return path_str, page_spec, remaining


def extract_inline_image(prompt: str) -> Tuple[str, Optional[Path], Optional[str]]:
    """
    Detects if /image, /img, /pic, /pdf-image, or /pdfimg <path> is embedded within the prompt string.
    Returns (cleaned_prompt, attached_image_path, optional_page_spec).
    """
    match = re.search(
        r'/(?:image|img|pic|pdf-image|pdfimg)\s+("([^"]+)"|(\S+))',
        prompt,
        re.IGNORECASE,
    )
    if not match:
        return prompt, None, None

    raw_img = match.group(2) or match.group(3)
    page_spec = None
    colon_match = re.search(r'\.pdf:([0-9,\-all*]+)$', raw_img, re.IGNORECASE)
    if colon_match:
        page_spec = colon_match.group(1)
        raw_img = raw_img[: colon_match.start() + 4]

    img_path = sanitize_file_path(raw_img)
    cleaned_prompt = prompt[: match.start()] + prompt[match.end():]
    cleaned_prompt = re.sub(r'\s+', ' ', cleaned_prompt).strip()
    return cleaned_prompt, img_path, page_spec


def parse_export_args(raw_args: str) -> Tuple[Optional[str], Optional[Path]]:
    """
    Parses arguments passed to /export.
    Supports:
      - /export                                     -> (None, None) (export last answer)
      - /export output.txt                          -> (None, Path("output.txt"))
      - /export "prompt" "output.txt"               -> ("prompt", Path("output.txt"))
      - /export "prompt" output.txt                 -> ("prompt", Path("output.txt"))
      - /export output.txt prompt                   -> ("prompt", Path("output.txt"))
      - /export "output.txt" prompt                 -> ("prompt", Path("output.txt"))
      - /export prompt output.txt                   -> ("prompt", Path("output.txt"))
    """
    raw = raw_args.strip()
    if not raw:
        return None, None

    # Pattern 1: "prompt" "output.txt" or "prompt" output.txt
    match1 = re.match(r'^"(.+)"\s+("([^"]+)"|(\S+))$', raw)
    if match1:
        prompt = match1.group(1).strip()
        filename = match1.group(3) or match1.group(4)
        return prompt, sanitize_file_path(filename)

    # Pattern 2: "output.txt" prompt
    match2 = re.match(r'^"([^"]+\.(?:txt|md|log|json|csv|html))"\s+(.+)$', raw, re.IGNORECASE)
    if match2:
        filename = match2.group(1).strip()
        prompt = match2.group(2).strip()
        if prompt.startswith('"') and prompt.endswith('"') and len(prompt) >= 2:
            prompt = prompt[1:-1]
        return prompt, sanitize_file_path(filename)

    # Pattern 3: output.txt prompt ...
    parts = raw.split(maxsplit=1)
    first_token = sanitize_file_path(parts[0])
    if any(str(first_token).lower().endswith(ext) for ext in (".txt", ".md", ".log", ".json", ".csv", ".html")):
        if len(parts) > 1:
            prompt = parts[1].strip()
            if prompt.startswith('"') and prompt.endswith('"') and len(prompt) >= 2:
                prompt = prompt[1:-1]
            return prompt, first_token
        else:
            return None, first_token

    # Pattern 4: prompt ... output.txt
    parts_r = raw.rsplit(maxsplit=1)
    last_token = sanitize_file_path(parts_r[-1])
    if any(str(last_token).lower().endswith(ext) for ext in (".txt", ".md", ".log", ".json", ".csv")):
        prompt = parts_r[0].strip()
        if prompt.startswith('"') and prompt.endswith('"') and len(prompt) >= 2:
            prompt = prompt[1:-1]
        return prompt, last_token

    # Single word without space ending with file extension
    sanitized_single = sanitize_file_path(raw)
    if any(str(sanitized_single).lower().endswith(ext) for ext in (".txt", ".md", ".log", ".json", ".csv")):
        return None, sanitized_single

    # Fallback: treat entire string as prompt to be generated
    return raw.strip('\"\''), None

