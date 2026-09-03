"""
File extraction and document handling utilities for OpenVINO GenAI Chat.
Supports text, code, JSON, CSV, Markdown, and PDF documents.
"""

import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TEXT_EXTENSIONS = {
    ".txt", ".md", ".json", ".csv", ".tsv", ".log", ".yaml", ".yml",
    ".xml", ".html", ".htm", ".css", ".py", ".js", ".jsx", ".ts",
    ".tsx", ".c", ".cpp", ".h", ".hpp", ".rs", ".go", ".java",
    ".cs", ".php", ".rb", ".sh", ".bat", ".ps1", ".sql", ".ini",
    ".toml", ".cfg", ".conf", ".rst", ".tex",
}

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff",
}


def is_image_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in IMAGE_EXTENSIONS


def is_document_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in TEXT_EXTENSIONS or ext == ".pdf"


def extract_text_from_bytes(filename: str, raw_bytes: bytes, max_chars: int = 50000) -> str:
    """Extracts plain text from raw file bytes (supports text files and PDFs)."""
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
            pages_text = []
            total_chars = 0
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                pages_text.append(f"--- Page {i + 1} ---\n{text}")
                total_chars += len(text)
                if total_chars > max_chars:
                    pages_text.append(f"\n[... Truncated after {max_chars:,} characters ...]")
                    break
            return "\n\n".join(pages_text).strip()
        except Exception as e:
            return f"[Error extracting PDF text: {e}]"

    # Standard text/code file
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            text = raw_bytes.decode(encoding)
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n[... Truncated after {max_chars:,} characters ...]"
            return text
        except UnicodeDecodeError:
            continue

    return "[Binary or unsupported file encoding]"


def extract_text_from_path(file_path: Path, max_chars: int = 50000) -> str:
    """Extracts plain text from a local filesystem path."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    raw_bytes = file_path.read_bytes()
    return extract_text_from_bytes(file_path.name, raw_bytes, max_chars=max_chars)


def format_attachments_context(files: List[Dict[str, str]]) -> str:
    """
    Formats a list of files ({filename, content}) into context prepended to user prompt.
    """
    if not files:
        return ""

    blocks = []
    for f in files:
        fname = f.get("filename", "file")
        content = f.get("content", "").strip()
        blocks.append(f"=== File: {fname} ===\n{content}\n=== End of {fname} ===")

    return "\n\n".join(blocks)

