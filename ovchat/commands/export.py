import time
from pathlib import Path

from ..parser import extract_inline_file, extract_inline_image, parse_export_args
from ..ui import UI
from .attachments import load_image_or_pdf_into_context
from .base import CommandContext, CommandResult


def handle_export(args_str: str, context: CommandContext) -> CommandResult:
    """
    Handles /export [file.txt] or /export "prompt" "file.txt".
    If a prompt is provided, generates the answer and saves it directly.
    If no prompt is provided, exports the last generated assistant response.
    """
    prompt_text, export_path = parse_export_args(args_str)

    if prompt_text:
        # Prompt specified: generate and export directly
        cleaned_prompt, inline_file = extract_inline_file(prompt_text)
        if inline_file:
            if not inline_file.exists():
                print(f"\n{UI.RED}Attached file not found: {inline_file}{UI.RESET}\n")
                return CommandResult()
            try:
                from ..files import extract_text_from_path

                context.pending_file_context = extract_text_from_path(inline_file)
                context.pending_file_name = inline_file.name
                print(
                    f"\n{UI.GREEN}Attached file: {inline_file.name} "
                    f"({len(context.pending_file_context):,} chars).{UI.RESET}"
                )
            except Exception as ex:
                print(f"\n{UI.RED}Failed to read attached file: {ex}{UI.RESET}\n")
                return CommandResult()

        cleaned_prompt, inline_img, page_spec = extract_inline_image(cleaned_prompt)
        if inline_img:
            if not load_image_or_pdf_into_context(inline_img, page_spec=page_spec, context=context):
                return CommandResult()

        if not export_path:
            ts = time.strftime("%Y%m%d_%H%M%S")
            export_path = Path(f"answer_{ts}.txt")

        return CommandResult(prompt_to_generate=cleaned_prompt, export_file=export_path)

    # No prompt provided: export the last assistant answer
    if not context.last_assistant_answer:
        print(f"\n{UI.YELLOW}No assistant answer available yet to export.{UI.RESET}\n")
        return CommandResult()

    if not export_path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        export_path = Path(f"answer_{ts}.txt")

    try:
        export_path.write_text(context.last_assistant_answer, encoding="utf-8")
        print(
            f"\n{UI.BOLD}{UI.GREEN}[Exported]{UI.RESET} "
            f"Last answer saved to: {export_path.resolve()}\n"
        )
    except Exception as ex:
        print(f"\n{UI.RED}Failed to export file: {ex}{UI.RESET}\n")

    return CommandResult()


def handle_export_chat(args_str: str, context: CommandContext) -> CommandResult:
    """Exports the entire conversation history transcript to a text file."""
    raw = args_str.strip()
    if raw:
        from ..parser import sanitize_file_path

        out_path = sanitize_file_path(raw)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = Path(f"chat_export_{ts}.txt")

    if not context.engine.history_messages:
        print(f"\n{UI.YELLOW}No conversation history to export.{UI.RESET}\n")
        return CommandResult()

    try:
        lines = []
        for msg in context.engine.history_messages:
            role = msg["role"].upper()
            content = msg["content"]
            lines.append(f"[{role}]\n{content}\n")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(
            f"\n{UI.BOLD}{UI.GREEN}[Exported]{UI.RESET} "
            f"Full chat history saved to: {out_path.resolve()}\n"
        )
    except Exception as ex:
        print(f"\n{UI.RED}Failed to export chat: {ex}{UI.RESET}\n")

    return CommandResult()

