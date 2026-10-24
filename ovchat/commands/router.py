from typing import Dict, List, Optional, Tuple, Union

from .attachments import handle_file, handle_image
from .base import CommandContext, CommandHandler, CommandResult
from .export import handle_export, handle_export_chat
from .session import (
    handle_clear,
    handle_hetero,
    handle_info,
    handle_load,
    handle_moe,
    handle_quit,
    handle_think,
    handle_unload,
)


class CommandRouter:
    """Dispatches user slash commands to registered handlers."""

    def __init__(self):
        self.registry: Dict[str, CommandHandler] = {}
        self._register_default_commands()

    def register(
        self,
        command_or_aliases: Union[str, List[str]],
        handler: CommandHandler,
    ) -> None:
        """Registers a handler for one or more command aliases."""
        if isinstance(command_or_aliases, str):
            aliases = [command_or_aliases]
        else:
            aliases = command_or_aliases

        for alias in aliases:
            cleaned = alias.strip().lower()
            if not cleaned.startswith("/"):
                cleaned = "/" + cleaned
            self.registry[cleaned] = handler

    def _register_default_commands(self) -> None:
        self.register(["/clear", "/reset"], handle_clear)
        self.register(["/info", "/status"], handle_info)
        self.register(["/think", "/reason", "/reasoning"], handle_think)
        self.register(["/moe", "/offload"], handle_moe)
        self.register(["/hetero", "/pipeline"], handle_hetero)
        self.register(["/unload", "/eject"], handle_unload)
        self.register(["/load", "/model", "/switch"], handle_load)
        self.register(["/quit", "/exit", "/q"], handle_quit)
        self.register(["/file", "/doc", "/attach"], handle_file)
        self.register(
            ["/image", "/img", "/pic", "/pdf-image", "/pdfimg"],
            handle_image,
        )
        self.register(["/export", "/save", "/export-answer"], handle_export)
        self.register(["/export-chat", "/save-chat"], handle_export_chat)

    def dispatch(
        self,
        raw_input: str,
        context: CommandContext,
    ) -> Optional[CommandResult]:
        """
        Parses raw user input.
        If it begins with a registered slash command, dispatches to the handler and returns CommandResult.
        Otherwise, returns None (prompt should be treated as regular chat message).
        """
        stripped = raw_input.strip()
        if not stripped.startswith("/"):
            return None

        # Split command token from arguments
        parts = stripped.split(maxsplit=1)
        cmd_token = parts[0].lower()
        args_str = parts[1] if len(parts) > 1 else ""

        handler = self.registry.get(cmd_token)
        if handler:
            return handler(args_str, context)

        # Unrecognized slash command: let caller decide or warn
        return None
