from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from ..engine import InferenceEngine
from ..history import ChatHistorySaver


@dataclass
class CommandContext:
    """Shared execution context provided to slash command handlers."""

    engine: InferenceEngine
    history_saver: ChatHistorySaver
    chat_session: Any = None  # Reference to Chat session for callbacks like load_model
    pending_file_context: Optional[str] = None
    pending_file_name: Optional[str] = None
    pending_image_info: Optional[str] = None
    last_assistant_answer: Optional[str] = None


@dataclass
class CommandResult:
    """Outcome of executing a slash command."""

    handled: bool = True
    should_exit: bool = False
    prompt_to_generate: Optional[str] = None
    export_file: Optional[Path] = None


# Handler type signature: (args_str: str, context: CommandContext) -> CommandResult
CommandHandler = Callable[[str, CommandContext], CommandResult]

