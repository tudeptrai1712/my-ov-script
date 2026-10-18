"""
Slash command dispatch system for interactive OpenVINO GenAI chat.
"""

from .base import CommandContext, CommandHandler, CommandResult
from .router import CommandRouter

__all__ = [
    "CommandContext",
    "CommandHandler",
    "CommandResult",
    "CommandRouter",
]

