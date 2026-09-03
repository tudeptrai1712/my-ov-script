import datetime
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import CHAT_HISTORY_DIR


class ChatHistorySaver:
    """Manages recording and persistent saving of chat sessions to disk."""

    def __init__(
        self,
        model_name: str,
        device: str,
        context_length: int,
        reasoning_enabled: bool,
        history_dir: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.context_length = context_length
        self.reasoning_enabled = reasoning_enabled
        self.history_dir = history_dir or CHAT_HISTORY_DIR
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.start_time = datetime.datetime.now()
        self.session_file = self._create_session_file()
        self.turns: List[Dict[str, Any]] = []

        # Write initial session header
        self.save()

    def _create_session_file(self) -> Path:
        safe_model = re.sub(r"[^\w\-.]", "_", self.model_name)
        timestamp = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        base_name = f"{timestamp}_{safe_model}"
        file_path = self.history_dir / f"{base_name}.md"

        counter = 1
        while file_path.exists():
            file_path = self.history_dir / f"{base_name}_{counter}.md"
            counter += 1

        return file_path

    def add_turn(
        self,
        user_prompt: str,
        assistant_response: str,
        metrics: Optional[Dict[str, Any]] = None,
        reasoning_used: bool = False,
    ) -> None:
        """Adds a completed conversation turn and immediately updates the file on disk."""
        turn_time = datetime.datetime.now()
        self.turns.append({
            "timestamp": turn_time.strftime("%H:%M:%S"),
            "user": user_prompt,
            "assistant": assistant_response,
            "metrics": metrics or {},
            "reasoning_used": reasoning_used,
        })
        self.save()

    def save(self) -> None:
        """Writes the current conversation transcript to the markdown file."""
        lines: List[str] = []

        # YAML Frontmatter
        lines.append("---")
        lines.append(f"started_at: '{self.start_time.isoformat()}'")
        lines.append(f"model: '{self.model_name}'")
        lines.append(f"device: '{self.device}'")
        lines.append(f"context_length: {self.context_length}")
        lines.append(f"reasoning_mode: {str(self.reasoning_enabled).lower()}")
        lines.append(f"total_turns: {len(self.turns)}")
        lines.append("---\n")

        # Document Header
        lines.append(f"# Chat Session: {self.model_name}\n")
        lines.append(f"- **Date**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **Device**: {self.device}")
        lines.append(f"- **Context Limit**: {self.context_length:,} tokens")
        lines.append(
            f"- **Reasoning Mode**: {'Enabled' if self.reasoning_enabled else 'Disabled'}\n"
        )
        lines.append("---\n")

        if not self.turns:
            lines.append("*(Session started - waiting for conversation...)*\n")
        else:
            for idx, turn in enumerate(self.turns, start=1):
                ts = turn.get("timestamp", "")
                reason_tag = " `[Thinking Mode]`" if turn.get("reasoning_used") else ""
                lines.append(f"### Turn {idx} ({ts})\n")
                lines.append(f"**User:**\n\n{turn['user']}\n")
                lines.append(f"**AI{reason_tag}:**\n\n{turn['assistant']}\n")

                m = turn.get("metrics", {})
                if m:
                    parts = []
                    if "input_tokens" in m and m["input_tokens"]:
                        parts.append(f"Input: {m['input_tokens']:,} tok")
                    if "output_tokens" in m and m["output_tokens"]:
                        parts.append(f"Output: {m['output_tokens']:,} tok")
                    if "throughput" in m and m.get("throughput", 0) > 0:
                        parts.append(f"Speed: {m['throughput']:.2f} tok/s")
                    if "ttft" in m and m.get("ttft", 0) > 0:
                        parts.append(f"TTFT: {m['ttft']:.0f} ms")
                    if parts:
                        lines.append(f"> *Metrics: {' | '.join(parts)}*\n")

                lines.append("---\n")

        content = "\n".join(lines)
        try:
            self.session_file.write_text(content, encoding="utf-8")
        except Exception:
            pass

    def reset(self) -> None:
        """Starts a new session file after conversation clear."""
        self.start_time = datetime.datetime.now()
        self.session_file = self._create_session_file()
        self.turns = []
        self.save()

