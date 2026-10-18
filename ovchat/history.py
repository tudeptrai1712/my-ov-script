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
        context_length: int = 32768,
        reasoning_enabled: bool = False,
        model_type: Optional[str] = None,
        precision: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        supports_reasoning: Optional[bool] = None,
        history_dir: Optional[Path] = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.context_length = context_length
        self.reasoning_enabled = reasoning_enabled
        self.model_type = model_type or "LLM"
        self.precision = precision or "INT4"
        self.max_new_tokens = max_new_tokens
        self.supports_reasoning = supports_reasoning
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

    def record_turn(
        self,
        user_prompt: str,
        assistant_response: str,
        ttft_sec: float = 0.0,
        generation_time_sec: float = 0.0,
        tokens_generated: int = 0,
        tokens_per_sec: float = 0.0,
        context_tokens_used: int = 0,
        attached_file: Optional[str] = None,
        attached_image: Optional[str] = None,
        reasoning_used: bool = False,
        **kwargs: Any,
    ) -> None:
        """Records a completed turn with detailed metrics and saves to disk."""
        metrics: Dict[str, Any] = {
            "input_tokens": context_tokens_used,
            "output_tokens": tokens_generated,
            "throughput": tokens_per_sec,
            "ttft": ttft_sec * 1000.0 if ttft_sec > 0 else 0,
            "generation_time_sec": generation_time_sec,
        }
        if attached_file:
            metrics["attached_file"] = attached_file
        if attached_image:
            metrics["attached_image"] = attached_image
        metrics.update(kwargs)

        self.add_turn(
            user_prompt=user_prompt,
            assistant_response=assistant_response,
            metrics=metrics,
            reasoning_used=reasoning_used,
        )

    def add_turn(
        self,
        user_prompt: str,
        assistant_response: str,
        metrics: Optional[Dict[str, Any]] = None,
        reasoning_used: bool = False,
        **kwargs: Any,
    ) -> None:
        """Adds a completed conversation turn and immediately updates the file on disk."""
        turn_time = datetime.datetime.now()
        turn_metrics = dict(metrics or {})
        turn_metrics.update(kwargs)

        self.turns.append({
            "timestamp": turn_time.strftime("%H:%M:%S"),
            "user": user_prompt,
            "assistant": assistant_response,
            "metrics": turn_metrics,
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
        lines.append(f"model_type: '{self.model_type}'")
        lines.append(f"precision: '{self.precision}'")
        lines.append(f"device: '{self.device}'")
        lines.append(f"context_length: {self.context_length}")
        if self.max_new_tokens:
            lines.append(f"max_new_tokens: {self.max_new_tokens}")
        lines.append(f"reasoning_mode: {str(self.reasoning_enabled).lower()}")
        lines.append(f"total_turns: {len(self.turns)}")
        lines.append("---\n")

        # Document Header
        lines.append(f"# Chat Session: {self.model_name}\n")
        lines.append(f"- **Date**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **Type**: {self.model_type} ({self.precision.upper()})")
        lines.append(f"- **Device**: {self.device}")
        lines.append(f"- **Context Limit**: {self.context_length:,} tokens")
        if self.max_new_tokens:
            lines.append(f"- **Max Output**: {self.max_new_tokens:,} tokens")
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
                    if m.get("attached_file"):
                        parts.append(f"File: {m['attached_file']}")
                    if m.get("attached_image"):
                        parts.append(f"Image: {m['attached_image']}")
                    
                    inp = m.get("input_tokens") or m.get("context_tokens_used")
                    if inp:
                        parts.append(f"Input: {inp:,} tok")
                    
                    outp = m.get("output_tokens") or m.get("tokens_generated")
                    if outp:
                        parts.append(f"Output: {outp:,} tok")
                    
                    speed = m.get("throughput") or m.get("tokens_per_sec", 0)
                    if speed and speed > 0:
                        parts.append(f"Speed: {speed:.2f} tok/s")
                    
                    ttft = m.get("ttft")
                    if ttft and ttft > 0:
                        parts.append(f"TTFT: {ttft:.0f} ms")
                    elif m.get("ttft_sec", 0) > 0:
                        parts.append(f"TTFT: {m['ttft_sec'] * 1000:.0f} ms")
                    
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

