"""
Reasoning and control-token streaming parser for OpenVINO GenAI models.
Detects and separates thinking/reasoning CoT streams from final answers
across OpenAI Harmony (e.g. gpt-oss-20b), Google Gemma 4, DeepSeek, and Qwen models.
"""

from typing import Callable, List, Optional, Tuple

THINKING_START_TAGS: List[str] = [
    "<|start|>assistant<|channel|>analysis<|message|>",
    "<|channel|>analysis<|message|>",
    "<|channel>thought\n",
    "<|channel>thought",
    "<think>",
    "<thought>",
    "<|thought|>",
]

CONTENT_START_TAGS: List[str] = [
    "<|end|><|start|>assistant<|channel|>final<|message|>",
    "<|start|>assistant<|channel|>final<|message|>",
    "<|channel|>final<|message|>",
    "<channel|>",
    "</think>",
    "</thought>",
    "<|/thought|>",
]

IGNORED_CONTROL_TAGS: List[str] = [
    "<|channel|>commentary json<|message|>",
    "<|channel|>commentary<|message|>",
    "<|start|>assistant",
    "<|start|>user",
    "<|start|>system",
    "<|start|>",
    "<|end|>",
    "<|return|>",
    "<|call|>",
    "<turn|>",
    "<|turn>model\n",
    "<|turn>model",
    "<|turn>user\n",
    "<|turn>user",
    "<|turn>",
    "<turn>",
    "<|im_start|>",
    "<|im_end|>",
]

# Sorted by descending length so compound tags match before substrings
ALL_TAGS: List[str] = sorted(
    THINKING_START_TAGS + CONTENT_START_TAGS + IGNORED_CONTROL_TAGS,
    key=len,
    reverse=True,
)


class StreamTokenParser:
    """
    Stateful incremental streamer that:
      1. Detects reasoning tokens vs final response tokens.
      2. Buffers potential partial delimiter prefixes to prevent leaking special tags.
      3. Dispatches clean tokens to on_thought and on_content callbacks.
    """

    def __init__(
        self,
        on_content: Optional[Callable[[str], None]] = None,
        on_thought: Optional[Callable[[str], None]] = None,
    ):
        self.on_content = on_content
        self.on_thought = on_thought
        self.mode: str = "content"  # "content" or "thought"
        self.buffer: str = ""
        self.thought_chunks: List[str] = []
        self.content_chunks: List[str] = []
        self._started: bool = False

    def feed(self, chunk: str) -> None:
        """Processes incoming streamed token chunk."""
        if not chunk:
            return
        self.buffer += chunk
        self._process_buffer(eof=False)

    def finish(self) -> None:
        """Flushes remaining buffered tokens at end of generation."""
        self._process_buffer(eof=True)

    def _emit(self, text: str) -> None:
        if not text:
            return
        if self.mode == "thought":
            self.thought_chunks.append(text)
            if self.on_thought:
                self.on_thought(text)
        else:
            self.content_chunks.append(text)
            if self.on_content:
                self.on_content(text)

    def _process_buffer(self, eof: bool) -> None:
        while self.buffer:
            # 1. Handle bare channel words at generation onset (from OpenVINO detokenizer)
            if not self._started:
                # If buffer starts with 'analysis'
                if self.buffer.startswith("analysis"):
                    self.buffer = self.buffer[len("analysis"):]
                    self.mode = "thought"
                    self._started = True
                    continue
                # If buffer starts with 'assistantfinal' or 'final'
                elif self.buffer.startswith("assistantfinal"):
                    self.buffer = self.buffer[len("assistantfinal"):]
                    self.mode = "content"
                    self._started = True
                    continue
                elif self.buffer.startswith("final"):
                    self.buffer = self.buffer[len("final"):]
                    self.mode = "content"
                    self._started = True
                    continue
                elif any(w.startswith(self.buffer) for w in ("analysis", "assistantfinal", "final")):
                    if not eof:
                        break  # Wait for full prefix
                    self._started = True
                else:
                    self._started = True

            # 2. Check for transition to final channel while in thought mode
            if self.mode == "thought":
                # Check for bare assistantfinal or final transition
                if self.buffer.startswith("assistantfinal"):
                    self.buffer = self.buffer[len("assistantfinal"):]
                    self.mode = "content"
                    continue
                elif self.buffer.startswith("<|channel|>final") or self.buffer.startswith("<channel|>"):
                    pass  # Will be handled by ALL_TAGS matching below
                elif any("assistantfinal".startswith(self.buffer) for _ in [0]):
                    if not eof:
                        break

            # 3. Delimiter tag search
            first_lt = self.buffer.find("<")
            if first_lt == -1:
                # Check if 'assistantfinal' exists in buffer while in thought mode
                if self.mode == "thought" and "assistantfinal" in self.buffer:
                    idx = self.buffer.find("assistantfinal")
                    self._emit(self.buffer[:idx])
                    self.buffer = self.buffer[idx + len("assistantfinal"):]
                    self.mode = "content"
                    continue

                self._emit(self.buffer)
                self.buffer = ""
                break

            # If there is normal text before '<', emit it immediately
            if first_lt > 0:
                pre = self.buffer[:first_lt]
                if self.mode == "thought" and "assistantfinal" in pre:
                    idx = pre.find("assistantfinal")
                    self._emit(pre[:idx])
                    self.buffer = self.buffer[idx + len("assistantfinal"):]
                    self.mode = "content"
                    continue

                self._emit(pre)
                self.buffer = self.buffer[first_lt:]

            # Buffer starts with '<'. Check against known delimiter tags.
            matched_tag = None
            for tag in ALL_TAGS:
                if self.buffer.startswith(tag):
                    matched_tag = tag
                    break

            if matched_tag:
                # If buffer matches matched_tag, check if buffer is also a prefix of a LONGER tag
                longer_prefix_exists = any(
                    tag.startswith(self.buffer) and len(tag) > len(matched_tag)
                    for tag in ALL_TAGS
                )
                if longer_prefix_exists and not eof:
                    break

                # Consume matched tag and transition state
                self.buffer = self.buffer[len(matched_tag):]
                if matched_tag in THINKING_START_TAGS:
                    self.mode = "thought"
                elif matched_tag in CONTENT_START_TAGS:
                    self.mode = "content"
                continue

            # Check if self.buffer is a prefix of any known tag
            is_prefix = any(tag.startswith(self.buffer) for tag in ALL_TAGS)
            if is_prefix and not eof:
                break

            # Emit '<' and advance
            self._emit(self.buffer[0])
            self.buffer = self.buffer[1:]


def split_reasoning_and_content(full_text: str) -> Tuple[Optional[str], str]:
    """
    Parses a complete model response string, separating thoughts from final content
    and stripping all control tags.
    """
    if not full_text:
        return None, ""

    parser = StreamTokenParser()
    parser.feed(full_text)
    parser.finish()

    thought_str = "".join(parser.thought_chunks).strip() if parser.thought_chunks else None
    content_str = "".join(parser.content_chunks).strip()

    # Safety fallback: if all text was filtered out or no tags present, preserve content
    if not content_str and not thought_str:
        content_str = full_text.strip()

    return thought_str, content_str
