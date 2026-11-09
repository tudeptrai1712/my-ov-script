import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
from unittest.mock import MagicMock

# Mock third-party native libraries if running in an environment without them
for mod in ("openvino", "openvino_genai", "PIL", "psutil", "fastapi", "pydantic"):
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

from ovchat.engine import GenerationResult, InferenceEngine
from ovchat.history import ChatHistorySaver
from ovchat.metadata import read_model_metadata
from ovchat.reasoning import (
    ALL_TAGS,
    CONTENT_START_TAGS,
    IGNORED_CONTROL_TAGS,
    THINKING_START_TAGS,
    StreamTokenParser,
    split_reasoning_and_content,
)


class TestReasoningParser(unittest.TestCase):
    def test_gpt_oss_reasoning_on(self):
        raw = (
            "<|start|>assistant<|channel|>analysis<|message|>"
            "Let's think through this step by step. 2 + 2 = 4."
            "<|end|><|start|>assistant<|channel|>final<|message|>"
            "The answer is 4.<|return|>"
        )
        thought, content = split_reasoning_and_content(raw)
        self.assertEqual(thought, "Let's think through this step by step. 2 + 2 = 4.")
        self.assertEqual(content, "The answer is 4.")

    def test_gpt_oss_reasoning_off(self):
        raw = "<|channel|>final<|message|>The answer is 4.<|return|>"
        thought, content = split_reasoning_and_content(raw)
        self.assertIsNone(thought)
        self.assertEqual(content, "The answer is 4.")

    def test_gpt_oss_detokenized_bare_words_on(self):
        raw = 'analysisThinking step 1.assistantfinalfour'
        thought, content = split_reasoning_and_content(raw)
        self.assertEqual(thought, "Thinking step 1.")
        self.assertEqual(content, "four")

    def test_gpt_oss_detokenized_bare_words_off(self):
        raw = 'finalfour'
        thought, content = split_reasoning_and_content(raw)
        self.assertIsNone(thought)
        self.assertEqual(content, "four")

    def test_gemma_4_reasoning_on(self):
        raw = (
            "<|channel>thought\n"
            "Analyzing user request...\n"
            "<channel|>"
            "Here is the answer.<turn|>"
        )
        thought, content = split_reasoning_and_content(raw)
        self.assertEqual(thought, "Analyzing user request...")
        self.assertEqual(content, "Here is the answer.")

    def test_gemma_4_reasoning_off(self):
        raw = "Here is the answer without thinking.<turn|>"
        thought, content = split_reasoning_and_content(raw)
        self.assertIsNone(thought)
        self.assertEqual(content, "Here is the answer without thinking.")

    def test_deepseek_reasoning_on(self):
        raw = (
            "<think>\n"
            "Thinking deeply about the universe\n"
            "</think>\n"
            "The answer is 42."
        )
        thought, content = split_reasoning_and_content(raw)
        self.assertEqual(thought, "Thinking deeply about the universe")
        self.assertEqual(content, "The answer is 42.")

    def test_mathematical_less_than_preservation(self):
        raw = "Formula: x < 5 and y < 10 and z < 15."
        thought, content = split_reasoning_and_content(raw)
        self.assertIsNone(thought)
        self.assertEqual(content, "Formula: x < 5 and y < 10 and z < 15.")

    def test_streaming_token_splits(self):
        streamed_thoughts = []
        streamed_content = []
        parser = StreamTokenParser(
            on_content=lambda s: streamed_content.append(s),
            on_thought=lambda s: streamed_thoughts.append(s),
        )

        chunks = [
            "<|start|>",
            "assistant<|channel|>",
            "analysis<|message|>",
            "Think",
            "ing step 1. ",
            "<|end|><|start|>ass",
            "istant<|channel|>final<|me",
            "ssage|>Final ",
            "response here.",
            "<|return|>",
        ]
        for ch in chunks:
            parser.feed(ch)
        parser.finish()

        self.assertEqual("".join(streamed_thoughts), "Thinking step 1. ")
        self.assertEqual("".join(streamed_content), "Final response here.")


class TestMetadataReasoningDetection(unittest.TestCase):
    def test_gpt_oss_metadata(self):
        model_path = Path(r"D:\AI models\openvino-genai\gpt-oss-20b-int4-ov")
        if model_path.exists():
            meta = read_model_metadata(model_path)
            self.assertTrue(
                meta.supports_reasoning,
                "gpt-oss-20b-int4-ov should be detected as supporting reasoning",
            )

    def test_gemma_4_metadata(self):
        model_path = Path(r"D:\AI models\openvino-genai\gemma-4-26b-a4b-it-int4-ov")
        if model_path.exists():
            meta = read_model_metadata(model_path)
            self.assertTrue(
                meta.supports_reasoning,
                "gemma-4-26b-a4b-it-int4-ov should be detected as supporting reasoning",
            )


class TestEngineReasoningContext(unittest.TestCase):
    def test_extra_context_toggling(self):
        engine = InferenceEngine(enable_reasoning=False)
        ctx_disabled = engine._build_extra_context()
        self.assertEqual(ctx_disabled["enable_thinking"], 0)
        self.assertTrue(all(isinstance(v, int) for v in ctx_disabled.values()))

        engine.set_reasoning(True)
        ctx_enabled = engine._build_extra_context()
        self.assertEqual(ctx_enabled["enable_thinking"], 1)
        self.assertTrue(all(isinstance(v, int) for v in ctx_enabled.values()))

        engine.set_reasoning(False)
        ctx_disabled2 = engine._build_extra_context()
        self.assertEqual(ctx_disabled2["enable_thinking"], 0)
        self.assertTrue(all(isinstance(v, int) for v in ctx_disabled2.values()))

    def test_extra_context_custom_effort(self):
        engine = InferenceEngine(enable_reasoning=True)
        ctx_override_off = engine._build_extra_context({"reasoning_effort": "none"})
        self.assertEqual(ctx_override_off["reasoning_effort"], 0)

        ctx_override_on = engine._build_extra_context({"reasoning_effort": "high"})
        self.assertEqual(ctx_override_on["reasoning_effort"], 1)
        self.assertTrue(all(isinstance(v, int) for v in ctx_override_on.values()))

    def test_engine_generate_with_reasoning_stream(self):
        engine = InferenceEngine(enable_reasoning=True)
        # Mock pipeline
        mock_pipe = MagicMock()
        def fake_generate(chat_hist, **kwargs):
            streamer = kwargs.get("streamer")
            chunks = [
                "<|channel|>analysis<|message|>",
                "Let's think: 1+1=2.",
                "<|end|><|start|>assistant<|channel|>final<|message|>",
                "The answer is 2.",
                "<|return|>",
            ]
            for c in chunks:
                streamer(c)
        mock_pipe.generate = fake_generate
        engine.pipe = mock_pipe

        thought_streamed = []
        content_streamed = []
        result = engine.generate(
            "What is 1+1?",
            streaming_callback=lambda s: content_streamed.append(s),
            reasoning_callback=lambda s: thought_streamed.append(s),
        )

        self.assertEqual(result.text, "The answer is 2.")
        self.assertEqual(result.reasoning, "Let's think: 1+1=2.")
        self.assertEqual("".join(thought_streamed), "Let's think: 1+1=2.")
        self.assertEqual("".join(content_streamed), "The answer is 2.")
        # Ensure clean content stored in history
        self.assertEqual(engine.history_messages[-1]["content"], "The answer is 2.")

    def test_engine_generate_without_reasoning_stream(self):
        engine = InferenceEngine(enable_reasoning=False)
        mock_pipe = MagicMock()
        def fake_generate(chat_hist, **kwargs):
            streamer = kwargs.get("streamer")
            chunks = [
                "<|channel|>final<|message|>",
                "The answer is 2.",
                "<|return|>",
            ]
            for c in chunks:
                streamer(c)
        mock_pipe.generate = fake_generate
        engine.pipe = mock_pipe

        thought_streamed = []
        content_streamed = []
        result = engine.generate(
            "What is 1+1?",
            streaming_callback=lambda s: content_streamed.append(s),
            reasoning_callback=lambda s: thought_streamed.append(s),
        )

        self.assertEqual(result.text, "The answer is 2.")
        self.assertIsNone(result.reasoning)
        self.assertEqual(thought_streamed, [])
        self.assertEqual("".join(content_streamed), "The answer is 2.")
        self.assertEqual(engine.history_messages[-1]["content"], "The answer is 2.")


class TestHistorySaverDetails(unittest.TestCase):
    def test_markdown_details_block(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            saver = ChatHistorySaver(
                model_name="test-model",
                device="GPU",
                history_dir=Path(tmpdir),
                reasoning_enabled=True,
            )
            saver.record_turn(
                user_prompt="Hello",
                assistant_response="Hi there!",
                reasoning="Let me think of a greeting.",
                reasoning_used=True,
            )
            content = saver.session_file.read_text(encoding="utf-8")
            self.assertIn("<details>", content)
            self.assertIn("Thinking Process", content)
            self.assertIn("Let me think of a greeting.", content)
            self.assertIn("Hi there!", content)


if __name__ == "__main__":
    unittest.main()
