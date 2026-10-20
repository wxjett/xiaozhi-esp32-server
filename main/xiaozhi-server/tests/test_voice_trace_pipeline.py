"""真实 LLM/TTS 处理代码的离线计时集成测试，外部请求全部替换。"""
import asyncio
import json
import queue
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

# tests/core 是普通包，不能让它遮蔽项目根目录的 core 命名空间包。
_test_dir = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p).resolve() != _test_dir]
sys.path.insert(0, str(_test_dir.parent))

from core.utils.voice_trace import VoiceTrace, current_trace

_logger_module = types.ModuleType("config.logger")
_logger_module.setup_logging = Mock(return_value=Mock())
with patch.dict(sys.modules, {"config.logger": _logger_module}):
    from core.providers.llm.openai import openai as llm_module
    from core.providers.tts import aliyun as tts_module
    from core.providers.tts import base as tts_base
    from core.providers.tts.dto.dto import TTSMessageDTO, SentenceType, ContentType
    from core.handle import sendAudioHandle as send_module


class CaptureLogger:
    def __init__(self):
        self.lines = []

    def bind(self, **kwargs):
        return self

    def info(self, message):
        self.lines.append(message)

    warning = info


class Stream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        return iter(self.chunks)

    def close(self):
        self.closed = True


def chunk(content=None, tools=None):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(
        delta=types.SimpleNamespace(content=content, tool_calls=tools)
    )])


class PipelineTests(unittest.TestCase):
    def test_llm_stream_preserves_reply_and_records_request_lifecycle(self):
        provider = llm_module.LLMProvider({
            "api_key": "unit-test-key", "base_url": "https://example.test/v1", "model_name": "test-model",
        })
        stream = Stream([chunk(), chunk("你好"), chunk("呀")])
        provider.client = types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=Mock(return_value=stream))
        ))
        trace = VoiceTrace(CaptureLogger())
        token = current_trace.set(trace)
        try:
            self.assertEqual(list(provider.response("session", [{"role": "user", "content": "hello"}])), ["你好", "呀"])
        finally:
            current_trace.reset(token)
        for event in ("llm_request", "llm_first_response", "llm_done"):
            self.assertIn((event, None), trace.events)
        self.assertTrue(stream.closed)

    def test_function_call_counts_as_response_without_faking_speakable_text(self):
        provider = llm_module.LLMProvider({
            "api_key": "unit-test-key", "base_url": "https://example.test/v1", "model_name": "test-model",
        })
        tools = [types.SimpleNamespace(index=0, function=types.SimpleNamespace(name="direct_answer", arguments="{}"))]
        stream = Stream([chunk(tools=tools)])
        provider.client = types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=Mock(return_value=stream))
        ))
        trace = VoiceTrace(CaptureLogger())
        token = current_trace.set(trace)
        try:
            self.assertEqual(list(provider.response_with_functions("session", [], [])), [(None, tools)])
        finally:
            current_trace.reset(token)
        self.assertIn(("llm_first_response", None), trace.events)
        self.assertNotIn(("reply_text", None), trace.events)

    def test_tts_queue_carries_trace_across_threads_without_changing_audio(self):
        trace = VoiceTrace(CaptureLogger())
        provider = tts_module.TTSProvider({"appkey": "test", "token": "test", "voice": "xiaoyun"}, True)
        conn = types.SimpleNamespace(
            sentence_id="sentence-1", stop_event=threading.Event(), client_abort=False, sample_rate=24000,
        )
        provider.conn = conn
        provider.opus_encoder = None

        class StopOnLast(queue.Queue):
            def get(self, *args, **kwargs):
                message = super().get(*args, **kwargs)
                if message.sentence_type == SentenceType.LAST:
                    conn.stop_event.set()
                return message

        provider.tts_text_queue = StopOnLast()
        token = current_trace.set(trace)
        try:
            for kind, content in ((SentenceType.FIRST, None), (SentenceType.MIDDLE, "你好。"), (SentenceType.LAST, None)):
                provider.tts_text_queue.put(TTSMessageDTO(
                    "sentence-1", kind, ContentType.TEXT if content else ContentType.ACTION, content,
                ))
        finally:
            current_trace.reset(token)
        response = types.SimpleNamespace(
            status_code=200, headers={"Content-Type": "audio/wav"}, content=b"unchanged-audio",
        )

        def encode(data, **kwargs):
            self.assertEqual(data, response.content)
            kwargs["callback"](b"opus-packet")

        with patch.object(tts_module.requests, "post", return_value=response), patch.object(tts_base, "audio_bytes_to_data_stream", side_effect=encode):
            worker = threading.Thread(target=provider.tts_text_priority_thread)
            worker.start()
            worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        for event in ("reply_text", "tts_dequeued", "tts_segment", "tts_request", "tts_audio", "tts_conversion", "tts_segment_done"):
            self.assertIn((event, None), trace.events)
        items = list(provider.tts_audio_queue.queue)
        self.assertIn((SentenceType.MIDDLE, b"opus-packet", None, "sentence-1"), items)
        self.assertIsNone(current_trace.get())


class AudioSendTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_packet_timestamp_is_after_websocket_send(self):
        trace = VoiceTrace(CaptureLogger())
        started, release = asyncio.Event(), asyncio.Event()

        async def send(packet):
            started.set()
            await release.wait()

        conn = types.SimpleNamespace(conn_from_mqtt_gateway=False, websocket=types.SimpleNamespace(send=send))
        flow = {"voice_trace": trace, "packet_count": 0, "sequence": 0}
        task = asyncio.create_task(send_module._do_send_audio(conn, b"audio", flow))
        await asyncio.wait_for(started.wait(), 1)
        self.assertNotIn(("audio_sent_first", None), trace.events)
        release.set()
        await task
        self.assertIn(("audio_sent_first", None), trace.events)
        self.assertEqual(trace.output_frames, 1)

    async def test_summary_waits_for_send_queue_and_stop_message(self):
        trace = VoiceTrace(CaptureLogger())
        trace.audio_sent(12)

        async def wait_audio(conn):
            self.assertFalse(trace.finished)

        async def send(message):
            self.assertFalse(trace.finished)
            self.assertEqual(json.loads(message)["state"], "stop")

        conn = types.SimpleNamespace(
            sentence_id="s1", session_id="session", config={},
            voice_traces={"s1": trace}, clearSpeakStatus=Mock(),
            websocket=types.SimpleNamespace(send=send),
        )
        with patch.object(send_module, "_wait_for_audio_completion", side_effect=wait_audio):
            await send_module.send_tts_message(conn, "stop")
        self.assertTrue(trace.finished)
        self.assertEqual(conn.voice_traces, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
