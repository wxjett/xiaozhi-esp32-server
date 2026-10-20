"""语音链路计时、线程上下文和凭证脱敏的离线测试。"""
import concurrent.futures
import json
import threading
import types
import unittest
from unittest.mock import patch

from config.logger import redact_log_message
from core.utils.voice_trace import (
    VoiceTrace, bind_sentence, current_trace, observe_audio, traced_chat,
)


class CaptureLogger:
    def __init__(self):
        self.lines = []

    def bind(self, **kwargs):
        return self

    def info(self, message):
        self.lines.append(message)

    warning = info

    def summary(self):
        return json.loads(next(line.split(" ", 1)[1] for line in self.lines if line.startswith("语音链路汇总 ")))


class VoiceTraceTests(unittest.TestCase):
    def test_summary_reports_separate_durations(self):
        log = CaptureLogger()
        trace = VoiceTrace(log, started=100)
        for event, at in (
            ("asr_connect", 100.1), ("asr_connected", 100.3),
            ("asr_start", 100.31), ("asr_ready", 100.5),
            ("asr_final", 101.8), ("chat_enter", 101.802),
            ("llm_request", 101.9), ("llm_first_response", 105.9),
            ("reply_text", 106), ("tts_dequeued", 106.01),
            ("tts_segment", 106.2), ("tts_request", 106.21), ("tts_audio", 106.91),
        ):
            trace.mark(event, at=at)
        trace.last_voice = 101
        with patch("core.utils.voice_trace.time.monotonic", return_value=106.95):
            trace.audio_sent(120)
        trace.finish("音频下发完成")
        summary = log.summary()
        self.assertEqual(summary["ASR建连毫秒"], 200)
        self.assertEqual(summary["最后人声帧到ASR结果毫秒_VAD估计"], 800)
        self.assertEqual(summary["LLM首响应毫秒"], 4000)
        self.assertEqual(summary["TTS首段排队毫秒"], 10)
        self.assertEqual(summary["首段分句等待毫秒"], 190)
        self.assertEqual(summary["TTS首段合成毫秒_含重试"], 700)
        self.assertEqual(summary["ASR结果到首包下发毫秒"], 5150)

    def test_missing_timing_is_null_and_finish_is_idempotent(self):
        log = CaptureLogger()
        trace = VoiceTrace(log)
        trace.finish("中断")
        trace.finish("音频下发完成")
        trace.mark("asr_final")
        self.assertEqual(len(log.lines), 1)
        self.assertIsNone(log.summary()["LLM首响应毫秒"])

    def test_audio_packet_counters_do_not_log_every_packet(self):
        log = CaptureLogger()
        trace = VoiceTrace(log)
        for _ in range(1000):
            trace.audio_sent(40)
        self.assertEqual(len(log.lines), 1)
        trace.finish("音频下发完成")
        self.assertEqual(log.summary()["下发Opus包数"], 1000)
        self.assertEqual(log.summary()["下发字节数"], 40000)

    def test_multiple_requests_keep_first_response_metric(self):
        log = CaptureLogger()
        trace = VoiceTrace(log, started=0)
        trace.mark("llm_request", span="llm1", at=1)
        trace.mark("llm_first_response", span="llm1", at=2)
        trace.mark("llm_request", span="llm2", at=3)
        trace.mark("llm_first_response", span="llm2", at=5)
        trace.finish("结束")
        self.assertEqual(log.summary()["LLM首响应毫秒"], 1000)
        self.assertEqual(len(log.lines), 5)

    def test_parallel_turns_do_not_share_thread_context(self):
        barrier = threading.Barrier(2)

        @traced_chat
        def work(conn, sentence):
            bind_sentence(conn, sentence)
            barrier.wait(timeout=2)
            return current_trace.get().id

        a = types.SimpleNamespace(voice_traces={})
        b = types.SimpleNamespace(voice_traces={})
        ta, tb = VoiceTrace(CaptureLogger()), VoiceTrace(CaptureLogger())
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            fa = pool.submit(work, a, "a", _voice_trace=ta)
            fb = pool.submit(work, b, "b", _voice_trace=tb)
            self.assertEqual(fa.result(timeout=3), ta.id)
            self.assertEqual(fb.result(timeout=3), tb.id)
        self.assertIs(a.voice_traces["a"], ta)
        self.assertIs(b.voice_traces["b"], tb)
        self.assertIsNone(current_trace.get())

    def test_input_history_excludes_idle_before_preroll(self):
        conn = types.SimpleNamespace(logger=CaptureLogger(), client_voice_stop=False)
        for i in range(20):
            observe_audio(conn, b"x" * 1920, False, received_at=10 + i * .06)
        observe_audio(conn, b"x" * 1920, True, received_at=11.2)
        self.assertAlmostEqual(conn.input_voice_trace.started, 10.6)
        self.assertEqual(conn.input_voice_trace.input_frames, 11)
        self.assertEqual(conn.input_voice_trace.input_bytes, 1920 * 11)

    def test_errors_do_not_include_provider_exception_body(self):
        log = CaptureLogger()
        trace = VoiceTrace(log)
        trace.error("LLM", ValueError("secret-key-in-provider-body"), 状态码=401)
        self.assertNotIn("secret-key-in-provider-body", log.lines[0])
        self.assertIn("ValueError", log.lines[0])

    def test_redacts_credentials_but_keeps_diagnostic_values(self):
        line = '''{'authorization': 'Bearer abc.def', "appkey":"private-app", "token":"private-token", "api_key":"private-api", "access_key_secret":"private-secret", "max_tokens":1024} https://example.test/?token=private-url&status=401'''
        safe = redact_log_message(line)
        for value in ("abc.def", "private-app", "private-token", "private-api", "private-secret", "private-url"):
            self.assertNotIn(value, safe)
        self.assertIn('"max_tokens":1024', safe)
        self.assertIn("status=401", safe)


if __name__ == "__main__":
    unittest.main(verbosity=2)
