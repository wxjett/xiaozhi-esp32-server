"""每轮语音链路的轻量计时；只记录元数据，不保存录音或凭证。"""
import contextvars
import functools
import json
import threading
import time
import uuid
from collections import deque


current_trace = contextvars.ContextVar("voice_trace", default=None)
current_segment = contextvars.ContextVar("voice_segment", default=None)

LABELS = {
    "audio_received": "收到本轮首个缓存音频包",
    "voice_start": "检测到人声",
    "voice_end": "VAD判断语音结束",
    "asr_connect": "开始连接ASR",
    "asr_connected": "ASR连接建立",
    "asr_start": "发送ASR开始请求",
    "asr_ready": "ASR服务就绪",
    "asr_audio_first": "首个音频包已发送给ASR",
    "asr_final": "ASR返回最终文字",
    "chat_dispatch": "文字提交对话线程",
    "chat_enter": "进入对话处理",
    "llm_request": "开始请求LLM",
    "llm_first_response": "LLM返回首个有效响应",
    "reply_text": "首段可播文字进入TTS队列",
    "llm_done": "LLM本次输出结束",
    "tts_dequeued": "开始处理首段回复文字",
    "tts_segment": "文字分段完成，开始处理语音片段",
    "tts_request": "发起TTS合成请求",
    "tts_audio": "TTS返回音频",
    "tts_conversion": "开始转换和编码音频",
    "tts_segment_done": "片段处理结束",
    "audio_queued_first": "首个音频包编码完成并入队",
    "audio_sent_first": "首个音频包已下发",
}


class VoiceTrace:
    def __init__(self, logger, started=None):
        self.id = uuid.uuid4().hex[:12]
        self.logger = logger.bind(tag="voice.trace")
        self.started = time.monotonic() if started is None else started
        self.events = {}
        self.last_events = {}
        self.spans = {}
        self.lock = threading.RLock()
        self.finished = False
        self.input_frames = self.input_bytes = 0
        self.output_frames = self.output_bytes = 0
        self.last_voice = self.last_audio_sent = None
        self.errors = 0

    def new_span(self, kind):
        with self.lock:
            self.spans[kind] = self.spans.get(kind, 0) + 1
            return f"{kind}{self.spans[kind]}"

    def mark(self, event, *, at=None, span=None, **fields):
        at = time.monotonic() if at is None else at
        with self.lock:
            key = (event, span)
            if self.finished or key in self.events:
                return
            self.events[key] = at
            self.events.setdefault((event, None), at)
            self.last_events[event] = at
            self.logger.info("语音链路 " + json.dumps({
                "轮次": self.id, "事件": LABELS.get(event, event),
                "事件码": event, "累计毫秒": round((at - self.started) * 1000, 1),
                **({"片段或请求": span} if span else {}), **fields,
            }, ensure_ascii=False))

    def error(self, stage, exc=None, **fields):
        # 不输出异常正文，服务商可能在正文中回显密钥。
        with self.lock:
            if not self.finished:
                self.errors += 1
                self.logger.warning("语音链路异常 " + json.dumps({
                    "轮次": self.id, "阶段": stage,
                    "异常类型": type(exc).__name__ if exc else None, **fields,
                }, ensure_ascii=False))

    def audio_sent(self, size):
        with self.lock:
            if self.finished:
                return
            self.output_frames += 1
            self.output_bytes += size
            self.last_audio_sent = time.monotonic()
            self.mark("audio_sent_first", at=self.last_audio_sent)

    def finish(self, status):
        with self.lock:
            if self.finished:
                return
            self.finished = True
            def elapsed(start, end):
                a = self.events.get((start, None)) if isinstance(start, str) else start
                b = self.events.get((end, None)) if isinstance(end, str) else end
                return round((b - a) * 1000, 1) if a is not None and b is not None else None
            summary = {
                "轮次": self.id, "状态": status,
                "ASR建连毫秒": elapsed("asr_connect", "asr_connected"),
                "ASR就绪等待毫秒": elapsed("asr_start", "asr_ready"),
                "ASR首包发出到结果毫秒_含讲话时长": elapsed("asr_audio_first", "asr_final"),
                "最后人声帧到ASR结果毫秒_VAD估计": elapsed(self.last_voice, "asr_final"),
                "ASR结果到对话入口毫秒": elapsed("asr_final", "chat_enter"),
                "LLM首响应毫秒": elapsed("llm_request", "llm_first_response"),
                "LLM到可播文字毫秒": elapsed("llm_request", "reply_text"),
                "LLM首次请求到末次输出结束毫秒": elapsed("llm_request", self.last_events.get("llm_done")),
                "TTS首段排队毫秒": elapsed("reply_text", "tts_dequeued"),
                "首段分句等待毫秒": elapsed("tts_dequeued", "tts_segment"),
                "TTS首段合成毫秒_含重试": elapsed("tts_request", "tts_audio"),
                "首段音频转换及入队毫秒": elapsed("tts_audio", "audio_queued_first"),
                "首个音频包发送排队毫秒": elapsed("audio_queued_first", "audio_sent_first"),
                "TTS首段返回到首包下发毫秒": elapsed("tts_audio", "audio_sent_first"),
                "ASR结果到首包下发毫秒": elapsed("asr_final", "audio_sent_first"),
                "最后人声帧到首包下发毫秒_VAD估计": elapsed(self.last_voice, "audio_sent_first"),
                "最后音频包下发累计毫秒": elapsed(self.started, self.last_audio_sent),
                "接收PCM帧数": self.input_frames, "接收音频毫秒": round(self.input_bytes / 32, 1),
                "下发Opus包数": self.output_frames, "下发字节数": self.output_bytes,
                "LLM请求次数": self.spans.get("llm", 0),
                "TTS请求次数": self.spans.get("tts_http", 0),
                "异常记录数": self.errors,
            }
            self.logger.info("语音链路汇总 " + json.dumps(summary, ensure_ascii=False))


def get_trace(conn, sentence_id=None):
    if sentence_id is not None:
        return getattr(conn, "voice_traces", {}).get(sentence_id)
    return current_trace.get() or getattr(conn, "input_voice_trace", None)


def observe_audio(conn, pcm, have_voice, received_at=None):
    received_at = time.monotonic() if received_at is None else received_at
    if not hasattr(conn, "voice_audio_history"):
        conn.voice_audio_history = deque(maxlen=10)
    trace = getattr(conn, "input_voice_trace", None)
    if trace is None and have_voice:
        history = list(conn.voice_audio_history)
        trace = VoiceTrace(conn.logger, history[0][0] if history else received_at)
        trace.input_frames = len(history)
        trace.input_bytes = sum(size for _, size in history)
        conn.input_voice_trace = trace
        trace.mark("audio_received", at=trace.started)
        trace.mark("voice_start", 接收队列等待毫秒=round((time.monotonic() - received_at) * 1000, 1))
    if trace:
        trace.input_frames += 1
        trace.input_bytes += len(pcm)
        if have_voice:
            trace.last_voice = received_at
        if getattr(conn, "client_voice_stop", False):
            trace.mark("voice_end")
    conn.voice_audio_history.append((received_at, len(pcm)))


def bind_sentence(conn, sentence_id):
    trace = current_trace.get()
    if trace:
        conn.voice_traces[sentence_id] = trace


def traced_chat(function):
    @functools.wraps(function)
    def wrapped(conn, *args, **kwargs):
        trace = kwargs.pop("_voice_trace", None) or get_trace(conn)
        token = current_trace.set(trace)
        if trace:
            trace.mark("chat_enter")
        try:
            return function(conn, *args, **kwargs)
        except Exception as exc:
            if trace:
                trace.error("对话处理", exc)
                trace.finish("异常")
            raise
        finally:
            current_trace.reset(token)
    return wrapped


def traced_tts_segment(function):
    @functools.wraps(function)
    def wrapped(provider, text, *args, **kwargs):
        trace = current_trace.get()
        if not trace:
            return function(provider, text, *args, **kwargs)
        segment = trace.new_span("tts")
        token = current_segment.set(segment)
        trace.mark("tts_segment", span=segment, 字符数=len(text))
        try:
            return function(provider, text, *args, **kwargs)
        finally:
            trace.mark("tts_segment_done", span=segment)
            current_segment.reset(token)
    return wrapped
