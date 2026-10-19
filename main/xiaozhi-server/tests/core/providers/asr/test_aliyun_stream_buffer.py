"""阿里云流式 ASR 的音频完整性回归测试，不访问云端服务。"""
import asyncio
import json
import sys
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

import websockets


# 可直接运行此文件，避免测试日志初始化读取生产配置。
_logger_module = types.ModuleType("config.logger")
_logger_module.setup_logging = Mock(return_value=Mock())
with patch.dict(sys.modules, {"config.logger": _logger_module}):
    from core.providers.asr import aliyun_stream

ASRProvider = aliyun_stream.ASRProvider


class FakeSocket:
    def __init__(self, ready_immediately=False):
        self.messages = []
        self.responses = asyncio.Queue()
        self.ready_immediately = ready_immediately
        self.send_hook = None

    async def send(self, message):
        if isinstance(message, bytes) and self.send_hook:
            await self.send_hook(message)
        self.messages.append(message)
        if isinstance(message, str):
            name = json.loads(message)["header"]["name"]
            if name == "StartTranscription" and self.ready_immediately:
                self.ready()
        # 强制让缓存补发和实时接收有机会交错执行。
        await asyncio.sleep(0)

    async def recv(self):
        message = await self.responses.get()
        if message is None:
            raise websockets.ConnectionClosedOK(None, None)
        return message

    def ready(self):
        self.responses.put_nowait(json.dumps({
            "header": {"name": "TranscriptionStarted", "status": 20000000}
        }))

    async def close(self):
        self.responses.put_nowait(None)

    @property
    def audio(self):
        return [message for message in self.messages if isinstance(message, bytes)]


class AliyunStreamBufferTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.provider = ASRProvider({"appkey": "test", "token": "test"}, True)
        self.conn = types.SimpleNamespace(
            asr_audio=[], client_listen_mode="auto", client_have_voice=True,
            client_voice_stop=False, stop_event=asyncio.Event(), logger=Mock(),
            asr=self.provider,
        )
        self.conn.reset_audio_states = self.conn.asr_audio.clear
        self.socket = FakeSocket()
        self.connect = patch.object(
            aliyun_stream.websockets, "connect",
            new=AsyncMock(side_effect=lambda *a, **kw: self.socket),
        )
        self.connect.start()

    async def asyncTearDown(self):
        self.conn.stop_event.set()
        task = self.provider.forward_task
        await self.provider.close()
        if task:
            await asyncio.wait_for(task, 1)
        self.connect.stop()

    async def wait_until(self, condition):
        async def wait():
            while not condition():
                await asyncio.sleep(0)
        await asyncio.wait_for(wait(), 1)

    async def feed(self, frames, have_voice=True):
        for frame in frames:
            await self.provider.receive_audio(self.conn, frame, have_voice)

    async def test_delayed_ready_keeps_all_frames_not_only_last_ten(self):
        frames = [bytes([i]) * 1920 for i in range(30)]
        await self.feed(frames)
        self.assertEqual(self.socket.audio, [])
        self.socket.ready()
        await self.wait_until(lambda: len(self.socket.audio) >= len(frames))
        self.assertEqual(self.socket.audio, frames)

    async def test_preserves_audio_before_vad_trigger(self):
        self.conn.client_have_voice = False
        pre_roll = [b"before-1", b"before-2", b"before-3"]
        await self.feed(pre_roll, have_voice=False)
        self.conn.client_have_voice = True
        await self.feed([b"speech-start"])
        self.socket.ready()
        expected = pre_roll + [b"speech-start"]
        await self.wait_until(lambda: len(self.socket.audio) >= len(expected))
        self.assertEqual(self.socket.audio, expected)

    async def test_fast_ready_does_not_duplicate_trigger_frame(self):
        self.socket.ready_immediately = True
        await self.feed([b"first"])
        await self.wait_until(lambda: self.provider.server_ready)
        await self.feed([b"second"])
        self.assertEqual(self.socket.audio, [b"first", b"second"])

    async def test_live_audio_cannot_overtake_cached_audio(self):
        frames = [bytes([i]) for i in range(20)]
        await self.feed(frames)
        sending = asyncio.Event()
        release = asyncio.Event()

        async def block_first(frame):
            if frame == frames[0]:
                sending.set()
                await release.wait()

        self.socket.send_hook = block_first
        self.socket.ready()
        await asyncio.wait_for(sending.wait(), 1)
        live = asyncio.create_task(self.feed([b"live-1", b"live-2"]))
        await asyncio.sleep(0)
        release.set()
        await asyncio.wait_for(live, 1)
        self.assertEqual(self.socket.audio, frames + [b"live-1", b"live-2"])

    async def test_manual_stop_waits_for_cached_audio(self):
        self.conn.client_listen_mode = "manual"
        frames = [b"manual-start", b"manual-end"]
        await self.feed(frames)
        await self.provider._send_stop_request()
        self.assertFalse(any(
            isinstance(m, str) and json.loads(m)["header"]["name"] == "StopTranscription"
            for m in self.socket.messages
        ))
        self.socket.ready()
        await self.wait_until(lambda: any(
            isinstance(m, str) and json.loads(m)["header"]["name"] == "StopTranscription"
            for m in self.socket.messages
        ))
        self.assertEqual(self.socket.audio, frames)
        self.assertEqual(json.loads(self.socket.messages[-1])["header"]["name"], "StopTranscription")

    async def test_cleanup_does_not_replay_previous_session(self):
        await self.feed([b"old-recording"])
        old_task = self.provider.forward_task
        await self.provider.close()
        await asyncio.wait_for(old_task, 1)
        self.assertEqual(self.conn.asr_audio, [])
        self.socket = FakeSocket()
        await self.feed([b"new-recording"])
        self.socket.ready()
        await self.wait_until(lambda: bool(self.socket.audio))
        self.assertEqual(self.socket.audio, [b"new-recording"])

    async def test_send_failure_clears_state_before_reconnect(self):
        self.socket.ready_immediately = True

        async def fail_send(frame):
            raise OSError("simulated disconnect")

        self.socket.send_hook = fail_send
        await self.feed([b"failed-session"])
        await self.wait_until(lambda: self.provider.asr_ws is None)
        await self.wait_until(lambda: not self.conn.asr_audio)
        self.socket = FakeSocket(ready_immediately=True)
        await self.feed([b"reconnected"])
        await self.wait_until(lambda: bool(self.socket.audio))
        self.assertEqual(self.socket.audio, [b"reconnected"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
