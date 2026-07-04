import asyncio
import json


class ProgressEmitter:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._queue: asyncio.Queue = asyncio.Queue()

    def emit(self, event: str, data: dict):
        payload = {"event": event, "data": data}
        self._loop.call_soon_threadsafe(self._queue.put_nowait, payload)

    async def __aiter__(self):
        while True:
            msg = await self._queue.get()
            yield f"event: {msg['event']}\ndata: {json.dumps(msg['data'])}\n\n"
            if msg["event"] == "done" or msg["event"] == "error":
                break
