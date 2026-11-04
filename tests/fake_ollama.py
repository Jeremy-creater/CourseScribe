"""A real local HTTP server that behaves like Ollama, scripted by each test."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Reply:
    def __init__(
        self,
        text: str = "",
        status: int = 200,
        delay: float = 0.0,
        raw: bytes | None = None,
        broken: str | None = None,
        fields: dict | None = None,
        close_service: bool = False,
    ) -> None:
        self.text = text
        self.status = status
        self.delay = delay
        self.raw = raw  # sent as the body as it is, instead of JSON
        # "no-reply": hang up after reading the request; "short-body": promise more bytes than are sent
        self.broken = broken
        self.fields = fields  # merged into a 200 answer, e.g. {"done_reason": "length"}
        # Ollama quits while working on this request: the service shuts itself down, so this
        # request is hung up on without an answer and later connections are refused.
        self.close_service = close_service


class FakeOllama:
    def __init__(
        self,
        respond: Callable[[str], Reply | str],
        models=("model-a",),
        capabilities=("completion",),
        tags_reply: Callable[[], Reply] | None = None,
        unload_reply: Callable[[], Reply] | None = None,
        show_reply: Callable[[], Reply] | None = None,
    ) -> None:
        self.respond = respond
        self.models = list(models)
        self.capabilities = list(capabilities)
        self.tags_reply = tags_reply
        self.unload_reply = unload_reply
        self.show_reply = show_reply  # None: /api/show always answers with the capabilities
        self.prompts: list[str] = []
        self.requests: list[dict] = []
        self.bodies: list[tuple[str, bytes]] = []  # (path, raw request bytes) of every POST
        self.log: list[tuple[float, str, str | None]] = []  # (time.perf_counter(), path, prompt)
        self._closing = threading.Event()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status: int, body: dict | bytes, missing: int = 0) -> None:
                data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Length", str(len(data) + missing))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except OSError:
                    pass  # the client gave up waiting

            def _reply(self, reply: Reply, normal: dict) -> None:
                if reply.close_service:
                    # Joined, so the port already refuses connections when this request is hung up on:
                    # a request sent again at once must find the service gone, not half closed.
                    closer = threading.Thread(target=fake.close, daemon=True)
                    closer.start()
                    closer.join()
                fake._closing.wait(reply.delay)
                if fake._closing.is_set():
                    return
                if reply.broken:
                    self.close_connection = True
                    if reply.broken == "short-body":
                        self._send(reply.status, json.dumps(normal).encode("utf-8"), missing=100)
                    return
                if reply.raw is not None:
                    self._send(reply.status, reply.raw)
                elif reply.status == 200:
                    self._send(200, {**normal, **(reply.fields or {})})
                else:
                    self._send(reply.status, {"error": reply.text})

            def do_GET(self):
                fake.log.append((time.perf_counter(), self.path, None))
                normal = {"models": [{"name": name} for name in fake.models]}
                self._reply(fake.tags_reply() if fake.tags_reply else Reply(), normal)

            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                fake.bodies.append((self.path, raw))
                body = json.loads(raw)
                fake.log.append((time.perf_counter(), self.path, body.get("prompt")))
                if self.path == "/api/show":
                    if fake.show_reply:
                        self._reply(fake.show_reply(), {"capabilities": fake.capabilities})
                        return
                    self._send(200, {"capabilities": fake.capabilities})
                    return
                fake.requests.append(body)
                if "prompt" not in body:  # an unload request has no prompt
                    self._reply(fake.unload_reply() if fake.unload_reply else Reply(), {"response": ""})
                    return
                fake.prompts.append(body["prompt"])
                reply = fake.respond(body["prompt"])
                if isinstance(reply, str):
                    reply = Reply(reply)
                self._reply(reply, {"response": reply.text})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        # A short poll keeps close() quick; shutdown() waits for one poll.
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    @property
    def closed(self) -> bool:
        return self._closing.is_set()

    def close(self) -> None:
        self._closing.set()
        self.server.shutdown()
        self.server.server_close()
