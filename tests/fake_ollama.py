"""A real local HTTP server that behaves like Ollama, scripted by each test."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Reply:
    def __init__(self, text: str = "", status: int = 200, delay: float = 0.0) -> None:
        self.text = text
        self.status = status
        self.delay = delay


class FakeOllama:
    def __init__(self, respond: Callable[[str], Reply | str], models=("model-a",), capabilities=("completion",)) -> None:
        self.respond = respond
        self.models = list(models)
        self.capabilities = list(capabilities)
        self.prompts: list[str] = []
        self.requests: list[dict] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status: int, body: dict) -> None:
                data = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except OSError:
                    pass  # the client gave up waiting

            def do_GET(self):
                self._send(200, {"models": [{"name": name} for name in fake.models]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/show":
                    self._send(200, {"capabilities": fake.capabilities})
                    return
                fake.requests.append(body)
                if "prompt" not in body:  # an unload request has no prompt
                    self._send(200, {"response": ""})
                    return
                fake.prompts.append(body["prompt"])
                reply = fake.respond(body["prompt"])
                if isinstance(reply, str):
                    reply = Reply(reply)
                time.sleep(reply.delay)
                self._send(reply.status, {"response": reply.text})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
