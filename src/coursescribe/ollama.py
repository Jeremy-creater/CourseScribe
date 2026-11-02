"""Talk to the Ollama service on this computer. Nothing else is ever contacted."""

from __future__ import annotations

import http.client
import json
import re
import socket
import urllib.error
import urllib.request
from urllib.parse import urlparse

DEFAULT_URL = "http://127.0.0.1:11434"
_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
# Large enough for an 800-character chunk and its prompt; qwen2.5:14b spills out of a
# 16 GB GPU onto the CPU above this size.
_NUM_CTX = 6144
_MAX_REASON_BYTES = 500


class OllamaError(Exception):
    """The service answered with something unusable."""


class OllamaDown(OllamaError):
    """The service cannot be reached."""


class OllamaTimeout(OllamaError):
    """The service did not answer in time."""


class OllamaModelMissing(OllamaError):
    """The service does not have this model."""


class OllamaBusy(OllamaError):
    """The service failed for now; the same request may work a little later."""


class OllamaClient:
    def __init__(self, url: str = DEFAULT_URL, timeout: float = 600.0, unload_timeout: float = 15.0) -> None:
        if urlparse(url).hostname not in _LOCAL_HOSTS:
            raise ValueError("Transcripts never leave this computer: Ollama must be local")
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.unload_timeout = unload_timeout
        # A system proxy could not reach a local service, and must not see transcripts.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self._can_think: dict[str, bool] = {}

    def _call(self, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.url + path, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with self._opener.open(request, timeout=self.timeout if timeout is None else timeout) as response:
                answer = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            message = _service_reason(exc)
            if exc.code == 404:
                raise OllamaModelMissing(message) from exc
            if exc.code == 429 or exc.code >= 500:  # out of memory, crashed or queue full: try later
                raise OllamaBusy(message) from exc
            raise OllamaError(message) from exc
        except urllib.error.URLError as exc:  # failed before the request was sent
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise OllamaTimeout(str(exc)) from exc
            raise OllamaDown(str(exc)) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise OllamaTimeout(str(exc)) from exc
        except (OSError, http.client.HTTPException) as exc:
            # The request was sent but the answer broke off: the service failed while working on it.
            raise OllamaBusy(str(exc) or type(exc).__name__) from exc
        except ValueError as exc:
            raise OllamaBusy("Service returned an unreadable answer") from exc
        if not isinstance(answer, dict):
            raise OllamaBusy("Service returned an unreadable answer")
        return answer

    def models(self) -> set[str]:
        names = {item.get("name", "") for item in self._call("/api/tags").get("models", [])}
        return names | {name.removesuffix(":latest") for name in names}

    def _thinks(self, model: str) -> bool:
        if model not in self._can_think:
            shown = self._call("/api/show", {"model": model})
            self._can_think[model] = "thinking" in shown.get("capabilities", [])
        return self._can_think[model]

    def generate(self, model: str, prompt: str) -> str:
        body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": _NUM_CTX, "temperature": 0.1},
        }
        if self._thinks(model):
            body["think"] = False
        reply = self._call("/api/generate", body).get("response", "")
        return _THINK_BLOCK.sub("", reply).strip()

    def unload(self, model: str) -> None:
        """Free the model's memory now rather than after the service's idle delay."""
        self._call("/api/generate", {"model": model, "keep_alive": 0}, timeout=self.unload_timeout)


def _service_reason(exc: urllib.error.HTTPError) -> str:
    """The status and the start of the service's own explanation, for the log."""
    try:
        body = exc.read(_MAX_REASON_BYTES)
    except (OSError, http.client.HTTPException):
        body = b""
    finally:
        exc.close()
    reason = body.decode("utf-8", errors="replace").strip()
    return f"Service returned {exc.code}: {reason}" if reason else f"Service returned {exc.code}"
