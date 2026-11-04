"""Talk to the Ollama service on this computer. Nothing else is ever contacted."""

from __future__ import annotations

import http.client
import json
import logging
import math
import re
import socket
import urllib.error
import urllib.request
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

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


class OllamaIncomplete(OllamaError):
    """The answer stopped at a length limit; it is not a whole answer."""


def _local_opener() -> urllib.request.OpenerDirector:
    """An opener for plain http only.

    A system proxy could not reach a local service, and must not see transcripts. build_opener()
    would also add an https handler, which loads the Windows certificate store (several
    milliseconds) for every client although nothing here ever uses https.
    """
    opener = urllib.request.OpenerDirector()
    for handler in (
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPHandler(),
        urllib.request.HTTPDefaultErrorHandler(),
        urllib.request.HTTPErrorProcessor(),
        urllib.request.UnknownHandler(),
    ):
        opener.add_handler(handler)
    return opener


class OllamaClient:
    def __init__(self, url: str = DEFAULT_URL, timeout: float = 600.0, unload_timeout: float = 15.0) -> None:
        parsed = urlparse(url)
        if parsed.hostname not in _LOCAL_HOSTS:
            raise ValueError("Transcripts never leave this computer: Ollama must be local")
        if parsed.scheme != "http":
            raise ValueError(f"Ollama on this computer is reached over plain http, not {parsed.scheme!r}")
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.unload_timeout = unload_timeout
        self._opener = _local_opener()
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

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        num_ctx: int = _NUM_CTX,
        num_predict: int | None = None,
        timeout: float | None = None,
    ) -> str:
        """The model's answer; ``OllamaIncomplete`` if it stopped at a length limit.

        ``num_predict=None`` sends no answer limit: Ollama's own -1 and -2 would remove it silently.
        """
        if not _is_count(num_ctx):
            raise ValueError(f"num_ctx must be a whole number of at least 1, not {num_ctx!r}")
        if num_predict is not None and not _is_count(num_predict):
            raise ValueError(f"num_predict must be None or a whole number of at least 1, not {num_predict!r}")
        if timeout is not None and not _is_seconds(timeout):
            raise ValueError(f"timeout must be None or a positive number of seconds, not {timeout!r}")
        options: dict = {"num_ctx": num_ctx, "temperature": 0.1}
        if num_predict is not None:
            options["num_predict"] = num_predict
        body = {"model": model, "prompt": prompt, "stream": False, "options": options}
        if self._thinks(model):
            body["think"] = False
        answer = self._call("/api/generate", body, timeout=timeout)
        reason = answer.get("done_reason")
        if reason == "length" or answer.get("done") is False:
            # Only the limits go in the message: logs may be pasted into reports, lectures must not.
            raise OllamaIncomplete(
                f"Answer cut off (done_reason={reason!r}, done={answer.get('done')!r}) "
                f"at num_ctx={num_ctx}, num_predict={num_predict}"
            )
        if reason not in (None, "stop"):
            logger.warning("Unknown done_reason from Ollama, taken as a whole answer: %r", reason)
        return _THINK_BLOCK.sub("", answer.get("response", "")).strip()

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


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _is_seconds(value: object) -> bool:
    """Zero would make the socket non-blocking and infinity is refused by it."""
    return (
        isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) and value > 0
    )
