"""LLM client abstraction, on-disk response cache and spend budget."""
from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol


class RateLimited(Exception):
    pass


class Timeout(Exception):
    pass


class ApiError(Exception):
    """Transient API failure (connection error, 408/409, 5xx/529): retried by run_agent."""


class FatalApiError(Exception):
    """The API rejected the request itself (HTTP 400/401/403/404/413/422, other non-retryable
    4xx, an enforced account spend limit, missing credentials). Retrying cannot help, so this
    is NOT an ``ApiError``: run_agent does not retry it or turn it into a failed agent; it
    propagates and aborts the command. The message is sanitised (no keys, URLs or headers)."""

    def __init__(self, status: int | None, kind: str, message: str):
        self.status, self.kind, self.detail = status, kind, message
        where = f"HTTP {status} {kind}" if status is not None else kind
        super().__init__(f"Anthropic API rejected the request ({where}): {message}")


_KEYLIKE = re.compile(r"sk-ant-[A-Za-z0-9_\-]+|(?i:bearer\s+\S+)"
                      r"|(?i:(?:api[_-]?key|authorization|token)\s*[=:]\s*\S+)")
_URL = re.compile(r"\b[a-z][a-z0-9+.-]*://\S+", re.IGNORECASE)
_TRANSIENT_STATUS = frozenset({408, 409, 429})


def sanitise(text: str, limit: int = 300) -> str:
    """API error text safe to print: key-like strings and URLs removed, one line, bounded."""
    text = _KEYLIKE.sub("<redacted>", str(text))
    text = _URL.sub("<url>", text)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _api_message(e) -> tuple[str, str]:
    """(error type, message) from the SDK error body, never its raw repr (which embeds the
    request); falls back to the exception class name."""
    body = getattr(e, "body", None)
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict) and err.get("message"):
        return str(err.get("type") or type(e).__name__), sanitise(err["message"])
    return type(e).__name__, type(e).__name__


class BudgetExceeded(Exception):
    pass


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str


class LLMClient(Protocol):
    def complete(self, system: str, user: str, max_tokens: int) -> LLMResponse: ...


class AnthropicClient:
    """Anthropic Messages API client. The API key is read by the SDK from the environment
    and is never stored or logged here.

    No sampling parameter (temperature/top_p/top_k) is ever sent: current models reject them
    with HTTP 400 and the API version this SDK sends rejects them on every model. Replays are
    deterministic because identical inputs are served from the on-disk cache (CachedLLM), not
    because of sampling settings. ``effort`` ("low".."max", Settings.effort) is sent as
    ``output_config.effort`` only when set; thinking is left at the model's default (adaptive
    on current models: its tokens count against ``max_tokens`` and are billed as output).
    ``request_tag`` names the non-default request settings; CachedLLM puts it in the cache key
    so a changed setting never replays responses made under another one."""

    def __init__(self, model: str, sdk=None, *, effort: str | None = None):
        self.model = model
        self.effort = effort
        self._sdk = sdk

    @property
    def request_tag(self) -> str:
        return f"effort={self.effort}" if self.effort else ""

    def _client(self):
        if self._sdk is None:
            import anthropic

            self._sdk = anthropic.Anthropic()
        return self._sdk

    def complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        import anthropic

        extra = {"output_config": {"effort": self.effort}} if self.effort else {}
        spend_limit = getattr(anthropic, "EnforcedSpendLimitReachedError", ())
        try:
            msg = self._client().messages.create(
                model=self.model,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=max_tokens,
                **extra,
            )
        except spend_limit as e:  # a 429 subclass: the account's spend limit, not a rate limit
            kind, text = _api_message(e)
            raise FatalApiError(getattr(e, "status_code", None), kind, text) from None
        except anthropic.RateLimitError as e:
            raise RateLimited(type(e).__name__) from None
        except anthropic.APITimeoutError as e:
            raise Timeout(type(e).__name__) from None
        except anthropic.APIStatusError as e:
            status = getattr(e, "status_code", None)
            if isinstance(status, int) and 400 <= status < 500 and status not in _TRANSIENT_STATUS:
                kind, text = _api_message(e)
                raise FatalApiError(status, kind, text) from None
            raise ApiError(f"{type(e).__name__} status={status if status is not None else '?'}") from None
        except anthropic.APIError as e:
            raise ApiError(type(e).__name__) from None
        except anthropic.AnthropicError as e:  # e.g. no credentials configured: not transient
            raise FatalApiError(None, type(e).__name__, "no usable Anthropic credentials or client "
                                "configuration (is ANTHROPIC_API_KEY set?)") from None
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return LLMResponse(text, msg.usage.input_tokens, msg.usage.output_tokens, msg.stop_reason or "")


class StubClient:
    """Deterministic offline client: fn(system, user) -> text. Token counts are ~chars/4."""

    def __init__(self, fn: Callable[[str, str], str]):
        self.fn = fn

    def complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        text = self.fn(system, user)
        return LLMResponse(text, max(1, (len(system) + len(user)) // 4), max(1, len(text) // 4), "end_turn")


class CachedLLM:
    """Raw-response cache keyed by sha256(model, system, user[, client.request_tag]) plus
    spend budget. The client's ``request_tag`` (non-default request settings such as effort)
    is part of the key only when non-empty, so existing keys are unchanged.

    A cache hit makes no client call and costs nothing. `bypass_cache=True` forces a live
    call and overwrites the entry, so retries never replay a cached bad response.
    """

    def __init__(self, client: LLMClient, cache_path: Path, model: str,
                 usd_per_mtok: tuple[float, float], budget_usd: float, *,
                 initial_spent_usd: float = 0.0):
        """`initial_spent_usd` carries spend from earlier runs (e.g. a backtest checkpoint)
        so the budget cap spans resumes."""
        self.client = client
        self.model = model
        self.request_tag = str(getattr(client, "request_tag", "") or "")
        self.usd_per_mtok = usd_per_mtok
        self.budget_usd = budget_usd
        self._spent = 0.0
        self._reserved = 0.0
        self._lock = threading.Lock()
        self.restore_spent(initial_spent_usd)
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as c, c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, text TEXT, "
                "input_tokens INTEGER, output_tokens INTEGER, stop_reason TEXT)"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.cache_path, timeout=30)

    @property
    def spent_usd(self) -> float:
        return self._spent

    def restore_spent(self, usd: float) -> None:
        """Raise the spend counter to at least `usd` (spend persisted by a checkpoint).
        Never lowers it: money already spent by this instance is not forgotten."""
        usd = float(usd)
        if not math.isfinite(usd) or usd < 0:
            raise ValueError(f"invalid spend {usd!r}")
        with self._lock:
            self._spent = max(self._spent, usd)

    def _key(self, system: str, user: str) -> str:
        h = hashlib.sha256()
        for part in (self.model, system, user) + ((self.request_tag,) if self.request_tag else ()):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def _cost(self, r: LLMResponse) -> float:
        return (r.input_tokens * self.usd_per_mtok[0] + r.output_tokens * self.usd_per_mtok[1]) / 1e6

    def _worst_case(self, system: str, user: str, max_tokens: int) -> float:
        in_tok = (len(system) + len(user)) // 4 + 1
        return (in_tok * self.usd_per_mtok[0] + max_tokens * self.usd_per_mtok[1]) / 1e6

    def call(self, system: str, user: str, max_tokens: int = 4096,
             bypass_cache: bool = False) -> LLMResponse:
        return self.call_with_meta(system, user, max_tokens, bypass_cache)[0]

    def call_with_meta(self, system: str, user: str, max_tokens: int = 4096,
                       bypass_cache: bool = False) -> tuple[LLMResponse, bool]:
        """Like call(), also returning whether the response was replayed from the cache."""
        key = self._key(system, user)
        if not bypass_cache:
            with closing(self._conn()) as c:
                row = c.execute(
                    "SELECT text, input_tokens, output_tokens, stop_reason FROM llm_cache WHERE key=?",
                    (key,)).fetchone()
            if row:
                return LLMResponse(*row), True
        estimate = self._worst_case(system, user, max_tokens)
        with self._lock:  # reserve worst case before the call; lock not held during the call
            if self._spent + self._reserved + estimate > self.budget_usd:
                raise BudgetExceeded(
                    f"call could cost up to ${estimate:.4f}; spent ${self._spent:.4f} "
                    f"(+${self._reserved:.4f} in flight) of ${self.budget_usd:.2f} budget")
            self._reserved += estimate
        resp = None
        try:
            resp = self.client.complete(system, user, max_tokens)
        finally:
            with self._lock:
                self._reserved -= estimate
                if resp is not None:
                    self._spent += self._cost(resp)
        with closing(self._conn()) as c, c:
            c.execute(
                "INSERT OR REPLACE INTO llm_cache VALUES (?,?,?,?,?)",
                (key, resp.text, resp.input_tokens, resp.output_tokens, resp.stop_reason))
        if self._spent > self.budget_usd:  # actual tokens exceeded the estimate
            raise BudgetExceeded(f"spent ${self._spent:.4f} exceeds ${self.budget_usd:.2f} budget")
        return resp, False
