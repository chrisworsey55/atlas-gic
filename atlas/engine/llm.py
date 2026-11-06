"""LLM client abstraction, on-disk response cache and spend budget."""
from __future__ import annotations

import hashlib
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
    pass


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
    """Anthropic Messages API client. Temperature is fixed to 0. The API key is read by the
    SDK from the environment and is never stored or logged here."""

    TEMPERATURE = 0

    def __init__(self, model: str, sdk=None):
        self.model = model
        self._sdk = sdk

    def _client(self):
        if self._sdk is None:
            import anthropic

            self._sdk = anthropic.Anthropic()
        return self._sdk

    def complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        import anthropic

        try:
            msg = self._client().messages.create(
                model=self.model,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=self.TEMPERATURE,
            )
        except anthropic.RateLimitError as e:
            raise RateLimited(type(e).__name__) from None
        except anthropic.APITimeoutError as e:
            raise Timeout(type(e).__name__) from None
        except anthropic.APIStatusError as e:
            raise ApiError(f"{type(e).__name__} status={getattr(e, 'status_code', '?')}") from None
        except anthropic.APIError as e:
            raise ApiError(type(e).__name__) from None
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
    """Raw-response cache keyed by sha256(model, system, user) plus spend budget.

    A cache hit makes no client call and costs nothing. `bypass_cache=True` forces a live
    call and overwrites the entry, so retries never replay a cached bad response.
    """

    def __init__(self, client: LLMClient, cache_path: Path, model: str,
                 usd_per_mtok: tuple[float, float], budget_usd: float):
        self.client = client
        self.model = model
        self.usd_per_mtok = usd_per_mtok
        self.budget_usd = budget_usd
        self._spent = 0.0
        self._reserved = 0.0
        self._lock = threading.Lock()
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

    def _key(self, system: str, user: str) -> str:
        h = hashlib.sha256()
        for part in (self.model, system, user):
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
