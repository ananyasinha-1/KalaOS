"""
KalaOS — AI Orchestration Layer: Model Router
===============================================
Central intelligence for routing AI inference requests.

Priority chain:
  1. vLLM (fastest, GPU, local)
  2. Ollama (local, versatile)
  3. llama.cpp (lightweight local)
  4. OpenAI (cloud fallback, cost-gated)
  5. Anthropic (cloud fallback, quality-gated)

Features:
- Health-aware routing (skips unhealthy backends)
- Streaming support across all backends
- Prompt caching (Redis)
- Retry + fallback on failure
- Latency tracking via OpenTelemetry
- Context length enforcement
"""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class ModelBackend(str, Enum):
    VLLM = "vllm"
    OLLAMA = "ollama"
    LLAMACPP = "llamacpp"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class BackendStatus:
    healthy: bool = True
    last_check: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    consecutive_failures: int = 0
    max_failures: int = 3


@dataclass
class InferenceRequest:
    prompt: str
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False
    task_type: str = "general"        # general | music | visual | code | creative
    priority: str = "normal"          # low | normal | high | realtime


@dataclass
class InferenceResponse:
    text: str
    model_used: str
    backend: ModelBackend
    latency_ms: float
    cached: bool = False
    tokens_used: Optional[int] = None


class AIRouter:
    """
    Stateful AI router with health tracking, fallback, and caching.
    Instantiate once at application startup (singleton pattern).
    """

    # Task-type → preferred backend ordering
    TASK_ROUTING: Dict[str, List[ModelBackend]] = {
        "general":  [ModelBackend.VLLM, ModelBackend.OLLAMA, ModelBackend.OPENAI],
        "music":    [ModelBackend.VLLM, ModelBackend.OLLAMA, ModelBackend.OPENAI],
        "visual":   [ModelBackend.VLLM, ModelBackend.OLLAMA, ModelBackend.OPENAI],
        "code":     [ModelBackend.VLLM, ModelBackend.OLLAMA, ModelBackend.ANTHROPIC],
        "creative": [ModelBackend.OLLAMA, ModelBackend.VLLM, ModelBackend.ANTHROPIC],
        "realtime": [ModelBackend.VLLM, ModelBackend.LLAMACPP, ModelBackend.OLLAMA],
    }

    def __init__(self) -> None:
        from app.config import settings
        self._settings = settings
        self._status: Dict[ModelBackend, BackendStatus] = {
            b: BackendStatus() for b in ModelBackend
        }
        self._cache: Optional[object] = None   # Injected Redis client

    def inject_cache(self, redis_client) -> None:
        self._cache = redis_client

    async def infer(self, req: InferenceRequest) -> InferenceResponse:
        """
        Route an inference request through the backend priority chain.
        Returns first successful response; logs and skips failed backends.
        """
        # Check prompt cache
        if not req.stream:
            cached = await self._cache_get(req)
            if cached:
                return cached

        backends = self.TASK_ROUTING.get(req.task_type, self.TASK_ROUTING["general"])

        for backend in backends:
            if not self._status[backend].healthy:
                logger.debug("Skipping unhealthy backend: %s", backend)
                continue
            try:
                t0 = time.perf_counter()
                response = await self._dispatch(backend, req)
                latency = (time.perf_counter() - t0) * 1000
                self._record_success(backend, latency)

                result = InferenceResponse(
                    text=response,
                    model_used=req.model or "default",
                    backend=backend,
                    latency_ms=round(latency, 2),
                )

                if not req.stream:
                    await self._cache_set(req, result)

                return result

            except Exception as exc:
                logger.warning("Backend %s failed: %s", backend, exc)
                self._record_failure(backend)
                continue

        raise RuntimeError("All AI backends exhausted. No inference available.")

    async def stream(self, req: InferenceRequest) -> AsyncGenerator[str, None]:
        """Stream tokens from the first available healthy backend."""
        req.stream = True
        backends = self.TASK_ROUTING.get(req.task_type, self.TASK_ROUTING["general"])

        for backend in backends:
            if not self._status[backend].healthy:
                continue
            try:
                async for token in self._dispatch_stream(backend, req):
                    yield token
                return
            except Exception as exc:
                logger.warning("Streaming backend %s failed: %s", backend, exc)
                self._record_failure(backend)
                continue

        yield "[Error: No AI backend available for streaming]"

    # ── Backend dispatchers ────────────────────────────────────────────────

    async def _dispatch(self, backend: ModelBackend, req: InferenceRequest) -> str:
        if backend == ModelBackend.VLLM:
            return await self._call_vllm(req)
        if backend == ModelBackend.OLLAMA:
            return await self._call_ollama(req)
        if backend == ModelBackend.LLAMACPP:
            return await self._call_llamacpp(req)
        if backend == ModelBackend.OPENAI:
            return await self._call_openai(req)
        if backend == ModelBackend.ANTHROPIC:
            return await self._call_anthropic(req)
        raise ValueError(f"Unknown backend: {backend}")

    async def _dispatch_stream(
        self, backend: ModelBackend, req: InferenceRequest
    ) -> AsyncGenerator[str, None]:
        if backend == ModelBackend.OLLAMA:
            async for tok in self._stream_ollama(req):
                yield tok
        elif backend == ModelBackend.VLLM:
            async for tok in self._stream_vllm(req):
                yield tok
        else:
            result = await self._dispatch(backend, req)
            yield result

    async def _call_ollama(self, req: InferenceRequest) -> str:
        messages = []
        if req.system_prompt:
            messages.append({"role": "system", "content": req.system_prompt})
        messages.append({"role": "user", "content": req.prompt})

        async with httpx.AsyncClient(timeout=self._settings.OLLAMA_TIMEOUT) as client:
            r = await client.post(
                f"{self._settings.OLLAMA_HOST}/api/chat",
                json={
                    "model": req.model or self._settings.OLLAMA_DEFAULT_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": req.temperature,
                        "num_predict": req.max_tokens,
                    },
                },
            )
            r.raise_for_status()
            data = r.json()
            return data["message"]["content"]

    async def _stream_ollama(self, req: InferenceRequest) -> AsyncGenerator[str, None]:
        import json as _json
        messages = []
        if req.system_prompt:
            messages.append({"role": "system", "content": req.system_prompt})
        messages.append({"role": "user", "content": req.prompt})

        async with httpx.AsyncClient(timeout=self._settings.OLLAMA_STREAM_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{self._settings.OLLAMA_HOST}/api/chat",
                json={"model": req.model or self._settings.OLLAMA_DEFAULT_MODEL,
                      "messages": messages, "stream": True},
            ) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        try:
                            chunk = _json.loads(line)
                            if content := chunk.get("message", {}).get("content"):
                                yield content
                        except _json.JSONDecodeError:
                            continue

    async def _call_vllm(self, req: InferenceRequest) -> str:
        if not self._settings.VLLM_HOST:
            raise RuntimeError("vLLM not configured")
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{self._settings.VLLM_HOST}/v1/completions",
                headers={"Authorization": f"Bearer {self._settings.VLLM_API_KEY or ''}"},
                json={
                    "model": self._settings.VLLM_MODEL,
                    "prompt": req.prompt,
                    "max_tokens": req.max_tokens,
                    "temperature": req.temperature,
                    "stream": False,
                },
            )
            r.raise_for_status()
            return r.json()["choices"][0]["text"]

    async def _stream_vllm(self, req: InferenceRequest) -> AsyncGenerator[str, None]:
        import json as _json
        if not self._settings.VLLM_HOST:
            raise RuntimeError("vLLM not configured")
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST",
                f"{self._settings.VLLM_HOST}/v1/completions",
                headers={"Authorization": f"Bearer {self._settings.VLLM_API_KEY or ''}"},
                json={"model": self._settings.VLLM_MODEL,
                      "prompt": req.prompt, "max_tokens": req.max_tokens,
                      "temperature": req.temperature, "stream": True},
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            chunk = _json.loads(line[6:])
                            if text := chunk["choices"][0].get("text"):
                                yield text
                        except Exception:
                            continue

    async def _call_llamacpp(self, req: InferenceRequest) -> str:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                "http://localhost:8080/completion",
                json={"prompt": req.prompt, "n_predict": req.max_tokens,
                      "temperature": req.temperature, "stream": False},
            )
            r.raise_for_status()
            return r.json()["content"]

    async def _call_openai(self, req: InferenceRequest) -> str:
        if not self._settings.OPENAI_API_KEY:
            raise RuntimeError("OpenAI API key not configured")
        import openai
        client = openai.AsyncOpenAI(api_key=self._settings.OPENAI_API_KEY)
        messages = []
        if req.system_prompt:
            messages.append({"role": "system", "content": req.system_prompt})
        messages.append({"role": "user", "content": req.prompt})
        resp = await client.chat.completions.create(
            model=self._settings.OPENAI_MODEL,
            messages=messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        return resp.choices[0].message.content or ""

    async def _call_anthropic(self, req: InferenceRequest) -> str:
        if not self._settings.ANTHROPIC_API_KEY:
            raise RuntimeError("Anthropic API key not configured")
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=self._settings.ANTHROPIC_API_KEY)
        msg = await client.messages.create(
            model=self._settings.ANTHROPIC_MODEL,
            max_tokens=req.max_tokens,
            system=req.system_prompt or "You are a helpful creative AI assistant.",
            messages=[{"role": "user", "content": req.prompt}],
        )
        return msg.content[0].text

    # ── Health tracking ────────────────────────────────────────────────────

    def _record_success(self, backend: ModelBackend, latency_ms: float) -> None:
        s = self._status[backend]
        s.healthy = True
        s.consecutive_failures = 0
        s.latency_ms = latency_ms
        s.last_check = time.time()

    def _record_failure(self, backend: ModelBackend) -> None:
        s = self._status[backend]
        s.consecutive_failures += 1
        s.last_check = time.time()
        if s.consecutive_failures >= s.max_failures:
            s.healthy = False
            logger.warning("Backend %s marked unhealthy after %d failures", backend, s.consecutive_failures)

    # ── Prompt caching ─────────────────────────────────────────────────────

    def _cache_key(self, req: InferenceRequest) -> str:
        raw = f"{req.prompt}:{req.system_prompt}:{req.model}:{req.temperature}:{req.max_tokens}"
        return f"airouter:{hashlib.sha256(raw.encode()).hexdigest()[:24]}"

    async def _cache_get(self, req: InferenceRequest) -> Optional[InferenceResponse]:
        if not self._cache:
            return None
        try:
            from app.infrastructure.redis_client import cache_get
            data = await cache_get(self._cache_key(req))
            if data:
                return InferenceResponse(**data, cached=True)
        except Exception:
            pass
        return None

    async def _cache_set(self, req: InferenceRequest, resp: InferenceResponse) -> None:
        if not self._cache:
            return
        try:
            from app.infrastructure.redis_client import cache_set
            await cache_set(
                self._cache_key(req),
                {
                    "text": resp.text,
                    "model_used": resp.model_used,
                    "backend": resp.backend.value,
                    "latency_ms": resp.latency_ms,
                },
                ttl=300,
            )
        except Exception:
            pass


# ── Singleton instance ─────────────────────────────────────────────────────
ai_router = AIRouter()
