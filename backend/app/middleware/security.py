"""
KalaOS — Middleware: Security Headers + Zero-Trust
===================================================
Production-grade security middleware:
- Security headers (CSP, HSTS, X-Frame, etc.)
- Request ID injection
- Prompt injection detection
- AI output sanitization utilities
- Request sanitization
"""
from __future__ import annotations

import re
import uuid
import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# ── Prompt injection patterns ─────────────────────────────────────────────

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"forget\s+(all\s+)?previous\s+(instructions?|context)",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"act\s+as\s+(a|an)\s+(?!artist|musician|creator|writer)",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode\s+enabled",
    r"disregard\s+(the\s+)?system\s+prompt",
    r"<\|?system\|?>",
    r"\[INST\].*?\[/INST\]",
    r"###\s*(system|human|assistant)\s*:",
]

_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS),
    re.IGNORECASE | re.DOTALL,
)


def detect_prompt_injection(text: str) -> bool:
    """Return True if the text contains prompt injection patterns."""
    return bool(_INJECTION_RE.search(text))


def sanitize_ai_output(text: str) -> str:
    """
    Basic sanitization of AI-generated output before returning to client.
    Strips known dangerous patterns from LLM responses.
    """
    # Remove potential script injection in AI output
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"javascript:", "", text, flags=re.IGNORECASE)
    text = re.sub(r"on\w+\s*=", "", text, flags=re.IGNORECASE)
    return text.strip()


# ── Security Headers Middleware ────────────────────────────────────────────

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Injects production-grade security headers on every response.
    Also attaches a unique X-Request-ID for distributed tracing.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Attach request ID for tracing
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        response: Response = await call_next(request)

        # Core security headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(self), camera=(self), "
            "payment=(), usb=(), magnetometer=()"
        )

        # HSTS (only in production over HTTPS)
        from app.config import settings
        if settings.is_production:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://fonts.googleapis.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://fonts.gstatic.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: blob: https:; "
            "media-src 'self' blob:; "
            "connect-src 'self' ws: wss: https:; "
            "worker-src 'self' blob:; "
            "frame-ancestors 'none';"
        )

        return response


# ── Request Logging Middleware ─────────────────────────────────────────────

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Structured request/response logging for observability."""

    _SKIP_PATHS = {"/health", "/metrics", "/favicon.ico"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self._SKIP_PATHS:
            return await call_next(request)

        import time
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "request_id": getattr(request.state, "request_id", "-"),
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "-"),
            },
        )
        return response
