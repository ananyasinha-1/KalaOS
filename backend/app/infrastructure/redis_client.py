"""
KalaOS — Infrastructure: Redis Client
======================================
Async Redis connection pool with:
- Connection pooling for performance
- Health-check helper
- Cache helper with TTL
- Pub/Sub channel helpers
- Graceful shutdown
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

import redis.asyncio as aioredis
from redis.asyncio import Redis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import RedisError

from app.config import settings

logger = logging.getLogger(__name__)

_pool: Optional[ConnectionPool] = None
_client: Optional[Redis] = None


async def init_redis() -> None:
    """Initialize the global Redis connection pool."""
    global _pool, _client
    _pool = ConnectionPool.from_url(
        settings.REDIS_URL,
        max_connections=50,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True,
        health_check_interval=30,
    )
    _client = Redis(connection_pool=_pool)
    # Verify connectivity
    await _client.ping()
    logger.info("Redis connected: %s", settings.REDIS_URL)


async def close_redis() -> None:
    """Gracefully close Redis connections."""
    global _client, _pool
    if _client:
        await _client.aclose()
    if _pool:
        await _pool.aclose()
    logger.info("Redis connection closed.")


def get_redis() -> Redis:
    """Return the shared Redis client. Must call init_redis() first."""
    if _client is None:
        raise RuntimeError("Redis not initialized. Call init_redis() at startup.")
    return _client


# ── Cache helpers ──────────────────────────────────────────────────────────

async def cache_get(key: str) -> Optional[Any]:
    """Get a JSON-decoded value from cache."""
    try:
        r = get_redis()
        raw = await r.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except (RedisError, json.JSONDecodeError) as exc:
        logger.warning("Cache get failed for key=%s: %s", key, exc)
        return None


async def cache_set(key: str, value: Any, ttl: int = settings.REDIS_CACHE_TTL) -> bool:
    """JSON-encode and store a value with TTL."""
    try:
        r = get_redis()
        await r.setex(key, ttl, json.dumps(value, default=str))
        return True
    except (RedisError, TypeError) as exc:
        logger.warning("Cache set failed for key=%s: %s", key, exc)
        return False


async def cache_delete(key: str) -> bool:
    """Delete a cache key."""
    try:
        r = get_redis()
        await r.delete(key)
        return True
    except RedisError as exc:
        logger.warning("Cache delete failed for key=%s: %s", key, exc)
        return False


async def cache_invalidate_prefix(prefix: str) -> int:
    """Delete all keys matching a prefix pattern."""
    try:
        r = get_redis()
        keys = await r.keys(f"{prefix}:*")
        if keys:
            return await r.delete(*keys)
        return 0
    except RedisError as exc:
        logger.warning("Cache invalidation failed for prefix=%s: %s", prefix, exc)
        return 0


# ── Pub/Sub helpers ────────────────────────────────────────────────────────

async def publish(channel: str, message: Any) -> None:
    """Publish a JSON message to a Redis channel."""
    try:
        r = get_redis()
        await r.publish(channel, json.dumps(message, default=str))
    except RedisError as exc:
        logger.error("Pub/Sub publish failed on channel=%s: %s", channel, exc)


@asynccontextmanager
async def subscribe(channel: str) -> AsyncGenerator[aioredis.client.PubSub, None]:
    """Context manager for subscribing to a Redis channel."""
    r = get_redis()
    pubsub = r.pubsub()
    await pubsub.subscribe(channel)
    try:
        yield pubsub
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.aclose()


# ── Rate limiting helpers ──────────────────────────────────────────────────

async def check_rate_limit(key: str, limit: int, window: int) -> tuple[bool, int]:
    """
    Sliding window rate limit check.
    Returns (allowed: bool, remaining: int).
    """
    try:
        r = get_redis()
        pipe = r.pipeline()
        pipe.incr(key)
        pipe.expire(key, window)
        count, _ = await pipe.execute()
        remaining = max(0, limit - count)
        return count <= limit, remaining
    except RedisError:
        # Fail open — don't block on Redis failure
        return True, limit


# ── Health check ───────────────────────────────────────────────────────────

async def redis_health() -> dict:
    """Return Redis health status for the /health endpoint."""
    try:
        r = get_redis()
        await r.ping()
        info = await r.info("server")
        return {
            "status": "healthy",
            "version": info.get("redis_version", "unknown"),
            "connected_clients": info.get("connected_clients", 0),
        }
    except Exception as exc:
        return {"status": "unhealthy", "error": str(exc)}
