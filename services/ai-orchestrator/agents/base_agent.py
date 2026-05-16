"""
KalaOS — AI Agent Runtime: Base Agent
========================================
All creative agents inherit from BaseAgent.

Design principles:
- Agents are async-first
- Each agent has isolated memory scope
- Tool calls are validated before execution
- Agents stream progress via async generators
- Failure recovery with checkpoint/resume
- No agent can access another agent's private state directly
  — only through the shared message bus
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """Typed message passed between agents on the collaboration bus."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    recipient: str = ""           # "" = broadcast
    message_type: str = "text"    # text | task | result | error | status
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolCall:
    tool_name: str
    arguments: Dict[str, Any]
    caller_agent: str


@dataclass
class AgentContext:
    """Shared context passed into an agent for a single run."""
    task: str
    session_id: str
    user_id: Optional[str] = None
    history: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    success: bool
    output: Any
    agent_id: str
    task: str
    duration_ms: float
    tool_calls: List[ToolCall] = field(default_factory=list)
    error: Optional[str] = None
    checkpoints: List[Dict] = field(default_factory=list)


class BaseAgent(ABC):
    """
    Abstract base for all KalaOS creative agents.

    Subclasses must implement:
    - agent_name (property)
    - agent_description (property)
    - allowed_tools (property)
    - _execute (async method)
    """

    def __init__(self, memory_store=None, tool_registry=None, message_bus=None):
        self._memory = memory_store
        self._tools = tool_registry
        self._bus = message_bus
        self._run_id: Optional[str] = None
        self._checkpoints: List[Dict] = []

    @property
    @abstractmethod
    def agent_name(self) -> str: ...

    @property
    @abstractmethod
    def agent_description(self) -> str: ...

    @property
    @abstractmethod
    def allowed_tools(self) -> List[str]: ...

    @abstractmethod
    async def _execute(
        self, ctx: AgentContext
    ) -> AsyncGenerator[str, None]: ...

    async def run(self, ctx: AgentContext) -> AgentResult:
        """
        Top-level run method — handles timing, error recovery, and checkpoints.
        """
        self._run_id = str(uuid.uuid4())
        self._checkpoints = []
        t0 = time.perf_counter()
        tool_calls: List[ToolCall] = []
        output_parts: List[str] = []

        logger.info(
            "Agent starting [agent=%s, session=%s, task=%.80s]",
            self.agent_name, ctx.session_id, ctx.task,
        )

        try:
            async for chunk in self._execute(ctx):
                output_parts.append(chunk)

            return AgentResult(
                success=True,
                output="".join(output_parts),
                agent_id=self.agent_name,
                task=ctx.task,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                tool_calls=tool_calls,
                checkpoints=self._checkpoints,
            )

        except Exception as exc:
            logger.exception("Agent %s failed", self.agent_name)
            return AgentResult(
                success=False,
                output=None,
                agent_id=self.agent_name,
                task=ctx.task,
                duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                error=str(exc),
                checkpoints=self._checkpoints,
            )

    async def stream_run(self, ctx: AgentContext) -> AsyncGenerator[str, None]:
        """Stream agent output tokens in real-time."""
        try:
            async for chunk in self._execute(ctx):
                yield chunk
        except Exception as exc:
            logger.exception("Agent %s stream error", self.agent_name)
            yield f"\n[Agent error: {exc}]"

    def _checkpoint(self, name: str, data: Any = None) -> None:
        """Save a named checkpoint for failure recovery."""
        self._checkpoints.append({
            "name": name,
            "timestamp": time.time(),
            "data": data,
        })
        logger.debug("Checkpoint [agent=%s, step=%s]", self.agent_name, name)

    async def _call_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Validated tool invocation — only allowed tools can be called.
        Prevents privilege escalation between agents.
        """
        if tool_name not in self.allowed_tools:
            raise PermissionError(
                f"Agent '{self.agent_name}' is not permitted to call tool '{tool_name}'"
            )
        if not self._tools:
            raise RuntimeError("Tool registry not injected into agent")
        return await self._tools.execute(tool_name, agent=self.agent_name, **kwargs)

    async def _recall_memory(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant memories from the vector memory store."""
        if not self._memory:
            return []
        return await self._memory.search(
            query=query, agent_scope=self.agent_name, top_k=top_k
        )

    async def _store_memory(self, key: str, value: Any) -> None:
        """Persist a memory entry for this agent."""
        if self._memory:
            await self._memory.store(
                key=key, value=value, agent_scope=self.agent_name
            )

    async def _broadcast(self, message: AgentMessage) -> None:
        """Publish a message to the agent collaboration bus."""
        if self._bus:
            message.sender = self.agent_name
            await self._bus.publish(message)

    async def _send(self, recipient: str, content: Any, message_type: str = "text") -> None:
        """Send a direct message to another agent."""
        await self._broadcast(AgentMessage(
            sender=self.agent_name,
            recipient=recipient,
            message_type=message_type,
            content=content,
        ))
