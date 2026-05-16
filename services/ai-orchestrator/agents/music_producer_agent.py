"""
KalaOS — Agent: Music Producer
================================
Specialized agent for AI-assisted music production workflows.

Capabilities:
- Prompt-to-composition analysis
- Genre-aware chord progression generation
- BPM and arrangement suggestions
- Mixing recommendations
- Mastering target guidance
- FL Studio workflow hints
- Collaboration with Creative Director agent
"""
from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, List

from services.ai_orchestrator.agents.base_agent import (
    AgentContext, BaseAgent
)
from services.ai_orchestrator.router.ai_router import InferenceRequest, ai_router

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """
You are the KalaOS Music Producer Agent — a world-class AI music production expert.

You have deep knowledge of:
- Music theory (harmony, melody, counterpoint, modulation)
- Genre conventions (trap, boom-bap, house, R&B, cinematic, experimental)
- DAW workflows (FL Studio, Ableton, Logic Pro)
- Mixing and mastering chains
- Audio signal processing (EQ, compression, reverb, saturation)
- Arrangement and song structure
- Emotional storytelling through music

You respond with:
- Actionable, specific production advice
- Real chord names, BPM values, and genre tags
- Step-by-step workflow recommendations
- Always in the context of the artist's creative vision

You NEVER:
- Suggest the artist copy someone else's work
- Give generic advice without context
- Override the artist's creative choices
- Generate or assist with copyright-infringing content
""".strip()


class MusicProducerAgent(BaseAgent):

    @property
    def agent_name(self) -> str:
        return "music_producer"

    @property
    def agent_description(self) -> str:
        return (
            "AI music production expert. Handles composition, arrangement, "
            "mixing, mastering, and genre-specific production advice."
        )

    @property
    def allowed_tools(self) -> List[str]:
        return [
            "kalacore.compose",
            "kalacore.produce",
            "kalacore.generate_beat",
            "kalacore.analyze_signal",
            "search.music_theory",
            "memory.recall",
            "memory.store",
        ]

    async def _execute(self, ctx: AgentContext) -> AsyncGenerator[str, None]:
        self._checkpoint("start", {"task": ctx.task[:200]})

        # Recall relevant past sessions for this user
        memories = await self._recall_memory(ctx.task, top_k=3)
        memory_context = ""
        if memories:
            memory_context = "\n\nPast session context:\n" + "\n".join(
                f"- {m.get('content', '')}" for m in memories
            )

        # Build enriched prompt
        prompt = f"""
Artist's request: {ctx.task}

Session metadata: {json.dumps(ctx.metadata, default=str)}
{memory_context}

Please provide detailed, actionable music production guidance. Include:
1. Compositional approach and structure
2. Specific chord progressions (with Roman numeral analysis)
3. Recommended BPM range and groove feel
4. Key production elements and sound design direction
5. Mixing priorities for this style
6. FL Studio / DAW workflow suggestions
7. Next steps for the artist
""".strip()

        self._checkpoint("inference_start")

        req = InferenceRequest(
            prompt=prompt,
            system_prompt=_SYSTEM_PROMPT,
            temperature=0.75,
            max_tokens=1500,
            stream=True,
            task_type="music",
        )

        output_buffer = []
        async for token in ai_router.stream(req):
            output_buffer.append(token)
            yield token

        self._checkpoint("inference_done")

        # Store this interaction in memory for future sessions
        await self._store_memory(
            key=f"session:{ctx.session_id}:music_advice",
            value={
                "task": ctx.task[:300],
                "summary": "".join(output_buffer)[:500],
            },
        )

        # Notify Creative Director if relevant
        await self._send(
            recipient="creative_director",
            content={"type": "music_advice_complete", "session": ctx.session_id},
            message_type="status",
        )
