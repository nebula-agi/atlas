from __future__ import annotations

import random
import re
from typing import Dict, List, Sequence

from .session_generator import Session, Utterance
from ..world.schema import World, Entity


class SessionLLM:
    """
    Minimal interface expected by the LLM-backed session generator.
    """

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class LlmSessionGenerator:
    """
    Generates sessions via an LLM using the latent world as context.
    """

    def __init__(self, world: World, rng: random.Random, model: SessionLLM):
        self.world = world
        self.rng = rng
        self.model = model

    def _entities_summary(self) -> str:
        lines: List[str] = []
        for e in self.world.entities.values():
            aliases = ", ".join(a.surface_form for a in e.aliases) or "(no aliases)"
            roles = ", ".join(r.role for r in e.roles) or "(no roles)"
            lines.append(f"- {e.id} [{e.type}]: aliases=({aliases}); roles=({roles}); persona={getattr(e, 'persona', '') or 'n/a'}")
        return "\n".join(lines) or "- (no entities)"

    def _facts_summary(self) -> str:
        return "\n".join(
            f"- fact: {f.subject_id} {f.predicate} {getattr(f.object, 'entity_id', f.object)} (conf={f.confidence})"
            for f in self.world.facts
        ) or "- (no facts)"

    def _events_summary(self) -> str:
        lines: List[str] = []
        for ev in self.world.events.values():
            steps = "; ".join(f"{s.actor.entity_id} {s.description} @t={s.time}" for s in ev.steps)
            lines.append(f"- event {ev.id} ({ev.name}): {steps or 'no steps'}")
        return "\n".join(lines) or "- (no events)"

    def _preferences_summary(self) -> str:
        return "\n".join(
            f"- pref: {p.subject_id} -> {p.signal} (scope={p.scope or 'general'}, conf={p.confidence})"
            for p in self.world.preferences
        ) or "- (no preferences)"

    def _prompts_from_template(self, session_idx: int) -> Dict[str, str]:
        """
        Build a structured prompt grounded in the richer world template.
        The LLM is asked to alternate user/assistant turns, inject noise, and avoid verbatim gold spans.
        """
        return {
            "session_idx": session_idx,
            "entities": self._entities_summary(),
            "facts": self._facts_summary(),
            "events": self._events_summary(),
            "preferences": self._preferences_summary(),
            "guidelines": (
                "Generate a 10-15 turn dialogue alternating user/assistant. "
                "Include 30-50% harmless noise unrelated to the world (small talk, reminders). "
                "Keep critical facts indirect; do NOT leak gold answers verbatim. "
                "Use aliases and indirect references; vary wording. "
                "Format strictly as:\nuser: ...\nassistant: ...\n"
            ),
        }

    def _build_prompt(self, session_idx: int) -> str:
        tpl = self._prompts_from_template(session_idx)
        return (
            f"Session #{tpl['session_idx']}\n"
            f"Entities:\n{tpl['entities']}\n"
            f"Facts:\n{tpl['facts']}\n"
            f"Events:\n{tpl['events']}\n"
            f"Preferences:\n{tpl['preferences']}\n"
            f"{tpl['guidelines']}"
        )

    def _parse_session(self, text: str, session_idx: int) -> Session:
        turns: List[Utterance] = []
        
        if not text or not text.strip():
            print(f"[SessionGenerator] Warning: Empty response for session {session_idx}")
        
        # Try multiple parsing strategies
        # Strategy 1: Standard "speaker: message" format (case-insensitive)
        for line in text.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            # Remove markdown bold/italic markers
            line = re.sub(r'\*\*|\*|__|_', '', line)
            speaker, msg = line.split(":", 1)
            speaker = speaker.strip().lower()
            # Handle variations like "User", "ASSISTANT", etc.
            if speaker in {"user", "u"}:
                turns.append(Utterance("user", msg.strip(), []))
            elif speaker in {"assistant", "a", "ai", "bot"}:
                turns.append(Utterance("assistant", msg.strip(), []))
        
        # Strategy 2: If no turns found, try matching "User:" or "Assistant:" anywhere in text
        if not turns:
            pattern = r'(?:^|\n)\s*\*?\*?(user|assistant|User|Assistant|USER|ASSISTANT)\*?\*?\s*:\s*(.+?)(?=\n\s*\*?\*?(?:user|assistant|User|Assistant|USER|ASSISTANT)\*?\*?\s*:|$)'
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for speaker, msg in matches:
                speaker = speaker.strip().lower()
                msg = msg.strip()
                if msg and speaker in {"user", "assistant"}:
                    turns.append(Utterance(speaker, msg[:500], []))  # Limit message length
        
        if not turns:
            print(f"[SessionGenerator] Warning: Could not parse session {session_idx}")
            print(f"[SessionGenerator] Raw response (first 300 chars): {text[:300]}")
            turns = [
                Utterance("user", "quick update on the project", []),
                Utterance("assistant", "noted.", []),
            ]
        
        # trim or pad to keep within 10-15 turns
        if len(turns) > 15:
            turns = turns[:15]
        while len(turns) < 10:
            turns.append(Utterance("assistant" if len(turns) % 2 else "user", "noted.", []))
        return Session(id=f"llm-{session_idx}", timestamp=session_idx, turns=turns)

    def generate_sessions(
        self,
        min_sessions: int = 30,
        min_total_tokens: int = 20_000,
    ) -> List[Session]:
        sessions: List[Session] = []
        total_tokens = 0
        session_idx = 0

        while len(sessions) < min_sessions or total_tokens < min_total_tokens:
            prompt = self._build_prompt(session_idx)
            raw = self.model.generate(prompt)
            session = self._parse_session(raw, session_idx)
            sessions.append(session)
            total_tokens += sum(len(u.text.split()) for u in session.turns)
            session_idx += 1
        return sessions

    @staticmethod
    def summarize_sessions(sessions: Sequence[Session]) -> str:
        return "\n".join(
            f"{s.timestamp:03d} [{s.id}] " + " | ".join(f"{u.speaker}: {u.text}" for u in s.turns) for s in sessions
        )


