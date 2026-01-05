from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Union


@dataclass
class TimeWindow:
    start_time: int
    end_time: Optional[int]

    def is_active(self, time: int) -> bool:
        if self.end_time is None:
            return time >= self.start_time
        return self.start_time <= time <= self.end_time


@dataclass
class EntityRef:
    entity_id: str
    alias: Optional[str] = None


@dataclass
class Alias:
    surface_form: str
    start_time: int
    end_time: Optional[int]
    confidence: float

    def is_active(self, time: int) -> bool:
        if self.end_time is None:
            return time >= self.start_time
        return self.start_time <= time <= self.end_time


@dataclass
class TimeScopedRole:
    role: str
    start_time: int
    end_time: Optional[int]

    def is_active(self, time: int) -> bool:
        if self.end_time is None:
            return time >= self.start_time
        return self.start_time <= time <= self.end_time


@dataclass
class Fact:
    subject_id: str
    predicate: str
    object: Union[str, EntityRef]
    validity: TimeWindow
    source: Literal["user", "system", "log"]
    confidence: float
    id: Optional[str] = None  # Optional ID for referencing
    supersedes: Optional[str] = None  # ID of fact this supersedes (for belief revision)


@dataclass
class Preference:
    subject_id: str
    signal: str
    scope: Optional[str]
    strength: float
    confidence: float
    inferred: bool
    rank: Optional[int] = None  # For preference hierarchies
    is_hard_constraint: bool = False  # Soft vs hard constraint
    first_mentioned: Optional[int] = None  # For preference drift detection
    last_mentioned: Optional[int] = None


@dataclass
class MemoryPolicy:
    fact_predicate: str
    decay_half_life: int
    retrievable_after_expiry: bool


@dataclass
class EventStep:
    description: str
    time: int
    actor: EntityRef


@dataclass
class CausalEdge:
    source_step: int
    target_step: int
    relation: str


@dataclass
class Event:
    id: str
    name: str
    steps: List[EventStep]
    participants: List[EntityRef]
    start_time: int
    end_time: int
    causal_links: List[CausalEdge]


# === Pillar 1: World Modeling - Task/State Tracking ===

@dataclass
class Task:
    id: str
    description: str
    assignee: Optional[EntityRef]
    status: Literal["open", "in_progress", "completed", "blocked"]
    created_time: int
    completed_time: Optional[int]
    blocked_by: List[str] = field(default_factory=list)  # Task IDs
    reminder_time: Optional[int] = None
    commitment_text: Optional[str] = None  # Original commitment phrasing


# === Pillar 2: Declarative Reasoning - Verbatim Recall ===

@dataclass
class VerbatimItem:
    id: str
    item_type: Literal["code_block", "error_message", "query", "config"]
    content: str
    description: str  # Semantic description for retrieval
    created_time: int
    context: Optional[str] = None  # Where/when this was mentioned


# === Pillar 3: Temporal & Episodic - Cyclical Patterns ===

@dataclass
class CyclicalPattern:
    id: str
    subject_id: str
    pattern_type: Literal["daily", "weekly", "monthly", "custom"]
    description: str  # "code reviews on Fridays"
    day_of_week: Optional[int] = None  # 0=Monday, 6=Sunday
    time_of_day: Optional[str] = None  # "morning", "afternoon", etc.
    exceptions: List[str] = field(default_factory=list)  # "never when it rains"


# === Pillar 6: Procedural Knowledge ===

@dataclass
class Lesson:
    id: str
    trigger_context: str  # When to surface this lesson
    lesson_text: str  # The actual lesson learned
    created_time: int
    source_event: Optional[str] = None  # Event ID that generated this lesson
    is_systemic: bool = False  # One-off vs systemic pattern


@dataclass
class Procedure:
    id: str
    name: str
    steps: List[str]  # Ordered steps
    conditions: Dict[str, str] = field(default_factory=dict)  # "if error X" -> "do Y"
    last_updated: Optional[int] = None


@dataclass
class ToolMethodMemory:
    id: str
    problem_context: str  # Description of the problem
    solution: str  # What worked
    worked: bool  # Did it work?
    created_time: int
    environment: Optional[str] = None  # "staging", "production", etc.


# === Pillar 5: Knowledge Boundaries ===

@dataclass
class NegativeKnowledge:
    """Topics explicitly NOT mentioned or known."""
    id: str
    topic: str
    description: str


@dataclass
class Entity:
    id: str
    type: Literal["person", "project", "service", "event", "organization"]
    aliases: List[Alias]
    attributes: List[Fact]
    roles: List[TimeScopedRole]
    persona: Optional[str] = None
    category_memberships: List[str] = field(default_factory=list)  # Inferred types

    def active_aliases(self, time: int) -> List[Alias]:
        return [a for a in self.aliases if a.is_active(time)]


class World:
    """
    Latent world representation never shown directly to the evaluated model.
    """

    def __init__(
        self,
        entities: Dict[str, Entity],
        facts: List[Fact],
        events: Dict[str, Event],
        preferences: List[Preference],
        memory_policies: List[MemoryPolicy],
        # New pillar support
        tasks: Optional[List[Task]] = None,
        verbatim_items: Optional[List[VerbatimItem]] = None,
        cyclical_patterns: Optional[List[CyclicalPattern]] = None,
        lessons: Optional[List[Lesson]] = None,
        procedures: Optional[List[Procedure]] = None,
        tool_memories: Optional[List[ToolMethodMemory]] = None,
        negative_knowledge: Optional[List[NegativeKnowledge]] = None,
    ) -> None:
        self.entities = entities
        self.facts = facts
        self.events = events
        self.preferences = preferences
        self.memory_policies = memory_policies
        # New pillar support
        self.tasks = tasks or []
        self.verbatim_items = verbatim_items or []
        self.cyclical_patterns = cyclical_patterns or []
        self.lessons = lessons or []
        self.procedures = procedures or []
        self.tool_memories = tool_memories or []
        self.negative_knowledge = negative_knowledge or []

    def get_entity_by_alias(self, surface_form: str, time: int) -> Optional[Entity]:
        for entity in self.entities.values():
            if any(alias.surface_form == surface_form and alias.is_active(time) for alias in entity.aliases):
                return entity
        return None

    def get_active_facts(self, entity_id: str, time: int) -> List[Fact]:
        return [
            fact
            for fact in self.facts
            if fact.subject_id == entity_id
            and fact.validity.is_active(time)
            and not self.is_fact_expired(fact, time)
        ]

    def get_event(self, event_id: str) -> Optional[Event]:
        return self.events.get(event_id)

    def get_open_tasks(self) -> List[Task]:
        return [t for t in self.tasks if t.status in ("open", "in_progress", "blocked")]

    def get_blocked_tasks(self) -> List[Task]:
        return [t for t in self.tasks if t.status == "blocked"]

    def get_superseding_fact(self, fact_id: str) -> Optional[Fact]:
        """Get the fact that supersedes the given fact."""
        for f in self.facts:
            if f.supersedes == fact_id:
                return f
        return None

    def get_ranked_preferences(self, subject_id: str) -> List[Preference]:
        """Get preferences sorted by rank (explicit ranking)."""
        prefs = [p for p in self.preferences if p.subject_id == subject_id and p.rank is not None]
        return sorted(prefs, key=lambda p: p.rank or 999)

    def get_hard_constraints(self, subject_id: str) -> List[Preference]:
        """Get hard constraints (non-negotiable preferences)."""
        return [p for p in self.preferences if p.subject_id == subject_id and p.is_hard_constraint]

    def infer_preferences(self, subject_id: str) -> List[Preference]:
        direct = [p for p in self.preferences if p.subject_id == subject_id]
        inferred = [
            Preference(
                subject_id=subject_id,
                signal=p.signal,
                scope=p.scope,
                strength=min(1.0, p.strength * 0.8),
                confidence=max(p.confidence * 0.8, 0.4),
                inferred=True,
            )
            for p in direct
            if not p.inferred
        ]
        return direct + inferred

    def is_fact_expired(self, fact: Fact, query_time: int) -> bool:
        if fact.validity.end_time is None:
            return False
        return query_time > fact.validity.end_time

    def resolve_alias(self, surface_form: str, time: int) -> Optional[str]:
        entity = self.get_entity_by_alias(surface_form, time)
        return entity.id if entity else None

    def related_entities(self, entity_id: str) -> Sequence[EntityRef]:
        refs: List[EntityRef] = []
        for fact in self.facts:
            if fact.subject_id == entity_id and isinstance(fact.object, EntityRef):
                refs.append(fact.object)
        return refs

    def get_lessons_for_context(self, context: str) -> List[Lesson]:
        """Find lessons that match a given context."""
        return [l for l in self.lessons if context.lower() in l.trigger_context.lower()]

    def get_procedure(self, name: str) -> Optional[Procedure]:
        """Get a procedure by name."""
        for p in self.procedures:
            if p.name.lower() == name.lower():
                return p
        return None

    def get_tool_memories_for_problem(self, problem: str) -> List[ToolMethodMemory]:
        """Find tool/method memories that match a problem context."""
        return [t for t in self.tool_memories if problem.lower() in t.problem_context.lower()]


