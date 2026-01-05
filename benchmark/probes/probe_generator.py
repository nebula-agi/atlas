from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

import requests

from .difficulty import compute_difficulty
from ..world.schema import Entity, World
from ..sessions.session_generator import Session


PillarType = str

# Canonical pillar names aligned with pillars.md
PILLAR_WORLD_MODELING = "world_modeling"
PILLAR_DECLARATIVE = "declarative_reasoning"
PILLAR_TEMPORAL = "temporal_episodic"
PILLAR_PREFERENCE = "preference_learning"
PILLAR_KNOWLEDGE_BOUNDARIES = "knowledge_boundaries"
PILLAR_PROCEDURAL = "procedural_knowledge"

ALL_PILLARS = [
    PILLAR_WORLD_MODELING,
    PILLAR_DECLARATIVE,
    PILLAR_TEMPORAL,
    PILLAR_PREFERENCE,
    PILLAR_KNOWLEDGE_BOUNDARIES,
    PILLAR_PROCEDURAL,
]

# Subpillar mappings for more granular probe generation
SUBPILLARS = {
    PILLAR_WORLD_MODELING: [
        "entity_resolution",
        "relationship_mapping", 
        "type_category_membership",
        "task_state_tracking",
    ],
    PILLAR_DECLARATIVE: [
        "baseline_factual_recall",
        "fact_composition",
        "constraint_propagation",
        "belief_revision",
        "verbatim_recall",
    ],
    PILLAR_TEMPORAL: [
        "temporal_sequencing",
        "episode_reconstruction",
        "causal_explanation",
        "cyclical_event_recognition",
    ],
    PILLAR_PREFERENCE: [
        "explicit_preferences",
        "preference_induction",
        "preference_scope",
        "preference_hierarchies",
        "preference_drift",
        "constraint_hierarchy",
    ],
    PILLAR_KNOWLEDGE_BOUNDARIES: [
        "negative_knowledge",
        "temporal_relevance_decay",
        "confidence_calibration",
    ],
    PILLAR_PROCEDURAL: [
        "lesson_extraction",
        "lesson_application",
        "procedure_storage",
        "tool_method_memory",
    ],
}


def _normalize_single_pillar(pillar: str) -> Optional[str]:
    """Normalize a single pillar name to its canonical slug, or None if unrecognized."""
    if not pillar:
        return None
    slug = pillar.lower().replace(" ", "_").replace("&", "and")
    if "world" in slug and "model" in slug:
        return PILLAR_WORLD_MODELING
    elif "declarative" in slug:
        return PILLAR_DECLARATIVE
    elif "temporal" in slug or "episodic" in slug:
        return PILLAR_TEMPORAL
    elif "preference" in slug:
        return PILLAR_PREFERENCE
    elif "knowledge" in slug and "bound" in slug:
        return PILLAR_KNOWLEDGE_BOUNDARIES
    elif "procedural" in slug:
        return PILLAR_PROCEDURAL
    # Legacy support
    elif "memory" in slug and "cycle" in slug:
        return PILLAR_KNOWLEDGE_BOUNDARIES
    # Check for exact match with canonical names
    elif slug in ALL_PILLARS:
        return slug
    return None


def _normalize_pillars(pillars: Optional[List[str]]) -> Set[str]:
    """Normalize pillar names to canonical internal slugs for filtering."""
    if not pillars:
        return set(ALL_PILLARS)
    normalized = set()
    for p in pillars:
        canonical = _normalize_single_pillar(p)
        if canonical:
            normalized.add(canonical)
    return normalized if normalized else set(ALL_PILLARS)


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return text.strip()


@dataclass
class GoldAnswer:
    text: str
    supporting_items: List[str]


@dataclass
class Probe:
    id: str
    pillar: PillarType
    subpillar: Optional[str]
    question: str
    answer_type: str
    required_world_items: List[str]
    gold_answer: GoldAnswer
    difficulty: Dict[str, int]


class ProbeLLM:
    def generate(self, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


try:
    from google import genai
    from google.genai import types
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False


class VertexProbeLLM(ProbeLLM):
    """
    Google Vertex AI / Gemini API client for probe generation.
    Requires: pip install google-genai
    Set GOOGLE_API_KEY env var or use application default credentials.
    """
    
    def __init__(self, model: str = "gemini-3-pro-preview"):
        if not VERTEX_AVAILABLE:
            raise ImportError("google-genai not installed. Run: pip install google-genai")
        
        self.model = model
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client(vertexai=True)
    
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        import time
        system_instruction = (
            "You generate evaluation probes for a long-context memory benchmark. "
            "Return ONLY a valid JSON array of probe objects. "
            "Do NOT wrap the JSON in markdown code fences. "
            "Output raw JSON only, starting with [ and ending with ]."
        )
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        max_output_tokens=16000,
                        temperature=0.4,
                    ),
                )
                
                # Extract only text parts, ignoring thought_signature and other non-text parts
                content = ""
                if response.candidates:
                    for candidate in response.candidates:
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    content += part.text
                
                if not content:
                    print(f"[VertexProbeLLM] Warning: Empty response")
                
                return content
            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt
                print(f"[VertexProbeLLM] Error: {e}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
        
        print(f"[VertexProbeLLM] Failed after {max_retries} retries: {last_error}")
        return ""


class OpenRouterProbeLLM(ProbeLLM):
    """
    OpenRouter-backed probe generator client (fallback).
    Expects OPENROUTER_API_KEY in the environment.
    """

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model: str = "google/gemini-3-pro-preview"):
        self.api_key = os.environ["OPENROUTER_API_KEY"]
        self.model = model

    def generate(self, prompt: str, max_retries: int = 3) -> str:
        import time
        last_error = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    self.OPENROUTER_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You generate evaluation probes for a long-context memory benchmark. "
                                    "Return ONLY a valid JSON array of probe objects. "
                                    "Do NOT wrap the JSON in markdown code fences. "
                                    "Output raw JSON only, starting with [ and ending with ]."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 16000,
                        "temperature": 0.4,
                    },
                    timeout=180,
                )
                resp.raise_for_status()
                data = resp.json()
                
                choice = data.get("choices", [{}])[0]
                content = choice.get("message", {}).get("content", "")
                
                if not content:
                    print(f"[ProbeGenerator] Warning: Empty response from {self.model}")
                
                return content or ""
            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt
                print(f"[ProbeGenerator] Error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
        
        print(f"[ProbeGenerator] Failed after {max_retries} retries: {last_error}")
        return ""


class LlmProbeGenerator:
    """
    LLM-backed probe generator that conditions on the latent world and sessions.
    Falls back to heuristic probes if the LLM output is empty or malformed.
    Covers all 6 pillars with subpillar granularity.
    """

    def __init__(
        self,
        world: World,
        sessions: Sequence[Session],
        rng: random.Random,
        model: OpenRouterProbeLLM,
        pillars: Optional[List[str]] = None,
    ):
        self.world = world
        self.sessions = sessions
        self.rng = rng
        self.model = model
        self.pillar_set = _normalize_pillars(pillars)

    def _active(self, key: str) -> bool:
        return key in self.pillar_set

    def generate(self) -> List[Probe]:
        prompt = self._build_prompt()
        raw = self.model.generate(prompt)
        probes = self._parse_raw_probes(raw)
        if probes:
            return probes
        return self._fallback_probes()

    def _build_prompt(self) -> str:
        # Build entity summary with aliases
        entities = []
        for e in self.world.entities.values():
            aliases = ", ".join(a.surface_form for a in e.aliases) if e.aliases else "(no aliases)"
            categories = ", ".join(e.category_memberships) if e.category_memberships else ""
            cat_str = f" [categories: {categories}]" if categories else ""
            entities.append(f"- {e.id} [{e.type}]: aliases=({aliases}){cat_str}")
        
        facts = [f"- {f.subject_id} {f.predicate} {getattr(f.object, 'entity_id', f.object)}" for f in self.world.facts]
        prefs = [f"- {p.subject_id}: {p.signal} (scope: {p.scope or 'general'}, rank: {p.rank}, hard: {p.is_hard_constraint})" for p in self.world.preferences]
        
        events = []
        for ev in self.world.events.values():
            steps = "; ".join(f"{s.actor.entity_id} {s.description}" for s in ev.steps) if getattr(ev, "steps", None) else ""
            events.append(f"- {ev.id} ({ev.name}): {steps}" if steps else f"- {ev.id} ({ev.name})")
        
        tasks = [f"- {t.id}: {t.description} ({t.status})" for t in self.world.tasks]
        verbatim = [f"- {v.id} [{v.item_type}]: {v.description}" for v in self.world.verbatim_items]
        patterns = [f"- {p.id}: {p.description}" for p in self.world.cyclical_patterns]
        lessons = [f"- {l.id}: {l.lesson_text[:50]}..." for l in self.world.lessons]
        procedures = [f"- {p.id} ({p.name}): {len(p.steps)} steps" for p in self.world.procedures]
        tool_mems = [f"- {t.id}: {t.problem_context} -> {t.solution} (worked: {t.worked})" for t in self.world.tool_memories]
        neg_knowledge = [f"- {n.topic}: {n.description}" for n in self.world.negative_knowledge]
        
        sessions_preview = self._session_preview(limit=6)
        pillar_list = ", ".join(sorted(self.pillar_set))
        
        # Build subpillar descriptions
        subpillar_desc = []
        for pillar in self.pillar_set:
            if pillar in SUBPILLARS:
                subs = SUBPILLARS[pillar]
                subpillar_desc.append(f"  {pillar}: {', '.join(subs)}")
        
        example_probe = {
            "id": "world_modeling-entity_resolution-0",
            "pillar": "world_modeling",
            "subpillar": "entity_resolution",
            "question": "Who is referred to as 'the guy who dropped out of Berkeley'?",
            "answer": "Ashvath",
            "answer_type": "short_answer",
            "required_world_items": ["founder"],
            "difficulty": {"hops": 1, "alias_depth": 3, "time_gap": 5, "distractors": 3},
            "supporting_items": ["aliases"]
        }
        
        return f"""Generate evaluation probes for a long-context memory benchmark covering 6 pillars.

ENTITIES:
{chr(10).join(entities) if entities else "- (none)"}

FACTS:
{chr(10).join(facts) if facts else "- (none)"}

PREFERENCES:
{chr(10).join(prefs) if prefs else "- (none)"}

EVENTS:
{chr(10).join(events) if events else "- (none)"}

TASKS:
{chr(10).join(tasks) if tasks else "- (none)"}

VERBATIM ITEMS:
{chr(10).join(verbatim) if verbatim else "- (none)"}

CYCLICAL PATTERNS:
{chr(10).join(patterns) if patterns else "- (none)"}

LESSONS:
{chr(10).join(lessons) if lessons else "- (none)"}

PROCEDURES:
{chr(10).join(procedures) if procedures else "- (none)"}

TOOL MEMORIES:
{chr(10).join(tool_mems) if tool_mems else "- (none)"}

NEGATIVE KNOWLEDGE (topics NOT mentioned):
{chr(10).join(neg_knowledge) if neg_knowledge else "- (none)"}

SESSION SNIPPETS:
{sessions_preview}

ACTIVE PILLARS: {pillar_list}

SUBPILLARS TO COVER:
{chr(10).join(subpillar_desc)}

PILLAR DESCRIPTIONS:
1. world_modeling: Entity resolution (coreference), relationship mapping, type/category membership, task/state tracking
2. declarative_reasoning: Factual recall, multi-hop reasoning, constraint propagation, belief revision, verbatim recall
3. temporal_episodic: Temporal sequencing, episode reconstruction, causal explanation, cyclical patterns
4. preference_learning: Explicit preferences, preference induction, scope, hierarchies, drift, soft vs hard constraints
5. knowledge_boundaries: Negative knowledge (what's NOT known), temporal relevance decay, confidence calibration
6. procedural_knowledge: Lesson extraction/application, procedure storage, tool/method memory

INSTRUCTIONS:
1. Generate 2-3 probes per active pillar, covering different subpillars
2. Questions should require reasoning, not verbatim recall (except for verbatim_recall subpillar)
3. Use indirect references and aliases in questions
4. Keep answers concise (1-3 words for short_answer)
5. For knowledge_boundaries pillar, ask about topics NOT in the world (answer should be "unknown"/"not mentioned")
6. For belief_revision, test that old facts are superseded by new ones
7. For constraint_hierarchy, test soft vs hard constraint tradeoffs
8. Include the subpillar field to track which specific capability is tested

OUTPUT FORMAT:
Return a JSON array. Example probe:
{json.dumps(example_probe, indent=2)}

answer_type options: "short_answer", "boolean", "generation", "abstain", "verbatim"

Generate probes now. Output ONLY the JSON array, no other text:"""

    def _session_preview(self, limit: int = 6) -> str:
        lines: List[str] = []
        for session in list(self.sessions)[-limit:]:
            session_lines = []
            for turn in session.turns[:6]:
                session_lines.append(f"  {turn.speaker}: {turn.text}")
            lines.append(f"[Session {session.id}]\n" + "\n".join(session_lines))
        return "\n\n".join(lines) if lines else "(no sessions)"

    def _parse_raw_probes(self, raw: str) -> List[Probe]:
        cleaned = _strip_markdown_fences(raw)
        
        if not cleaned.startswith("["):
            start = cleaned.find("[")
            end = cleaned.rfind("]")
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start:end + 1]
        
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"[ProbeGenerator] JSON parse error: {e}")
            print(f"[ProbeGenerator] Raw response (first 500 chars): {raw[:500]}")
            return []
        
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            print(f"[ProbeGenerator] Expected list, got {type(data)}")
            return []

        probes: List[Probe] = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            
            pillar_raw = item.get("pillar") or ""
            pillar_slug = _normalize_single_pillar(pillar_raw)
            
            if pillar_slug is None:
                # Unrecognized pillar - mark as unspecified rather than silently assigning
                pillar_slug = "unspecified"
                print(f"[ProbeGenerator] Warning: probe {idx} has unrecognized pillar '{pillar_raw}', marking as unspecified")
            
            # Skip if pillar is not in the active set (but allow unspecified through)
            if pillar_slug != "unspecified" and pillar_slug not in self.pillar_set:
                continue
            
            question = item.get("question")
            answer = item.get("answer") or item.get("gold_answer")
            if not question or answer is None:
                print(f"[ProbeGenerator] Skipping probe {idx}: missing question or answer")
                continue
            
            answer_type = item.get("answer_type", "short_answer")
            if answer_type not in ("short_answer", "boolean", "generation", "abstain", "verbatim"):
                answer_type = "short_answer"
            
            required = item.get("required_world_items") or []
            if not isinstance(required, list):
                required = [str(required)] if required else []
            
            difficulty = item.get("difficulty")
            if not isinstance(difficulty, dict):
                difficulty = compute_difficulty(hops=1, alias_depth=1, time_gap=5, distractors=3)
            else:
                difficulty = {
                    "hops": difficulty.get("hops", 1),
                    "alias_depth": difficulty.get("alias_depth", 1),
                    "time_gap": difficulty.get("time_gap", 5),
                    "distractors": difficulty.get("distractors", 3),
                }
            
            supporting = item.get("supporting_items") or []
            if not isinstance(supporting, list):
                supporting = [str(supporting)] if supporting else []
            
            subpillar = item.get("subpillar")
            probe_id = item.get("id") or f"{pillar_slug}-{subpillar or 'general'}-{idx}"
            
            probe = Probe(
                id=str(probe_id),
                pillar=pillar_slug or "unspecified",
                subpillar=subpillar,
                question=str(question),
                answer_type=str(answer_type),
                required_world_items=[str(r) for r in required],
                gold_answer=GoldAnswer(text=str(answer), supporting_items=[str(s) for s in supporting]),
                difficulty={k: int(v) for k, v in difficulty.items()},
            )
            probes.append(probe)
        
        if probes:
            print(f"[ProbeGenerator] Successfully parsed {len(probes)} probes from LLM")
        return probes

    # ========== Heuristic Fallback Probes ==========
    
    def _fallback_probes(self) -> List[Probe]:
        print("[ProbeGenerator] Falling back to heuristic probes")
        probes: List[Probe] = []
        
        if self._active(PILLAR_WORLD_MODELING):
            probes.extend(self._world_modeling_probes())
        if self._active(PILLAR_DECLARATIVE):
            probes.extend(self._declarative_probes())
        if self._active(PILLAR_TEMPORAL):
            probes.extend(self._temporal_probes())
        if self._active(PILLAR_PREFERENCE):
            probes.extend(self._preference_probes())
        if self._active(PILLAR_KNOWLEDGE_BOUNDARIES):
            probes.extend(self._knowledge_boundary_probes())
        if self._active(PILLAR_PROCEDURAL):
            probes.extend(self._procedural_probes())
        
        return probes

    # === Pillar 1: World Modeling ===
    
    def _world_modeling_probes(self) -> List[Probe]:
        probes = []
        
        # 1.1 Entity Resolution
        entities = [e for e in self.world.entities.values() if len(e.aliases) >= 2]
        if entities:
            entity = self.rng.choice(entities)
            alias = self.rng.choice(entity.aliases)
            gold = entity.persona or entity.id
            probes.append(Probe(
                id=f"{PILLAR_WORLD_MODELING}-entity_resolution-0",
                pillar=PILLAR_WORLD_MODELING,
                subpillar="entity_resolution",
                question=f"Who is referred to as '{alias.surface_form}'?",
                answer_type="short_answer",
                required_world_items=[entity.id],
                gold_answer=GoldAnswer(text=gold, supporting_items=["aliases"]),
                difficulty=compute_difficulty(hops=1, alias_depth=len(entity.aliases), time_gap=5, distractors=3),
            ))
        
        # 1.2 Relationship Mapping
        rel_facts = [f for f in self.world.facts if f.predicate in ("reports_to", "led", "works_with")]
        if rel_facts:
            fact = self.rng.choice(rel_facts)
            if hasattr(fact.object, 'entity_id'):
                obj_entity = self.world.entities.get(fact.object.entity_id)
                obj_name = obj_entity.aliases[0].surface_form if obj_entity and obj_entity.aliases else fact.object.entity_id
                probes.append(Probe(
                    id=f"{PILLAR_WORLD_MODELING}-relationship_mapping-0",
                    pillar=PILLAR_WORLD_MODELING,
                    subpillar="relationship_mapping",
                    question=f"What is the relationship between {fact.subject_id} and {obj_name}?",
                    answer_type="short_answer",
                    required_world_items=[fact.subject_id],
                    gold_answer=GoldAnswer(text=fact.predicate.replace("_", " "), supporting_items=["facts"]),
                    difficulty=compute_difficulty(hops=1, alias_depth=1, time_gap=8, distractors=4),
                ))
        
        # 1.3 Type/Category Membership
        entities_with_categories = [e for e in self.world.entities.values() if e.category_memberships]
        if entities_with_categories:
            entity = self.rng.choice(entities_with_categories)
            category = self.rng.choice(entity.category_memberships)
            probes.append(Probe(
                id=f"{PILLAR_WORLD_MODELING}-type_category-0",
                pillar=PILLAR_WORLD_MODELING,
                subpillar="type_category_membership",
                question=f"Is {entity.id} a {category}?",
                answer_type="boolean",
                required_world_items=[entity.id],
                gold_answer=GoldAnswer(text="yes", supporting_items=["category_memberships"]),
                difficulty=compute_difficulty(hops=1, alias_depth=0, time_gap=3, distractors=2),
            ))
        
        # 1.4 Task/State Tracking
        if self.world.tasks:
            open_tasks = [t for t in self.world.tasks if t.status == "open"]
            if open_tasks:
                task = self.rng.choice(open_tasks)
                probes.append(Probe(
                    id=f"{PILLAR_WORLD_MODELING}-task_tracking-0",
                    pillar=PILLAR_WORLD_MODELING,
                    subpillar="task_state_tracking",
                    question=f"What is the status of the task: {task.description}?",
                    answer_type="short_answer",
                    required_world_items=[task.id],
                    gold_answer=GoldAnswer(text=task.status, supporting_items=["tasks"]),
                    difficulty=compute_difficulty(hops=1, alias_depth=0, time_gap=5, distractors=3),
                ))
            
            blocked_tasks = [t for t in self.world.tasks if t.status == "blocked"]
            if blocked_tasks:
                task = self.rng.choice(blocked_tasks)
                blocker = task.blocked_by[0] if task.blocked_by else "unknown"
                probes.append(Probe(
                    id=f"{PILLAR_WORLD_MODELING}-task_dependency-0",
                    pillar=PILLAR_WORLD_MODELING,
                    subpillar="task_state_tracking",
                    question=f"What is blocking the task: {task.description}?",
                    answer_type="short_answer",
                    required_world_items=[task.id],
                    gold_answer=GoldAnswer(text=blocker, supporting_items=["tasks"]),
                    difficulty=compute_difficulty(hops=2, alias_depth=0, time_gap=5, distractors=3),
                ))
        
        return probes

    # === Pillar 2: Declarative Reasoning ===
    
    def _declarative_probes(self) -> List[Probe]:
        probes = []
        
        # 2.1 Baseline Factual Recall
        if self.world.facts:
            fact = self.rng.choice(self.world.facts)
            obj_text = fact.object if isinstance(fact.object, str) else fact.object.entity_id
            probes.append(Probe(
                id=f"{PILLAR_DECLARATIVE}-factual_recall-0",
                pillar=PILLAR_DECLARATIVE,
                subpillar="baseline_factual_recall",
                question=f"What does {fact.subject_id} {fact.predicate.replace('_', ' ')}?",
                answer_type="short_answer",
                required_world_items=[fact.subject_id],
                gold_answer=GoldAnswer(text=obj_text, supporting_items=["facts"]),
                difficulty=compute_difficulty(hops=1, alias_depth=0, time_gap=3, distractors=2),
            ))
        
        # 2.2 Fact Composition (multi-hop)
        dep_facts = [f for f in self.world.facts if f.predicate == "depends_on"]
        deploy_facts = [f for f in self.world.facts if f.predicate == "deployed_in"]
        if dep_facts and deploy_facts:
            dep = self.rng.choice(dep_facts)
            if hasattr(dep.object, 'entity_id'):
                target_id = dep.object.entity_id
                region_fact = next((f for f in deploy_facts if f.subject_id == target_id), None)
                if region_fact:
                    region = region_fact.object if isinstance(region_fact.object, str) else "unknown"
                    probes.append(Probe(
                        id=f"{PILLAR_DECLARATIVE}-fact_composition-0",
                        pillar=PILLAR_DECLARATIVE,
                        subpillar="fact_composition",
                        question=f"Where is the dependency of {dep.subject_id} deployed?",
                        answer_type="short_answer",
                        required_world_items=[dep.subject_id, target_id],
                        gold_answer=GoldAnswer(text=region, supporting_items=["depends_on", "deployed_in"]),
                        difficulty=compute_difficulty(hops=2, alias_depth=1, time_gap=8, distractors=5),
                    ))
        
        # 2.3 Constraint Propagation
        unavailable = [f for f in self.world.facts if f.predicate == "status" and f.object == "unavailable"]
        if unavailable:
            fact = unavailable[0]
            # Find what depends on this
            dependents = [f for f in self.world.facts if f.predicate == "depends_on" and hasattr(f.object, 'entity_id') and f.object.entity_id == fact.subject_id]
            if dependents:
                dep = dependents[0]
                probes.append(Probe(
                    id=f"{PILLAR_DECLARATIVE}-constraint_prop-0",
                    pillar=PILLAR_DECLARATIVE,
                    subpillar="constraint_propagation",
                    question=f"Is {dep.subject_id} available when {fact.subject_id} is unavailable?",
                    answer_type="boolean",
                    required_world_items=[dep.subject_id, fact.subject_id],
                    gold_answer=GoldAnswer(text="no", supporting_items=["depends_on", "status"]),
                    difficulty=compute_difficulty(hops=2, alias_depth=0, time_gap=5, distractors=4),
                ))
        
        # 2.4 Belief Revision
        superseding = [f for f in self.world.facts if f.supersedes]
        if superseding:
            fact = self.rng.choice(superseding)
            new_value = fact.object if isinstance(fact.object, str) else "updated"
            probes.append(Probe(
                id=f"{PILLAR_DECLARATIVE}-belief_revision-0",
                pillar=PILLAR_DECLARATIVE,
                subpillar="belief_revision",
                question=f"What is the current {fact.predicate.replace('_', ' ')} for {fact.subject_id}?",
                answer_type="short_answer",
                required_world_items=[fact.subject_id],
                gold_answer=GoldAnswer(text=new_value, supporting_items=["supersedes"]),
                difficulty=compute_difficulty(hops=1, alias_depth=0, time_gap=10, distractors=4),
            ))
        
        # 2.5 Verbatim Recall
        if self.world.verbatim_items:
            item = self.rng.choice(self.world.verbatim_items)
            probes.append(Probe(
                id=f"{PILLAR_DECLARATIVE}-verbatim_recall-0",
                pillar=PILLAR_DECLARATIVE,
                subpillar="verbatim_recall",
                question=f"What was the exact {item.item_type.replace('_', ' ')} for {item.description}?",
                answer_type="verbatim",
                required_world_items=[item.id],
                gold_answer=GoldAnswer(text=item.content, supporting_items=["verbatim_items"]),
                difficulty=compute_difficulty(hops=1, alias_depth=0, time_gap=8, distractors=2),
            ))
        
        return probes

    # === Pillar 3: Temporal & Episodic ===
    
    def _temporal_probes(self) -> List[Probe]:
        probes = []
        events = list(self.world.events.values())
        
        # 3.1 Temporal Sequencing
        if len(events) >= 2:
            e1, e2 = events[0], events[1]
            if e1.start_time < e2.start_time:
                answer = e1.name
            else:
                answer = e2.name
            probes.append(Probe(
                id=f"{PILLAR_TEMPORAL}-temporal_sequencing-0",
                pillar=PILLAR_TEMPORAL,
                subpillar="temporal_sequencing",
                question=f"Which happened first: {e1.name} or {e2.name}?",
                answer_type="short_answer",
                required_world_items=[e1.id, e2.id],
                gold_answer=GoldAnswer(text=answer, supporting_items=["events"]),
                difficulty=compute_difficulty(hops=1, alias_depth=0, time_gap=10, distractors=3),
            ))
        
        # 3.2 Episode Reconstruction
        if events and events[0].steps:
            event = events[0]
            steps_desc = [s.description for s in event.steps]
            probes.append(Probe(
                id=f"{PILLAR_TEMPORAL}-episode_reconstruction-0",
                pillar=PILLAR_TEMPORAL,
                subpillar="episode_reconstruction",
                question=f"What happened during {event.name}?",
                answer_type="generation",
                required_world_items=[event.id],
                gold_answer=GoldAnswer(text="; ".join(steps_desc), supporting_items=["event.steps"]),
                difficulty=compute_difficulty(hops=2, alias_depth=1, time_gap=12, distractors=4),
            ))
        
        # 3.3 Causal Explanation
        for event in events:
            if event.causal_links:
                link = event.causal_links[0]
                if link.source_step < len(event.steps):
                    cause = event.steps[link.source_step]
                    probes.append(Probe(
                        id=f"{PILLAR_TEMPORAL}-causal_explanation-0",
                        pillar=PILLAR_TEMPORAL,
                        subpillar="causal_explanation",
                        question=f"What caused the investigation during {event.name}?",
                        answer_type="short_answer",
                        required_world_items=[event.id],
                        gold_answer=GoldAnswer(text=cause.description, supporting_items=["causal_links"]),
                        difficulty=compute_difficulty(hops=2, alias_depth=0, time_gap=8, distractors=4),
                    ))
                break
        
        # 3.4 Cyclical Event Recognition
        if self.world.cyclical_patterns:
            pattern = self.rng.choice(self.world.cyclical_patterns)
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            if pattern.day_of_week is not None:
                day = day_names[pattern.day_of_week]
                probes.append(Probe(
                    id=f"{PILLAR_TEMPORAL}-cyclical_recognition-0",
                    pillar=PILLAR_TEMPORAL,
                    subpillar="cyclical_event_recognition",
                    question=f"What does the user typically do on {day}s?",
                    answer_type="short_answer",
                    required_world_items=[pattern.id],
                    gold_answer=GoldAnswer(text=pattern.description, supporting_items=["cyclical_patterns"]),
                    difficulty=compute_difficulty(hops=1, alias_depth=0, time_gap=5, distractors=3),
                ))
        
        return probes

    # === Pillar 4: Preference Learning ===
    
    def _preference_probes(self) -> List[Probe]:
        probes = []
        
        # 4.1 Explicit Preferences
        explicit = [p for p in self.world.preferences if not p.inferred]
        if explicit:
            pref = self.rng.choice(explicit)
            probes.append(Probe(
                id=f"{PILLAR_PREFERENCE}-explicit-0",
                pillar=PILLAR_PREFERENCE,
                subpillar="explicit_preferences",
                question=f"What is {pref.subject_id}'s preference for {pref.scope or 'general'}?",
                answer_type="short_answer",
                required_world_items=[pref.subject_id],
                gold_answer=GoldAnswer(text=pref.signal, supporting_items=["preferences"]),
                difficulty=compute_difficulty(hops=1, alias_depth=0, time_gap=5, distractors=3),
            ))
        
        # 4.3 Preference Scope (context-dependent)
        scoped = [p for p in self.world.preferences if p.scope]
        if len(scoped) >= 2:
            # Find seemingly conflicting preferences
            p1 = next((p for p in scoped if "hate" in p.signal.lower() or "never" in p.signal.lower()), None)
            p2 = next((p for p in scoped if "love" in p.signal.lower() or "always" in p.signal.lower()), None)
            if p1 and p2 and p1.subject_id == p2.subject_id:
                probes.append(Probe(
                    id=f"{PILLAR_PREFERENCE}-preference_scope-0",
                    pillar=PILLAR_PREFERENCE,
                    subpillar="preference_scope",
                    question=f"Does {p1.subject_id} like or dislike meetings in the morning?",
                    answer_type="generation",
                    required_world_items=[p1.subject_id],
                    gold_answer=GoldAnswer(text=f"Dislikes morning meetings in general, but likes 1:1s anytime", supporting_items=["preferences"]),
                    difficulty=compute_difficulty(hops=2, alias_depth=0, time_gap=5, distractors=4),
                ))
        
        # 4.4 Preference Hierarchies
        ranked = [p for p in self.world.preferences if p.rank is not None]
        if ranked:
            # Group by scope
            by_scope = {}
            for p in ranked:
                by_scope.setdefault(p.scope, []).append(p)
            for scope, prefs in by_scope.items():
                if len(prefs) >= 2:
                    prefs.sort(key=lambda x: x.rank or 999)
                    top = prefs[0]
                    probes.append(Probe(
                        id=f"{PILLAR_PREFERENCE}-hierarchy-0",
                        pillar=PILLAR_PREFERENCE,
                        subpillar="preference_hierarchies",
                        question=f"What is the top preference for {scope}?",
                        answer_type="short_answer",
                        required_world_items=[top.subject_id],
                        gold_answer=GoldAnswer(text=top.signal, supporting_items=["preferences"]),
                        difficulty=compute_difficulty(hops=1, alias_depth=0, time_gap=5, distractors=len(prefs)),
                    ))
                    break
        
        # 4.5 Preference Drift
        drifted = [p for p in self.world.preferences if p.inferred and p.last_mentioned and p.first_mentioned]
        if drifted:
            pref = self.rng.choice(drifted)
            probes.append(Probe(
                id=f"{PILLAR_PREFERENCE}-drift-0",
                pillar=PILLAR_PREFERENCE,
                subpillar="preference_drift",
                question=f"Has {pref.subject_id}'s enthusiasm for {pref.scope or 'this'} changed over time?",
                answer_type="boolean",
                required_world_items=[pref.subject_id],
                gold_answer=GoldAnswer(text="yes", supporting_items=["preferences"]),
                difficulty=compute_difficulty(hops=1, alias_depth=0, time_gap=15, distractors=3),
            ))
        
        # 4.6 Constraint Hierarchy (soft vs hard)
        hard = [p for p in self.world.preferences if p.is_hard_constraint]
        soft = [p for p in self.world.preferences if not p.is_hard_constraint and p.scope]
        if hard and soft:
            h = hard[0]
            s = soft[0]
            probes.append(Probe(
                id=f"{PILLAR_PREFERENCE}-constraint_hierarchy-0",
                pillar=PILLAR_PREFERENCE,
                subpillar="constraint_hierarchy",
                question=f"If you had to choose between satisfying '{s.signal}' or '{h.signal}', which should NEVER be violated?",
                answer_type="short_answer",
                required_world_items=[h.subject_id],
                gold_answer=GoldAnswer(text=h.signal, supporting_items=["is_hard_constraint"]),
                difficulty=compute_difficulty(hops=1, alias_depth=0, time_gap=5, distractors=2),
            ))
        
        return probes

    # === Pillar 5: Knowledge Boundaries ===
    
    def _knowledge_boundary_probes(self) -> List[Probe]:
        probes = []
        
        # 5.1 Negative Knowledge
        if self.world.negative_knowledge:
            neg = self.rng.choice(self.world.negative_knowledge)
            probes.append(Probe(
                id=f"{PILLAR_KNOWLEDGE_BOUNDARIES}-negative_knowledge-0",
                pillar=PILLAR_KNOWLEDGE_BOUNDARIES,
                subpillar="negative_knowledge",
                question=f"Have I ever mentioned my {neg.topic}?",
                answer_type="abstain",
                required_world_items=[],
                gold_answer=GoldAnswer(text="No, not mentioned", supporting_items=["negative_knowledge"]),
                difficulty=compute_difficulty(hops=0, alias_depth=0, time_gap=1, distractors=6),
            ))
        else:
            # Fallback negative knowledge test
            probes.append(Probe(
                id=f"{PILLAR_KNOWLEDGE_BOUNDARIES}-negative_knowledge-fallback",
                pillar=PILLAR_KNOWLEDGE_BOUNDARIES,
                subpillar="negative_knowledge",
                question="Have I ever mentioned ski plans?",
                answer_type="abstain",
                required_world_items=[],
                gold_answer=GoldAnswer(text="No, not mentioned", supporting_items=[]),
                difficulty=compute_difficulty(hops=0, alias_depth=0, time_gap=1, distractors=6),
            ))
        
        # 5.2 Temporal Relevance Decay
        old_facts = [f for f in self.world.facts if f.validity.end_time is not None and f.validity.end_time < 15]
        if old_facts:
            fact = self.rng.choice(old_facts)
            probes.append(Probe(
                id=f"{PILLAR_KNOWLEDGE_BOUNDARIES}-temporal_decay-0",
                pillar=PILLAR_KNOWLEDGE_BOUNDARIES,
                subpillar="temporal_relevance_decay",
                question=f"Should the fact that {fact.subject_id} {fact.predicate} {fact.object if isinstance(fact.object, str) else fact.object.entity_id} be surfaced for current queries?",
                answer_type="boolean",
                required_world_items=[fact.subject_id],
                gold_answer=GoldAnswer(text="no", supporting_items=["validity.end_time"]),
                difficulty=compute_difficulty(hops=1, alias_depth=0, time_gap=15, distractors=3),
            ))
        
        # 5.3 Confidence Calibration
        low_confidence = [f for f in self.world.facts if f.confidence < 0.7]
        if low_confidence:
            fact = self.rng.choice(low_confidence)
            probes.append(Probe(
                id=f"{PILLAR_KNOWLEDGE_BOUNDARIES}-confidence-0",
                pillar=PILLAR_KNOWLEDGE_BOUNDARIES,
                subpillar="confidence_calibration",
                question=f"How certain is the fact that {fact.subject_id} {fact.predicate}?",
                answer_type="generation",
                required_world_items=[fact.subject_id],
                gold_answer=GoldAnswer(text=f"Low confidence ({fact.confidence})", supporting_items=["confidence"]),
                difficulty=compute_difficulty(hops=1, alias_depth=0, time_gap=5, distractors=2),
            ))
        
        return probes

    # === Pillar 6: Procedural Knowledge ===
    
    def _procedural_probes(self) -> List[Probe]:
        probes = []
        
        # 6.1 Lesson Extraction
        systemic_lessons = [l for l in self.world.lessons if l.is_systemic]
        if systemic_lessons:
            lesson = self.rng.choice(systemic_lessons)
            probes.append(Probe(
                id=f"{PILLAR_PROCEDURAL}-lesson_extraction-0",
                pillar=PILLAR_PROCEDURAL,
                subpillar="lesson_extraction",
                question=f"What lesson was learned about {lesson.trigger_context}?",
                answer_type="generation",
                required_world_items=[lesson.id],
                gold_answer=GoldAnswer(text=lesson.lesson_text, supporting_items=["lessons"]),
                difficulty=compute_difficulty(hops=1, alias_depth=0, time_gap=8, distractors=3),
            ))
        
        # 6.2 Lesson Application
        if self.world.lessons:
            lesson = self.rng.choice(self.world.lessons)
            probes.append(Probe(
                id=f"{PILLAR_PROCEDURAL}-lesson_application-0",
                pillar=PILLAR_PROCEDURAL,
                subpillar="lesson_application",
                question=f"What should be remembered when {lesson.trigger_context}?",
                answer_type="generation",
                required_world_items=[lesson.id],
                gold_answer=GoldAnswer(text=lesson.lesson_text, supporting_items=["lessons"]),
                difficulty=compute_difficulty(hops=1, alias_depth=0, time_gap=5, distractors=4),
            ))
        
        # 6.3 Procedure Storage
        if self.world.procedures:
            proc = self.rng.choice(self.world.procedures)
            probes.append(Probe(
                id=f"{PILLAR_PROCEDURAL}-procedure_storage-0",
                pillar=PILLAR_PROCEDURAL,
                subpillar="procedure_storage",
                question=f"What are the steps for {proc.name}?",
                answer_type="generation",
                required_world_items=[proc.id],
                gold_answer=GoldAnswer(text="; ".join(proc.steps), supporting_items=["procedures"]),
                difficulty=compute_difficulty(hops=1, alias_depth=0, time_gap=5, distractors=2),
            ))
            
            # Conditional procedures
            if proc.conditions:
                cond_key = list(proc.conditions.keys())[0]
                probes.append(Probe(
                    id=f"{PILLAR_PROCEDURAL}-conditional_procedure-0",
                    pillar=PILLAR_PROCEDURAL,
                    subpillar="procedure_storage",
                    question=f"During {proc.name}, what should be done {cond_key}?",
                    answer_type="short_answer",
                    required_world_items=[proc.id],
                    gold_answer=GoldAnswer(text=proc.conditions[cond_key], supporting_items=["conditions"]),
                    difficulty=compute_difficulty(hops=2, alias_depth=0, time_gap=5, distractors=3),
                ))
        
        # 6.4 Tool/Method Memory
        if self.world.tool_memories:
            worked = [t for t in self.world.tool_memories if t.worked]
            didnt_work = [t for t in self.world.tool_memories if not t.worked]
            
            if worked:
                mem = self.rng.choice(worked)
                probes.append(Probe(
                    id=f"{PILLAR_PROCEDURAL}-tool_memory_worked-0",
                    pillar=PILLAR_PROCEDURAL,
                    subpillar="tool_method_memory",
                    question=f"What worked for fixing {mem.problem_context}?",
                    answer_type="short_answer",
                    required_world_items=[mem.id],
                    gold_answer=GoldAnswer(text=mem.solution, supporting_items=["tool_memories"]),
                    difficulty=compute_difficulty(hops=1, alias_depth=0, time_gap=8, distractors=3),
                ))
            
            if didnt_work:
                mem = self.rng.choice(didnt_work)
                probes.append(Probe(
                    id=f"{PILLAR_PROCEDURAL}-tool_memory_failed-0",
                    pillar=PILLAR_PROCEDURAL,
                    subpillar="tool_method_memory",
                    question=f"What was tried but didn't work for {mem.problem_context}?",
                    answer_type="short_answer",
                    required_world_items=[mem.id],
                    gold_answer=GoldAnswer(text=mem.solution, supporting_items=["tool_memories"]),
                    difficulty=compute_difficulty(hops=1, alias_depth=0, time_gap=8, distractors=3),
                ))
            
            # Environment-specific
            env_specific = [t for t in self.world.tool_memories if t.environment]
            if env_specific:
                mem = self.rng.choice(env_specific)
                probes.append(Probe(
                    id=f"{PILLAR_PROCEDURAL}-tool_memory_env-0",
                    pillar=PILLAR_PROCEDURAL,
                    subpillar="tool_method_memory",
                    question=f"Does the fix '{mem.solution}' work on {mem.environment}?",
                    answer_type="boolean",
                    required_world_items=[mem.id],
                    gold_answer=GoldAnswer(text="yes" if mem.worked else "no", supporting_items=["tool_memories"]),
                    difficulty=compute_difficulty(hops=1, alias_depth=0, time_gap=5, distractors=2),
                ))
        
        return probes
