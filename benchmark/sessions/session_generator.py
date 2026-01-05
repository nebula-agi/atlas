from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from .templates import TEMPLATES
from ..world.schema import Entity, EntityRef, Event, World


class SessionType(str, Enum):
    # Pillar 1: World Modeling
    INTRO = "INTRO"
    PROJECT_UPDATE = "PROJECT_UPDATE"
    ADMIN_CHANGE = "ADMIN_CHANGE"
    TASK_TRACKING = "TASK_TRACKING"
    RELATIONSHIP_UPDATE = "RELATIONSHIP_UPDATE"
    
    # Pillar 2: Declarative Reasoning
    FACT_CHAIN = "FACT_CHAIN"
    BELIEF_REVISION = "BELIEF_REVISION"
    VERBATIM_SHARE = "VERBATIM_SHARE"
    COUNTERFACTUAL = "COUNTERFACTUAL"
    
    # Pillar 3: Temporal & Episodic
    INCIDENT_LOG = "INCIDENT_LOG"
    TEMPORAL_NARRATIVE = "TEMPORAL_NARRATIVE"
    CYCLICAL_PATTERN = "CYCLICAL_PATTERN"
    CAUSAL_EXPLANATION = "CAUSAL_EXPLANATION"
    
    # Pillar 4: Preference Learning
    PREFERENCE_SIGNAL = "PREFERENCE_SIGNAL"
    PREFERENCE_RANKED = "PREFERENCE_RANKED"
    PREFERENCE_CONTEXTUAL = "PREFERENCE_CONTEXTUAL"
    HARD_CONSTRAINT = "HARD_CONSTRAINT"
    
    # Pillar 5: Knowledge Boundaries
    REFLECTION = "REFLECTION"
    STALE_CONTEXT = "STALE_CONTEXT"
    
    # Pillar 6: Procedural Knowledge
    LESSON_LEARNED = "LESSON_LEARNED"
    PROCEDURE_SHARE = "PROCEDURE_SHARE"
    TOOL_MEMORY = "TOOL_MEMORY"
    
    # Noise
    NOISE = "NOISE"


@dataclass
class Utterance:
    speaker: str
    text: str
    grounded_refs: List[EntityRef]


@dataclass
class Session:
    id: str
    timestamp: int
    turns: List[Utterance]


class SessionGenerator:
    """
    Generates long-context sessions while hiding gold answers.
    Produces multi-turn sessions (10–15 turns) and injects noise to keep info density low.
    Covers all 6 pillars with comprehensive session types.
    """

    def __init__(
        self,
        world: World,
        rng: random.Random,
        min_turns: int = 10,
        max_turns: int = 15,
    ) -> None:
        self.world = world
        self.rng = rng
        self.min_turns = min_turns
        self.max_turns = max_turns

    def _choose_template(self, key: str) -> str:
        if key not in TEMPLATES:
            return f"[template:{key}]"
        return self.rng.choice(TEMPLATES[key])

    def _pick_entity_by_type(self, type_: str) -> Optional[Entity]:
        candidates = [e for e in self.world.entities.values() if e.type == type_]
        if not candidates:
            all_entities = list(self.world.entities.values())
            return self.rng.choice(all_entities) if all_entities else None
        return self.rng.choice(candidates)

    def _indirect_entity_ref(self, entity: Entity, time: int) -> Tuple[str, EntityRef]:
        aliases = entity.active_aliases(time)
        if aliases:
            alias = self.rng.choice(aliases)
            return alias.surface_form, EntityRef(entity_id=entity.id, alias=alias.surface_form)
        return self._choose_template("entity_reference"), EntityRef(entity_id=entity.id)

    # === Pillar 1: World Modeling Sessions ===
    
    def _generate_intro(self, timestamp: int) -> Session:
        person = self._pick_entity_by_type("person")
        project = self._pick_entity_by_type("project")
        if not person or not project:
            return self._generate_noise(timestamp)
        surface, ref = self._indirect_entity_ref(person, timestamp)
        turns = [
            Utterance("user", f"Hi, I'm working with {surface} on {project.aliases[0].surface_form if project.aliases else project.id}.", [ref]),
            Utterance("assistant", "Got it. I'll keep track of your projects.", []),
        ]
        return Session(id=f"intro-{timestamp}", timestamp=timestamp, turns=turns)

    def _generate_project_update(self, timestamp: int) -> Session:
        dependency = next((f for f in self.world.facts if f.predicate == "depends_on" and isinstance(f.object, EntityRef)), None)
        if dependency:
            project = self.world.entities.get(dependency.subject_id, self._pick_entity_by_type("project"))
            dep_entity = self.world.entities.get(dependency.object.entity_id, self._pick_entity_by_type("service"))
        else:
            project = self._pick_entity_by_type("project")
            dep_entity = self._pick_entity_by_type("service")

        if not project or not dep_entity:
            return self._generate_noise(timestamp)

        surface_a, ref_a = self._indirect_entity_ref(project, timestamp)
        surface_b, ref_b = self._indirect_entity_ref(dep_entity, timestamp)
        template = self._choose_template("dependency")
        line = template.format(A=surface_a, B=surface_b)
        turns = [
            Utterance("user", f"Quick update: {line}.", [ref_a, ref_b]),
            Utterance("assistant", "Noted. I'll watch that dependency.", []),
        ]
        return Session(id=f"update-{timestamp}", timestamp=timestamp, turns=turns)

    def _generate_admin_change(self, timestamp: int) -> Session:
        candidate = None
        for entity in self.world.entities.values():
            if entity.roles:
                candidate = entity
                break
        if not candidate:
            candidate = self._pick_entity_by_type("person")
        if not candidate:
            return self._generate_noise(timestamp)
        surface, ref = self._indirect_entity_ref(candidate, timestamp)
        turns = [
            Utterance("user", f"{surface} is now leading a project.", [ref]),
            Utterance("assistant", "I'll record that change.", []),
        ]
        return Session(id=f"admin-{timestamp}", timestamp=timestamp, turns=turns)

    def _generate_task_tracking(self, timestamp: int) -> Session:
        """Pillar 1.4: Task/State Tracking"""
        if not self.world.tasks:
            return self._generate_noise(timestamp)
        task = self.rng.choice(self.world.tasks)
        assignee_text = ""
        refs = []
        if task.assignee:
            entity = self.world.entities.get(task.assignee.entity_id)
            if entity:
                surface, ref = self._indirect_entity_ref(entity, timestamp)
                assignee_text = f"{surface} needs to "
                refs.append(ref)
        
        if task.status == "open":
            text = f"{assignee_text}{task.description}. {task.commitment_text or 'Please follow up.'}"
        elif task.status == "blocked":
            blocker = task.blocked_by[0] if task.blocked_by else "something"
            text = f"The {task.description} is blocked by {blocker}."
        elif task.status == "completed":
            text = f"Good news: finished the {task.description}."
        else:
            text = f"Working on {task.description}."
        
        turns = [
            Utterance("user", text, refs),
            Utterance("assistant", "I've noted that task update.", []),
        ]
        return Session(id=f"task-{timestamp}", timestamp=timestamp, turns=turns)

    def _generate_relationship_update(self, timestamp: int) -> Session:
        """Pillar 1.2: Relationship Mapping"""
        # Find a relationship fact
        rel_fact = next((f for f in self.world.facts if f.predicate in ("reports_to", "led", "works_with")), None)
        if rel_fact and isinstance(rel_fact.object, EntityRef):
            subject = self.world.entities.get(rel_fact.subject_id)
            obj = self.world.entities.get(rel_fact.object.entity_id)
            if subject and obj:
                subj_surface, subj_ref = self._indirect_entity_ref(subject, timestamp)
                obj_surface, obj_ref = self._indirect_entity_ref(obj, timestamp)
                template = self._choose_template("relationship")
                text = template.format(A=subj_surface, B=obj_surface)
                turns = [
                    Utterance("user", text, [subj_ref, obj_ref]),
                    Utterance("assistant", "Noted that relationship.", []),
                ]
                return Session(id=f"relationship-{timestamp}", timestamp=timestamp, turns=turns)
        return self._generate_noise(timestamp)

    # === Pillar 2: Declarative Reasoning Sessions ===

    def _generate_fact_chain(self, timestamp: int) -> Session:
        """Pillar 2.2: Fact Composition"""
        # Find chained dependencies
        dep_facts = [f for f in self.world.facts if f.predicate == "depends_on"]
        deploy_facts = [f for f in self.world.facts if f.predicate == "deployed_in"]
        
        if dep_facts and deploy_facts:
            dep = self.rng.choice(dep_facts)
            if isinstance(dep.object, EntityRef):
                target_id = dep.object.entity_id
                region_fact = next((f for f in deploy_facts if f.subject_id == target_id), None)
                if region_fact:
                    project = self.world.entities.get(dep.subject_id)
                    service = self.world.entities.get(target_id)
                    if project and service:
                        proj_surface, proj_ref = self._indirect_entity_ref(project, timestamp)
                        svc_surface, svc_ref = self._indirect_entity_ref(service, timestamp)
                        region = region_fact.object if isinstance(region_fact.object, str) else "unknown"
                        text = f"{proj_surface} uses {svc_surface}, which is deployed in {region}."
                        turns = [
                            Utterance("user", text, [proj_ref, svc_ref]),
                            Utterance("assistant", "Got it, I'll remember that dependency chain.", []),
                        ]
                        return Session(id=f"factchain-{timestamp}", timestamp=timestamp, turns=turns)
        return self._generate_noise(timestamp)

    def _generate_belief_revision(self, timestamp: int) -> Session:
        """Pillar 2.4: Belief Revision"""
        # Find superseding facts
        superseding = [f for f in self.world.facts if f.supersedes]
        if superseding:
            fact = self.rng.choice(superseding)
            new_value = fact.object if isinstance(fact.object, str) else "updated"
            if fact.predicate == "name":
                text = f"Actually my name is {new_value} now."
            elif fact.predicate == "lives_in":
                text = f"Correction: I moved to {new_value}."
            else:
                text = f"Update: {fact.predicate} is now {new_value}."
            turns = [
                Utterance("user", text, []),
                Utterance("assistant", "Noted, I've updated that information.", []),
            ]
            return Session(id=f"revision-{timestamp}", timestamp=timestamp, turns=turns)
        return self._generate_noise(timestamp)

    def _generate_verbatim_share(self, timestamp: int) -> Session:
        """Pillar 2.5: Verbatim Recall"""
        if not self.world.verbatim_items:
            return self._generate_noise(timestamp)
        item = self.rng.choice(self.world.verbatim_items)
        
        if item.item_type == "code_block":
            text = f"Here's the {item.description}: `{item.content}`"
        elif item.item_type == "error_message":
            text = f"The exact error was: {item.content}"
        elif item.item_type == "query":
            text = f"Save this SQL query for later: {item.content}"
        else:
            text = f"Our config: {item.content}"
        
        turns = [
            Utterance("user", text, []),
            Utterance("assistant", f"I've saved that {item.item_type.replace('_', ' ')}.", []),
        ]
        return Session(id=f"verbatim-{timestamp}", timestamp=timestamp, turns=turns)

    def _generate_counterfactual(self, timestamp: int) -> Session:
        """Pillar 2.4: Counterfactual resistance"""
        counterfactual = self._choose_template("counterfactual")
        turns = [
            Utterance("user", counterfactual, []),
            Utterance("assistant", "I understand that's hypothetical, not a fact.", []),
        ]
        return Session(id=f"counterfactual-{timestamp}", timestamp=timestamp, turns=turns)

    # === Pillar 3: Temporal & Episodic Sessions ===

    def _generate_incident(self, timestamp: int) -> Session:
        incident = self._choose_template("incident")
        event = next(iter(self.world.events.values()), None)
        if event and event.steps:
            step = self.rng.choice(event.steps)
            actor = self.world.entities.get(step.actor.entity_id, self._pick_entity_by_type("person"))
            if actor:
                actor_surface, actor_ref = self._indirect_entity_ref(actor, timestamp)
                turns = [
                    Utterance("user", f"Remember that {incident}. {actor_surface} {step.description}.", [actor_ref]),
                    Utterance("assistant", f"Noted, that was during {event.name}.", []),
                ]
                return Session(id=f"incident-{timestamp}", timestamp=timestamp, turns=turns)
        return self._generate_noise(timestamp)

    def _generate_temporal_narrative(self, timestamp: int) -> Session:
        """Pillar 3.1/3.2: Temporal Sequencing & Episode Reconstruction"""
        events = list(self.world.events.values())
        if len(events) >= 2:
            e1, e2 = self.rng.sample(events, 2)
            if e1.start_time < e2.start_time:
                text = f"The {e1.name} happened before {e2.name}."
            else:
                text = f"The {e2.name} happened before {e1.name}."
            turns = [
                Utterance("user", text, []),
                Utterance("assistant", "I've noted that timeline.", []),
            ]
            return Session(id=f"temporal-{timestamp}", timestamp=timestamp, turns=turns)
        elif events:
            event = events[0]
            if event.steps and len(event.steps) >= 2:
                s1, s2 = event.steps[0], event.steps[-1]
                text = f"During {event.name}, first {s1.description}, then later {s2.description}."
                turns = [
                    Utterance("user", text, []),
                    Utterance("assistant", "Got the sequence.", []),
                ]
                return Session(id=f"temporal-{timestamp}", timestamp=timestamp, turns=turns)
        return self._generate_noise(timestamp)

    def _generate_cyclical_pattern(self, timestamp: int) -> Session:
        """Pillar 3.4: Cyclical Event Recognition"""
        if not self.world.cyclical_patterns:
            return self._generate_noise(timestamp)
        pattern = self.rng.choice(self.world.cyclical_patterns)
        
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if pattern.day_of_week is not None:
            day = day_names[pattern.day_of_week]
            text = f"I {pattern.description.replace('on Fridays', f'on {day}s').replace('on Mondays', f'on {day}s')}."
        elif pattern.time_of_day:
            text = f"I {pattern.description} in the {pattern.time_of_day}."
        else:
            text = f"I regularly {pattern.description}."
        
        if pattern.exceptions:
            text += f" But {pattern.exceptions[0]}."
        
        turns = [
            Utterance("user", text, []),
            Utterance("assistant", "I'll remember that pattern.", []),
        ]
        return Session(id=f"cyclical-{timestamp}", timestamp=timestamp, turns=turns)

    def _generate_causal_explanation(self, timestamp: int) -> Session:
        """Pillar 3.3: Causal Explanation"""
        events = list(self.world.events.values())
        for event in events:
            if event.causal_links:
                link = self.rng.choice(event.causal_links)
                if link.source_step < len(event.steps) and link.target_step < len(event.steps):
                    cause = event.steps[link.source_step]
                    effect = event.steps[link.target_step]
                    text = f"During {event.name}, {cause.description} {link.relation} which led to {effect.description}."
                    turns = [
                        Utterance("user", text, []),
                        Utterance("assistant", "I understand that causal chain.", []),
                    ]
                    return Session(id=f"causal-{timestamp}", timestamp=timestamp, turns=turns)
        return self._generate_noise(timestamp)

    # === Pillar 4: Preference Learning Sessions ===

    def _generate_preference_signal(self, timestamp: int) -> Session:
        pref = self._choose_template("preference")
        if self.world.preferences:
            preference = self.rng.choice(self.world.preferences)
            subject = self.world.entities.get(preference.subject_id)
            if subject:
                surface, ref = self._indirect_entity_ref(subject, timestamp)
                phrasing = f"{surface} prefers: {preference.signal}"
                turns = [
                    Utterance("user", phrasing, [ref]),
                    Utterance("assistant", "Acknowledged.", []),
                ]
                return Session(id=f"pref-{timestamp}", timestamp=timestamp, turns=turns)
        
        turns = [
            Utterance("user", f"For status updates, {pref}.", []),
            Utterance("assistant", "Acknowledged.", []),
        ]
        return Session(id=f"pref-{timestamp}", timestamp=timestamp, turns=turns)

    def _generate_preference_ranked(self, timestamp: int) -> Session:
        """Pillar 4.4: Preference Hierarchies"""
        ranked = [p for p in self.world.preferences if p.rank is not None]
        ranked.sort(key=lambda x: x.rank or 999)
        
        if len(ranked) >= 2:
            # Find preferences with same scope
            scopes = {}
            for p in ranked:
                if p.scope:
                    scopes.setdefault(p.scope, []).append(p)
            
            for scope, prefs in scopes.items():
                if len(prefs) >= 2:
                    p1, p2 = prefs[0], prefs[1]
                    text = f"For {scope}: I prefer {p1.signal.split()[-1]} over {p2.signal.split()[-1]}."
                    turns = [
                        Utterance("user", text, []),
                        Utterance("assistant", "I've noted that preference ranking.", []),
                    ]
                    return Session(id=f"prefrank-{timestamp}", timestamp=timestamp, turns=turns)
        
        return self._generate_preference_signal(timestamp)

    def _generate_preference_contextual(self, timestamp: int) -> Session:
        """Pillar 4.3: Preference Scope"""
        contextual = [p for p in self.world.preferences if p.scope]
        if contextual:
            pref = self.rng.choice(contextual)
            text = f"For {pref.scope.replace('_', ' ')}: {pref.signal}."
            turns = [
                Utterance("user", text, []),
                Utterance("assistant", "Noted that context-specific preference.", []),
            ]
            return Session(id=f"prefctx-{timestamp}", timestamp=timestamp, turns=turns)
        return self._generate_preference_signal(timestamp)

    def _generate_hard_constraint(self, timestamp: int) -> Session:
        """Pillar 4.6: Constraint Hierarchy"""
        hard = [p for p in self.world.preferences if p.is_hard_constraint]
        soft = [p for p in self.world.preferences if not p.is_hard_constraint and p.scope]
        
        if hard:
            constraint = self.rng.choice(hard)
            text = f"Important: {constraint.signal}. This is non-negotiable."
            turns = [
                Utterance("user", text, []),
                Utterance("assistant", "Understood, I'll treat that as an absolute constraint.", []),
            ]
            return Session(id=f"hardconstraint-{timestamp}", timestamp=timestamp, turns=turns)
        elif soft:
            pref = self.rng.choice(soft)
            text = f"{pref.signal}, but it's not a dealbreaker."
            turns = [
                Utterance("user", text, []),
                Utterance("assistant", "Got it, that's a preference but flexible.", []),
            ]
            return Session(id=f"softconstraint-{timestamp}", timestamp=timestamp, turns=turns)
        return self._generate_preference_signal(timestamp)

    # === Pillar 5: Knowledge Boundaries Sessions ===

    def _generate_reflection(self, timestamp: int) -> Session:
        pref = self._choose_template("preference")
        turns = [
            Utterance("user", f"Note to self: {pref}.", []),
            Utterance("assistant", "Understood, I'll adapt.", []),
        ]
        return Session(id=f"reflection-{timestamp}", timestamp=timestamp, turns=turns)

    def _generate_stale_context(self, timestamp: int) -> Session:
        """Pillar 5.2: Temporal Relevance Decay"""
        stale = self._choose_template("stale_context")
        turns = [
            Utterance("user", stale, []),
            Utterance("assistant", "Thanks for the context update.", []),
        ]
        return Session(id=f"stale-{timestamp}", timestamp=timestamp, turns=turns)

    # === Pillar 6: Procedural Knowledge Sessions ===

    def _generate_lesson_learned(self, timestamp: int) -> Session:
        """Pillar 6.1: Lesson Extraction"""
        if not self.world.lessons:
            return self._generate_noise(timestamp)
        lesson = self.rng.choice(self.world.lessons)
        
        if lesson.is_systemic:
            text = f"Important lesson: {lesson.lesson_text}"
        else:
            text = f"One-off issue: {lesson.lesson_text}"
        
        turns = [
            Utterance("user", text, []),
            Utterance("assistant", "I've recorded that lesson.", []),
        ]
        return Session(id=f"lesson-{timestamp}", timestamp=timestamp, turns=turns)

    def _generate_procedure_share(self, timestamp: int) -> Session:
        """Pillar 6.3: Procedure Storage"""
        if not self.world.procedures:
            return self._generate_noise(timestamp)
        proc = self.rng.choice(self.world.procedures)
        
        steps_text = ", then ".join(proc.steps[:3])
        text = f"To {proc.name}: first {steps_text}..."
        
        if proc.conditions:
            cond_key = self.rng.choice(list(proc.conditions.keys()))
            text += f" Also, {cond_key}: {proc.conditions[cond_key]}."
        
        turns = [
            Utterance("user", text, []),
            Utterance("assistant", f"I've saved the procedure for '{proc.name}'.", []),
        ]
        return Session(id=f"procedure-{timestamp}", timestamp=timestamp, turns=turns)

    def _generate_tool_memory(self, timestamp: int) -> Session:
        """Pillar 6.4: Tool/Method Memory"""
        if not self.world.tool_memories:
            return self._generate_noise(timestamp)
        memory = self.rng.choice(self.world.tool_memories)
        
        if memory.worked:
            text = f"When we had {memory.problem_context}, {memory.solution}."
        else:
            text = f"We tried {memory.solution} for {memory.problem_context}, but that didn't help."
        
        if memory.environment:
            text += f" (This was on {memory.environment}.)"
        
        turns = [
            Utterance("user", text, []),
            Utterance("assistant", "I'll remember that for future reference.", []),
        ]
        return Session(id=f"toolmem-{timestamp}", timestamp=timestamp, turns=turns)

    # === Noise & Utilities ===

    def _generate_noise(self, timestamp: int) -> Session:
        noise = self._choose_template("noise")
        turns = [Utterance("user", noise, []), Utterance("assistant", "ok", [])]
        return Session(id=f"noise-{timestamp}", timestamp=timestamp, turns=turns)

    def _pad_with_noise(self, session: Session) -> Session:
        desired = self.rng.randint(self.min_turns, self.max_turns)
        turns = list(session.turns)
        speakers = ["user", "assistant"]
        while len(turns) < desired:
            speaker = speakers[len(turns) % 2]
            noise_text = self._choose_template("noise")
            turns.append(Utterance(speaker, noise_text, []))
        return Session(id=session.id, timestamp=session.timestamp, turns=turns)

    def _sample_session_type(self) -> SessionType:
        # Weighted sampling across all session types for balanced pillar coverage
        weights = {
            # Pillar 1: World Modeling
            SessionType.INTRO: 1,
            SessionType.PROJECT_UPDATE: 3,
            SessionType.ADMIN_CHANGE: 2,
            SessionType.TASK_TRACKING: 3,
            SessionType.RELATIONSHIP_UPDATE: 2,
            
            # Pillar 2: Declarative Reasoning
            SessionType.FACT_CHAIN: 3,
            SessionType.BELIEF_REVISION: 2,
            SessionType.VERBATIM_SHARE: 3,
            SessionType.COUNTERFACTUAL: 1,
            
            # Pillar 3: Temporal & Episodic
            SessionType.INCIDENT_LOG: 3,
            SessionType.TEMPORAL_NARRATIVE: 2,
            SessionType.CYCLICAL_PATTERN: 2,
            SessionType.CAUSAL_EXPLANATION: 2,
            
            # Pillar 4: Preference Learning
            SessionType.PREFERENCE_SIGNAL: 2,
            SessionType.PREFERENCE_RANKED: 2,
            SessionType.PREFERENCE_CONTEXTUAL: 2,
            SessionType.HARD_CONSTRAINT: 2,
            
            # Pillar 5: Knowledge Boundaries
            SessionType.REFLECTION: 2,
            SessionType.STALE_CONTEXT: 2,
            
            # Pillar 6: Procedural Knowledge
            SessionType.LESSON_LEARNED: 3,
            SessionType.PROCEDURE_SHARE: 2,
            SessionType.TOOL_MEMORY: 2,
            
            # Noise
            SessionType.NOISE: 6,
        }
        population = list(weights.keys())
        probs = [weights[p] for p in population]
        total = sum(probs)
        pick = self.rng.uniform(0, total)
        acc = 0.0
        for p, w in zip(population, probs):
            acc += w
            if pick <= acc:
                return p
        return SessionType.NOISE

    def _token_estimate(self, text: str) -> int:
        return max(1, len(text.split()))

    def generate_sessions(
        self,
        min_sessions: int = 30,
        min_total_tokens: int = 20_000,
        max_info_density: float = 0.2,
    ) -> List[Session]:
        sessions: List[Session] = []
        total_tokens = 0
        timestamp = 0

        while len(sessions) < min_sessions or total_tokens < min_total_tokens:
            s_type = self._sample_session_type()
            if s_type == SessionType.INTRO and any(s.id.startswith("intro") for s in sessions):
                s_type = SessionType.PROJECT_UPDATE

            session = self._dispatch_session_type(s_type, timestamp)
            session = self._pad_with_noise(session)
            density = self._estimate_density(session)
            if density <= max_info_density or s_type == SessionType.NOISE:
                sessions.append(session)
                total_tokens += sum(self._token_estimate(u.text) for u in session.turns)
            timestamp += 1

        return sessions

    def _dispatch_session_type(self, s_type: SessionType, timestamp: int) -> Session:
        """Dispatch to the appropriate session generator based on type."""
        dispatch_map = {
            # Pillar 1
            SessionType.INTRO: self._generate_intro,
            SessionType.PROJECT_UPDATE: self._generate_project_update,
            SessionType.ADMIN_CHANGE: self._generate_admin_change,
            SessionType.TASK_TRACKING: self._generate_task_tracking,
            SessionType.RELATIONSHIP_UPDATE: self._generate_relationship_update,
            # Pillar 2
            SessionType.FACT_CHAIN: self._generate_fact_chain,
            SessionType.BELIEF_REVISION: self._generate_belief_revision,
            SessionType.VERBATIM_SHARE: self._generate_verbatim_share,
            SessionType.COUNTERFACTUAL: self._generate_counterfactual,
            # Pillar 3
            SessionType.INCIDENT_LOG: self._generate_incident,
            SessionType.TEMPORAL_NARRATIVE: self._generate_temporal_narrative,
            SessionType.CYCLICAL_PATTERN: self._generate_cyclical_pattern,
            SessionType.CAUSAL_EXPLANATION: self._generate_causal_explanation,
            # Pillar 4
            SessionType.PREFERENCE_SIGNAL: self._generate_preference_signal,
            SessionType.PREFERENCE_RANKED: self._generate_preference_ranked,
            SessionType.PREFERENCE_CONTEXTUAL: self._generate_preference_contextual,
            SessionType.HARD_CONSTRAINT: self._generate_hard_constraint,
            # Pillar 5
            SessionType.REFLECTION: self._generate_reflection,
            SessionType.STALE_CONTEXT: self._generate_stale_context,
            # Pillar 6
            SessionType.LESSON_LEARNED: self._generate_lesson_learned,
            SessionType.PROCEDURE_SHARE: self._generate_procedure_share,
            SessionType.TOOL_MEMORY: self._generate_tool_memory,
            # Noise
            SessionType.NOISE: self._generate_noise,
        }
        
        generator = dispatch_map.get(s_type, self._generate_noise)
        return generator(timestamp)

    def _estimate_density(self, session: Session) -> float:
        grounded = sum(len(u.grounded_refs) for u in session.turns)
        total = max(1, sum(self._token_estimate(u.text) for u in session.turns))
        return grounded / total

    @staticmethod
    def summarize_sessions(sessions: Sequence[Session]) -> str:
        return "\n".join(
            f"{s.timestamp:03d} [{s.id}] " + " | ".join(f"{u.speaker}: {u.text}" for u in s.turns) for s in sessions
        )


