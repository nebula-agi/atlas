from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from .schema import (
    Alias,
    Entity,
    EntityRef,
    Event,
    EventStep,
    Fact,
    MemoryPolicy,
    Preference,
    TimeScopedRole,
    TimeWindow,
    World,
    CausalEdge,
)


def _tw(d: Dict) -> TimeWindow:
    return TimeWindow(start_time=d["start_time"], end_time=d.get("end_time"))


def _alias_list(raw: List[Dict]) -> List[Alias]:
    return [
        Alias(
            surface_form=a["surface_form"],
            start_time=a.get("start_time", 0),
            end_time=a.get("end_time"),
            confidence=a.get("confidence", 0.8),
        )
        for a in raw
    ]


def _roles(raw: List[Dict]) -> List[TimeScopedRole]:
    return [
        TimeScopedRole(role=r["role"], start_time=r.get("start_time", 0), end_time=r.get("end_time"))
        for r in raw
    ]


def _entity_ref(obj: Union[Dict, str]) -> EntityRef:
    if isinstance(obj, str):
        return EntityRef(entity_id=obj)
    return EntityRef(entity_id=obj["entity_id"], alias=obj.get("alias"))


def _facts(raw: List[Dict]) -> List[Fact]:
    facts: List[Fact] = []
    for f in raw:
        obj = f["object"]
        facts.append(
            Fact(
                subject_id=f["subject_id"],
                predicate=f["predicate"],
                object=_entity_ref(obj) if isinstance(obj, dict) else obj,
                validity=_tw(f["validity"]),
                source=f.get("source", "user"),
                confidence=f.get("confidence", 0.7),
            )
        )
    return facts


def _event_steps(raw: List[Dict]) -> List[EventStep]:
    return [
        EventStep(description=s["description"], time=s["time"], actor=_entity_ref(s["actor"]))
        for s in raw
    ]


def _causal_links(raw: Optional[List[Dict]]) -> List[CausalEdge]:
    if not raw:
        return []
    return [
        CausalEdge(source_step=link["source_step"], target_step=link["target_step"], relation=link["relation"])
        for link in raw
    ]


def _events(raw: Dict[str, Dict]) -> Dict[str, Event]:
    events: Dict[str, Event] = {}
    for event_id, e in raw.items():
        events[event_id] = Event(
            id=event_id,
            name=e["name"],
            steps=_event_steps(e.get("steps", [])),
            participants=[_entity_ref(p) for p in e.get("participants", [])],
            start_time=e.get("start_time", 0),
            end_time=e.get("end_time", 0),
            causal_links=_causal_links(e.get("causal_links")),
        )
    return events


def _preferences(raw: List[Dict]) -> List[Preference]:
    return [
        Preference(
            subject_id=p["subject_id"],
            signal=p["signal"],
            scope=p.get("scope"),
            strength=p.get("strength", 0.8),
            confidence=p.get("confidence", 0.7),
            inferred=p.get("inferred", False),
        )
        for p in raw
    ]


def _policies(raw: List[Dict]) -> List[MemoryPolicy]:
    return [
        MemoryPolicy(
            fact_predicate=p["fact_predicate"],
            decay_half_life=p.get("decay_half_life", 60),
            retrievable_after_expiry=p.get("retrievable_after_expiry", True),
        )
        for p in raw
    ]


def load_world_from_dict(data: Dict) -> World:
    entities: Dict[str, Dict] = {}
    for entity_id, e in data["entities"].items():
        entities[entity_id] = Entity(
            id=entity_id,
            type=e["type"],
            aliases=_alias_list(e.get("aliases", [])),
            attributes=[],
            roles=_roles(e.get("roles", [])),
            persona=e.get("persona"),
        )

    facts = _facts(data.get("facts", []))
    for fact in facts:
        if fact.subject_id in entities:
            entities[fact.subject_id].attributes.append(fact)

    events = _events(data.get("events", {}))
    preferences = _preferences(data.get("preferences", []))
    policies = _policies(data.get("memory_policies", []))

    return World(
        entities=entities,
        facts=facts,
        events=events,
        preferences=preferences,
        memory_policies=policies,
    )


def load_world(path: Path) -> World:
    # Explicit UTF-8 read to avoid Windows locale decoding errors.
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "world_id" in data:
        # Template-shaped world; convert to internal schema.
        converted = _convert_template_to_internal(data)
        return load_world_from_dict(converted)
    return load_world_from_dict(data)


def _convert_template_to_internal(data: Dict) -> Dict:
    """
    Convert a template-style world (see benchmark/world_template.json) into
    the minimal internal structure expected by the benchmark.
    This is a lossy mapping focused on session/probe generation.
    """

    def _alias_list_template(raw: List[Dict]) -> List[Dict]:
        aliases: List[Dict] = []
        for a in raw or []:
            aliases.append(
                {
                    "surface_form": a.get("text", "") or a.get("surface_form", ""),
                    "start_time": 0,
                    "end_time": None,
                    "confidence": 0.8,
                }
            )
        return aliases

    def _entity_type(t: str) -> str:
        t = (t or "").lower()
        if "person" in t:
            return "person"
        if "project" in t:
            return "project"
        if "service" in t or "system" in t:
            return "service"
        if "event" in t:
            return "event"
        return "person"

    entities: Dict[str, Entity] = {}
    for e in data.get("entities", []):
        entity_id = e.get("id", "").replace("ent:", "") or f"ent_{len(entities)}"
        entities[entity_id] = {
            "id": entity_id,
            "type": _entity_type(e.get("type", "")),
            "aliases": _alias_list_template(e.get("aliases", [])),
            "attributes": [],
            "roles": [],
            "persona": e.get("canonical_name") or None,
        }

    facts: List[Dict] = []

    # Map relations -> facts with EntityRef objects.
    for rel in data.get("relations", []):
        subj = (rel.get("subject") or {}).get("ref", "")
        obj_ref = (rel.get("object") or {}).get("ref", "")
        if subj and obj_ref:
            facts.append(
                {
                    "subject_id": subj.replace("ent:", ""),
                    "predicate": rel.get("type", "related_to"),
                    "object": {"entity_id": obj_ref.replace("ent:", "")},
                    "validity": {"start_time": 0, "end_time": None},
                    "source": "user",
                    "confidence": 0.7,
                }
            )

    # Map facts list -> internal facts.
    for f in data.get("facts", []):
        subj = (f.get("subject") or {}).get("ref", "")
        obj = f.get("object", {})
        if not subj or "value" not in obj:
            continue
        val = obj.get("value")
        val_type = obj.get("value_type", "")
        obj_value: Union[str, Dict] = val
        if isinstance(val, dict) and val.get("ref"):
            obj_value = {"entity_id": val["ref"].replace("ent:", "")}
        elif val_type == "ref" and isinstance(val, str):
            obj_value = {"entity_id": val.replace("ent:", "")}
        facts.append(
            {
                "subject_id": subj.replace("ent:", ""),
                "predicate": f.get("predicate", "related_to"),
                "object": obj_value,
                "validity": {"start_time": 0, "end_time": None},
                "source": "user",
                "confidence": 0.7,
            }
        )

    # Attach attributes back to entities for quick lookups.
    for fact in facts:
        sid = fact.get("subject_id")
        if sid in entities:
            entities[sid]["attributes"].append(fact)

    # Map episodes -> events (best effort).
    events: Dict[str, Dict] = {}
    for ep in data.get("episodes", []):
        ev_id = ep.get("id", "").replace("ep:", "") or f"event_{len(events)}"
        steps_raw = ep.get("sequence", [])
        steps: List[EventStep] = []
        for idx, s in enumerate(steps_raw):
            actor_ref = None
            links = s.get("links") or []
            if links:
                for link in links:
                    if isinstance(link, str) and link.startswith("ent:"):
                        actor_ref = {"entity_id": link.replace("ent:", "")}
                        break
            if not actor_ref:
                actor_ref = {"entity_id": "unknown"}
            steps.append(
                {
                    "description": s.get("summary", s.get("kind", "step")),
                    "time": idx,
                    "actor": actor_ref,
                }
            )
        participants = [
            {"entity_id": p.get("ref", "").replace("ent:", "")}
            for p in ep.get("participants", [])
            if p.get("ref")
        ]
        events[ev_id] = {
            "id": ev_id,
            "name": ep.get("title", ep.get("type", "event")),
            "steps": steps,
            "participants": participants,
            "start_time": 0,
            "end_time": len(steps_raw) or 1,
            "causal_links": [],
        }

    # Map preferences.
    preferences: List[Dict] = []
    for pref in data.get("preferences", []):
        holder = (pref.get("holder") or {}).get("ref") or pref.get("id") or "unknown"
        preferences.append(
            {
                "subject_id": holder.replace("ent:", ""),
                "signal": pref.get("statement") or pref.get("topic") or "unspecified",
                "scope": ",".join(pref.get("scope", {}).get("context_tags", [])) or None,
                "strength": 0.8,
                "confidence": 0.75,
                "inferred": pref.get("kind", "") == "induced",
            }
        )

    # Memory policies (optional) from staleness overrides.
    memory_policies: List[Dict] = []
    staleness = (data.get("knowledge_boundaries") or {}).get("staleness", {})
    for override in staleness.get("topic_overrides", []) or []:
        topic = override.get("topic", "general")
        memory_policies.append(
            {
                "fact_predicate": topic,
                "decay_half_life": int(override.get("half_life_days", 90)),
                "retrievable_after_expiry": True,
            }
        )

    return {
        "entities": entities,
        "facts": facts,
        "events": events,
        "preferences": preferences,
        "memory_policies": memory_policies,
    }


