from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# Default pillars aligned with pillars.md
DEFAULT_PILLARS = [
    "world_modeling",
    "declarative_reasoning",
    "temporal_episodic",
    "preference_learning",
    "knowledge_boundaries",
    "procedural_knowledge",
]


@dataclass
class Subpillar:
    id: str
    name: str
    description: str


@dataclass
class Pillar:
    id: str
    name: str
    description: str
    subpillars: List[Subpillar]


def parse_pillars(md_path: Path) -> List[str]:
    """
    Parse pillar names from a markdown file by reading ### headings.
    Returns slugs suitable for probe generation, e.g. 'world_modeling'.
    """
    if not md_path.exists():
        return DEFAULT_PILLARS
    
    # Explicit UTF-8 read to avoid Windows cp1252 decode errors on smart quotes.
    text = md_path.read_text(encoding="utf-8")
    pillars: List[str] = []
    
    for line in text.splitlines():
        # Match ### **Pillar N: Name** format
        m = re.match(r"^###\s+\*\*Pillar\s+\d+:\s+(.+?)\*\*", line.strip())
        if m:
            name = m.group(1).strip()
            # Convert to slug
            slug = name.lower()
            slug = slug.replace("&", "and").replace(" ", "_")
            slug = re.sub(r"[^a-z0-9_]", "", slug)
            # Normalize known variations
            slug = _normalize_pillar_slug(slug)
            if slug and slug not in pillars:
                pillars.append(slug)
    
    # Fallback to default pillars if file is empty or no pillars found
    if not pillars:
        pillars = DEFAULT_PILLARS
    
    return pillars


def _normalize_pillar_slug(slug: str) -> str:
    """Normalize pillar slug to canonical form."""
    if "world" in slug and "model" in slug:
        return "world_modeling"
    if "declarative" in slug:
        return "declarative_reasoning"
    if "temporal" in slug or "episodic" in slug:
        return "temporal_episodic"
    if "preference" in slug:
        return "preference_learning"
    if "knowledge" in slug and "bound" in slug:
        return "knowledge_boundaries"
    if "procedural" in slug:
        return "procedural_knowledge"
    return slug


def parse_pillars_detailed(md_path: Path) -> List[Pillar]:
    """
    Parse detailed pillar structure including subpillars from markdown.
    Returns list of Pillar objects with full hierarchy.
    """
    if not md_path.exists():
        return _default_pillars_detailed()
    
    text = md_path.read_text(encoding="utf-8")
    pillars: List[Pillar] = []
    current_pillar: Optional[Pillar] = None
    current_description_lines: List[str] = []
    
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Match ### **Pillar N: Name** format
        pillar_match = re.match(r"^###\s+\*\*Pillar\s+(\d+):\s+(.+?)\*\*", line)
        if pillar_match:
            # Save previous pillar if exists
            if current_pillar:
                pillars.append(current_pillar)
            
            pillar_num = pillar_match.group(1)
            pillar_name = pillar_match.group(2).strip()
            pillar_slug = _normalize_pillar_slug(
                pillar_name.lower().replace("&", "and").replace(" ", "_")
            )
            
            # Get description from italicized line after heading
            description = ""
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                desc_match = re.match(r"^_(.+)_$", next_line)
                if desc_match:
                    description = desc_match.group(1)
                    i += 1
            
            current_pillar = Pillar(
                id=pillar_slug,
                name=pillar_name,
                description=description,
                subpillars=[],
            )
            i += 1
            continue
        
        # Match **N.M Subpillar Name** format
        subpillar_match = re.match(r"^\*\*(\d+\.\d+)\s+(.+?)\*\*", line)
        if subpillar_match and current_pillar:
            sub_id = subpillar_match.group(1)
            sub_name = subpillar_match.group(2).strip()
            sub_slug = sub_name.lower().replace(" ", "_").replace("/", "_")
            sub_slug = re.sub(r"[^a-z0-9_]", "", sub_slug)
            
            # Collect description from following lines until next heading or empty
            description_lines = []
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if next_line.startswith("**") or next_line.startswith("###") or next_line.startswith("---"):
                    break
                if next_line.startswith("-"):
                    description_lines.append(next_line[1:].strip())
                j += 1
            
            current_pillar.subpillars.append(Subpillar(
                id=sub_slug,
                name=sub_name,
                description="; ".join(description_lines) if description_lines else "",
            ))
        
        i += 1
    
    # Save last pillar
    if current_pillar:
        pillars.append(current_pillar)
    
    if not pillars:
        return _default_pillars_detailed()
    
    return pillars


def _default_pillars_detailed() -> List[Pillar]:
    """Return default pillar structure when markdown is unavailable."""
    return [
        Pillar(
            id="world_modeling",
            name="World Modeling",
            description="Ability to resolve entities, map relationships, track state, and maintain a coherent ontology over time.",
            subpillars=[
                Subpillar("entity_resolution", "Entity Resolution", "Coreference across sessions"),
                Subpillar("relationship_mapping", "Relationship Mapping", "Directional relationships and updates"),
                Subpillar("type_category_membership", "Type/Category Membership", "Explicit and inferred types"),
                Subpillar("task_state_tracking", "Task/State Tracking", "Open tasks, dependencies, completion status"),
            ],
        ),
        Pillar(
            id="declarative_reasoning",
            name="Declarative Reasoning",
            description="Ability to reason over explicit facts in the world model.",
            subpillars=[
                Subpillar("baseline_factual_recall", "Baseline Factual Recall", "Single-hop retrieval"),
                Subpillar("fact_composition", "Fact Composition", "Multi-hop chains"),
                Subpillar("constraint_propagation", "Constraint Propagation", "Property inheritance"),
                Subpillar("belief_revision", "Belief Revision", "Supersession and updates"),
                Subpillar("verbatim_recall", "Verbatim Recall", "Exact retrieval of code/errors/queries"),
            ],
        ),
        Pillar(
            id="temporal_episodic",
            name="Temporal & Episodic Reasoning",
            description="Ability to reason about what happened, when, in what order, and why.",
            subpillars=[
                Subpillar("temporal_sequencing", "Temporal Sequencing", "Relative ordering"),
                Subpillar("episode_reconstruction", "Episode Reconstruction", "Multi-session synthesis"),
                Subpillar("causal_explanation", "Causal Explanation", "Inferred causation"),
                Subpillar("cyclical_event_recognition", "Cyclical Event Recognition", "Pattern identification"),
            ],
        ),
        Pillar(
            id="preference_learning",
            name="Preference Learning",
            description="Ability to learn patterns from repeated experience and apply them contextually.",
            subpillars=[
                Subpillar("explicit_preferences", "Explicit Preferences", "Stated directly"),
                Subpillar("preference_induction", "Preference Induction", "Pattern extraction"),
                Subpillar("preference_scope", "Preference Scope", "Context-dependent preferences"),
                Subpillar("preference_hierarchies", "Preference Hierarchies", "Explicit ranking"),
                Subpillar("preference_drift", "Preference Drift", "Detecting decay via absence"),
                Subpillar("constraint_hierarchy", "Constraint Hierarchy", "Soft vs hard constraints"),
            ],
        ),
        Pillar(
            id="knowledge_boundaries",
            name="Knowledge Boundaries",
            description="Ability to know what is and isn't known, and what should or shouldn't be surfaced.",
            subpillars=[
                Subpillar("negative_knowledge", "Negative Knowledge", "Absence detection"),
                Subpillar("temporal_relevance_decay", "Temporal Relevance Decay", "Stale context"),
                Subpillar("confidence_calibration", "Confidence Calibration", "Expressing uncertainty"),
            ],
        ),
        Pillar(
            id="procedural_knowledge",
            name="Procedural Knowledge",
            description="Ability to learn operational lessons from experience and apply them to future situations.",
            subpillars=[
                Subpillar("lesson_extraction", "Lesson Extraction", "Generalizing from incidents"),
                Subpillar("lesson_application", "Lesson Application", "Surfacing relevant lessons"),
                Subpillar("procedure_storage", "Procedure Storage", "Multi-step and conditional procedures"),
                Subpillar("tool_method_memory", "Tool/Method Memory", "What worked and didn't work"),
            ],
        ),
    ]


def get_pillar_info() -> Dict[str, Dict]:
    """
    Get pillar metadata for documentation and reporting.
    Returns dict mapping pillar slug to info dict.
    """
    pillars = _default_pillars_detailed()
    return {
        p.id: {
            "name": p.name,
            "description": p.description,
            "subpillars": {s.id: {"name": s.name, "description": s.description} for s in p.subpillars},
        }
        for p in pillars
    }


