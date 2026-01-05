from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .pillars import parse_pillars
from .probes.probe_generator import LlmProbeGenerator, VertexProbeLLM, OpenRouterProbeLLM, Probe
from .sessions.llm_session_generator import LlmSessionGenerator
from .sessions.session_generator import Session
from .world.loader import load_world
from .world.schema import World
from .sessions.openrouter_client import VertexLLM, OpenRouterLLM


def build_benchmark(
    seed: int,
    world_path: Optional[Path],
    pillars_path: Optional[Path],
    min_sessions: int = 5,
    min_tokens: int = 800,
) -> Dict[str, Any]:
    """
    Build a complete benchmark bundle: world, sessions, and probes.
    
    Args:
        seed: Random seed for reproducibility
        world_path: Path to world JSON file
        pillars_path: Path to pillars markdown file
        min_sessions: Minimum number of sessions to generate
        min_tokens: Minimum total tokens across all sessions
    
    Returns:
        Dictionary with 'world', 'sessions', and 'probes' keys
    """
    rng = random.Random(seed)
    world = load_world(world_path)
    
    print(f"[Benchmark] Loaded world with {len(world.entities)} entities, {len(world.facts)} facts")
    
    session_generator = LlmSessionGenerator(world, rng, model=VertexLLM())
    sessions = session_generator.generate_sessions(
        min_sessions=min_sessions, 
        min_total_tokens=min_tokens
    )
    print(f"[Benchmark] Generated {len(sessions)} sessions")
    
    pillars = parse_pillars(pillars_path) if pillars_path else None
    probe_generator = LlmProbeGenerator(
        world, sessions, rng, 
        model=VertexProbeLLM(), 
        pillars=pillars
    )
    probes = probe_generator.generate()
    print(f"[Benchmark] Generated {len(probes)} probes")
    
    return {"world": world, "sessions": sessions, "probes": probes}


def serialize_probe(p: Probe) -> Dict[str, Any]:
    """Serialize a Probe to a JSON-compatible dictionary."""
    return {
        "id": p.id,
        "pillar": p.pillar,
        "subpillar": p.subpillar,
        "question": p.question,
        "answer_type": p.answer_type,
        "gold_answer": {
            "text": p.gold_answer.text,
            "supporting_items": p.gold_answer.supporting_items,
        },
    }


def serialize_session(s: Session) -> Dict[str, Any]:
    """Serialize a Session to a JSON-compatible dictionary."""
    return {
        "id": s.id,
        "timestamp": s.timestamp,
        "turns": [{"speaker": t.speaker, "text": t.text} for t in s.turns],
    }


def serialize_world(world: World) -> Dict[str, Any]:
    """Serialize key world info for output."""
    return {
        "entities": list(world.entities.keys()),
        "facts_count": len(world.facts),
        "events_count": len(world.events),
        "preferences_count": len(world.preferences),
    }


def run(
    seed: int,
    output: Path,
    world_path: Optional[Path],
    pillars_path: Optional[Path],
    min_sessions: int = 5,
    min_tokens: int = 800,
) -> None:
    """
    Generate benchmark data and write to output file.
    """
    bundle = build_benchmark(
        seed,
        world_path=world_path,
        pillars_path=pillars_path,
        min_sessions=min_sessions,
        min_tokens=min_tokens,
    )
    
    world: World = bundle["world"]
    sessions: List[Session] = bundle["sessions"]
    probes: List[Probe] = bundle["probes"]

    output.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "seed": seed,
        "world_summary": serialize_world(world),
        "num_sessions": len(sessions),
        "num_probes": len(probes),
        "sessions": [serialize_session(s) for s in sessions],
        "probes": [serialize_probe(p) for p in probes],
    }
    
    output.write_text(json.dumps(payload, indent=2))
    print(f"[Benchmark] Output written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate long-context memory benchmark data (worlds, sessions, probes)."
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducibility")
    parser.add_argument("--output", type=Path, default=Path("benchmark_result.json"), help="Output JSON file")
    parser.add_argument("--world", type=Path, default=None, help="Path to world JSON file")
    parser.add_argument("--pillars", type=Path, default=Path("benchmark/pillars.md"), help="Path to pillars markdown")
    parser.add_argument("--min-sessions", type=int, default=5, help="Minimum sessions to generate")
    parser.add_argument("--min-tokens", type=int, default=800, help="Minimum total tokens")
    
    args = parser.parse_args()
    run(
        seed=args.seed,
        output=args.output,
        world_path=args.world,
        pillars_path=args.pillars,
        min_sessions=args.min_sessions,
        min_tokens=args.min_tokens,
    )


if __name__ == "__main__":
    main()


