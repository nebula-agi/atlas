#!/usr/bin/env python3
"""
Pre-ingest all benchmark files into Mem0.

This script ingests all sessions from benchmark files into Mem0, using the
filename as a unique user_id. After running this, wait for Mem0 to process
everything (check the dashboard), then run evaluations with --skip-ingest.

Usage:
    python ingest_mem0.py simple              # Ingest all simple test files
    python ingest_mem0.py complex             # Ingest all complex test files
    python ingest_mem0.py --file benchmark_001.json  # Ingest specific file
    python ingest_mem0.py --list              # List what would be ingested
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path

# Add benchmark module to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Optional: load variables from a local .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    from mem0 import MemoryClient
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    MemoryClient = None  # type: ignore[assignment]


def get_user_id_for_file(filepath: Path) -> str:
    """Generate a unique user_id based on filename."""
    # Use filename without extension as user_id
    # e.g., "benchmark_nebula_001.json" -> "benchmark_nebula_001"
    return filepath.stem


def load_benchmark_file(filepath: Path) -> dict:
    """Load a benchmark JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def ingest_file(client: MemoryClient, filepath: Path, dry_run: bool = False) -> int:
    """Ingest all sessions from a benchmark file into Mem0."""
    user_id = get_user_id_for_file(filepath)
    
    try:
        data = load_benchmark_file(filepath)
    except Exception as e:
        print(f"  ERROR loading {filepath.name}: {e}")
        return 0
    
    sessions = data.get("sessions", [])
    if not sessions:
        print(f"  {filepath.name}: No sessions found")
        return 0
    
    if dry_run:
        print(f"  {filepath.name}: {len(sessions)} sessions -> user_id: {user_id}")
        return len(sessions)
    
    print(f"  Ingesting {filepath.name} ({len(sessions)} sessions) as user_id: {user_id}")
    
    # Sort sessions by timestamp
    sorted_sessions = sorted(sessions, key=lambda s: s.get("timestamp", 0))
    
    ingested = 0
    for session in sorted_sessions:
        session_id = session.get("id", "unknown")
        session_timestamp = session.get("timestamp", 0)
        turns = session.get("turns", [])
        
        # Build messages array in the format Mem0 expects
        messages = []
        for turn in turns:
            speaker = turn.get("speaker", "user")
            text = turn.get("text", "")
            role = "user" if speaker == "user" else "assistant"
            messages.append({"role": role, "content": text})
        
        if not messages:
            continue
        
        try:
            client.add(
                messages=messages,
                user_id=user_id,
                metadata={
                    "session_id": session_id,
                    "timestamp": session_timestamp,
                    "source_file": filepath.name,
                }
            )
            ingested += 1
        except Exception as e:
            print(f"    ERROR ingesting session {session_id}: {e}")
    
    print(f"    -> Submitted {ingested}/{len(sessions)} sessions")
    return ingested


def get_test_files(test_set: str, specific_file: str = None) -> list:
    """Get list of benchmark files to ingest."""
    if specific_file:
        # Check in both directories
        for dir_name in ["simple_test_set", "complex_test_set"]:
            path = Path(dir_name) / specific_file
            if path.exists():
                return [path]
        # Try as direct path
        path = Path(specific_file)
        if path.exists():
            return [path]
        print(f"ERROR: File not found: {specific_file}")
        return []
    
    if test_set == "simple":
        test_dir = Path("simple_test_set")
    elif test_set == "complex":
        test_dir = Path("complex_test_set")
    else:
        print(f"ERROR: Unknown test set: {test_set}")
        return []
    
    if not test_dir.exists():
        print(f"ERROR: Directory not found: {test_dir}")
        return []
    
    return sorted(test_dir.glob("*.json"))


def main():
    parser = argparse.ArgumentParser(
        description="Pre-ingest benchmark files into Mem0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest_mem0.py simple                    # Ingest all simple test files
  python ingest_mem0.py complex                   # Ingest all complex test files  
  python ingest_mem0.py simple --max-files 5      # Ingest first 5 simple files
  python ingest_mem0.py --file benchmark_001.json # Ingest specific file
  python ingest_mem0.py simple --list             # List files without ingesting

After ingestion, check the Mem0 dashboard to verify processing is complete,
then run evaluations with: python run_eval.py simple --adapter mem0 --skip-ingest
        """
    )
    
    parser.add_argument(
        "test_set",
        nargs="?",
        choices=["simple", "complex"],
        help="Which test set to ingest",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Specific file to ingest",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to ingest",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List files that would be ingested (dry run)",
    )
    
    args = parser.parse_args()
    
    if not args.test_set and not args.file:
        parser.print_help()
        sys.exit(1)

    if not args.list and not MEM0_AVAILABLE:
        print("ERROR: mem0 package not installed. Run: pip install mem0ai")
        sys.exit(1)
    
    # Get API key
    api_key = os.environ.get("MEM0_API_KEY")
    
    if not api_key and not args.list:
        print("ERROR: MEM0_API_KEY not set. Set the environment variable (or put it in a local .env).")
        sys.exit(1)
    
    # Get files to ingest
    files = get_test_files(args.test_set, args.file)
    if not files:
        sys.exit(1)
    
    if args.max_files:
        files = files[:args.max_files]
    
    print(f"\n{'='*60}")
    print(f"MEM0 BULK INGESTION")
    print(f"{'='*60}")
    print(f"Files to ingest: {len(files)}")
    print(f"Mode: {'DRY RUN (listing only)' if args.list else 'INGESTING'}")
    print(f"{'='*60}\n")
    
    if args.list:
        # Dry run - just list
        total_sessions = 0
        for filepath in files:
            sessions = ingest_file(None, filepath, dry_run=True)
            total_sessions += sessions
        
        print(f"\n{'='*60}")
        print(f"SUMMARY: {len(files)} files, {total_sessions} total sessions")
        print(f"{'='*60}")
        return
    
    # Real ingestion
    client = MemoryClient(api_key=api_key)
    
    total_sessions = 0
    for i, filepath in enumerate(files):
        print(f"\n[{i+1}/{len(files)}]", end="")
        sessions = ingest_file(client, filepath)
        total_sessions += sessions
    
    print(f"\n{'='*60}")
    print(f"INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"Files processed: {len(files)}")
    print(f"Sessions submitted: {total_sessions}")
    print(f"\nNEXT STEPS:")
    print(f"1. Check Mem0 dashboard to verify all sessions are processed")
    print(f"2. Wait until processing is complete (may take several minutes)")
    print(f"3. Run evaluation with: python run_eval.py {args.test_set or 'simple'} --adapter mem0 --skip-ingest")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

