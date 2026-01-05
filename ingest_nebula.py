#!/usr/bin/env python3
"""
Pre-ingest all benchmark files into Nebula.

This script ingests all sessions from benchmark files into Nebula:
- By default, all files in a test set are ingested into a single collection
  (e.g. "nebula_benchmark_complex") to avoid hammering the collections API.
- Use --per-file-collections if you explicitly want one collection per file.
- Each session gets its own memory_id
- All messages in the same session share the same memory_id

Usage:
    python ingest_nebula.py simple              # Ingest all simple test files
    python ingest_nebula.py complex             # Ingest all complex test files
    python ingest_nebula.py --file benchmark_001.json  # Ingest specific file
    python ingest_nebula.py --list              # List what would be ingested
"""

from __future__ import annotations

import os
import sys
import json
import time
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
    from nebula import Nebula
    NEBULA_AVAILABLE = True
except ImportError:
    NEBULA_AVAILABLE = False
    Nebula = None  # type: ignore[assignment]


def get_collection_name_for_file(filepath: Path) -> str:
    """Generate a unique collection name based on filename."""
    # Use filename without extension as collection name
    # e.g., "benchmark_nebula_001.json" -> "benchmark_nebula_001"
    return filepath.stem


def load_benchmark_file(filepath: Path) -> dict:
    """Load a benchmark JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


COLLECTION_ID_CACHE_PATH = Path("nebula_collection_ids.json")
_COLLECTION_ID_CACHE: dict[str, str] | None = None


def _load_collection_id_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            out: dict[str, str] = {}
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                    out[k] = v
            return out
    except Exception:
        pass
    return {}


def _save_collection_id_cache(path: Path, cache: dict[str, str]) -> None:
    # Best-effort persistence; ingestion can proceed even if this fails.
    try:
        path.write_text(
            json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
    except Exception:
        pass


def _get_collection_id_cache() -> dict[str, str]:
    global _COLLECTION_ID_CACHE
    if _COLLECTION_ID_CACHE is None:
        _COLLECTION_ID_CACHE = _load_collection_id_cache(COLLECTION_ID_CACHE_PATH)
    return _COLLECTION_ID_CACHE


def _find_collection_id_via_list(client: Nebula, collection_name: str) -> str | None:
    """Fallback: scan collections list to find a name → id match (no name-based API lookup)."""
    try:
        limit = 200
        offset = 0
        while True:
            cols = client.list_collections(limit=limit, offset=offset)
            if not cols:
                return None
            for c in cols:
                name = getattr(c, "name", None)
                cid = getattr(c, "id", None)
                if name == collection_name and cid:
                    return str(cid)
            if len(cols) < limit:
                return None
            offset += limit
    except Exception:
        return None


def create_or_get_collection(client: Nebula, collection_name: str, max_retries: int = 5) -> str:
    """Create collection (no name lookup) and return its UUID.

    To avoid flaky/slow name-based lookups, we:
    - Prefer a local cache file (nebula_collection_ids.json) if present
    - Otherwise create the collection and cache its id
    - If creation conflicts (already exists), we fall back to list+match (still no name endpoint)
    """
    cache = _get_collection_id_cache()
    cached = cache.get(collection_name)
    if cached:
        print(f"    Using cached collection id: {collection_name} (UUID: {cached})")
        return cached

    last_error: Exception | None = None

    for attempt in range(max_retries):
        if attempt > 0:
            wait_time = min(2 ** attempt, 30)  # Exponential backoff, max 30s
            print(f"    Retry {attempt}/{max_retries-1} after {wait_time}s...")
            time.sleep(wait_time)

        # Create new collection (fast path)
        try:
            result = client.create_collection(name=collection_name)
            if hasattr(result, "id"):
                collection_id = result.id
            elif isinstance(result, dict):
                collection_id = result.get("id")
            else:
                collection_id = str(result)
            collection_id = str(collection_id)
            print(f"    Created collection: {collection_name} (UUID: {collection_id})")

            cache[collection_name] = collection_id
            _save_collection_id_cache(COLLECTION_ID_CACHE_PATH, cache)
            return collection_id
        except Exception as e:
            last_error = e
            msg = str(e).lower()
            # Some upstream gateways return HTML (e.g. 504) which causes JSONDecodeError
            # inside the SDK; treat that as a retryable server timeout.
            doc_text = ""
            try:
                doc = getattr(e, "doc", "")
                if isinstance(doc, str):
                    doc_text = doc.lower()
            except Exception:
                doc_text = ""

            # If it might already exist (conflict) OR the create request timed out,
            # try to recover without name-based lookup endpoints.
            if any(token in msg for token in ("already exists", "conflict", "409", "timed out", "timeout")) or any(
                token in doc_text for token in ("504", "gateway time-out", "gateway timeout", "<html")
            ):
                existing_id = _find_collection_id_via_list(client, collection_name)
                if existing_id:
                    print(
                        f"    Using existing collection (via list): {collection_name} (UUID: {existing_id})"
                    )
                    cache[collection_name] = existing_id
                    _save_collection_id_cache(COLLECTION_ID_CACHE_PATH, cache)
                    return existing_id

            print(f"    ERROR: Collection creation failed: {e}")
            if attempt == 0:
                print(f"      Exception type: {type(e).__name__}")
                print(f"      Exception args: {e.args}")
                print(f"      All attributes: {vars(e) if hasattr(e, '__dict__') else 'N/A'}")

    raise ValueError(
        f"Could not create collection '{collection_name}' after {max_retries} attempts: {last_error}"
    )


def ingest_file(
    client: Nebula | None,
    filepath: Path,
    *,
    collection_id: str | None = None,
    collection_display: str | None = None,
    per_file_collections: bool = False,
    dry_run: bool = False,
) -> dict:
    """Ingest all sessions from a benchmark file into Nebula."""
    collection_name = get_collection_name_for_file(filepath)
    
    try:
        data = load_benchmark_file(filepath)
    except Exception as e:
        print(f"  ERROR loading {filepath.name}: {e}")
        return {"sessions": 0, "messages": 0}
    
    sessions = data.get("sessions", [])
    if not sessions:
        print(f"  {filepath.name}: No sessions found")
        return {"sessions": 0, "messages": 0}
    
    total_messages = sum(len(s.get("turns", [])) for s in sessions)
    
    if dry_run:
        target = (
            collection_display
            or (f"UUID:{collection_id}" if collection_id else None)
            or (collection_name if per_file_collections else "UUID:<provided at runtime>")
        )
        print(
            f"  {filepath.name}: {len(sessions)} sessions, {total_messages} messages -> collection: {target}"
        )
        return {"sessions": len(sessions), "messages": total_messages}
    
    print(f"  Ingesting {filepath.name} ({len(sessions)} sessions, {total_messages} messages)")
    
    # Determine collection_id
    if per_file_collections:
        if client is None:
            raise ValueError("client is required for ingestion")
        collection_id = create_or_get_collection(client, collection_name)
        print(f"    Using collection (per-file): {collection_name} (UUID: {collection_id})")
    else:
        if not collection_id:
            raise ValueError("collection_id is required when not using per-file collections")
    
    # Sort sessions by timestamp
    sorted_sessions = sorted(sessions, key=lambda s: s.get("timestamp", 0))
    
    ingested_sessions = 0
    ingested_messages = 0
    
    for session in sorted_sessions:
        session_id = session.get("id", "unknown")
        session_timestamp = session.get("timestamp", 0)
        turns = session.get("turns", [])
        
        if not turns:
            continue
        
        # Build conversation text with role labels
        conversation_text = ""
        for turn in turns:
            speaker = turn.get("speaker", "user")
            text = turn.get("text", "")
            role_label = "User" if speaker == "user" else "Assistant"
            conversation_text += f"{role_label}: {text}\n"
        
        # Retry logic - try up to 3 times per session
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # NOTE:
                # The current Nebula Python SDK sends `collection_ref` for POST /v1/memories,
                # but this Nebula API expects `collection_id` (otherwise returns 422 with
                # "Field required: collection_id"). We therefore call the underlying request
                # method with the correct payload shape.
                payload = {
                    "collection_id": collection_id,
                    "engram_type": "document",
                    "raw_text": conversation_text.strip(),
                    "metadata": {
                        "session_id": str(session_id),
                        "timestamp": str(session_timestamp),
                        "source_file": filepath.name,
                        "source_file_stem": filepath.stem,
                        "num_turns": str(len(turns)),
                    },
                    "ingestion_mode": "fast",
                }
                response = client._make_request("POST", "/v1/memories", json_data=payload)

                memory_id = None
                task_id = None
                if isinstance(response, dict):
                    results = response.get("results")
                    if isinstance(results, dict):
                        memory_id = results.get("id") or results.get("engram_id")
                        task_id = results.get("task_id")
                if not memory_id:
                    memory_id = str(response)

                ingested_messages += len(turns)
                ingested_sessions += 1
                task_info = f", task_id: {task_id}" if task_id else ""
                print(
                    f"    Session {session_id}: {len(turns)} messages ingested (memory_id: {memory_id}{task_info})"
                )
                break  # Success, move to next session
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"    RETRY {attempt + 1}/{max_retries} for session {session_id}: {e}")
                    time.sleep(1)  # Wait 1 second before retry
                else:
                    print(f"    ERROR (after {max_retries} attempts) session {session_id}: {e}")
                    # Print more details about the error (NebulaException carries a `details` dict)
                    details = getattr(e, "details", None)
                    if details:
                        print(f"      Details: {details}")
    
    print(f"    -> Ingested {ingested_sessions}/{len(sessions)} sessions, {ingested_messages}/{total_messages} messages")
    # Small delay between files to avoid overwhelming the server
    time.sleep(1)
    return {"sessions": ingested_sessions, "messages": ingested_messages}


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
        description="Pre-ingest benchmark files into Nebula",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest_nebula.py simple                    # Ingest all simple test files
  python ingest_nebula.py complex                   # Ingest all complex test files  
  python ingest_nebula.py simple --max-files 5      # Ingest first 5 simple files
  python ingest_nebula.py --file benchmark_001.json # Ingest specific file
  python ingest_nebula.py simple --list             # List files without ingesting

After ingestion, run evaluations with: python run_eval.py simple --adapter nebula --skip-ingest
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
        "--start-from",
        type=int,
        default=1,
        help="Start from file N (1-based, skip earlier files)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List files that would be ingested (dry run)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Nebula HTTP timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--collection-id",
        type=str,
        default=None,
        help="Ingest into an existing Nebula collection UUID (avoids collection create/lookup).",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help="Collection name to create/use when not providing --collection-id (default: nebula_benchmark_<test_set>).",
    )
    parser.add_argument(
        "--per-file-collections",
        action="store_true",
        help="Create/use a separate collection per benchmark file (slow; may cause timeouts).",
    )
    
    args = parser.parse_args()
    
    if not args.test_set and not args.file:
        parser.print_help()
        sys.exit(1)

    if not args.list and not NEBULA_AVAILABLE:
        print("ERROR: nebula package not installed. Run: pip install nebula-client")
        sys.exit(1)
    
    # Get API key
    api_key = os.environ.get("NEBULA_API_KEY")
    
    if not api_key and not args.list:
        print("ERROR: NEBULA_API_KEY not set. Set the environment variable (or put it in a local .env).")
        sys.exit(1)
    
    # Get files to ingest
    files = get_test_files(args.test_set, args.file)
    if not files:
        sys.exit(1)
    
    # Apply start-from (convert 1-based to 0-based index)
    start_idx = max(0, args.start_from - 1)
    if start_idx > 0:
        print(f"Skipping first {start_idx} files (--start-from {args.start_from})")
        files = files[start_idx:]
    
    if args.max_files:
        files = files[:args.max_files]
    
    print(f"\n{'='*60}")
    print(f"NEBULA BULK INGESTION")
    print(f"{'='*60}")
    print(f"Files to ingest: {len(files)}")
    print(f"Mode: {'DRY RUN (listing only)' if args.list else 'INGESTING'}")
    print(f"{'='*60}\n")
    
    if args.list:
        # Dry run - just list
        total_sessions = 0
        total_messages = 0

        # Best-effort display of target collection strategy
        if args.per_file_collections:
            collection_display = "(per-file collections)"
        elif args.collection_id:
            collection_display = f"UUID:{args.collection_id}"
        else:
            base = args.collection_name or (f"nebula_benchmark_{args.test_set}" if args.test_set else "nebula_benchmark")
            collection_display = f"{base} (UUID created at runtime)"

        for filepath in files:
            result = ingest_file(
                None,
                filepath,
                dry_run=True,
                per_file_collections=args.per_file_collections,
                collection_id=args.collection_id,
                collection_display=collection_display,
            )
            total_sessions += result["sessions"]
            total_messages += result["messages"]
        
        print(f"\n{'='*60}")
        print(f"SUMMARY: {len(files)} files, {total_sessions} sessions, {total_messages} messages")
        print(f"{'='*60}")
        return
    
    # Real ingestion
    client = Nebula(api_key=api_key, timeout=args.timeout)

    # Determine target collection once (unless using per-file collections)
    target_collection_id = None
    if not args.per_file_collections:
        if args.collection_id:
            target_collection_id = args.collection_id
        else:
            base = args.collection_name or (
                f"nebula_benchmark_{args.test_set}" if args.test_set else "nebula_benchmark"
            )
            print(f"Using single collection name: {base}")
            target_collection_id = create_or_get_collection(client, base)
            print(f"Using single collection UUID: {target_collection_id}")
    
    total_sessions = 0
    total_messages = 0
    failed_files = []
    total_files = len(files) + start_idx  # Total files in full set
    for i, filepath in enumerate(files):
        file_num = start_idx + i + 1  # Actual file number
        print(f"\n[{file_num}/{total_files}]", end="")
        try:
            result = ingest_file(
                client,
                filepath,
                collection_id=target_collection_id,
                per_file_collections=args.per_file_collections,
            )
            total_sessions += result["sessions"]
            total_messages += result["messages"]
        except Exception as e:
            print(f"  FAILED: {filepath.name} - {e}")
            failed_files.append(filepath.name)
            # Add a longer delay after a failure to let the server recover
            print(f"  Waiting 10s before continuing...")
            time.sleep(10)
    
    print(f"\n{'='*60}")
    print(f"INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"Files processed: {len(files)}")
    print(f"Files succeeded: {len(files) - len(failed_files)}")
    if failed_files:
        print(f"Files FAILED: {len(failed_files)}")
        for f in failed_files:
            print(f"  - {f}")
    print(f"Sessions ingested: {total_sessions}")
    print(f"Messages ingested: {total_messages}")
    print(f"\nNEXT STEPS:")
    print(f"Run evaluation with: python run_eval.py {args.test_set or 'simple'} --adapter nebula --skip-ingest")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

