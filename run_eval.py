#!/usr/bin/env python3
"""
Convenience script for running the benchmark evaluation.

Usage:
    python run_eval.py simple    # Evaluate all adapters on simple test set
    python run_eval.py complex   # Evaluate all adapters on complex test set
    python run_eval.py simple --adapter naive_rag  # Evaluate only naive_rag
    python run_eval.py both      # Evaluate on both test sets
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Optional: load variables from a local .env file (recommended for open-source usage)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is optional; environment variables still work without it.
    pass

from benchmark.evaluation.run_evaluation import (
    evaluate_all_adapters,
    evaluate_single_adapter,
    setup_environment,
    ADAPTER_TYPES,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation on memory systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_eval.py simple                           # All adapters on simple test set
  python run_eval.py complex                          # All adapters on complex test set
  python run_eval.py both                             # All adapters on both test sets
  python run_eval.py simple --adapter naive_rag      # Only naive_rag on simple
  python run_eval.py simple --adapter no_rag --max-files 5  # Quick test
        """
    )
    
    parser.add_argument(
        "test_set",
        choices=["simple", "complex", "both"],
        help="Which test set to evaluate",
    )
    parser.add_argument(
        "--adapter",
        choices=ADAPTER_TYPES,
        help="Specific adapter to evaluate (default: all)",
    )
    parser.add_argument(
        "--adapters",
        nargs="+",
        choices=ADAPTER_TYPES,
        help="List of adapters to evaluate",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of benchmark files to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--no-llm-eval",
        action="store_true",
        help="Disable LLM judge (uses heuristic scoring instead)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Show retrieved context for each probe in the terminal",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingestion (use with pre-ingested Mem0 data from ingest_mem0.py)",
    )
    
    args = parser.parse_args()
    
    # Check environment
    print("\n" + "=" * 60)
    print("NEBULA MEMORY BENCHMARK EVALUATION")
    print("=" * 60)
    
    if not setup_environment():
        print("\n[WARNING] Some environment variables are not set.")
        print(
            "   Set one of: GOOGLE_API_KEY (or Vertex ADC), OPENROUTER_API_KEY, or OPENAI_API_KEY for LLM access."
        )
    
    # Determine test sets
    project_dir = Path(__file__).parent
    test_sets = []
    
    if args.test_set in ["simple", "both"]:
        test_sets.append(project_dir / "simple_test_set")
    if args.test_set in ["complex", "both"]:
        test_sets.append(project_dir / "complex_test_set")
    
    # Validate directories
    for test_dir in test_sets:
        if not test_dir.exists():
            print(f"Error: Test directory not found: {test_dir}")
            sys.exit(1)
    
    # Determine adapters
    if args.adapter:
        adapters_list = [args.adapter]
    elif args.adapters:
        adapters_list = args.adapters
    else:
        adapters_list = None  # All adapters
    
    # Run evaluation
    all_results = {}
    
    for test_dir in test_sets:
        print(f"\n{'#' * 60}")
        print(f"# TEST SET: {test_dir.name}")
        print(f"{'#' * 60}")
        
        if args.adapter:
            result = evaluate_single_adapter(
                adapter_type=args.adapter,
                benchmark_dir=test_dir,
                output_dir=args.output_dir,
                max_files=args.max_files,
                verbose=not args.quiet,
                use_llm_eval=not args.no_llm_eval,
                show_context=args.show_context,
                skip_ingest=args.skip_ingest,
            )
            all_results[f"{test_dir.name}_{args.adapter}"] = result
        else:
            result = evaluate_all_adapters(
                benchmark_dir=test_dir,
                output_dir=args.output_dir,
                max_files=args.max_files,
                verbose=not args.quiet,
                use_llm_eval=not args.no_llm_eval,
                adapters=adapters_list,
                show_context=args.show_context,
                skip_ingest=args.skip_ingest,
            )
            all_results[test_dir.name] = result
    
    # Final summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {args.output_dir}/")
    
    return all_results


if __name__ == "__main__":
    main()

