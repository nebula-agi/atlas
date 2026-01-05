#!/usr/bin/env python3
"""
Main evaluation script for benchmarking memory systems.

Usage:
    python -m benchmark.evaluation.run_evaluation --adapter mem0 --benchmark-dir simple_test_set
    python -m benchmark.evaluation.run_evaluation --all --benchmark-dir simple_test_set
    python -m benchmark.evaluation.run_evaluation --adapter naive_rag --benchmark-dir complex_test_set --max-files 10
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    # Optional: load variables from a local .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from .evaluator import BenchmarkEvaluator, run_evaluation
from .memory_adapters import create_adapter


ADAPTER_TYPES = ["mem0", "supermemory", "naive_rag", "no_rag", "nebula"]


def setup_environment():
    """Check and print environment setup status."""
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)
    
    # Check API keys
    env_vars = {
        "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY"),
        "MEM0_API_KEY": os.environ.get("MEM0_API_KEY"),
        "SUPERMEMORY_API_KEY": os.environ.get("SUPERMEMORY_API_KEY"),
        "NEBULA_API_KEY": os.environ.get("NEBULA_API_KEY"),
        "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
    }
    
    for key, value in env_vars.items():
        status = "[OK] Set" if value else "[X] Not set"
        print(f"  {key}: {status}")
    
    # Check LLM availability (OpenRouter is now the priority)
    llm_available = env_vars["OPENROUTER_API_KEY"] or env_vars["GOOGLE_API_KEY"] or env_vars["OPENAI_API_KEY"]
    if not llm_available:
        print("\n[WARNING] No LLM API key found. Set OPENROUTER_API_KEY (recommended), GOOGLE_API_KEY, or OPENAI_API_KEY")
        return False
    
    if env_vars["OPENROUTER_API_KEY"]:
        print("\n[OK] Using OpenRouter with GPT-4o")
    
    print("=" * 60)
    return True


def evaluate_single_adapter(
    adapter_type: str,
    benchmark_dir: Path,
    output_dir: Path,
    max_files: Optional[int] = None,
    verbose: bool = True,
    search_top_k: int = 10,
    use_llm_eval: bool = True,  # LLM judge is now default
    show_context: bool = False,  # Show retrieved context in output
    skip_ingest: bool = False,  # Skip ingestion for pre-ingested data
) -> Dict[str, Any]:
    """
    Evaluate a single adapter against the benchmark directory.
    """
    print(f"\n{'#' * 60}")
    print(f"# Evaluating: {adapter_type.upper()}")
    print(f"{'#' * 60}")
    
    try:
        # Create adapter with appropriate kwargs
        adapter_kwargs = {}
        
        if adapter_type == "mem0":
            api_key = os.environ.get("MEM0_API_KEY")
            if not api_key:
                print(f"[WARNING] Skipping {adapter_type}: MEM0_API_KEY not set")
                return {"error": "MEM0_API_KEY not set", "adapter_type": adapter_type}
            adapter_kwargs["api_key"] = api_key
        
        elif adapter_type == "supermemory":
            api_key = os.environ.get("SUPERMEMORY_API_KEY")
            if not api_key:
                print(f"[WARNING] Skipping {adapter_type}: SUPERMEMORY_API_KEY not set")
                return {"error": "SUPERMEMORY_API_KEY not set", "adapter_type": adapter_type}
            adapter_kwargs["api_key"] = api_key

        elif adapter_type == "nebula":
            api_key = os.environ.get("NEBULA_API_KEY")
            if not api_key:
                print(f"[WARNING] Skipping {adapter_type}: NEBULA_API_KEY not set")
                return {"error": "NEBULA_API_KEY not set", "adapter_type": adapter_type}
            adapter_kwargs["api_key"] = api_key
        
        # Create unique user ID for this evaluation run
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if skip_ingest:
            # When skipping ingestion, we'll set user_id per-file later
            # Use a placeholder for now
            adapter_kwargs["user_id"] = "placeholder_will_be_set_per_file"
            print(f"[SKIP-INGEST] Will use filename-based user_ids for pre-ingested data")
        else:
            adapter_kwargs["user_id"] = f"benchmark_{adapter_type}_{run_id}"
        
        # Create adapter
        adapter = create_adapter(adapter_type, **adapter_kwargs)
        
        # Create evaluator
        evaluator = BenchmarkEvaluator(
            adapter=adapter,
            verbose=verbose,
            use_llm_eval=use_llm_eval,
            search_top_k=search_top_k,
            show_context=show_context,
            skip_ingest=skip_ingest,
        )
        
        # Run evaluation
        results = evaluator.evaluate_directory(
            directory=benchmark_dir,
            max_files=max_files,
        )
        
        # Generate output path
        benchmark_name = benchmark_dir.name
        output_path = output_dir / f"results_{adapter_type}_{benchmark_name}_{run_id}.json"
        
        # Generate report
        report = BenchmarkEvaluator.generate_report(results, output_path)
        
        print(f"\n{'=' * 50}")
        print(f"[OK] Results saved to: {output_path}")
        print(f"  Overall accuracy: {report['overall_summary']['overall_accuracy']:.1%}")
        print(f"  Total probes: {report['total_probes']}")
        print(f"  (JSON contains full details: questions, answers, scores)")
        print(f"{'=' * 50}")
        
        return report
        
    except ImportError as e:
        # Optional dependencies (SDKs) may not be installed; skip cleanly.
        print(f"[WARNING] Skipping {adapter_type}: {e}")
        return {"error": str(e), "adapter_type": adapter_type}
    except Exception as e:
        print(f"[ERROR] Error evaluating {adapter_type}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "adapter_type": adapter_type}


def evaluate_all_adapters(
    benchmark_dir: Path,
    output_dir: Path,
    max_files: Optional[int] = None,
    verbose: bool = True,
    search_top_k: int = 10,
    use_llm_eval: bool = True,  # LLM judge is now default
    adapters: Optional[List[str]] = None,
    show_context: bool = False,  # Show retrieved context in output
    skip_ingest: bool = False,  # Skip ingestion for pre-ingested data
) -> Dict[str, Any]:
    """
    Evaluate all (or selected) adapters against the benchmark directory.
    """
    adapters_to_run = adapters or ADAPTER_TYPES
    
    all_reports = {}
    for adapter_type in adapters_to_run:
        report = evaluate_single_adapter(
            adapter_type=adapter_type,
            benchmark_dir=benchmark_dir,
            output_dir=output_dir,
            max_files=max_files,
            verbose=verbose,
            search_top_k=search_top_k,
            use_llm_eval=use_llm_eval,
            show_context=show_context,
            skip_ingest=skip_ingest,
        )
        all_reports[adapter_type] = report
    
    # Generate comparison report
    comparison = generate_comparison_report(all_reports, benchmark_dir, output_dir)
    
    return {
        "individual_reports": all_reports,
        "comparison": comparison,
    }


def generate_comparison_report(
    reports: Dict[str, Dict[str, Any]],
    benchmark_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Generate a comparison report across all adapters.
    """
    print(f"\n{'=' * 60}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    
    comparison = {
        "generated_at": datetime.now().isoformat(),
        "benchmark_dir": str(benchmark_dir),
        "adapters": {},
    }
    
    # Extract key metrics
    for adapter_type, report in reports.items():
        if "error" in report:
            comparison["adapters"][adapter_type] = {
                "status": "error",
                "error": report["error"],
            }
            print(f"  {adapter_type:15} - ERROR: {report['error']}")
        else:
            summary = report.get("overall_summary", {})
            accuracy = summary.get("overall_accuracy", 0)
            count = summary.get("count", 0)
            
            comparison["adapters"][adapter_type] = {
                "status": "success",
                "overall_accuracy": accuracy,
                "total_probes": count,
                "by_answer_type": summary.get("by_answer_type", {}),
                "by_pillar": summary.get("by_pillar", {}),
            }
            print(f"  {adapter_type:15} - Accuracy: {accuracy:.2%} ({count} probes)")
    
    # Save comparison report
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = output_dir / f"comparison_{benchmark_dir.name}_{run_id}.json"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Comparison report saved to: {comparison_path}")
    
    return comparison


def print_detailed_report(report: Dict[str, Any]):
    """Print a detailed breakdown of the evaluation report."""
    print(f"\n{'=' * 60}")
    print("DETAILED BREAKDOWN")
    print(f"{'=' * 60}")
    
    summary = report.get("overall_summary", {})
    
    # By answer type
    print("\nBy Answer Type:")
    for atype, stats in summary.get("by_answer_type", {}).items():
        acc = stats.get("accuracy", 0)
        cnt = stats.get("count", 0)
        print(f"  {atype:20} - {acc:.2%} ({cnt} probes)")
    
    # By pillar
    print("\nBy Pillar:")
    for pillar, stats in summary.get("by_pillar", {}).items():
        acc = stats.get("accuracy", 0)
        cnt = stats.get("count", 0)
        print(f"  {pillar:25} - {acc:.2%} ({cnt} probes)")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark evaluation for memory systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate Mem0 on simple test set
  python -m benchmark.evaluation.run_evaluation --adapter mem0 --benchmark-dir simple_test_set

  # Evaluate all adapters on complex test set
  python -m benchmark.evaluation.run_evaluation --all --benchmark-dir complex_test_set

  # Evaluate specific adapters with limited files
  python -m benchmark.evaluation.run_evaluation --adapters naive_rag no_rag --benchmark-dir simple_test_set --max-files 5

  # Evaluate with heuristic scoring (faster, less accurate)
  python -m benchmark.evaluation.run_evaluation --adapter naive_rag --benchmark-dir simple_test_set --no-llm-eval
        """
    )
    
    # Adapter selection
    adapter_group = parser.add_mutually_exclusive_group()
    adapter_group.add_argument(
        "--adapter",
        choices=ADAPTER_TYPES,
        help="Single adapter to evaluate",
    )
    adapter_group.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all adapters",
    )
    adapter_group.add_argument(
        "--adapters",
        nargs="+",
        choices=ADAPTER_TYPES,
        help="List of adapters to evaluate",
    )
    
    # Benchmark configuration
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        required=True,
        help="Directory containing benchmark JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results"),
        help="Directory to save evaluation results (default: evaluation_results)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of benchmark files to evaluate",
    )
    
    # Evaluation options
    parser.add_argument(
        "--search-top-k",
        type=int,
        default=5,
        help="Number of results to retrieve for each query (default: 5)",
    )
    parser.add_argument(
        "--no-llm-eval",
        action="store_true",
        help="Disable LLM judge and use heuristic scoring instead (faster but less accurate)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed breakdown after evaluation",
    )
    
    args = parser.parse_args()
    
    # Validate benchmark directory
    if not args.benchmark_dir.exists():
        print(f"Error: Benchmark directory not found: {args.benchmark_dir}")
        sys.exit(1)
    
    # Check environment
    if not setup_environment():
        print("Warning: Some environment variables are not set. Some adapters may fail.")
    
    # Determine which adapters to run
    verbose = not args.quiet
    
    if args.adapter:
        # Single adapter
        report = evaluate_single_adapter(
            adapter_type=args.adapter,
            benchmark_dir=args.benchmark_dir,
            output_dir=args.output_dir,
            max_files=args.max_files,
            verbose=verbose,
            search_top_k=args.search_top_k,
            use_llm_eval=not args.no_llm_eval,
        )
        if args.detailed and "error" not in report:
            print_detailed_report(report)
    
    elif args.all or args.adapters:
        # Multiple adapters
        adapters_list = args.adapters if args.adapters else None
        result = evaluate_all_adapters(
            benchmark_dir=args.benchmark_dir,
            output_dir=args.output_dir,
            max_files=args.max_files,
            verbose=verbose,
            search_top_k=args.search_top_k,
            use_llm_eval=not args.no_llm_eval,
            adapters=adapters_list,
        )
        if args.detailed:
            for adapter_type, report in result["individual_reports"].items():
                if "error" not in report:
                    print(f"\n--- {adapter_type.upper()} ---")
                    print_detailed_report(report)
    
    else:
        parser.print_help()
        print("\nError: Please specify --adapter, --adapters, or --all")
        sys.exit(1)
    
    print("\n[OK] Evaluation complete!")


if __name__ == "__main__":
    main()

