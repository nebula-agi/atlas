"""
Core benchmark evaluator module.

Orchestrates the evaluation process:
1. Load benchmark files
2. Ingest sessions into memory system
3. Run probes and collect answers
4. Score answers against gold standards
5. Generate evaluation reports
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict

from .memory_adapters import MemoryAdapter, create_adapter
from .metrics import compute_answer_score, aggregate_scores


@dataclass
class ProbeResult:
    """Result of evaluating a single probe."""
    probe_id: str
    pillar: str
    subpillar: str
    question: str
    answer_type: str
    gold_answer: str
    predicted_answer: str
    retrieved_context: List[str]
    score: float
    match_type: str
    search_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float


@dataclass
class BenchmarkResult:
    """Result of evaluating a single benchmark file."""
    benchmark_file: str
    adapter_type: str
    num_sessions: int
    num_probes: int
    ingestion_time_s: float
    total_evaluation_time_s: float
    probe_results: List[ProbeResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    stored_memories: List[Dict[str, Any]] = field(default_factory=list)  # Memories extracted by the adapter
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_file": self.benchmark_file,
            "adapter_type": self.adapter_type,
            "num_sessions": self.num_sessions,
            "num_probes": self.num_probes,
            "ingestion_time_s": self.ingestion_time_s,
            "total_evaluation_time_s": self.total_evaluation_time_s,
            "probe_results": [asdict(p) for p in self.probe_results],
            "summary": self.summary,
            "stored_memories": self.stored_memories,
        }


class BenchmarkEvaluator:
    """
    Main evaluator class for running benchmarks against memory systems.
    """
    
    def __init__(
        self,
        adapter: MemoryAdapter,
        verbose: bool = True,
        use_llm_eval: bool = True,  # Now defaults to True - uses LLM judge for binary scoring
        search_top_k: int = 10,
        show_context: bool = False,  # Show retrieved context in terminal output
        skip_ingest: bool = False,  # Skip ingestion for pre-ingested data
    ):
        """
        Initialize the evaluator.
        
        Args:
            adapter: The memory adapter to evaluate
            verbose: Whether to print progress
            use_llm_eval: Whether to use LLM judge for binary scoring (default: True)
            search_top_k: Number of results to retrieve for each query
            show_context: Whether to show retrieved context for each probe
            skip_ingest: Skip ingestion (for pre-ingested Mem0 data)
        """
        self.adapter = adapter
        self.verbose = verbose
        self.use_llm_eval = use_llm_eval
        self.search_top_k = search_top_k
        self.show_context = show_context
        self.skip_ingest = skip_ingest
    
    def log(self, message: str) -> None:
        """Print message if verbose mode is on."""
        if self.verbose:
            print(message)
    
    def load_benchmark(self, filepath: Path) -> Dict[str, Any]:
        """Load a benchmark JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def evaluate_benchmark(self, filepath: Path) -> BenchmarkResult:
        """
        Evaluate a single benchmark file.
        
        Args:
            filepath: Path to the benchmark JSON file
        
        Returns:
            BenchmarkResult with all probe results and summary
        """
        filepath = Path(filepath)
        self.log(f"\n{'='*60}")
        self.log(f"Evaluating: {filepath.name}")
        self.log(f"{'='*60}")
        
        # Load benchmark
        benchmark = self.load_benchmark(filepath)
        sessions = benchmark.get("sessions", [])
        probes = benchmark.get("probes", [])
        
        self.log(f"Sessions: {len(sessions)}, Probes: {len(probes)}")
        
        if self.skip_ingest:
            # Skip ingestion - use pre-ingested data
            # For Mem0, the user_id should already be set to match the filename
            self.log("[SKIP] Skipping ingestion - using pre-ingested data")
            self.log(f"[SKIP] Adapter user_id: {self.adapter.user_id}")
            ingestion_time = 0.0
        else:
            # Clear previous state
            self.adapter.clear()
            
            # Ingest sessions
            self.log("Ingesting sessions...")
            ingest_start = time.time()
            self.adapter.ingest_sessions(sessions)
            ingestion_time = time.time() - ingest_start
            self.log(f"Ingestion completed in {ingestion_time:.2f}s")
            
            # Allow time for indexing (some systems need this)
            time.sleep(1)
        
        # Evaluate probes
        self.log("Evaluating probes...")
        eval_start = time.time()
        probe_results = []
        
        for i, probe in enumerate(probes):
            probe_result = self._evaluate_probe(probe, i + 1, len(probes))
            probe_results.append(probe_result)
        
        total_eval_time = time.time() - eval_start
        
        # Compute summary
        result_dicts = [
            {
                "score": r.score,
                "answer_type": r.answer_type,
                "pillar": r.pillar,
            }
            for r in probe_results
        ]
        summary = aggregate_scores(result_dicts)
        
        self.log(f"\n{'-' * 50}")
        self.log(f"RESULTS: {summary['total_score']:.0f}/{summary['count']} correct ({summary['overall_accuracy']:.1%})")
        self.log(f"Time: {total_eval_time:.2f}s")
        self.log(f"{'-' * 50}")
        
        # Get stored memories from adapter if available
        stored_memories = getattr(self.adapter, 'stored_memories', [])
        
        return BenchmarkResult(
            benchmark_file=str(filepath),
            adapter_type=type(self.adapter).__name__,
            num_sessions=len(sessions),
            num_probes=len(probes),
            ingestion_time_s=ingestion_time,
            total_evaluation_time_s=total_eval_time,
            probe_results=probe_results,
            summary=summary,
            stored_memories=stored_memories,
        )
    
    def _evaluate_probe(self, probe: Dict[str, Any], current: int, total: int) -> ProbeResult:
        """Evaluate a single probe."""
        probe_id = probe.get("id", "unknown")
        pillar = probe.get("pillar", "unknown")
        subpillar = probe.get("subpillar", "unknown")
        question = probe.get("question", "")
        answer_type = probe.get("answer_type", "short_answer")
        gold_answer = probe.get("gold_answer", {})
        gold_text = gold_answer.get("text", "")
        
        self.log(f"\n  [{current}/{total}] {probe_id}")
        self.log(f"    Question: {question[:100]}{'...' if len(question) > 100 else ''}")
        
        # Search for relevant context (track latency)
        search_start = time.time()
        context = self.adapter.search(question, top_k=self.search_top_k)
        search_latency = (time.time() - search_start) * 1000  # ms
        
        # Display retrieved context if show_context is enabled
        if self.show_context:
            if context:
                self.log(f"    Retrieved Context ({len(context)} chunks):")
                for i, ctx in enumerate(context[:3]):  # Show top 3
                    ctx_preview = ctx[:200].replace('\n', ' ')
                    self.log(f"      [{i+1}] {ctx_preview}{'...' if len(ctx) > 200 else ''}")
            else:
                self.log(f"    Retrieved Context: (none)")
        
        # Generate answer (track latency)
        gen_start = time.time()
        predicted = self.adapter.generate_answer(question, context)
        generation_latency = (time.time() - gen_start) * 1000  # ms
        
        total_latency = search_latency + generation_latency
        
        # Score the answer
        score_result = compute_answer_score(
            prediction=predicted,
            gold_answer=gold_answer,
            answer_type=answer_type,
            use_llm_eval=self.use_llm_eval,
        )
        
        score = score_result.get("score", 0.0)
        match_type = score_result.get("match_type", "unknown")
        
        # Display results for auditing
        score_icon = "[OK]" if score == 1 else "[X]"
        self.log(f"    Gold Answer:  {gold_text}")
        self.log(f"    Model Answer: {predicted[:150]}{'...' if len(predicted) > 150 else ''}")
        self.log(f"    Result: {score_icon} {'CORRECT' if score == 1 else 'INCORRECT'} ({match_type})")
        
        return ProbeResult(
            probe_id=probe_id,
            pillar=pillar,
            subpillar=subpillar,
            question=question,
            answer_type=answer_type,
            gold_answer=gold_text,
            predicted_answer=predicted,
            retrieved_context=context[:3],  # Keep top 3 for logging
            score=score,
            match_type=match_type,
            search_latency_ms=search_latency,
            generation_latency_ms=generation_latency,
            total_latency_ms=total_latency,
        )
    
    def evaluate_directory(
        self,
        directory: Path,
        pattern: str = "*.json",
        max_files: Optional[int] = None,
    ) -> List[BenchmarkResult]:
        """
        Evaluate all benchmark files in a directory.
        
        Args:
            directory: Path to directory containing benchmark files
            pattern: Glob pattern for benchmark files
            max_files: Maximum number of files to evaluate (None for all)
        
        Returns:
            List of BenchmarkResult objects
        """
        directory = Path(directory)
        files = sorted(directory.glob(pattern))
        
        if max_files:
            files = files[:max_files]
        
        self.log(f"\nFound {len(files)} benchmark files in {directory}")
        
        results = []
        for filepath in files:
            try:
                # Set user_id per-file so each benchmark gets its own container/namespace
                # This ensures memories from different files don't mix
                filename_user_id = filepath.stem  # filename without extension
                self.adapter.user_id = filename_user_id
                self.log(f"Set user_id/container to: {filename_user_id}")
                
                result = self.evaluate_benchmark(filepath)
                results.append(result)
            except Exception as e:
                self.log(f"Error evaluating {filepath}: {e}")
                continue
        
        return results
    
    @staticmethod
    def generate_report(
        results: List[BenchmarkResult],
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: List of BenchmarkResult objects
            output_path: Optional path to save the report as JSON
        
        Returns:
            Report dictionary with aggregate statistics
        """
        if not results:
            return {"error": "No results to report"}
        
        # Aggregate all probe results
        all_probe_results = []
        for result in results:
            for probe in result.probe_results:
                all_probe_results.append({
                    "score": probe.score,
                    "answer_type": probe.answer_type,
                    "pillar": probe.pillar,
                    "benchmark_file": result.benchmark_file,
                })
        
        overall_summary = aggregate_scores(all_probe_results)
        
        # Collect all stored memories from all benchmark results
        all_stored_memories = []
        for r in results:
            all_stored_memories.extend(r.stored_memories)
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "adapter_type": results[0].adapter_type if results else "unknown",
            "num_benchmarks": len(results),
            "total_probes": len(all_probe_results),
            "overall_summary": overall_summary,
            "stored_memories": all_stored_memories,  # All memories extracted by the adapter
            "benchmark_summaries": [
                {
                    "file": r.benchmark_file,
                    "accuracy": r.summary.get("overall_accuracy", 0),
                    "num_probes": r.num_probes,
                    "ingestion_time_s": r.ingestion_time_s,
                    "eval_time_s": r.total_evaluation_time_s,
                }
                for r in results
            ],
            "detailed_results": [r.to_dict() for r in results],
        }
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report


def run_evaluation(
    adapter_type: str,
    benchmark_dir: Path,
    output_path: Optional[Path] = None,
    max_files: Optional[int] = None,
    verbose: bool = True,
    **adapter_kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to run a complete evaluation.
    
    Args:
        adapter_type: One of 'mem0', 'supermemory', 'naive_rag', 'no_rag', 'nebula'
        benchmark_dir: Directory containing benchmark JSON files
        output_path: Optional path to save results
        max_files: Maximum number of files to evaluate
        verbose: Whether to print progress
        **adapter_kwargs: Additional arguments for the adapter
    
    Returns:
        Evaluation report dictionary
    """
    # Create adapter
    adapter = create_adapter(adapter_type, **adapter_kwargs)
    
    # Create evaluator
    evaluator = BenchmarkEvaluator(adapter=adapter, verbose=verbose)
    
    # Run evaluation
    results = evaluator.evaluate_directory(
        directory=benchmark_dir,
        max_files=max_files,
    )
    
    # Generate report
    report = BenchmarkEvaluator.generate_report(results, output_path)
    
    return report

