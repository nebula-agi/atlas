"""
Evaluation module for benchmarking memory systems.

This module provides adapters for various memory systems (Mem0, Supermemory, Naive RAG, No RAG)
and evaluation logic to test them against the benchmark probes.
"""

from .memory_adapters import (
    MemoryAdapter,
    Mem0Adapter,
    SupermemoryAdapter,
    NaiveRAGAdapter,
    NoRAGAdapter,
)
from .evaluator import BenchmarkEvaluator
from .metrics import compute_answer_score, normalize_answer

__all__ = [
    "MemoryAdapter",
    "Mem0Adapter",
    "SupermemoryAdapter", 
    "NaiveRAGAdapter",
    "NoRAGAdapter",
    "BenchmarkEvaluator",
    "compute_answer_score",
    "normalize_answer",
]

