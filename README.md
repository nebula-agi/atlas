## Atlas: Cognitive-Based Benchmarking for Agentic Memory Systems

### Overview

- **`benchmark/`**: benchmark generator + evaluation code (adapters, scoring, metrics)
- **`simple_test_set/`** and **`complex_test_set/`**: JSON test sets
- **`results/`**: saved runs / example results
- **Entry points**:
  - `run_eval.py` (recommended)
  - `ingest_mem0.py` (pre-ingest Mem0)
  - `ingest_nebula.py` (pre-ingest Nebula)

### Setup

- **Python**: 3.10+
- **Install**:

```bash
pip install -r requirements.txt
```

- **Credentials**: copy `env.example` to `.env` and fill in what you use (or export env vars).
  - **LLM (set at least one)**: `OPENROUTER_API_KEY` or `GOOGLE_API_KEY` or `OPENAI_API_KEY`
  - **Memory systems (only if you run those adapters)**: `MEM0_API_KEY`, `SUPERMEMORY_API_KEY`, `NEBULA_API_KEY`

### Run evaluation

```bash
# all adapters, simple test set
python run_eval.py simple

# single adapter
python run_eval.py complex --adapter naive_rag

# quicker smoke run
python run_eval.py simple --adapter no_rag --max-files 3 --no-llm-eval
```

Results are written to `results/` by default.

### Pre-ingestion (optional)

Use this if you want to ingest once and then evaluate with `--skip-ingest`.

```bash
# Mem0
python ingest_mem0.py simple
python run_eval.py simple --adapter mem0 --skip-ingest

# Nebula
python ingest_nebula.py simple
python run_eval.py simple --adapter nebula --skip-ingest
```

### License

MIT License (see `LICENSE`).
