"""KV Cache Memory Optimizer — benchmarking engine.

Loads a pretrained model and benchmarks different KV cache strategies,
measuring GPU memory, latency, throughput, and output consistency.
"""

import gc
import json
import csv
import time
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from memory_estimator import estimate_kv_cache_memory, get_kv_cache_params

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
RESULTS_DIR = Path(__file__).parent / "results"
TEST_PROMPTS = [
    "The theory of general relativity describes",
    "In computer science, a hash table is",
    "The water cycle begins when",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_mem_mb() -> dict[str, float]:
    """Current GPU memory stats in MB. Returns zeros on CPU."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0, "peak_allocated_mb": 0, "peak_reserved_mb": 0}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2,
    }


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"gpu_name": "N/A (CPU)", "gpu_vram_gb": 0}
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_vram_gb": round(props.total_memory / 1024**3, 2),
    }


def get_env_info() -> dict:
    import transformers
    return {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        **get_gpu_info(),
    }

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str = DEFAULT_MODEL, device: torch.device | None = None):
    """Load model and tokenizer. Uses float16 on GPU, float32 on CPU."""
    device = device or get_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()
    return model, tokenizer

# ---------------------------------------------------------------------------
# Cache strategy definitions
# ---------------------------------------------------------------------------

@dataclass
class CacheStrategy:
    name: str
    description: str
    generate_kwargs: dict = field(default_factory=dict)
    setup_error: str | None = None  # set if strategy can't run


def get_cache_strategies(model_config) -> list[CacheStrategy]:
    """Build list of cache strategies, checking availability."""
    strategies = []

    # 1. Dynamic cache (baseline) — always available
    strategies.append(CacheStrategy(
        name="dynamic",
        description="Standard dynamically-growing KV cache (baseline)",
        generate_kwargs={},  # default behavior
    ))

    # 2. Quantized cache (quanto, 4-bit)
    quanto_error = None
    try:
        from transformers import QuantizedCache  # noqa: F401
        import optimum.quanto  # noqa: F401
    except ImportError as e:
        quanto_error = f"Missing dependency: {e}"

    strategies.append(CacheStrategy(
        name="quantized_quanto_4bit",
        description="Quantized KV cache via optimum-quanto backend (4-bit)",
        generate_kwargs={"cache_implementation": "quantized",
                         "cache_config": {"backend": "quanto", "nbits": 4}},
        setup_error=quanto_error,
    ))

    # 3. Quantized cache (quanto, 2-bit)
    strategies.append(CacheStrategy(
        name="quantized_quanto_2bit",
        description="Quantized KV cache via optimum-quanto backend (2-bit)",
        generate_kwargs={"cache_implementation": "quantized",
                         "cache_config": {"backend": "quanto", "nbits": 2}},
        setup_error=quanto_error,
    ))

    # 4. Offloaded cache via StaticCache(offloading=True)
    offload_error = None
    try:
        from transformers import StaticCache
        import inspect
        sig = inspect.signature(StaticCache.__init__)
        if "offloading" not in sig.parameters:
            offload_error = "StaticCache does not support offloading parameter in this version"
    except ImportError as e:
        offload_error = f"Missing dependency: {e}"

    strategies.append(CacheStrategy(
        name="offloaded",
        description="Static cache with CPU offloading (trades latency for GPU memory)",
        generate_kwargs={"cache_implementation": "offloaded"},
        setup_error=offload_error if offload_error else None,
    ))

    # 5. Static cache
    static_error = None
    try:
        from transformers import StaticCache  # noqa: F401
        if torch.cuda.is_available():
            import triton  # noqa: F401
    except ImportError as e:
        static_error = f"Static cache requires triton for compilation: {e}"

    strategies.append(CacheStrategy(
        name="static",
        description="Pre-allocated static KV cache (fixed max length)",
        generate_kwargs={"cache_implementation": "static"},
        setup_error=static_error,
    ))

    return strategies

# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    timestamp: str = ""
    model_name: str = ""
    strategy: str = ""
    strategy_description: str = ""
    batch_size: int = 1
    prompt_length: int = 0
    max_new_tokens: int = 0
    generated_token_count: int = 0
    # Memory (MB)
    model_memory_mb: float = 0
    baseline_allocated_mb: float = 0
    baseline_reserved_mb: float = 0
    peak_allocated_mb: float = 0
    peak_reserved_mb: float = 0
    incremental_memory_mb: float = 0
    theoretical_cache_mb: float = 0
    # Timing
    ttft_ms: float = 0
    total_latency_ms: float = 0
    tokens_per_second: float = 0
    # Output consistency
    output_text: str = ""
    output_token_ids: str = ""  # stored as JSON list
    match_ratio: float = -1  # -1 = baseline or not compared
    prefix_match_length: int = -1
    full_match: bool = False
    # Status
    status: str = "pending"
    error: str = ""
    device: str = ""
    dtype: str = ""
    # Environment
    gpu_name: str = ""
    torch_version: str = ""
    transformers_version: str = ""


def _measure_ttft(model, input_ids, attention_mask, generate_kwargs) -> float:
    """Measure time-to-first-token by generating exactly 1 token."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            do_sample=False,
            **generate_kwargs,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def run_single_benchmark(
    model,
    tokenizer,
    strategy: CacheStrategy,
    prompt: str,
    batch_size: int = 1,
    max_new_tokens: int = 50,
    model_memory_mb: float = 0,
) -> BenchmarkResult:
    """Run a single benchmark: one strategy, one prompt, one batch size."""
    device = next(model.parameters()).device
    env = get_env_info()
    result = BenchmarkResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_name=model.config._name_or_path,
        strategy=strategy.name,
        strategy_description=strategy.description,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        model_memory_mb=model_memory_mb,
        device=str(device),
        dtype=str(next(model.parameters()).dtype),
        gpu_name=env.get("gpu_name", ""),
        torch_version=env.get("torch_version", ""),
        transformers_version=env.get("transformers_version", ""),
    )

    if strategy.setup_error:
        result.status = "unsupported"
        result.error = strategy.setup_error
        return result

    try:
        # Tokenize
        inputs = tokenizer(
            [prompt] * batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        result.prompt_length = inputs["input_ids"].shape[1]

        # Theoretical estimate
        total_seq_len = result.prompt_length + max_new_tokens
        dtype_str = "float16" if device.type == "cuda" else "float32"
        if "nbits" in strategy.generate_kwargs.get("cache_config", {}):
            nbits = strategy.generate_kwargs["cache_config"]["nbits"]
            dtype_str = f"int{nbits}" if nbits in (4, 8) else dtype_str
        estimate = estimate_kv_cache_memory(model.config, total_seq_len, batch_size, dtype_str)
        result.theoretical_cache_mb = estimate.total_mb

        cleanup()
        reset_peak_memory()

        # Baseline memory before generation
        result.baseline_allocated_mb = gpu_mem_mb()["allocated_mb"]
        result.baseline_reserved_mb = gpu_mem_mb()["reserved_mb"]

        # --- Warm-up run ---
        with torch.no_grad():
            model.generate(
                **inputs, max_new_tokens=2, do_sample=False,
                **strategy.generate_kwargs,
            )
        cleanup()
        reset_peak_memory()
        result.baseline_allocated_mb = gpu_mem_mb()["allocated_mb"]

        # --- TTFT ---
        result.ttft_ms = _measure_ttft(
            model, inputs["input_ids"], inputs["attention_mask"],
            strategy.generate_kwargs,
        )
        cleanup()
        reset_peak_memory()
        result.baseline_allocated_mb = gpu_mem_mb()["allocated_mb"]

        # --- Main generation ---
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                **strategy.generate_kwargs,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        # Memory after generation
        mem = gpu_mem_mb()
        result.peak_allocated_mb = mem["peak_allocated_mb"]
        result.peak_reserved_mb = mem["peak_reserved_mb"]
        result.incremental_memory_mb = result.peak_allocated_mb - result.baseline_allocated_mb

        # Timing
        result.total_latency_ms = elapsed * 1000
        generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
        result.generated_token_count = len(generated_ids)
        if result.generated_token_count > 0:
            result.tokens_per_second = result.generated_token_count / elapsed

        # Output
        result.output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        result.output_token_ids = json.dumps(generated_ids.tolist())

        result.status = "success"

    except torch.cuda.OutOfMemoryError as e:
        result.status = "oom"
        result.error = str(e)
        cleanup()
    except Exception as e:
        result.status = "error"
        result.error = f"{type(e).__name__}: {e}"
        cleanup()

    return result

# ---------------------------------------------------------------------------
# Output consistency comparison
# ---------------------------------------------------------------------------

def compare_outputs(baseline: BenchmarkResult, other: BenchmarkResult) -> None:
    """Compare other's output to baseline, updating match fields in-place."""
    if baseline.status != "success" or other.status != "success":
        return

    base_ids = json.loads(baseline.output_token_ids) if baseline.output_token_ids else []
    other_ids = json.loads(other.output_token_ids) if other.output_token_ids else []

    if not base_ids and not other_ids:
        other.match_ratio = 1.0
        other.prefix_match_length = 0
        other.full_match = True
        return

    # Prefix match
    prefix_len = 0
    for a, b in zip(base_ids, other_ids):
        if a == b:
            prefix_len += 1
        else:
            break
    other.prefix_match_length = prefix_len

    # Match ratio: fraction of matching tokens over the longer sequence
    max_len = max(len(base_ids), len(other_ids))
    matches = sum(a == b for a, b in zip(base_ids, other_ids))
    other.match_ratio = matches / max_len if max_len > 0 else 1.0

    other.full_match = (base_ids == other_ids)

# ---------------------------------------------------------------------------
# Full benchmark suite
# ---------------------------------------------------------------------------

def get_model_memory_mb(model) -> float:
    """Approximate model parameter memory in MB."""
    total = 0
    for p in model.parameters():
        total += p.nelement() * p.element_size()
    return total / 1024**2


def run_benchmark_suite(
    model_name: str = DEFAULT_MODEL,
    context_lengths: list[int] | None = None,
    batch_sizes: list[int] | None = None,
    max_new_tokens: int = 50,
    strategies: list[str] | None = None,
    progress_callback=None,
) -> list[BenchmarkResult]:
    """Run benchmarks across strategies, context lengths, and batch sizes.

    Args:
        progress_callback: optional callable(status_str, fraction) for UI updates
    """
    if context_lengths is None:
        context_lengths = [128, 256, 512]
    if batch_sizes is None:
        batch_sizes = [1]

    def progress(msg, frac=0):
        if progress_callback:
            progress_callback(msg, frac)

    progress("Loading model...", 0)
    device = get_device()
    model, tokenizer = load_model(model_name, device)
    model_mem = get_model_memory_mb(model)
    all_strategies = get_cache_strategies(model.config)

    # Filter strategies if specified
    if strategies:
        all_strategies = [s for s in all_strategies if s.name in strategies]

    results: list[BenchmarkResult] = []
    total_runs = len(all_strategies) * len(context_lengths) * len(batch_sizes)
    run_idx = 0

    for strat in all_strategies:
        for ctx_len in context_lengths:
            for bs in batch_sizes:
                run_idx += 1
                label = f"[{run_idx}/{total_runs}] {strat.name} ctx={ctx_len} bs={bs}"
                progress(label, run_idx / total_runs)

                # Build a prompt of approximately ctx_len tokens
                prompt = _make_prompt(tokenizer, ctx_len)

                result = run_single_benchmark(
                    model, tokenizer, strat, prompt,
                    batch_size=bs,
                    max_new_tokens=max_new_tokens,
                    model_memory_mb=model_mem,
                )
                results.append(result)
                cleanup()

    # Compare outputs against dynamic baseline
    _compare_all_outputs(results)

    # Save results
    progress("Saving results...", 1.0)
    save_results(results)

    # Unload model
    del model
    cleanup()

    return results


def _make_prompt(tokenizer, target_length: int) -> str:
    """Create a prompt that tokenizes to approximately target_length tokens."""
    # Repeat a base sentence to reach target length
    base = "The study of artificial intelligence encompasses machine learning, natural language processing, and computer vision. "
    prompt = base
    for _ in range(200):
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) >= target_length:
            # Trim to exact length and decode back
            trimmed_ids = ids[:target_length]
            return tokenizer.decode(trimmed_ids, skip_special_tokens=True)
        prompt += base
    return prompt


def _compare_all_outputs(results: list[BenchmarkResult]):
    """Compare all results against the dynamic baseline at matching config."""
    baselines = {}
    for r in results:
        if r.strategy == "dynamic" and r.status == "success":
            key = (r.prompt_length, r.batch_size)
            baselines[key] = r

    for r in results:
        if r.strategy == "dynamic":
            continue
        key = (r.prompt_length, r.batch_size)
        if key in baselines:
            compare_outputs(baselines[key], r)

# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------

def save_results(results: list[BenchmarkResult], tag: str = "") -> Path:
    """Save results to CSV and JSON in the results/ directory."""
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"benchmark_{ts}" + (f"_{tag}" if tag else "")

    csv_path = RESULTS_DIR / f"{stem}.csv"
    json_path = RESULTS_DIR / f"{stem}.json"

    dicts = [asdict(r) for r in results]

    # CSV
    if dicts:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=dicts[0].keys())
            writer.writeheader()
            writer.writerows(dicts)

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dicts, f, indent=2, default=str)

    return csv_path


def load_latest_results() -> list[dict] | None:
    """Load the most recent JSON results file."""
    RESULTS_DIR.mkdir(exist_ok=True)
    json_files = sorted(RESULTS_DIR.glob("benchmark_*.json"), reverse=True)
    if not json_files:
        return None
    with open(json_files[0], encoding="utf-8") as f:
        return json.load(f)


def load_all_results() -> list[dict]:
    """Load all benchmark records from all JSON result files."""
    RESULTS_DIR.mkdir(exist_ok=True)
    json_files = sorted(RESULTS_DIR.glob("benchmark_*.json"), reverse=True)
    all_records = []
    seen = set()
    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    key = (item.get("timestamp"), item.get("strategy"), item.get("prompt_length"), item.get("batch_size"))
                    if key not in seen:
                        seen.add(key)
                        all_records.append(item)
        except Exception:
            continue
    return all_records


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KV Cache Memory Optimizer Benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--strategies", nargs="+", default=None,
                        help="Filter strategies by name")
    args = parser.parse_args()

    def cli_progress(msg, frac):
        print(f"  {frac*100:5.1f}% | {msg}")

    print(f"Environment: {get_env_info()}")
    print(f"Running benchmark: model={args.model}")

    results = run_benchmark_suite(
        model_name=args.model,
        context_lengths=args.context_lengths,
        batch_sizes=args.batch_sizes,
        max_new_tokens=args.max_new_tokens,
        strategies=args.strategies,
        progress_callback=cli_progress,
    )

    print(f"\n{'='*80}")
    print(f"{'Strategy':<28} {'Ctx':>5} {'BS':>3} {'Peak MB':>9} {'Incr MB':>9} "
          f"{'Latency ms':>11} {'Tok/s':>7} {'Match':>6} {'Status':<10}")
    print(f"{'='*80}")
    for r in results:
        match_str = f"{r.match_ratio:.2f}" if r.match_ratio >= 0 else "base"
        print(f"{r.strategy:<28} {r.prompt_length:>5} {r.batch_size:>3} "
              f"{r.peak_allocated_mb:>9.1f} {r.incremental_memory_mb:>9.1f} "
              f"{r.total_latency_ms:>11.1f} {r.tokens_per_second:>7.1f} "
              f"{match_str:>6} {r.status:<10}")
        if r.error:
            print(f"  |-> {r.error[:100]}")
