# KV Cache Memory Optimizer

Benchmark and compare KV cache optimization strategies for LLM inference on consumer GPUs.

## What This Does

Measures how different KV cache implementations affect GPU memory usage, generation latency, throughput, and output quality when running a pretrained LLM. This is a **benchmarking project**, not a training project.

### KV Cache Background

During autoregressive generation, transformer models cache the Key and Value projections from each attention layer to avoid recomputation. This "KV cache" grows linearly with sequence length and can dominate GPU memory on long contexts.

**Theoretical KV cache memory:**

```
bytes = 2 × layers × kv_heads × head_dim × tokens × batch_size × bytes_per_element
```

For Qwen2.5-0.5B (24 layers, 2 KV heads, 64 head dim) at 1024 tokens in float16:
`2 × 24 × 2 × 64 × 1024 × 1 × 2 = 12 MB`

### Cache Strategies Tested

| Strategy | Description | Status |
|----------|-------------|--------|
| **Dynamic** | Standard growing KV cache (baseline) | Supported (built-in) |
| **Quantized (4-bit)** | Quantized KV cache via `optimum-quanto` | Supported (`optimum-quanto`) |
| **Quantized (2-bit)** | Quantized KV cache via `optimum-quanto` | Supported (`optimum-quanto`) |
| **Offloaded** | Static cache with CPU offloading | Supported (built-in) |
| **Static** | Pre-allocated fixed-size cache | Unsupported on Windows (requires `triton`) |

## Setup

### Requirements

- Windows 10/11
- Python 3.10+
- NVIDIA GPU with CUDA support (tested on RTX 2050 / 4 GB VRAM)
- ~2 GB disk space for model weights

### Installation (PowerShell)

```powershell
# Create and activate virtual environment
cd "c:\Users\tanis\Projects\KV Cache Memory Optimizer"
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

If you already have PyTorch with CUDA installed system-wide, you can skip the venv and just install the extra packages:

```powershell
pip install quanto streamlit plotly pandas pyyaml
```

## Usage

### Run a Quick Benchmark (CLI)

```powershell
# Small test — dynamic cache only, 128 tokens
python benchmark.py --context-lengths 128 --strategies dynamic

# Full benchmark — all strategies, multiple context lengths
python benchmark.py --context-lengths 128 256 512 --batch-sizes 1

# Specific strategies
python benchmark.py --strategies dynamic quantized_quanto_4bit offloaded
```

### Launch the Dashboard

```powershell
streamlit run app.py
```

Then click **"▶️ Start Benchmark"** in the dashboard to run benchmarks interactively. Results appear as charts and tables.

### Command-Line Options

```
--model              Model name (default: Qwen/Qwen2.5-0.5B)
--context-lengths    Token counts to test (default: 128 256)
--batch-sizes        Batch sizes to test (default: 1)
--max-new-tokens     Tokens to generate per run (default: 50)
--strategies         Filter strategies by name
```

## Output

Results are saved to `results/` as both CSV and JSON:

```
results/benchmark_20261029_220500.csv
results/benchmark_20261029_220500.json
```

Each row contains: strategy, context length, batch size, peak GPU memory, incremental memory, theoretical cache size, latency, tokens/sec, output match ratio, and status.

## Project Structure

```
├── app.py               # Streamlit dashboard
├── benchmark.py         # Model loading, cache strategies, profiling, CLI
├── memory_estimator.py  # Theoretical KV cache memory calculations
├── requirements.txt     # Dependencies
├── README.md            # This file
└── results/             # Benchmark output (CSV/JSON)
```

## Metrics Collected

- **Peak allocated GPU memory** — maximum `torch.cuda.max_memory_allocated()`
- **Incremental memory** — peak allocated minus pre-generation baseline
- **Theoretical cache memory** — formula-based estimate from model config
- **Time to first token (TTFT)** — latency for generating 1 token
- **Total generation latency** — wall-clock time for full generation
- **Tokens per second** — generated tokens / total time
- **Output match ratio** — token-level agreement with dynamic-cache baseline

### Important Notes on Memory Measurement

- Incremental GPU memory ≠ exact KV cache memory. Temporary tensors, CUDA workspace allocations, and allocator fragmentation contribute to the measured value.
- Theoretical estimates assume dense storage without metadata, scales, or padding overhead.
- Quantized caches may use more memory than the raw bit-width suggests due to quantization scales and residual buffers.
- All GPU measurements use `torch.cuda.synchronize()` before timing and `reset_peak_memory_stats()` before each run.

## Hardware Limitations

- **4 GB VRAM** limits practical context lengths. Expect OOM at ~2048 tokens depending on batch size and strategy.
- Quantized cache reduces KV cache memory but not model weights — the model itself uses ~990 MB in float16.
- Offloaded cache trades GPU memory for increased latency.
- CPU-only runs will not produce GPU memory data.

## Interpreting Results

- **Dynamic cache** provides the baseline. All other strategies are compared against it.
- **Lower peak memory** with similar throughput = good optimization.
- **Higher latency** with lower memory = expected tradeoff for offloading.
- **Match ratio < 1.0** for quantized caches is expected — reduced precision changes logits slightly.
- **Match ratio = 1.0** for static cache is expected — same precision, just pre-allocated.

## Known Limitations

- Only tests greedy decoding (`do_sample=False`).
- Does not benchmark continuous batching or paged attention (requires vLLM).
- Theoretical estimates for quantized caches don't account for quantization metadata.
- TTFT measurement includes a small overhead from the `generate()` call setup.

