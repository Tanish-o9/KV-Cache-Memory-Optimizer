# ⚡ KV Cache Memory Optimizer

### Benchmarking LLM Inference Memory, Latency & Performance Trade-offs

An experimental toolkit for analyzing and optimizing **Key-Value (KV) cache memory** during Large Language Model (LLM) inference. Compare cache strategies, measure GPU memory consumption, evaluate generation performance, and visualize results through an interactive Streamlit dashboard.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?logo=huggingface&logoColor=black" alt="Hugging Face"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Project-ML%20Optimization-success" alt="ML Optimization"/>
</p>

---

## 📌 Overview

During autoregressive text generation, an LLM stores previously computed attention keys and values in GPU memory. This KV cache avoids recomputing attention states for every generated token, improving inference efficiency.

However, KV cache memory grows with sequence length and batch size. Under limited GPU memory, this can restrict context length, concurrent requests, and inference capacity.

**KV Cache Memory Optimizer** investigates this problem by benchmarking different cache strategies and measuring their trade-offs in memory consumption, latency, throughput, and output consistency.

The project combines theoretical memory estimation with actual measurements rather than relying solely on theoretical compression ratios.

## 🎯 Objectives

- Measure KV cache memory consumption across different context lengths and batch sizes.
- Compare Dynamic, Quantized, Offloaded, and Static cache implementations where supported.
- Evaluate peak GPU memory, generation latency, and token throughput.
- Compare optimized outputs against a dynamic-cache baseline.
- Validate theoretical KV cache estimates against measured memory.
- Visualize experiments using an interactive Streamlit dashboard.
- Identify practical trade-offs between memory efficiency and inference performance.

## 🧠 Understanding KV Cache

Transformers generate text one token at a time. During self-attention, each layer computes key (K) and value (V) tensors for the input tokens.

Instead of recomputing these tensors for all previous tokens at every generation step, the model can store and reuse them in a KV cache.

**Without caching:** Previously computed attention states may need to be recomputed.

**With caching:** Previously computed keys and values are reused, reducing redundant computation.

The trade-off is memory consumption: as the sequence grows, more K and V tensors must be stored.

## 📐 KV Cache Memory Formula

For a conventional decoder-only transformer, the approximate uncompressed KV cache size is:

\[
M_{\mathrm{KV}} =
2 \times L \times H_{\mathrm{KV}} \times D_{\mathrm{head}}
\times T \times B \times S
\]

Where:

| Symbol | Description |
|---|---|
| \(L\) | Number of transformer layers |
| \(H_{\mathrm{KV}}\) | Number of key-value heads |
| \(D_{\mathrm{head}}\) | Dimension of each attention head |
| \(T\) | Number of cached tokens |
| \(B\) | Batch size |
| \(S\) | Bytes per stored element |
| \(2\) | Accounts for keys and values |

The formula estimates raw cache tensor storage. Actual GPU memory can be higher because of quantization metadata, temporary tensors, workspaces, allocator overhead, and reserved memory.

For static caches, allocated capacity may exceed the number of tokens currently stored. Quantized caches may also require additional metadata and residual buffers.

## ⚙️ Cache Strategies

| Strategy | Approach | Main trade-off |
|---|---|---|
| Dynamic Cache | Grows as tokens are generated | Flexible allocation, no inherent compression |
| Quantized Cache | Stores K/V using lower-precision representations | Reduced cache storage, possible quality or latency impact |
| Offloaded Cache | Moves cache data between GPU and CPU memory | Lower GPU memory pressure, possible transfer overhead |
| Static Cache | Reserves a predetermined cache capacity | Predictable allocation, potentially better compilation compatibility |

Availability depends on the model architecture, Transformers version, quantization backend, and device.

Unsupported strategies should be reported explicitly rather than silently replaced with another implementation.

## 🏗️ Architecture

The project follows a lightweight Python architecture:

```text
KV Cache Memory Optimizer
│
├── app.py
│   └── Streamlit dashboard and visualizations
│
├── benchmark.py
│   └── Model loading, cache strategies and benchmarks
│
├── memory_estimator.py
│   └── Theoretical KV cache memory calculations
│
├── requirements.txt
│   └── Project dependencies
│
├── README.md
│   └── Documentation
│
└── results/
    └── Benchmark CSV files
```

The actual file structure may differ slightly depending on the implementation.

## 📊 Benchmark Metrics

The benchmarking pipeline evaluates the following metrics:

**Memory**
- Peak allocated GPU memory.
- Peak reserved GPU memory.
- KV cache tensor memory when measurable.
- Theoretical cache memory estimates.
- Memory savings relative to a matching baseline.

**Inference performance**
- Time to first token (TTFT), when reliably measured.
- Total generation latency.
- Average inter-token latency, where supported.
- Tokens generated per second.

**Output consistency**
- Token-level agreement with the dynamic-cache baseline.
- Length of the identical initial token sequence.
- Full sequence match.
- Decoded outputs for selected prompts.

**Reliability**
- Supported and unsupported strategy status.
- Out-of-memory failures.
- Benchmark configuration and environment details.

Measurements should use comparable prompts, decoding settings, batch sizes, and generation lengths.

## 🖥️ Interactive Dashboard

The Streamlit dashboard provides a central interface for inspecting benchmark experiments.

Expected capabilities include:

- GPU and model information.
- Cache strategy selection.
- Context length and batch size configuration.
- Benchmark execution and progress feedback.
- Memory and latency comparison charts.
- Throughput analysis.
- Theoretical versus measured memory visualization.
- Benchmark history and CSV export.
- Output consistency analysis.

All displayed benchmark metrics should come from actual runs. Empty states should be shown when no measurements are available.

> Add an actual screenshot of your dashboard here after the application is running. Avoid presenting mock metrics as real experimental results.

## 🛠️ Tech Stack

- **Python** — implementation and experiment orchestration.
- **PyTorch** — model inference and GPU memory profiling.
- **Hugging Face Transformers** — pretrained LLMs and KV cache implementations.
- **Streamlit** — interactive dashboard.
- **Plotly or Matplotlib** — charts and result visualization.
- **Pandas** — benchmark result processing.
- **CUDA / NVIDIA tools** — GPU execution and supplementary memory monitoring.

## 🚀 Getting Started

### Prerequisites

- Python version compatible with the selected PyTorch and Transformers releases.
- Git.
- NVIDIA GPU and a compatible CUDA-enabled PyTorch installation for GPU benchmarking.
- Sufficient available memory for the selected model and workload.

A smaller pretrained model such as `Qwen/Qwen2.5-0.5B` is a reasonable starting point for a GPU with 4 GB VRAM, although actual feasibility depends on model loading, dtype, cache strategy, and other GPU allocations.

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd kv-cache-memory-optimizer
```

Replace the placeholder with your actual repository URL.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, consult your system's execution-policy restrictions rather than changing security settings blindly. You can also run the environment's Python executable directly.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build compatible with your GPU and CUDA environment if the requirements file does not already specify an appropriate build. Verify the installation before benchmarking.

### 4. Verify GPU availability

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

A CPU fallback may be useful for functional testing, but CPU measurements must not be presented as GPU benchmark results.

### 5. Launch the dashboard

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit in your browser.

## 🧪 Running Experiments

Start with a small workload to verify that the model and cache implementation work.

Suggested initial configuration:

| Parameter | Initial value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` |
| Cache strategy | Dynamic |
| Context length | 128–256 tokens |
| Batch size | 1 |
| Decoding | Greedy |
| Generated tokens | 32–64 |
| Repetitions | 3 or more after the smoke test |

After verifying the baseline:

1. Run the same configuration using a supported quantized cache.
2. Compare offloaded cache if compatible.
3. Test static cache where supported.
4. Increase context length gradually.
5. Test larger batches only when sufficient GPU memory is available.
6. Save the actual results and inspect the dashboard.

Benchmarking should include warm-up runs, CUDA synchronization, repeated measurements, and consistent generation settings.

## 📈 How to Interpret Results

### Memory savings

\[
\text{Memory Saved (\%)} =
\frac{M_{\mathrm{baseline}}-M_{\mathrm{optimized}}}
{M_{\mathrm{baseline}}}\times100
\]

Use comparable measurements and clearly state whether the values refer to cache tensor storage, incremental memory, or total peak GPU memory.

### Latency overhead

\[
\text{Latency Overhead (\%)} =
\frac{t_{\mathrm{optimized}}-t_{\mathrm{baseline}}}
{t_{\mathrm{baseline}}}\times100
\]

Positive values indicate additional latency; negative values indicate lower measured latency.

### Throughput

\[
\text{Throughput} =
\frac{\text{Generated Tokens}}{\text{Generation Time}}
\]

State whether throughput is measured per request or across a batch.

### Output consistency

Compare generated token IDs under identical prompts and deterministic decoding settings. Numerical changes introduced by cache quantization can alter generated tokens, even when the decoded answers appear similar.

## 🔬 Experimental Considerations

- Cache memory and total GPU memory are different metrics.
- Static cache allocation depends on the configured maximum capacity.
- Quantization overhead depends on the backend and implementation.
- Offloading can reduce GPU memory usage without reducing total system memory.
- Different model architectures use different KV head counts.
- CUDA allocator reservation can exceed currently allocated tensor memory.
- Performance results vary with GPU hardware, software versions, prompt length, and workload.
- A smaller cache does not automatically guarantee faster generation.

The goal is to measure these effects rather than assume that one strategy is always best.

## ⚠️ Limitations

- Cache strategy support varies across model architectures and library versions.
- Small models may not show dramatic end-to-end memory savings because model weights and other allocations contribute to total GPU memory.
- Quantized cache implementations may introduce additional dependencies and numerical differences.
- Results obtained on one GPU cannot be generalized to all inference hardware.
- Theoretical memory estimates do not replace direct measurements.

## 🔮 Future Improvements

- Benchmark additional model architectures.
- Compare different supported cache quantization precisions.
- Add multi-request workloads and concurrency analysis.
- Investigate prefix caching and cache eviction.
- Compare optimized cache behavior across GPU hardware.
- Export reproducible benchmark reports.
- Explore integration with production inference engines such as vLLM where the environment supports it.

## 💼 Resume Description

**KV Cache Memory Optimizer | Python, PyTorch, Hugging Face Transformers, Streamlit**

- Developed an LLM inference benchmarking toolkit to analyze KV cache memory consumption across different context lengths and batch sizes.
- Benchmarked supported Dynamic, Quantized, Offloaded, and Static cache implementations to evaluate memory–latency trade-offs.
- Built a theoretical memory estimator and an interactive dashboard to visualize GPU memory usage, throughput, latency, and output consistency.

Update these bullets with verified quantitative results once experiments are complete.

## 🤝 Contributing

Contributions, bug reports, and suggestions are welcome. Please include the model configuration, software versions, hardware details, and reproduction steps when reporting benchmark issues.

## 📄 License

Add a license appropriate for your project before distributing it publicly. Also review the pretrained model's license and usage terms separately.

---

**Built to explore a practical question in LLM engineering: how can inference memory be reduced without unnecessarily sacrificing performance or output quality?**
