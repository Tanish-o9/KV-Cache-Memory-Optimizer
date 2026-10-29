"""Theoretical KV cache memory estimation from model config."""

from dataclasses import dataclass


@dataclass
class KVCacheEstimate:
    """Result of a theoretical KV cache memory estimate."""
    num_layers: int
    num_kv_heads: int
    head_dim: int
    sequence_length: int
    batch_size: int
    bytes_per_element: int
    dtype: str
    total_bytes: int

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 ** 2)

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024 ** 3)


def get_kv_cache_params(config) -> tuple[int, int, int]:
    """Extract (num_layers, num_kv_heads, head_dim) from a HF model config.

    Handles standard MHA, GQA, and MQA architectures.
    """
    num_layers = config.num_hidden_layers

    # KV heads: use num_key_value_heads if available (GQA/MQA), else num_attention_heads (MHA)
    num_kv_heads = getattr(config, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = config.num_attention_heads

    # Head dimension: explicit or derived
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads

    return num_layers, num_kv_heads, head_dim


def estimate_kv_cache_memory(
    config,
    sequence_length: int,
    batch_size: int = 1,
    dtype: str = "float16",
) -> KVCacheEstimate:
    """Estimate theoretical KV cache memory.

    Formula: 2 × layers × kv_heads × head_dim × seq_len × batch_size × bytes_per_element

    The factor of 2 accounts for both key and value tensors.
    """
    dtype_bytes = {"float16": 2, "bfloat16": 2, "float32": 4, "int8": 1, "int4": 0.5}
    bytes_per_element = dtype_bytes.get(dtype)
    if bytes_per_element is None:
        raise ValueError(f"Unknown dtype '{dtype}'. Supported: {list(dtype_bytes.keys())}")

    num_layers, num_kv_heads, head_dim = get_kv_cache_params(config)

    total_bytes = int(
        2 * num_layers * num_kv_heads * head_dim
        * sequence_length * batch_size * bytes_per_element
    )

    return KVCacheEstimate(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        sequence_length=sequence_length,
        batch_size=batch_size,
        bytes_per_element=int(bytes_per_element) if bytes_per_element >= 1 else bytes_per_element,
        dtype=dtype,
        total_bytes=total_bytes,
    )

