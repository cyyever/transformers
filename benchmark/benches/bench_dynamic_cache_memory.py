"""
Benchmark: DynamicLayer (torch.cat) vs PreallocatedDynamicLayer (pre-allocated buffers)

Measures peak GPU memory during simulated decode loops and full model generation.
Run on a CUDA machine:

    python benchmark/benches/bench_dynamic_cache_memory.py

Optional flags:
    --num-layers 32       Number of simulated layers (default: 32)
    --num-heads 32        Number of KV heads (default: 32)
    --head-dim 128        Head dimension (default: 128)
    --prefill-len 512     Prefill sequence length (default: 512)
    --decode-steps 512    Number of decode steps (default: 512)
    --batch-size 1        Batch size (default: 1)
    --dtype float16       Dtype: float16 or bfloat16 (default: float16)
    --model MODEL_ID      Also run full model generation benchmark (e.g. HuggingFaceTB/SmolLM2-135M-Instruct)
    --max-new-tokens 128  Tokens to generate in model benchmark (default: 128)
"""

import argparse
import gc

import torch

from transformers.cache_utils import (
    Cache,
    CacheLayerMixin,
    DynamicLayer,
)


def get_gpu_mem_mb():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def reset_gpu_mem():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def bench_layer_class(
    layer_class: type[CacheLayerMixin],
    *,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    prefill_len: int,
    decode_steps: int,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict:
    """Simulate a decode loop with the given layer class and measure peak memory."""
    reset_gpu_mem()

    # Create layers
    layers = [layer_class() for _ in range(num_layers)]

    # Prefill: one big update per layer
    prefill_k = torch.randn(batch_size, num_heads, prefill_len, head_dim, dtype=dtype, device=device)
    prefill_v = torch.randn(batch_size, num_heads, prefill_len, head_dim, dtype=dtype, device=device)
    for layer in layers:
        layer.update(prefill_k, prefill_v)
    del prefill_k, prefill_v

    after_prefill = get_gpu_mem_mb()
    reset_gpu_mem()
    # Re-measure from current state (peak stats reset, but tensors still allocated)

    # Decode: one token at a time
    for step in range(decode_steps):
        token_k = torch.randn(batch_size, num_heads, 1, head_dim, dtype=dtype, device=device)
        token_v = torch.randn(batch_size, num_heads, 1, head_dim, dtype=dtype, device=device)
        for layer in layers:
            layer.update(token_k, token_v)
        del token_k, token_v

    peak_during_decode = get_gpu_mem_mb()

    # Verify correctness
    total_seq_len = prefill_len + decode_steps
    for i, layer in enumerate(layers):
        assert layer.get_seq_length() == total_seq_len, (
            f"Layer {i}: expected seq_len={total_seq_len}, got {layer.get_seq_length()}"
        )

    # Cleanup
    del layers
    reset_gpu_mem()

    return {
        "after_prefill_mb": after_prefill,
        "peak_during_decode_mb": peak_during_decode,
    }


def bench_model_generation(model_id: str, layer_class: type[CacheLayerMixin], max_new_tokens: int) -> dict:
    """Run full model generation and measure peak GPU memory."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    reset_gpu_mem()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    device = model.device

    # Warm up
    inputs = tokenizer("Hello", return_tensors="pt").to(device)
    model.generate(**inputs, max_new_tokens=5, cache_implementation="dynamic")

    reset_gpu_mem()

    # Build cache with the specified layer class
    cache = DynamicCache(config=model.config)
    # Replace the layer_class_to_replicate so new layers use our class
    cache.layer_class_to_replicate = layer_class
    # Clear any pre-built layers so they get rebuilt with our class
    cache.layers = []

    prompt = "Explain the theory of relativity in simple terms. Start from the basics and work your way up."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    model.generate(**inputs, max_new_tokens=max_new_tokens, past_key_values=cache, do_sample=False)
    peak = get_gpu_mem_mb()

    del model, tokenizer, cache, inputs
    reset_gpu_mem()

    return {"peak_mb": peak}


def main():
    parser = argparse.ArgumentParser(description="Benchmark DynamicLayer memory usage")
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--prefill-len", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--model", type=str, default=None, help="Model ID for full generation benchmark")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "This benchmark requires CUDA"
    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    # Try to import PreallocatedDynamicLayer; if not yet implemented, skip that part
    try:
        from transformers.cache_utils import PreallocatedDynamicLayer

        has_prealloc = True
    except ImportError:
        has_prealloc = False
        print("PreallocatedDynamicLayer not found — running baseline only.\n")

    # ── Isolated layer benchmark ──────────────────────────────────────────
    common = dict(
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        prefill_len=args.prefill_len,
        decode_steps=args.decode_steps,
        batch_size=args.batch_size,
        dtype=dtype,
        device=device,
    )

    total_seq = args.prefill_len + args.decode_steps
    kv_bytes = 2 * args.num_layers * args.batch_size * args.num_heads * total_seq * args.head_dim * dtype.itemsize
    theoretical_kv_mb = kv_bytes / (1024 * 1024)

    print("=" * 70)
    print("DynamicLayer Memory Benchmark")
    print("=" * 70)
    print(f"Config: {args.num_layers} layers, {args.num_heads} heads, head_dim={args.head_dim}")
    print(f"Prefill: {args.prefill_len} tokens, Decode: {args.decode_steps} steps, Batch: {args.batch_size}")
    print(f"Dtype: {args.dtype}")
    print(f"Theoretical KV cache at final seq_len={total_seq}: {theoretical_kv_mb:.1f} MB")
    print()

    print("Running DynamicLayer (torch.cat) baseline...")
    baseline = bench_layer_class(DynamicLayer, **common)
    print(f"  Peak during decode: {baseline['peak_during_decode_mb']:.1f} MB")
    print()

    if has_prealloc:
        print("Running PreallocatedDynamicLayer...")
        optimized = bench_layer_class(PreallocatedDynamicLayer, **common)
        print(f"  Peak during decode: {optimized['peak_during_decode_mb']:.1f} MB")
        print()

        saved = baseline["peak_during_decode_mb"] - optimized["peak_during_decode_mb"]
        pct = 100 * saved / baseline["peak_during_decode_mb"] if baseline["peak_during_decode_mb"] > 0 else 0
        print(f"Savings: {saved:.1f} MB ({pct:.1f}%)")
        print(f"Theoretical KV: {theoretical_kv_mb:.1f} MB | "
              f"Baseline peak: {baseline['peak_during_decode_mb']:.1f} MB | "
              f"Optimized peak: {optimized['peak_during_decode_mb']:.1f} MB")
    print()

    # ── Full model benchmark ─────────────────────────────────────────────
    if args.model:
        print("=" * 70)
        print(f"Full Model Generation Benchmark: {args.model}")
        print("=" * 70)

        print("Running with DynamicLayer...")
        baseline_model = bench_model_generation(args.model, DynamicLayer, args.max_new_tokens)
        print(f"  Peak: {baseline_model['peak_mb']:.1f} MB")

        if has_prealloc:
            print("Running with PreallocatedDynamicLayer...")
            optimized_model = bench_model_generation(args.model, PreallocatedDynamicLayer, args.max_new_tokens)
            print(f"  Peak: {optimized_model['peak_mb']:.1f} MB")

            saved = baseline_model["peak_mb"] - optimized_model["peak_mb"]
            pct = 100 * saved / baseline_model["peak_mb"] if baseline_model["peak_mb"] > 0 else 0
            print(f"\nSavings: {saved:.1f} MB ({pct:.1f}%)")


if __name__ == "__main__":
    main()
