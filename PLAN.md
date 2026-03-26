# Plan: Optimize Peak GPU Memory During Inference

## Corrected analysis of DynamicLayer.update()

`DynamicLayer.update()` uses `torch.cat` on every decode step. During this call, old
(size N) and new (size N+1) tensors coexist briefly.

However, **layers update sequentially** — only one layer's K+V spikes at a time. For
Llama-7B (32 layers) at 4K context in float16, the transient is ~64 MB (one layer's
K+V), not ~2 GB. That's ~3% of total KV cache. The CUDA caching allocator also often
reuses the freed block, further reducing the actual spike.

**Pre-allocation would save single-digit % on peak memory for KV cache updates alone.**
The real benefits are reduced fragmentation and fewer allocator calls over long sequences,
not a dramatic peak reduction.

This means the biggest memory wins during inference lie elsewhere.

## Where peak memory actually occurs

1. **Prefill attention (eager)**: materializes full `[batch, heads, seq, seq]` attention
   matrix — O(seq²) memory. For 4K context, 32 heads: 4096² × 32 × 4B = **2 GB**.
   Flash/SDPA attention avoids this, but eager is still the fallback.

2. **Logits materialization**: `[batch, seq, vocab_size]` at prefill. For vocab=128K,
   seq=4K, float32: 4096 × 128K × 4B = **2 GB**. Already mitigated: generation extracts
   only `[:, -1, :]` then `del outputs`.

3. **Chunked prefill intermediate logits**: when using `prefill_chunk_size`, intermediate
   `outputs` objects (containing logits for each chunk) are not freed between chunks.
   Each chunk's logits live until Python GC collects them.

4. **KV cache torch.cat fragmentation**: repeated alloc/free of incrementally growing
   tensors fragments the CUDA memory pool. Over long sequences (8K+ tokens), this can
   cause OOM even when total live memory fits in VRAM, because the allocator can't find
   a contiguous block.

## Plan

### Change 1: PreallocatedDynamicLayer (fragmentation + allocator overhead)

Even though the per-step transient spike is small (~3%), the cumulative fragmentation
from thousands of alloc/free cycles is real. Pre-allocation eliminates this entirely.

Add `PreallocatedDynamicLayer(DynamicLayer)` in `src/transformers/cache_utils.py`:
- `_key_buffer` / `_value_buffer`: pre-allocated backing tensors
- `_seq_len` / `_capacity`: track valid length vs buffer size
- `self.keys` / `self.values`: always views of valid portion (external compatibility)
- Growth: 2× capacity when full, O(log N) resizes total
- Override all methods that touch keys/values: `update`, `get_seq_length`, `crop`,
  `reset`, `offload`, `prefetch`, `reorder_cache`, `batch_repeat_interleave`,
  `batch_select_indices`

Wire into `DynamicCache.__init__` for non-sliding layers (lines 962, 979, 986).

**Not changed**: `DynamicSlidingWindowLayer` (bounded by window), `QuantizedLayer`
(own update logic), `StaticLayer` (already pre-allocated).

Expected impact: eliminates CUDA memory fragmentation from KV cache growth. Measurable
on long sequences (4K+ decode steps). Modest peak reduction (~3-5%) from removing
per-step transient.

### Change 2: Free intermediate outputs in chunked prefill

In `src/transformers/generation/utils.py` line 3778, add `del outputs` inside the
chunked prefill loop (before the next iteration), matching the existing pattern at
line 2787 in the decode loop.

Each intermediate chunk's `outputs` contains logits of shape `[batch, chunk_size, vocab]`.
For chunk_size=1024, vocab=128K, float32: **512 MB per chunk** sitting in memory until GC.

Expected impact: significant for large-vocab models with chunked prefill. Frees
chunk_size × vocab × 4B per chunk immediately instead of waiting for GC.

## Verification

`benchmark/benches/bench_dynamic_cache_memory.py` (already committed) measures:
1. Peak GPU memory for `DynamicLayer` vs `PreallocatedDynamicLayer` in isolation
2. Optional full-model generation comparison

Run on CUDA machine:
```bash
# Isolated layer benchmark (default: 32 layers, 512 prefill, 512 decode)
python benchmark/benches/bench_dynamic_cache_memory.py

# Long-sequence stress test (where fragmentation matters more)
python benchmark/benches/bench_dynamic_cache_memory.py --prefill-len 2048 --decode-steps 4096

# Full model
python benchmark/benches/bench_dynamic_cache_memory.py --model HuggingFaceTB/SmolLM2-135M-Instruct
```

## Risks

- `self.keys`/`self.values` are views of buffer → external code sees correct data
- `DynamicCache.__iter__` yields `layer.keys` → correct because views are synced
- Beam search ops on full buffer: correct but wastes compute on unused capacity (negligible)
- Buffer wastes up to 2× memory right after a resize — bounded and transient
- `offload`/`prefetch` moves full buffer including unused portion — minor bandwidth cost
