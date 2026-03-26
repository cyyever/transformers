# Plan: Optimize Peak GPU Memory During Inference

## Problem

`DynamicLayer.update()` uses `torch.cat` on every decode step, which allocates a new
tensor of size N+1, copies N elements from the old cache, then frees the old tensor.
During this operation, both old and new tensors coexist, causing a **transient peak of
~2× the KV cache size on every single decode step**.

For Llama-7B at 4K context in float16, this means ~2 GB of transient memory pressure
per step — often the difference between fitting in VRAM and OOM.

## Solution: PreallocatedDynamicLayer

Add a new `PreallocatedDynamicLayer` class that pre-allocates KV cache buffers with
geometric growth (2×), eliminating the `torch.cat` transient spike on ~99% of decode steps.

### Design

- `_key_buffer` / `_value_buffer`: pre-allocated backing tensors
- `_seq_len`: tracks how many tokens are valid
- `self.keys` / `self.values`: always views of the valid portion (for external compatibility)
- Growth: when buffer is full, allocate 2× capacity and copy once
- `_sync_views()`: updates `self.keys`/`self.values` to point at valid buffer portion

### Memory profile comparison

| Metric                  | DynamicLayer (torch.cat) | PreallocatedDynamicLayer  |
|-------------------------|--------------------------|---------------------------|
| Transient spike/step    | +N (every step)          | 0 (most steps)            |
| Resize spike (rare)     | N/A                      | ~3× current (O(log N) times) |
| Buffer overhead         | 0                        | ≤ N unused slots          |
| **Peak across decode**  | **~2N every step**       | **≤ 2N buffer, no spike** |

## Changes

### 1. `src/transformers/cache_utils.py`

Add `PreallocatedDynamicLayer(DynamicLayer)` with:
- `__init__`: init `_key_buffer`, `_value_buffer`, `_seq_len`, `_capacity`
- `lazy_initialization`: allocate buffer sized to first input
- `_ensure_capacity(needed)`: geometric growth
- `_sync_views()`: set `self.keys`/`self.values` as buffer views
- `update()`: in-place copy + capacity check (no torch.cat)
- Override: `get_seq_length`, `crop`, `reset`, `offload`, `prefetch`,
  `reorder_cache`, `batch_repeat_interleave`, `batch_select_indices`

Update `DynamicCache.__init__` to use `PreallocatedDynamicLayer` instead of
`DynamicLayer` for non-sliding layers (lines 962, 979, 986).

**Not changed**: `DynamicSlidingWindowLayer` (bounded by window size, torch.cat
overhead is small and fixed), `QuantizedLayer` (has its own update logic),
`StaticLayer` (already pre-allocated).

### 2. `src/transformers/generation/utils.py`

Add `del outputs` inside the chunked prefill loop (line 3778-3781) to free
intermediate logits between chunks, matching the existing pattern at line 2787.

## Verification

`benchmarks/benchmark_dynamic_cache_memory.py` — GPU-runnable script that:
1. Measures peak GPU memory for both `DynamicLayer` and `PreallocatedDynamicLayer`
   in isolation (simulated decode loop)
2. Measures peak GPU memory for full model generation with `DynamicCache`
   using both layer types
3. Reports savings in MB and percentage

Run: `python benchmarks/benchmark_dynamic_cache_memory.py`

## Risks / Things to watch

- `self.keys`/`self.values` are views → external code that accesses them directly
  gets correct data (the valid portion), verified via `_sync_views()`
- `DynamicCache.__iter__` yields `layer.keys` → correct because views are synced
- Beam search (`reorder_cache`, `batch_repeat_interleave`): operates on full buffer
  but produces correct results; slight waste on unused capacity, negligible
- `offload`/`prefetch`: moves full buffer including unused portion; minor bandwidth
  waste, bounded by 2× factor
