#!/usr/bin/env python3
"""
Benchmark: tilelang paged_mqa_logits vs lightop paged_mqa_logits.

API (matches lightop gemmopt.paged_mqa_logits):
    logits = paged_mqa_logits(q, fused_kv_cache, weights, context_lens,
                              block_table, schedule_meta, max_context_len,
                              clean_logits=True)

Where:
    q:               [batch_size, next_n, heads, head_dim]  BF16
    fused_kv_cache:  [num_blocks, block_size, 1, head_dim]  BF16 (block_size=64)
    weights:         [batch_size * next_n, heads]            float32
    context_lens:    [batch_size]                            int32
    block_table:     [batch_size, max_num_blocks]            int32
    schedule_meta:   [num_sms+1, 2] or None                 int32 (ignored)
    max_context_len: int

Returns:
    logits:          [batch_size * next_n, max_context_len]  float32
"""

import argparse
import random
import sys
from typing import Optional

import tilelang
from tilelang import language as T
import torch

# ============================================================================
# Constants
# ============================================================================
LDS_LIMIT = 80 * 1024          # Maximum shared memory per block (bytes)
BLOCK_KV = 64                  # Number of tokens per physical KV block


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


# ============================================================================
# Kernel: paged_mqa_logits (BF16)
# ============================================================================

def _pick_tile_config(heads: int, head_dim: int):
    """
    Pick the best K-token tile size and pipeline stages under the LDS budget.

    Strategy (greedy, in priority order):
      1. More pipeline stages  → better latency hiding
      2. Smaller tile_k_token  → less LDS, more SM residency headroom

    Returns (tile_k_token, num_stages, lds_bytes_used).
    """
    dim = head_dim
    q_smem_bytes = heads * dim * 2  # Q sits in LDS full-time; not pipelined

    # Collect all (tile_k_token, stages) combinations that fit in LDS
    candidates = []
    for tile_n in (32, 64):
        for stages in (2, 1, 0):
            k_smem_bytes = (stages + 1) * tile_n * dim * 2
            lds_used = q_smem_bytes + k_smem_bytes
            if lds_used <= LDS_LIMIT:
                candidates.append((tile_n, stages, lds_used))

    if not candidates:
        # No config fits — minimum tile with no pipelining as fallback
        return 32, 0, q_smem_bytes + 32 * dim * 2

    # Sort: prefer more stages, then smaller tile_n
    candidates.sort(key=lambda x: (-x[1], x[0]))
    return candidates[0]


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def paged_mqa_logits_kernel(
    heads: int,
    head_dim: int,
    tile_k_token: int = 64,
    num_stages: int = 1,
    threads: int = 256,
    policy: str = "square",
):
    """
    TileLang BF16 paged MQA logits kernel.

    Grid: (max_num_blocks, batch_size) — one block per logical KV page (64 tokens).
    Each block handles all heads for 1 query position.

    Workflow:
      1. Load query [heads, head_dim] and weights [1, heads] into shared memory.
      2. Look up physical KV page id from block_table.
      3. For each tile of the 64 KV rows:
         - Load KV tile into shared memory (with pipelining).
         - GEMM: [tile_k_token, head_dim] × [heads, head_dim]^T → [tile_k_token, heads].
         - Apply ReLU and head weights, then reduce-sum over heads → [tile_k_token].
         - Write results to global logits.
    """
    dim = head_dim
    dtype = T.bfloat16
    accum_dtype = T.float32
    index_dtype = T.int32

    K_PACK = 1   # For RDNA (gfx936); CDNA could use 2
    gemm_policy = T.GemmWarpPolicy.FullRow if policy == "full_row" else T.GemmWarpPolicy.Square

    # Dynamic dimensions
    batch_size = T.dynamic("batch_size")
    num_blocks = T.dynamic("num_blocks")
    max_num_blocks = T.dynamic("max_num_blocks")
    max_context_len = T.dynamic("max_context_len")

    @T.prim_func
    def kernel(
        q: T.Tensor([batch_size, heads, dim], dtype),
        kv_cache: T.Tensor([num_blocks, BLOCK_KV, 1, dim], dtype),
        logits: T.Tensor([batch_size, max_context_len], accum_dtype),
        weights: T.Tensor([batch_size, heads], accum_dtype),
        block_table: T.Tensor([batch_size, max_num_blocks], index_dtype),
    ):
        # Each thread block handles one logical KV page of one batch sample
        with T.Kernel(max_num_blocks, batch_size, threads=threads) as (kv_logical_page_idx, batch_idx):
            # 1. Translate logical page index to physical page id via page table
            phys_kv_page_id = block_table[batch_idx, kv_logical_page_idx]
            # Global token offset for this logical page (start position in the context sequence)
            global_offset = kv_logical_page_idx * BLOCK_KV

            # Allocate shared memory for query, key tile, and fragments
            q_smem = T.alloc_shared([heads, dim], dtype)
            k_smem = T.alloc_shared([tile_k_token, dim], dtype)
            s = T.alloc_fragment([tile_k_token, heads], accum_dtype)      # GEMM output
            logits_tile = T.alloc_fragment([tile_k_token], accum_dtype)   # per-token logits
            w_frag = T.alloc_fragment([heads], accum_dtype)               # head weights

            # Load query and weights for this batch once
            T.copy(q[batch_idx, 0:heads, 0:dim], q_smem)
            T.copy(weights[batch_idx, 0:heads], w_frag)

            # Process the 64 keys in tiles of size tile_k_token
            num_tiles = T.ceildiv(BLOCK_KV, tile_k_token)
            for tile_idx in T.Pipelined(num_tiles, num_stages=num_stages):
                kv_row_start = tile_idx * tile_k_token
                # Load key tile from physical KV page into shared memory
                T.copy(
                    kv_cache[phys_kv_page_id, kv_row_start:kv_row_start + tile_k_token, 0, 0:dim],
                    k_smem
                )

                # GEMM: k_smem [tile_k_token, dim] × q_smem [heads, dim]^T → s [tile_k_token, heads]
                T.clear(s)
                T.gemm(
                    k_smem, q_smem, s,
                    k_pack=K_PACK,
                    transpose_B=True,
                    policy=gemm_policy,
                )

                # Apply ReLU and head weights, then reduce-sum over heads
                for row_in_tile, head_idx in T.Parallel(tile_k_token, heads):
                    s[row_in_tile, head_idx] = (
                        T.max(s[row_in_tile, head_idx], T.cast(0, accum_dtype))
                        * w_frag[head_idx]
                    )

                T.reduce_sum(s, logits_tile, dim=1, clear=True)

                # Write results to global logits (only if within context length bounds)
                for row_in_tile in T.Parallel(tile_k_token):
                    global_pos = global_offset + kv_row_start + row_in_tile
                    if global_pos < max_context_len:
                        logits[batch_idx, global_pos] = logits_tile[row_in_tile]

    return kernel


@tilelang.jit
def clean_paged_logits(threads: int = 256, block_K: int = 4096):
    """
    Mask positions >= context_len with -inf for each batch row.
    """
    batch_size = T.dynamic("batch_size")
    max_context_len = T.dynamic("max_context_len")
    dtype = T.float
    indices_dtype = T.int32

    @T.prim_func
    def kernel(
        logits: T.Tensor([batch_size, max_context_len], dtype),
        context_lens: T.Tensor([batch_size], indices_dtype),
    ):
        with T.Kernel(batch_size, threads=threads) as batch_idx:
            tid = T.thread_binding(0, threads, thread="threadIdx.x")
            ctx_len = context_lens[batch_idx]

            # Loop over chunks of size block_K
            for chunk in T.Pipelined(T.ceildiv(max_context_len, block_K)):
                # Each thread handles several positions within the chunk
                for i in T.serial(block_K // threads):
                    pos = chunk * block_K + i * threads + tid
                    if pos >= ctx_len:
                        logits[batch_idx, pos] = -T.infinity(dtype)

    return kernel


# ============================================================================
# PyTorch Reference Implementation
# ============================================================================

def ref_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_model_len: int,
):
    """
    Pure PyTorch implementation for correctness validation.
    Follows the same algorithm as lightop's ref_fp8_paged_mqa_logits.
    """
    batch_size, next_n, heads, head_dim = q.shape
    block_size = kv_cache.shape[1]

    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float('-inf'),
        device=q.device,
        dtype=torch.float32,
    )
    ctx_list = context_lens.tolist()

    for b in range(batch_size):
        ctx_len = ctx_list[b]
        q_offsets = torch.arange(ctx_len - next_n, ctx_len, device=q.device)
        weight_slice = weights[b * next_n:(b + 1) * next_n, :].transpose(0, 1).contiguous()

        for block_rk in range(ceil_div(ctx_len, block_size)):
            phys_block = block_table[b, block_rk].item()
            qx = q[b].to(torch.float32)                     # [next_n, heads, head_dim]
            kx = kv_cache[phys_block].to(torch.float32).squeeze(1)  # [block_size, head_dim]

            k_offsets = torch.arange(block_rk * block_size,
                                     (block_rk + 1) * block_size,
                                     device=q.device)

            # causal mask: only positions <= query offset are valid
            mask = (k_offsets[None, :] < ctx_len) & (k_offsets[None, :] <= q_offsets[:, None])

            # Compute scores: [next_n, heads, block_size]
            scores = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.T).to(logits.dtype),
                float('-inf'),
            )
            scores = torch.relu(scores) * weight_slice[..., None]
            scores = scores.sum(dim=0)   # sum over heads → [next_n, block_size]

            # Write to logits, again masking invalid positions
            logits[b * next_n:(b + 1) * next_n,
                   block_rk * block_size:(block_rk + 1) * block_size] = \
                torch.where(k_offsets[None, :] <= q_offsets[:, None],
                            scores,
                            float('-inf'))

    return logits


# ============================================================================
# TileLang High-Level Interface
# ============================================================================

_kernel_cache = {}


def run_tilelang(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_context_len: int,
    clean_logits: bool = True,
):
    """
    Run the tilelang paged_mqa_logits kernel (BF16). Requires next_n == 1.
    """
    batch_size, next_n, heads, head_dim = q.shape
    assert next_n == 1, "Only next_n=1 is supported."

    # Pick optimal tiling configuration
    tile_k_token, num_stages, lds_bytes = _pick_tile_config(heads, head_dim)

    # For very small batches, we can use a larger tile and FullRow policy
    use_full_row = False
    if batch_size <= 4 and tile_k_token == 32:
        q_smem = heads * head_dim * 2
        k_smem_64 = 64 * head_dim * 2
        if q_smem + k_smem_64 <= LDS_LIMIT:
            tile_k_token = 64
            num_stages = 0
            lds_bytes = q_smem + k_smem_64
            use_full_row = True

    # Cache and compile the kernel
    cache_key = (heads, head_dim, tile_k_token, num_stages, use_full_row)
    if cache_key not in _kernel_cache:
        policy_str = "full_row" if use_full_row else "square"
        print(f"  [tilelang config] H={heads}, dim={head_dim}, "
              f"tile_k_token={tile_k_token}, num_stages={num_stages}, policy={policy_str}, "
              f"LDS={lds_bytes / 1024:.1f}KB / {LDS_LIMIT / 1024:.0f}KB")
        _kernel_cache[cache_key] = paged_mqa_logits_kernel(
            heads=heads,
            head_dim=head_dim,
            tile_k_token=tile_k_token,
            num_stages=num_stages,
            policy=policy_str,
        )

    logits_kernel = _kernel_cache[cache_key]

    # Allocate output logits (pre-filled with -inf if cleaning)
    if clean_logits:
        logits = torch.full(
            [batch_size, max_context_len],
            float("-inf"),
            device=q.device,
            dtype=torch.float32,
        )
    else:
        logits = torch.empty(
            [batch_size, max_context_len],
            device=q.device,
            dtype=torch.float32,
        )

    # Launch kernel
    logits_kernel(
        q.view(batch_size, heads, head_dim),  # flatten next_n
        kv_cache,
        logits,
        weights.view(batch_size, heads),
        block_table,
    )

    # Optionally mask positions beyond context length
    if clean_logits:
        clean_kernel = clean_paged_logits()
        clean_kernel(logits, context_lens)

    return logits.view(batch_size * next_n, max_context_len)


# ============================================================================
# Public API (vLLM-compatible)
# ============================================================================

def paged_mqa_logits(
    q: torch.Tensor,
    fused_kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_metadata: Optional[torch.Tensor] = None,
    max_context_len: int = 0,
    clean_logits: bool = True,
) -> torch.Tensor:
    """
    Public interface matching lightop.gemmopt.paged_mqa_logits.
    """
    if max_context_len <= 0:
        max_context_len = fused_kv_cache.shape[1] * block_table.shape[1]

    return run_tilelang(
        q,
        fused_kv_cache,
        weights,
        context_lens,
        block_table,
        max_context_len,
        clean_logits=clean_logits,
    )


# ============================================================================
# Lightop Integration (optional)
# ============================================================================

try:
    from lightop import gemmopt
    HAS_LIGHTOP = True
except ImportError:
    HAS_LIGHTOP = False


def run_lightop(q, kv_cache, weights, context_lens, block_table,
                max_context_len, clean_logits=True):
    """Run lightop reference implementation."""
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    schedule_meta = gemmopt.get_paged_mqa_logits_metadata(
        context_lens, block_kv=64, num_sms=num_sms,
    )
    return gemmopt.paged_mqa_logits(
        q, kv_cache, weights, context_lens, block_table,
        schedule_meta, max_context_len, clean_logits=clean_logits,
    )


# ============================================================================
# Correctness & Benchmark Helpers
# ============================================================================

def compute_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.data.double(), b.data.double()
    norm_sum = (a * a + b * b).sum()
    if norm_sum == 0:
        return 1.0
    return (2 * (a * b).sum() / norm_sum).item()


def calc_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return 1.0 - compute_correlation(a, b)


def bench_event(fn, warmup=10, rep=50):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return sum(times) / len(times)


# ============================================================================
# Test Data Generation
# ============================================================================

def create_test_data(
    batch_size: int,
    next_n: int,
    heads: int,
    head_dim: int,
    avg_ctx_len: int,
    max_model_len: int,
    device: str = "cuda",
):
    torch.manual_seed(42)

    block_size = BLOCK_KV
    total_blocks_needed = batch_size * ceil_div(int(avg_ctx_len * 1.1), block_size)
    num_blocks = total_blocks_needed * 3

    q = torch.randn(batch_size, next_n, heads, head_dim, device=device, dtype=torch.bfloat16)
    kv_cache = torch.randn(num_blocks, block_size, 1, head_dim, device=device, dtype=torch.bfloat16)
    weights = torch.randn(batch_size * next_n, heads, device=device, dtype=torch.float32)

    context_lens = torch.randint(
        int(0.9 * avg_ctx_len), int(1.1 * avg_ctx_len), (batch_size,),
        device=device,
    ).to(torch.int32)

    max_num_blocks = (context_lens.max().item() + block_size - 1) // block_size
    block_table = torch.zeros(batch_size, max_num_blocks, device=device, dtype=torch.int32)

    # Randomly assign physical blocks
    block_pool = list(range(num_blocks))
    random.shuffle(block_pool)
    counter = 0
    for i in range(batch_size):
        ctx_len = context_lens[i].item()
        for j in range(ceil_div(ctx_len, block_size)):
            block_table[i, j] = block_pool[counter]
            counter += 1

    return q, kv_cache, weights, context_lens, block_table


# ============================================================================
# Correctness Validation
# ============================================================================

CORRECTNESS_CASES = [
    ("bs1_H32_D128",         1, 1, 32, 128, 256),
    ("bs1_H64_D128",         1, 1, 64, 128, 256),
    ("bs2_H32_D128",         2, 1, 32, 128, 512),
    ("bs4_H32_D128",         4, 1, 32, 128, 1024),
    ("bs1_H32_D128_long",    1, 1, 32, 128, 4096),
    ("bs1_H32_D128_tiny",    1, 1, 32, 128, 32),
    ("bs1_H32_D128_small",   1, 1, 32, 128, 64),
    ("unalign_99",           1, 1, 32, 128, 99),
    ("unalign_257",          1, 1, 32, 128, 257),
    ("unalign_500_H64",      1, 1, 64, 128, 500),
    ("bs8_H32_D128",         8, 1, 32, 128, 2048),
    ("bs32_H32_D128",       32, 1, 32, 128, 1024),
]


def run_correctness_check(device="cuda"):
    all_pass = True
    failed = []

    print("=" * 90)
    print("Correctness Validation: tilelang vs lightop")
    print("=" * 90)

    header = f"{'Case':<22} {'B':>4} {'H':>4} {'D':>4} {'avg_ctx':>8}  {'tl_vs_lo':>12}  {'mask':>8}  {'status':>6}"
    print(header)
    print("-" * 90)

    for name, batch_size, next_n, heads, head_dim, avg_ctx in CORRECTNESS_CASES:
        max_model_len = avg_ctx * 2
        q, kv_cache, weights, context_lens, block_table = create_test_data(
            batch_size, next_n, heads, head_dim, avg_ctx, max_model_len, device,
        )

        logits_tl = run_tilelang(
            q, kv_cache, weights, context_lens, block_table, max_model_len,
        )

        if not HAS_LIGHTOP:
            print("  lightop not available, skipping")
            continue

        logits_lo = run_lightop(
            q, kv_cache, weights, context_lens, block_table, max_model_len, clean_logits=True,
        )

        tl_neginf = (logits_tl == float('-inf'))
        lo_neginf = (logits_lo == float('-inf'))
        mask_match = torch.equal(tl_neginf, lo_neginf)

        both_neginf = tl_neginf & lo_neginf
        diff = calc_diff(
            logits_tl.masked_fill(both_neginf, 0),
            logits_lo.masked_fill(both_neginf, 0),
        )

        status = "PASS" if (diff < 1e-3 and mask_match) else "FAIL"
        max_abs_err = (logits_tl.float() - logits_lo.float()).abs().max().item()
        print(f"{name:<22} {batch_size:>4} {heads:>4} {head_dim:>4} {avg_ctx:>8}  "
              f"{diff:>12.2e}  {str(mask_match):>8}  [{status}]  max_err={max_abs_err:.4e}")

        if status == "FAIL":
            # Debug info for first failure
            print(f"    DEBUG tl[0,:8]={logits_tl[0,:8].tolist()}")
            print(f"    DEBUG lo[0,:8]={logits_lo[0,:8].tolist()}")
            # Also check reference
            logits_ref = ref_paged_mqa_logits(
                q, kv_cache, weights, context_lens, block_table, max_model_len,
            )
            print(f"    DEBUG ref[0,:8]={logits_ref[0,:8].tolist()}")
            break

        if status == "FAIL":
            all_pass = False
            failed.append(name)

    print("-" * 90)
    if all_pass:
        print("ALL PASSED")
    else:
        print(f"FAILURES: {', '.join(failed)}")
    return all_pass


# ============================================================================
# Benchmark
# ============================================================================

BENCHMARK_CASES = [
    ("bs1_H32_D128_4k",      1, 1, 32, 128, 4096),
    ("bs64_H32_D128_4k",    64, 1, 32, 128, 4096),
    ("bs128_H32_D128_4k",  128, 1, 32, 128, 4096),
    ("bs1_H64_D128_4k",      1, 1, 64, 128, 4096),
    ("bs64_H64_D128_4k",    64, 1, 64, 128, 4096),
    ("bs1_H32_D128_72k",     1, 1, 32, 128, 72000),
    ("bs1_H64_D128_72k",     1, 1, 64, 128, 72000),
]


def run_benchmark(name, batch_size, next_n, heads, head_dim, avg_ctx, warmup, rep, device):
    print(f"\n{'=' * 70}")
    print(f"Config: {name}")
    print(f"  B={batch_size}, next_n={next_n}, H={heads}, D={head_dim}, avg_ctx~{avg_ctx}")

    max_model_len = avg_ctx * 2
    q, kv_cache, weights, context_lens, block_table = create_test_data(
        batch_size, next_n, heads, head_dim, avg_ctx, max_model_len, device,
    )

    sum_lens = context_lens.to(torch.int64).sum().item()
    tflops = 2 * sum_lens * next_n * heads * head_dim / 1e12

    # TileLang
    tl_fn = lambda: run_tilelang(
        q, kv_cache, weights, context_lens, block_table, max_model_len, clean_logits=False,
    )
    tl_ms = bench_event(tl_fn, warmup=warmup, rep=rep)
    print(f"  tilelang:  {tl_ms:8.3f} ms  ({tflops / (tl_ms * 1e-3):.2f} TFLOPS)")

    # Lightop
    lo_ms = None
    if HAS_LIGHTOP:
        lo_fn = lambda: run_lightop(
            q, kv_cache, weights, context_lens, block_table, max_model_len, clean_logits=False,
        )
        lo_ms = bench_event(lo_fn, warmup=warmup, rep=rep)
        speedup = tl_ms / lo_ms
        print(f"  lightop:   {lo_ms:8.3f} ms  ({tflops / (lo_ms * 1e-3):.2f} TFLOPS)")
        print(f"  speedup:   {speedup:8.2f}x  (lightop vs tilelang)")

    return {
        "name": name, "B": batch_size, "H": heads, "D": head_dim, "avg_ctx": avg_ctx,
        "tl_ms": tl_ms, "lo_ms": lo_ms,
        "tl_tflops": tflops / (tl_ms * 1e-3),
        "lo_tflops": tflops / (lo_ms * 1e-3) if lo_ms else None,
        "speedup": tl_ms / lo_ms if lo_ms else None,
        "sum_lens": sum_lens,
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark tilelang vs lightop HIP ASM paged_mqa_logits (BF16)"
    )
    parser.add_argument("--validate", action="store_true", help="Run correctness validation")
    parser.add_argument("--benchmark", action="store_true", default=False, help="Run performance benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=50, help="Measurement iterations")
    parser.add_argument("--csv", type=str, default=None, help="Save results to CSV")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA device not available.")
        sys.exit(1)

    device = "cuda"
    prop = torch.cuda.get_device_properties(0)
    gcn = getattr(prop, 'gcnArchName', prop.name)
    print(f"Device: {prop.name} ({gcn})")
    print(f"tilelang:   available")
    print(f"lightop:    {'available' if HAS_LIGHTOP else 'NOT available'}")

    if args.validate:
        run_correctness_check(device)
        if not args.benchmark:
            return

    results = []
    for name, batch_size, next_n, heads, head_dim, avg_ctx in BENCHMARK_CASES:
        r = run_benchmark(
            name, batch_size, next_n, heads, head_dim, avg_ctx,
            args.warmup, args.rep, device,
        )
        results.append(r)

    print("\n" + "=" * 95)
    print("Summary")
    print("=" * 95)
    header = f"{'Case':<28} {'B':>5} {'H':>4} {'D':>4} {'avg_ctx':>8}  {'tilelang(ms)':>12}  {'lightop(ms)':>12}  {'speedup':>8}"
    print(header)
    print("-" * 95)
    for r in results:
        lo_str = f"{r['lo_ms']:.3f}" if r['lo_ms'] else "N/A"
        sp_str = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"
        print(f"{r['name']:<28} {r['B']:>5} {r['H']:>4} {r['D']:>4} {r['avg_ctx']:>8}  "
              f"{r['tl_ms']:>12.3f}  {lo_str:>12}  {sp_str:>8}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Case", "B", "H", "D", "avg_ctx",
                             "tilelang_ms", "lightop_ms", "speedup",
                             "tl_tflops", "lo_tflops", "sum_lens"])
            for r in results:
                writer.writerow([
                    r['name'], r['B'], r['H'], r['D'], r['avg_ctx'],
                    f"{r['tl_ms']:.6f}",
                    f"{r['lo_ms']:.6f}" if r['lo_ms'] else "",
                    f"{r['speedup']:.4f}" if r['speedup'] else "",
                    f"{r['tl_tflops']:.4f}",
                    f"{r['lo_tflops']:.4f}" if r['lo_tflops'] else "",
                    r['sum_lens'],
                ])
        print(f"\nResults saved to: {args.csv}")


if __name__ == "__main__":
    main()