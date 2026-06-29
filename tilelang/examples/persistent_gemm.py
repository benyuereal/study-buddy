"""
Persistent GEMM implementation (V3 only).
"""

import torch
import tilelang as tl
import tilelang.language as T
from tilelang.intrinsics import get_swizzle_layout
from tilelang.layout.swizzle import make_linear_layout, make_hcu_swizzled_layout
from tvm import DataType


def get_persistent_configs(M, N, K):
    """
    Generate a list of kernel tuning configuration dictionaries for persistent GEMM.
    """
    param_dict = {
        "block_M": [64, 128, 256],
        "block_N": [64, 128, 256],
        "block_K": [32, 64],
        "num_stages": [0, 2, 3],
        "thread_num": [128, 256],
        "wgs_per_cu": [1, 2, 4],
        "group_size": [1, 2, 4],
    }
    from perf.gemm.utils import _generate_configs_from_product
    return _generate_configs_from_product(param_dict)


def get_best_persistent_config(M, N, K):
    """
    Autotune persistent GEMM kernel and return the best configuration.
    """
    from perf.gemm.utils import _run_autotuner

    def kernel(
        block_M=None,
        block_N=None,
        block_K=None,
        num_stages=None,
        thread_num=None,
        wgs_per_cu=None,
        group_size=None,
    ):
        dtype = "float16"
        accum_dtype = "float"

        grid_size = wgs_per_cu * 80

        @T.prim_func
        def main(
                A: T.Tensor((M, K), dtype),
                B: T.Tensor((N, K), dtype),
                C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(grid_size, threads=thread_num) as (block_id):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

                m_blocks = T.ceildiv(M, block_M)
                n_blocks = T.ceildiv(N, block_N)
                waves = T.ceildiv(m_blocks * n_blocks, grid_size)

                for w in T.serial(waves):
                    tile_id = grid_size * w + block_id
                    bx = (tile_id // group_size) % m_blocks
                    by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size

                    if bx * block_M < M and by * block_N < N:
                        T.clear(C_local)
                        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                            T.copy(A[bx * block_M, k * block_K], A_shared, coalesced_width=8)
                            T.copy(B[by * block_N, k * block_K], B_shared, coalesced_width=8)
                            T.gemm(A_shared, B_shared, C_local, k_pack=2, transpose_B=True)

                        T.copy(C_local, C[bx * block_M, by * block_N])

        return main

    return _run_autotuner(
        kernel,
        get_persistent_configs(M, N, K),
        pass_configs={"tl.enable_aggressive_shared_memory_merge": True},
    )


# Impl:
#   preload A/B to register swizzled before T.gemm
#   use T.persistent instead of swizzle manually
@tl.jit(out_idx=[-1], pass_configs={
     tl.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    },
)
def gemm_persistent_v3(M, N, K, block_M, block_N, block_K,
                    num_stages, thread_num, group_size=8, wgs_per_cu=2,
                    dtype="float16", accum_dtype="float"):
    cu_num = torch.cuda.get_device_properties("cuda").multi_processor_count
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    grid_size = T.min(m_blocks * n_blocks, wgs_per_cu * cu_num)

    split_n = 2
    sub_block_N = block_N // split_n

    @T.prim_func
    def _gemm_persistent(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_size, threads=thread_num) as (block_id):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared_0 = T.alloc_shared((sub_block_N, block_K), dtype)

            A_local_0 = T.alloc_fragment((block_M, block_K), dtype)
            A_local_0_ = T.alloc_fragment((block_M, block_K), dtype)

            B_local_0 = T.alloc_fragment((sub_block_N, block_K), dtype)
            B_local_1 = T.alloc_fragment((sub_block_N, block_K), dtype)

            B_local_0_ = T.alloc_fragment((sub_block_N, block_K), dtype)
            B_local_1_ = T.alloc_fragment((sub_block_N, block_K), dtype)

            C_local_0 = T.alloc_fragment((block_M, sub_block_N), accum_dtype)
            C_local_1 = T.alloc_fragment((block_M, sub_block_N), accum_dtype)

            C_shared_0 = T.alloc_shared((block_M, sub_block_N), dtype)
            T.annotate_layout({
                C_shared_0: tl.layout.make_hcu_swizzled_layout(C_shared_0, major_pack=2),
                B_shared_0: tl.layout.make_hcu_swizzled_layout(B_shared_0, major_pack=2),
                A_shared: tl.layout.make_hcu_swizzled_layout(A_shared, major_pack=2),
            })

            # bx: N, by: M
            for bx, by in T.Persistent(
                [T.ceildiv(N, block_N), T.ceildiv(M, block_M)],
                wgs_per_cu * cu_num,
                block_id,
                group_size=1
            ):
                if by * block_M < M and bx * block_N < N:
                    T.clear(C_local_0)
                    T.clear(C_local_1)
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                        T.copy(A[by * block_M, k * block_K], A_local_0, coalesced_width=8)
                        # A Block swizzle
                        T.copy(A_local_0, A_shared)

                        # preload B Block N_0
                        T.copy(B[bx * block_N, k * block_K], B_local_0, coalesced_width=8)
                        # preload B Block N_1
                        T.copy(B[bx * block_N + sub_block_N, k * block_K], B_local_1, coalesced_width=8)

                        # B Block N_0 swizzle
                        T.copy(B_local_0, B_shared_0)
                        T.copy(B_shared_0, B_local_0_)

                        # B Block N_1 swizzle
                        T.copy(B_local_1, B_shared_0)
                        T.copy(B_shared_0, B_local_1_)

                        # A local
                        T.copy(A_shared, A_local_0_)

                        T.gemm(A_local_0_, B_local_0_, C_local_0, k_pack=2, transpose_B=True)
                        T.gemm(A_local_0_, B_local_1_, C_local_1, k_pack=2, transpose_B=True)

                    T.copy(C_local_0, C_shared_0)
                    T.copy(C_shared_0, C[by * block_M, bx * block_N])
                    T.copy(C_local_1, C_shared_0)
                    T.copy(C_shared_0, C[by * block_M, bx * block_N + sub_block_N])

    return _gemm_persistent


# Note: Use pass_configs={"tl.disable_safe_memory_legalize": True} to disable safe memory legalize
#       during using vectorized with swizzled layout.
@tl.jit(out_idx=[-1])
def gemm_persistent(M, N, K, block_M, block_N, block_K,
                    num_stages, thread_num, group_size=8, wgs_per_cu=2,
                    dtype="float16", accum_dtype="float"):
    cu_num = torch.cuda.get_device_properties("cuda").multi_processor_count
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    grid_size = T.min(m_blocks * n_blocks, wgs_per_cu * cu_num)
    waves = T.ceildiv(m_blocks * n_blocks, grid_size)

    @T.prim_func
    def _gemm_persistent(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_size, threads=thread_num) as (block_id):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            for w in T.serial(waves):
                tile_id = grid_size * w + block_id
                bx = (tile_id // group_size) % m_blocks
                by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size

                if bx * block_M < M and by * block_N < N:
                    T.clear(C_local)
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                        T.copy(A[bx * block_M, k * block_K], A_shared, coalesced_width=8)
                        T.copy(B[by * block_N, k * block_K], B_shared, coalesced_width=8)
                        T.gemm(A_shared, B_shared, C_local, k_pack=2, transpose_B=True)

                    T.copy(C_local, C[bx * block_M, by * block_N])

    return _gemm_persistent
