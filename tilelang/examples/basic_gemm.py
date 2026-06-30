import torch
import tilelang.language as T
from tilelang import jit


@jit(out_idx=[-1])
def gemm(M: int, N: int, K: int, BM: int = 128, BN: int = 128, BK: int = 32,
         dtype: str = "float16", accum_dtype: str = "float32"):
    """基础分块 GEMM：每个 Block 处理一个 tile。"""

    @T.prim_func
    def gemm_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            A_s = T.alloc_shared((BM, BK), dtype)
            B_s = T.alloc_shared((BK, BN), dtype)
            C_f = T.alloc_fragment((BM, BN), accum_dtype)
            T.clear(C_f)

            for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
                T.copy(A[by * BM, ko * BK], A_s)
                T.copy(B[ko * BK, bx * BN], B_s)
                T.gemm(A_s, B_s, C_f)

            T.copy(C_f, C[by * BM, bx * BN])

    return gemm_kernel


if __name__ == "__main__":
    M, N, K = 1024, 1024, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    kernel = gemm(M, N, K)
    c_tilelang = kernel(a, b)

    c_torch = a @ b

    torch.testing.assert_close(c_tilelang, c_torch, atol=1e-3, rtol=1e-3)
    max_diff = (c_tilelang.float() - c_torch.float()).abs().max().item()
    print(f"M={M}, N={N}, K={K}, max_diff={max_diff:.6f}, passed.")
