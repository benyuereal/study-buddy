import torch
import tilelang.language as T
from tilelang import jit


@jit
def add(N: int, block: int = 256, dtype: str = "float32"):
    """GPU 向量加法，演示 TileLang 最基础的编程模式。"""

    @T.prim_func
    def add_kernel(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                gi = bx * block + i
                if gi < N:
                    C[gi] = A[gi] + B[gi]

    return add_kernel


if __name__ == "__main__":
    N = 1 << 20  # 1M elements
    a = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)
    c_tilelang = torch.empty(N, device="cuda", dtype=torch.float32)

    kernel = add(N)
    kernel(a, b, c_tilelang)

    c_torch = a + b

    torch.testing.assert_close(c_tilelang, c_torch)
    print(f"N={N}, max_diff={(c_tilelang - c_torch).abs().max().item():.6f}, passed.")
