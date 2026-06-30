# TileLang 核心关键字与概念

## 文档定位

本文档是 TileLang 关键字的**速查手册**，逐个解释每个原语的语法、语义、与 GPU 硬件的对应关系，以及使用场景。面向具备 GPU 基础知识的开发者（Expert 层级），提供线程级精确控制。

阅读前请先理解 [GPU 基础概念](./gpu-basic-knowledge.md)（Grid、Block、Wave、Shared Memory/LDS、寄存器、Tiling 等）。本文档以海光 DCU 为主要目标硬件，LDS 默认 64 KB/CU，Wave 为 64 线程。

---

## 前言：TileLang 编程模型

### 什么是 TileLang

TileLang 是一个**面向 GPU/CPU 高性能计算的领域特定语言（DSL）**，以 Pythonic 语法提供显式的硬件内存管理、线程级并行控制和软件流水线能力。它构建于 TVM 之上，通过多层 IR 逐步降级到硬件特定的可执行代码。支持 NVIDIA（CUDA）、AMD（ROCm/HIP）、海光 DCU（HCU）、华为昇腾（AscendC）、Apple Metal 等主流 GPU 后端。

TileLang 的核心思想是将 **Tile（分块）** 作为一等公民，开发者显式指定：

- **数据在哪里**：`T.alloc_shared`（LDS）、`T.alloc_fragment`（寄存器）、`T.alloc_local`（线程私有）
- **数据怎么搬**：`T.copy`（同步）、`T.async_copy`（异步）
- **怎么算**：`T.gemm`（矩阵乘）、逐元素运算、归约

这三类原语贯穿本文档始终，后续章节按"定义 Kernel → 分配内存 → 搬运数据 → 计算 → 控制流 → 布局优化 → 同步 → 调试 → 编译配置"的顺序逐一展开。

### 编程模型图示：多层分块 GEMM 全景

下图完整展示了 TileLang 的分层分块 GEMM 编程模型，打通 **GPU 硬件分层存储** 与 **TileLang DSL 语法** 的映射关系：

![TileLang 多层分块 GEMM 编程模型](assets/tilelang-gemm-model.png)

**左半部分 (a) — 硬件原理**：

- **存储金字塔**（绿色→红色→青蓝）：寄存器文件（最快、最小）→ Shared Memory/LDS（片上高速缓存）→ Global Memory（容量最大、延迟最高）
- **蓝色弧形箭头**：数据流转路径 Global → Shared → Register，体现分层缓存分块
- **两层网格**：下层 Global Memory 存储完整大矩阵（红色小块 = 当前 Block 待处理的 tile）；上层 Shared Memory 存放搬运来的 tile 副本，再拆分为更小块送入寄存器执行计算。核心语义：`C_tile += A_tile × B_tile`

**右半部分 (b) — TileLang 代码映射**（从上到下，每段代码对应一层硬件操作）：

| 代码块 | 对应硬件层级 | 关键原语 |
|--------|------------|---------|
| Kernel 启动配置 | Grid/Block 线程组织 | `T.Kernel(grid_x, grid_y, threads=128)` |
| 内存缓冲区分配 | Shared Memory + 寄存器 | `T.alloc_shared`（红色图例）、`T.alloc_fragment`（绿色图例） |
| 流水线主循环 | 加载与计算重叠 | `T.Pipelined(..., num_stages=3)` |
| Global → Shared 数据搬运 | 图左蓝色弧形箭头 | `T.copy(A[..], A_shared)` |
| GEMM 计算 | 寄存器级乘加 | `T.gemm(A_shared, B_shared, C_local)` |
| 结果写回 Global Memory | 寄存器 → 全局显存 | `T.copy(C_local, C[...])` |

这张图的核心价值：**一行 TileLang 原语对应一层硬件操作**，显式控制从全局显存到寄存器的完整数据流，是理解本文档所有原语设计动机的起点。

---

## 一、基础骨架

Decorator 定义函数，Kernel 配置启动参数，工具函数编译时求值。

| 原语 | 作用 |
|------|------|
| `@T.prim_func` | 将 Python 函数标记为 IR 函数，参数用 `T.Tensor(shape, dtype)` 标注。返回 IR 对象，不直接执行。 |
| `@tl.jit` | JIT 编译包装器。`out_idx` 指定输出参数索引（如 `[-1]`），`pass_configs` 传递编译选项。 |
| `T.Kernel` | 定义 Grid/Block 组织。`threads` 支持 `int` 或 `tuple`；DCU 上应为 64 的倍数，最大 1024。 |
| `T.dynamic` | 声明符号变量，如 `T.dynamic("M N K")`。 |
| `T.ceildiv` | 向上取整除，也写作 `T.cdiv`。 |
| `T.min` | 编译时取两表达式的最小值，生成 Min 节点而非运行时分支。 |

下面是一个完整的向量加法示例，把三者串起来：

```python
from tilelang import jit
import tilelang.language as T

@jit
def add(N: int, block: int = 256, dtype: str = "float32"):
    @T.prim_func                                         # ❶ 函数装饰器
    def add_kernel(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:  # ❷ Kernel 启动配置
            for i in T.Parallel(block):
                gi = bx * block + i                       # ❸ 编译时工具函数
                if gi < N:                                #   T.ceildiv 计算 grid size
                    C[gi] = A[gi] + B[gi]
    return add_kernel
```

运行输出示例：

```
[TileLang] completes to compile kernel `add_kernel`
N=1048576, max_diff=0.000000, passed.
```

完整代码见 [examples/vector_add.py](examples/vector_add.py)。

---

## 二、内存分配

### `T.alloc_shared` — 分配 Shared Memory 缓冲区

```python
T.alloc_shared(shape: tuple[int, int], dtype: str, scope: str = "shared.dyn") → Buffer
```

在 GPU 的 LDS（Local Data Share）上分配二维缓冲区。LDS 是片上 SRAM，当前 Block 独占，Block 内所有线程共享，其他 Block 不可访问。生命周期为 Block 执行期间，执行完毕自动释放。

```python
A_shared = T.alloc_shared((BM, BK), "float16")
B_shared = T.alloc_shared((BK, BN), "float16")
```

**参数说明**：

- `shape`：缓冲区形状，通常为 `(block_M, block_K)` 或 `(block_K, block_N)`
- `dtype`：数据类型，与输入矩阵一致（如 `"float16"`）
- `scope`：内存作用域，默认 `"shared.dyn"`，一般不需要改

**容量约束**：单 Buffer 的 LDS 占用 = `shape[0] × shape[1] × sizeof(dtype)` 字节。配合 `T.Pipelined` 的 `num_stages` 做多缓冲时，总占用为：

```
LDS 总占用 = num_stages × (BM×BK + BK×BN) × sizeof(dtype)
```

以基础 GEMM 为例（`BM=128, BN=128, BK=32, num_stages=3, dtype=float16`）：

```
LDS 占用 = 3 × (128×32 + 32×128) × 2 bytes = 48 KB
```

DCU 单 CU 的 LDS 上限为 64 KB，48 KB 已占用 75%。**调大 BM/BN/BK 或 num_stages 前，先用这个公式估算 LDS 是否超限。**

### `T.alloc_fragment` — 分配寄存器级缓冲区

```python
T.alloc_fragment(shape: tuple[int, int], dtype: str, scope: str = "local.fragment") → Buffer
```

在寄存器文件上分配缓冲区，用于存储 GEMM 累加器或输入数据。寄存器是 GPU 存储层级中最快的一级（~1 cycle 延迟）。

`alloc_fragment` 分配的不是单个线程的寄存器，而是**整个 Thread Block 的寄存器文件**中的一块 — TileLang 的 Layout Inference 会自动推导每个线程持有 fragment 的哪个子矩阵。

```python
C_f = T.alloc_fragment((BM, BN), "float32")
```

**参数说明**：

- `shape`：累加器形状，通常为 `(BM, BN)`。也可用于存储输入数据（如 `A_local`），此时形状为 `(BM, BK)` 或 `(BK, BN)`
- `dtype`：累加器用 `"float32"`（即使输入是 float16），避免多次累加精度丢失
- `scope`：默认 `"local.fragment"`，一般不需要改

**编码要点**：

- `T.gemm` 做的是 `C += A × B`（累加），寄存器分配后内容是未定义的，使用前**必须** `T.clear`：

  ```python
  C_f = T.alloc_fragment((BM, BN), "float32")
  T.clear(C_f)   # 必须——不清零，结果 = 垃圾值 + 有效结果 = 不可预测
  ```

- 如果设置了 `T.gemm(..., clear_accum=True)`，可以省略手动的 `T.clear(C_f)`

### `T.alloc_local` — 分配线程私有标量

```python
T.alloc_local(shape: list[int], dtype: str, scope: str = "local") → Buffer
```

在线程私有内存中分配标量或小数组。典型用途是存储归约操作的中间标量（scale、max、LSE 等）。

```python
scale   = T.alloc_local([1], "float32")
lse_max = T.alloc_local([1], "float32")
```

**参数说明**：

- `shape`：通常为 `[1]`（标量），也可为 `[n]`（小数组）
- `dtype`：通常 `"float32"`
- `scope`：默认 `"local"`

**注意**：local memory 在 GPU 上可能溢出到 Global Memory（通过 L1/L2 cache 缓冲），频繁访问会影响性能。

### `T.clear` / `T.fill` — 缓冲区初始化

```python
T.clear(buf)           # 所有元素置零，等价于 T.fill(buf, 0)
T.fill(buf, value)     # 所有元素填充为指定值
```

在计算前将缓冲区初始化为已知状态。

```python
T.clear(C_f)                                     # GEMM 累加器清零
T.fill(buf, -T.infinity("float32"))              # max 归约初始化
T.fill(buf, T.infinity("float32"))               # min 归约初始化
```

| 场景 | 写法 | 原因 |
|------|------|------|
| GEMM 累加器 | `T.clear(C_f)` | `C += A×B`，不清零结果不可预测 |
| 求最大值 | `T.fill(buf, -T.infinity("float32"))` | 任何值 > -∞，首轮即被替换 |
| 求和归约 | `T.clear(buf)` | 累加从 0 开始 |
| 求最小值 | `T.fill(buf, T.infinity("float32"))` | 任何值 < +∞，首轮即被替换 |

---

## 三、数据搬运

### `T.copy` — 同步数据搬运

```python
T.copy(
    src: Buffer | BufferLoad | BufferRegion,
    dst: Buffer | BufferLoad | BufferRegion,
    *,
    coalesced_width: int | None = None,
    disable_tma: bool = False,
    eviction_policy: Literal["evict_normal", "evict_first", "evict_last"] | None = None,
    loop_layout: Any | None = None,
)
```

`T.copy` 是 TileLang 中**统一的数据搬运原语**，支持任意内存层级之间的传输。语义上是同步的：调用后 `dst` 中的数据保证可用。

编译器会根据源/目标类型和目标架构自动选择最优底层指令（SIMT copy、ldmatrix、cp.async、TMA 等），并自动做 coalesce 保证 Global Memory 合并访问。

| 搬运方向 | 代码 | 典型场景 |
|---------|------|---------|
| Global → Shared | `T.copy(A[by*BM, ko*BK], A_s)` | 分块 GEMM 主循环，每次加载一个 tile 到 LDS |
| Shared → Fragment | `T.copy(A_shared, A_local)` | swizzle 中转：Shared 读取到寄存器 |
| Fragment → Shared | `T.copy(A_local, A_shared)` | swizzle 中转：寄存器写入 Shared（做 swizzle 变换） |
| Global → Fragment | `T.copy(A[...], A_local, coalesced_width=8)` | 直接加载到寄存器，绕开 Shared Memory |
| Fragment → Global | `T.copy(C_f, C[by*BM, bx*BN])` | GEMM 结果写回全局内存 |

**`coalesced_width`**：指定 Global Memory 访问时的合并宽度（以元素为单位）。`coalesced_width=8` 表示编译器会尝试用 128-bit（8×16bit）向量化加载指令一次读取 8 个 fp16 元素。

其他参数（`disable_tma`、`eviction_policy`、`loop_layout`）为进阶用法，分别用于禁用 TMA 加速、控制 L2 cache 驱逐策略、提供并行布局提示。基础 GEMM 不需要用到。

### `T.async_copy` — 显式异步搬运（进阶）

```python
T.async_copy(
    src: Buffer | BufferLoad | BufferRegion,
    dst: Buffer | BufferLoad | BufferRegion,
    *,
    coalesced_width: int | None = None,
    loop_layout: Any | None = None,
)
```

发起异步 Global → Shared 拷贝（通过 `cp.async` 指令），**不会自动等待完成**。必须手动调用 `T.ptx_wait_group` 来同步。

与 `T.copy` 的关键区别：`T.copy` 自动插入 `commit` + `wait` 保证同步语义；`T.async_copy` 只发出 `cp.async` + `commit_group`，不插入 `wait`。

```python
T.async_copy(A[by * BM, ko * BK], A_s)   # 发起异步拷贝，不等待
# ⋯ 做其他与 A_s 无关的计算 ⋯
T.ptx_wait_group(0)                        # 等待所有异步拷贝完成
T.gemm(A_s, B_s, C_f)                      # 现在 A_s 中的数据保证可用
```

大多数场景下 `T.Pipelined` 的自动流水线已经足够，不需要直接用 `T.async_copy`。仅当需要精细控制异步预取时机（如 warp-specialized 代码）时才使用。

### `T.transpose` — Shared Memory 转置

```python
T.transpose(src: Buffer, dst: Buffer)
```

将 Shared Memory 中的 2D 缓冲区原地逻辑转置：`dst[j, i] = src[i, j]`。两个缓冲区必须都在 Shared Memory 且至少 2 维。

---

## 四、计算

### `T.gemm` — 矩阵乘法原语

```python
T.gemm(
    A: Buffer,
    B: Buffer,
    C: Buffer,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    k_pack: int = 1,
    mbar: BarrierType | None = None,
    use_tf32: bool = False,
)
```

在 Shared Memory 或 Fragment 上的两个输入矩阵做矩阵乘法，结果累加到 Fragment 累加器：`C += A @ B`。

- **A、B**：输入矩阵，可位于 Shared Memory（`T.alloc_shared`）或 Fragment（`T.alloc_fragment`）
- **C**：累加器，**必须**位于 Fragment（`T.alloc_fragment`），**必须**先 `T.clear`（见第二章）
- 编译器根据 GPU 架构自动降级为最优硬件指令（DCU M-FMA、NVIDIA Tensor Core 等）

```python
# 基础用法：Shared Memory 输入，结果累加到 Fragment
T.gemm(A_shared, B_shared, C_f)

# 进阶用法：Fragment 输入 + 参数控制
T.gemm(A_local, B_local, C_f, k_pack=2, transpose_B=True,
       policy=T.GemmWarpPolicy.FullCol)
```

**`transpose_A` / `transpose_B`**：当输入矩阵的存储格式与计算格式不一致时使用。例如 DCU 上 B 矩阵通常按 N×K（列主序）存储，设置 `transpose_B=True` 告诉编译器 B 已经在逻辑上被转置了，避免手动转置。

**`k_pack`**：K 维度打包因子。`k_pack=2` 表示每次迭代处理 K 维度的 2 个元素，利用向量化指令一次完成多个乘加。值需与数据类型和目标架构匹配。

**`policy: GemmWarpPolicy`**：控制 `T.gemm` 内部 Warp 在 M、N 维度上的分区方式，直接影响每个 warp 处理的 tile 形状和寄存器压力：

| 策略 | 含义 | 适用场景 |
|------|------|---------|
| `Square`（默认） | M 和 N 之间平衡分配 warp | 通用场景，方阵或接近方阵 |
| `FullRow` | 所有 warp 沿 M 方向排列 | M 远大于 N |
| `FullCol` | 所有 warp 沿 N 方向排列 | N 远大于 M（如 attention 中 N 较小） |
| `FullColK` | 所有 warp 沿 N 和 K 方向排列 | 需更细粒度的 K 并行 |

**`clear_accum`**：如果设为 `True`，编译器会在 GEMM 前自动清零累加器，可以省略手动的 `T.clear(C_f)`。

**`use_tf32`**：在 HCU 目标上启用 TF32 矩阵指令，牺牲少量精度换取更高吞吐。

### 逐元素数学运算

以下原语可在 `prim_func` 内直接使用，对应 GPU 的逐元素运算指令。大部分映射到标准 TIR 算子，由编译器根据目标架构生成对应硬件指令。

| 原语 | 用途 | 典型场景 |
|------|------|---------|
| `T.max(a, b)` | 取两数最大值 | ReLU、attention score 裁剪 |
| `T.exp(x)` / `T.exp2(x)` | 指数 / 2 的指数 | softmax、attention |
| `T.log(x)` / `T.log2(x)` | 自然对数 / 以 2 为底对数 | 交叉熵、log-softmax |
| `T.rsqrt(x)` | 倒数平方根 | LayerNorm、RMSNorm |
| `T.sigmoid(x)` | Sigmoid 激活 | 门控机制 |
| `T.cast(x, dtype)` | 类型转换 | 精度切换（如 fp16 → fp32） |
| `T.infinity(dtype)` | 该类型的无穷大值 | 归约初始化 |
| `T.clamp(x, lo, hi)` | 限制在 [lo, hi] 范围 | 数值稳定性 |
| `T.dp4a(A, B, C)` | 4 元素点积累加 | INT8 量化计算 |

### `T.if_then_else` — 条件选择

```python
T.if_then_else(cond: PrimExpr, true_val: PrimExpr, false_val: PrimExpr) → PrimExpr
```

三元条件表达式，等价于 `cond ? true_val : false_val`。**不是控制流分支**，而是 IR 表达式，确保所有线程执行同一指令。

```python
# 边界检查：越界时写 0
C[gi] = T.if_then_else(gi < N, A[gi] + B[gi], 0.0)
```

### 归约操作

```python
T.reduce_sum(buffer: Buffer, out: Buffer, dim: int = -1, clear: bool = True)
T.reduce_max(buffer: Buffer, out: Buffer, dim: int = -1, clear: bool = True, nan_propagate: bool = False)
T.reduce_min(buffer: Buffer, out: Buffer, dim: int = -1, clear: bool = True, nan_propagate: bool = False)
```

- **`buffer`**：输入缓冲区
- **`out`**：输出缓冲区（通常用 `T.alloc_local` 分配）
- **`dim`**：归约维度，`-1` 表示归约所有维度
- **`clear`**：`True` 时自动将 `out` 初始化为单位元（sum→0, max→-inf, min→+inf）
- **`nan_propagate`**（max/min 专用）：float16/bfloat16 下控制 NaN 传播行为

```python
buf = T.alloc_fragment((BM, BN), "float32")
out = T.alloc_local([1], "float32")
T.reduce_sum(buf, out, dim=-1)       # 归约所有元素求和，结果在 out[0]
T.reduce_max(buf, out, dim=0)        # 沿 dim=0 取最大值
```

### `T.reshape` / `T.view` — 形状/类型重解释

```python
T.reshape(src: Buffer, shape: tuple[int, ...]) → Buffer
T.view(src: Buffer, shape: tuple[int, ...] | None = None, dtype: DType | None = None) → Buffer
```

两个原语都创建共享底层存储的新 Buffer 视图，不产生数据拷贝。`reshape` 只改形状（总比特数不变）；`view` 可以同时改形状和数据类型。



### 第二~四章总结：基础 GEMM

下面用前三章的全部知识——内存分配（`alloc_shared`/`alloc_fragment`/`clear`）、数据搬运（`T.copy`）、计算（`T.gemm`）——组合出一个完整的基础 GEMM（每个 Block 处理一个 tile）：

```python
@T.prim_func
def gemm(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
        A_s = T.alloc_shared((BM, BK), "float16")     # 二、内存分配
        B_s = T.alloc_shared((BK, BN), "float16")
        C_f = T.alloc_fragment((BM, BN), "float32")
        T.clear(C_f)

        for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
            T.copy(A[by * BM, ko * BK], A_s)           # 三、数据搬运
            T.copy(B[ko * BK, bx * BN], B_s)
            T.gemm(A_s, B_s, C_f)                      # 四、计算

        T.copy(C_f, C[by * BM, bx * BN])               # 结果写回
```

运行输出示例：

```
Loading tilelang libs from dev root: /workspace/test/tilelang/build
M=1024, N=1024, K=512, max_diff=0.031250, passed.
```

这是 TileLang 最经典的 GEMM 骨架：Global→Shared→gemm→Global。控制流优化（`T.Pipelined`、`T.Persistent`）将在下一章展开。

---

## 五、控制流

### 控制流选择总览

| 原语 | 执行模式 | 何时使用 |
|------|---------|---------|
| `T.serial(n)` | 串行，n 次迭代按顺序执行 | 迭代之间有依赖，或不需要并行/预取 |
| `T.unroll(n)` | 串行 + 编译时展开循环体 | 迭代次数很少且固定（如 4、8），消除循环开销 |
| `T.Parallel(n)` | 并行，n 次迭代分配到所有线程 | 迭代间无依赖，适合 Element-wise 操作 |
| `T.Pipelined(n, num_stages)` | 串行 + 多阶段预取，计算和加载重叠 | 循环体内有"加载数据 → 计算"模式，想隐藏访存延迟 |
| `T.Persistent(dims, grid, id, group_size)` | 持久化调度，少量 Block 循环处理大量 tile | Block 数少于 tile 数时，Block 自动轮询处理多个 tile |

### 选择流程图

```
迭代之间有数据依赖吗？
  ├── 有依赖 → 必须串行
  │     ├── 循环体内是先加载数据再做计算吗？
  │     │     ├── 是，且迭代次数多 → T.Pipelined
  │     │     └── 否，纯计算或迭代少 → T.serial
  │     └── 迭代次数很少且固定吗？
  │           └── 是 → T.unroll（替代 T.serial）
  └── 无依赖 → 可以并行
        └── T.Parallel

Block 数量 < 实际 tile 数量，想让 Block 循环处理多个 tile？
  └── T.Persistent
```

### `T.serial` — 串行循环

```python
for i in T.serial(N):
    result[i] = result[i - 1] + data[i]  # 迭代间有依赖，必须串行
```

对应普通 for 循环，每次迭代顺序执行。

### `T.unroll` — 编译时展开

```python
for i in T.unroll(4):
    x[i] = x[i] + 1
# 编译后等价于：
# x[0] = x[0] + 1
# x[1] = x[1] + 1
# x[2] = x[2] + 1
# x[3] = x[3] + 1
```

编译器在编译时把循环体复制 n 份，消除循环计数器和分支跳转开销。只适合循环次数很小且固定时使用。

### `T.Parallel` — 并行循环

```python
for i in T.Parallel(N):
    C[i] = A[i] + B[i]  # 每个元素计算独立，可以并行
```

将 N 次迭代并行分配给 Block 内所有线程。Layout Inference 自动决定每个线程执行哪些迭代。注意 `T.gemm` 内部已隐含并行分配，不需要用 `T.Parallel` 包裹。

### `T.Pipelined` — 软件流水线

```python
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    T.copy(A[...], A_shared)
    T.copy(B[...], B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

串行执行，但通过多阶段预取让**计算和数据加载重叠**：

```
T.serial（无流水线）:
  [加载1][计算1]          [加载2][计算2]          [加载3][计算3]
  ← GPU 等数据 →

T.Pipelined(num_stages=3):
  [加载1][加载2][加载3]
  [等待 ][计算1][计算2][计算3]  ← 加载和计算重叠
         [加载4][加载5][加载6]
```

- **代价**：LDS 占用 = `num_stages × (A_shared + B_shared)`
- **调优**：在 LDS 容量允许的前提下，增加 `num_stages` 可以更好地隐藏访存延迟

### `T.Persistent` — 持久化 Block 调度

```python
for bx, by in T.Persistent(
    [T.ceildiv(N, block_N), T.ceildiv(M, block_M)],  # tile 网格尺寸
    wgs_per_cu * cu_num,                               # 实际 Block 数量
    block_id,                                          # 当前 Block 的一维 ID
    group_size=1                                       # tile 分组大小
):
    ...
```

`T.Persistent` 是持久化 Kernel 的核心原语，解决的问题是：

> **当 Grid 中的 Block 数量少于实际 tile 数量时，让每个 Block 自动轮询处理多个 tile。**

持久化 Kernel 只启动少量 Block（等于 CU 数量 × wgs_per_cu），每个 Block 通过 `T.Persistent` 自动获取下一个要处理的 tile 坐标 `(bx, by)`，处理完后继续获取下一个，直到所有 tile 处理完毕。相比普通 GEMM（有多少 tile 就启动多少 Block），减少了 GPU 硬件调度器的 Block 创建/分配/回收开销。

**参数说明**：

- 第一个参数 `[tile_x, tile_y]`：二维 tile 网格的尺寸（总共 tile_x × tile_y 个 tile）
- 第二个参数：实际启动的 Block 总数（通常 = `wgs_per_cu × cu_num`，即每个 CU 上驻留 wgs_per_cu 个 Block）
- 第三个参数 `block_id`：当前 Block 在一维 Grid 中的 ID（来自 `T.Kernel(grid_size, ...) as (block_id)`）
- `group_size`：tile 分组大小，控制相邻 tile 是否合并处理。`group_size=1` 表示每个 Block 每次只拿一个 tile

---

## 六、布局注解（Layout Annotation）

### `T.annotate_layout` — 声明缓冲区的内存布局

```python
C_shared = T.alloc_shared((block_M, sub_block_N), dtype)
T.annotate_layout({
    C_shared: tl.layout.make_hcu_swizzled_layout(C_shared, major_pack=2),
    B_shared: tl.layout.make_hcu_swizzled_layout(B_shared, major_pack=2),
    A_shared: tl.layout.make_hcu_swizzled_layout(A_shared, major_pack=2),
})
```

`T.annotate_layout` 告诉编译器某个 Shared Memory 缓冲区的**数据排布方式**（layout），编译器根据这个信息生成高效的向量化读写指令。

#### 为什么需要 Swizzle 布局

正常情况下，Shared Memory 中的数据按行优先顺序排列。当多个线程同时访问同一 bank 的不同地址时，会发生 **bank conflict**，导致访问串行化、带宽下降。

**Swizzle 布局**通过对地址做 XOR 变换，将原本可能冲突的访问分散到不同的 bank，减少 bank conflict。具体来说：

- `make_hcu_swizzled_layout(buf, major_pack=2)`：为 DCU（HCU）生成 swizzle 布局
- `major_pack=2`：在 N 维度（major dimension）上每 2 个元素打包，控制 swizzle 的粒度

**什么时候需要 annotate_layout**：

- 当 `T.gemm` 的输入是 Fragment（而不是 Shared Memory）时，你需要手动控制数据在 Shared Memory 中的布局，因为数据路径变为 Global → Fragment → Shared → Fragment → gemm，编译器无法自动推断 Shared Memory 的布局
- 当用 `T.copy` 做 Fragment ↔ Shared 传输时，正确的 swizzle 布局可以让 `ds_write`/`ds_read` 指令以向量化方式执行

### `tl.layout.make_hcu_swizzled_layout` — DCU Swizzle 布局

```python
tl.layout.make_hcu_swizzled_layout(buffer, major_pack=2)
```

为 DCU 的 Shared Memory 缓冲区生成 swizzle 地址变换，返回一个 Layout 对象传给 `T.annotate_layout`。`major_pack` 控制向量化宽度。

---

## 七、同步与线程索引

### `T.sync_threads` / `T.sync_warp` — 线程同步

```python
T.sync_threads(barrier_id: int | None = None, arrive_count: int | None = None)
T.sync_warp(mask: int | None = None)
```

- `sync_threads`：Block 内所有线程同步（`__syncthreads`）。可选 `barrier_id` 指定屏障编号，`arrive_count` 指定到达线程数。
- `sync_warp`：Wave 内线程同步（`__syncwarp`，DCU 上 64 线程）。可选 `mask` 指定参与线程掩码。

Shared Memory 写入后、读取前通常需要同步。但 TileLang 的 `T.copy` 和 `T.gemm` 在大多数场景下已自动插入必要的同步。**当手动做 Fragment ↔ Shared 的数据流转时**，需要显式插入同步：

```python
T.copy(C_local_0, C_shared_0)
T.sync_threads()                                 # 确保所有线程写入完成
T.copy(C_shared_0, C[by * block_M, bx * block_N])
T.sync_threads()                                 # 确保 C_shared_0 可以被下一轮复用
T.copy(C_local_1, C_shared_0)
```

### `T.get_thread_binding` — 线程索引

```python
T.get_thread_binding(dim: int = 0) → Var            # 单个维度索引
T.get_thread_bindings() → list[Var]                  # 返回 [threadIdx.x, threadIdx.y]
```

获取当前线程在 Block 内的索引。`dim=0`→threadIdx.x, `dim=1`→threadIdx.y, `dim=2`→threadIdx.z。用于手动控制线程级数据分配（如 sparse attention 中的 gather/scatter 操作）。大多数情况下 Layout Inference 会自动分配，不需要手动获取。

---

## 八、调试

### `T.print` — 打印缓冲区内容

```python
T.print(obj: Buffer | PrimExpr | None = None, msg: str = "", warp_group_id: int = 0, warp_id: int = 0)
```

从单个线程（由 `warp_group_id` 和 `warp_id` 指定）打印缓冲区或标量的内容，避免多线程输出洪水。不指定 `obj` 时可只打印消息字符串。

```python
T.print(C_f, msg='accumulator:')
T.print(A_s, msg='A tile:')
T.print(msg='reached checkpoint')
```

### `T.device_assert` — 设备端断言

```python
T.device_assert(condition: PrimExpr, msg: str = "", no_stack_info: bool = False)
```

在 GPU 上执行条件检查，`condition` 为假时触发断言并打印 `msg`。`no_stack_info=True` 可省略堆栈信息以减少输出量。用于调试边界条件或数值异常。

```python
T.device_assert(gi < N, msg='index out of range')
```

---

## 九、编译 Pass 配置

### `tl.PassConfigKey` — 编译 Pass 开关

```python
@tl.jit(out_idx=[-1], pass_configs={
    tl.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    tl.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
})
```

编译 Pass 配置键，控制 TileLang 编译器的优化行为。常用的包括：

| 配置键 | 作用 |
|-------|------|
| `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE` | 激进合并 Shared Memory 分配，减少 LDS 碎片，让多个小缓冲区共享同一块 LDS 空间 |
| `TL_DISABLE_THREAD_STORAGE_SYNC` | 禁用线程存储同步，在确定不需要 `__syncthreads` 的场景下消除多余的同步屏障 |

---

## 十、持久化 GEMM 完整示例

下面是将前九章全部知识融会贯通的完整示例：持久化调度 + Swizzle 布局优化 + Fragment 中转数据流。

```python
@tl.jit(out_idx=[-1], pass_configs={
    tl.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
})
def gemm_persistent(M, N, K, block_M, block_N, block_K,
                    num_stages, thread_num, wgs_per_cu=2,
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
        B: T.Tensor((N, K), dtype),   # 注意：B 是 N×K 存储（转置）
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(grid_size, threads=thread_num) as (block_id):
            # Shared Memory 分配
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared_0 = T.alloc_shared((sub_block_N, block_K), dtype)

            # Fragment 分配：输入缓存（用于手动数据流转）
            A_local_0 = T.alloc_fragment((block_M, block_K), dtype)
            A_local_0_ = T.alloc_fragment((block_M, block_K), dtype)
            B_local_0 = T.alloc_fragment((sub_block_N, block_K), dtype)
            B_local_1 = T.alloc_fragment((sub_block_N, block_K), dtype)
            B_local_0_ = T.alloc_fragment((sub_block_N, block_K), dtype)
            B_local_1_ = T.alloc_fragment((sub_block_N, block_K), dtype)

            # Fragment 分配：输出累加器（split_n 份，各算一半 N）
            C_local_0 = T.alloc_fragment((block_M, sub_block_N), dtype="float32")
            C_local_1 = T.alloc_fragment((block_M, sub_block_N), dtype="float32")

            # Shared Memory 分配：输出中转缓冲区
            C_shared_0 = T.alloc_shared((block_M, sub_block_N), dtype)
            T.annotate_layout({
                C_shared_0: tl.layout.make_hcu_swizzled_layout(C_shared_0, major_pack=2),
                B_shared_0: tl.layout.make_hcu_swizzled_layout(B_shared_0, major_pack=2),
                A_shared: tl.layout.make_hcu_swizzled_layout(A_shared, major_pack=2),
            })

            # 持久化调度：少量 Block 轮询处理所有 tile
            for bx, by in T.Persistent(
                [n_blocks, m_blocks],
                grid_size,
                block_id,
                group_size=1
            ):
                if by * block_M < M and bx * block_N < N:
                    T.clear(C_local_0)
                    T.clear(C_local_1)

                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                        # 数据路径：Global → Fragment → Shared(swizzle) → Fragment → gemm
                        # 目的：通过 swizzle 布局减少 bank conflict
                        T.copy(A[by * block_M, k * block_K], A_local_0, coalesced_width=8)
                        T.copy(A_local_0, A_shared)

                        T.copy(B[bx * block_N, k * block_K], B_local_0, coalesced_width=8)
                        T.copy(B[bx * block_N + sub_block_N, k * block_K], B_local_1, coalesced_width=8)

                        T.copy(B_local_0, B_shared_0)
                        T.copy(B_shared_0, B_local_0_)
                        T.copy(B_local_1, B_shared_0)
                        T.copy(B_shared_0, B_local_1_)

                        T.copy(A_shared, A_local_0_)

                        T.gemm(A_local_0_, B_local_0_, C_local_0, k_pack=2, transpose_B=True)
                        T.gemm(A_local_0_, B_local_1_, C_local_1, k_pack=2, transpose_B=True)

                    # 写回：Fragment → Shared → Global（通过 Shared 中转确保 coalesced）
                    T.copy(C_local_0, C_shared_0)
                    T.copy(C_shared_0, C[by * block_M, bx * block_N])
                    T.copy(C_local_1, C_shared_0)
                    T.copy(C_shared_0, C[by * block_M, bx * block_N + sub_block_N])

    return _gemm_persistent
```

### 关键设计要点

1. **持久化调度**：`grid_size = wgs_per_cu × cu_num`，只启动少量 Block，每个 Block 通过 `T.Persistent` 循环处理多个 tile。好处是减少 Block 调度开销，让 CU 始终有活干。

2. **N 维度拆分（split_n）**：将 `block_N` 拆成两半（`sub_block_N`），每次只分配一半 N 的 Shared Memory，减少 LDS 占用。

3. **Fragment 中转 + Swizzle**：数据路径是 `Global → Fragment → Shared(swizzle) → Fragment → gemm`。Shared Memory 是 swizzle 布局的，`T.gemm` 的输入是 Fragment。编译器根据 `T.annotate_layout` 知道 Shared Memory 的 swizzle 排布，从而生成高效的 `ds_read`/`ds_write` 指令。

4. **B 转置存储**：B 矩阵存储为 N×K（而非 K×N），通过 `transpose_B=True` 告诉编译器无需手动转置。

5. **结果写回通过 Shared 中转**：`Fragment → Shared → Global`，确保 Global Memory 写入是 coalesced 的。

---

## 附录：常用原语速查

### 函数定义
| 原语 | 用途 |
|------|------|
| `@T.prim_func` | 定义 TileLang IR 函数 |
| `@tl.jit(out_idx, pass_configs)` | JIT 编译包装器 |
| `T.dynamic(name)` | 定义编译时符号变量 |
| `T.ceildiv(a, b)` | 编译时向上取整除 |

### 内存与搬运
| 原语 | 用途 |
|------|------|
| `T.alloc_shared(shape, dtype)` | 分配 Shared Memory 缓冲区 |
| `T.alloc_fragment(shape, dtype)` | 分配寄存器级缓冲区 |
| `T.alloc_local(shape, dtype)` | 分配线程私有标量/小数组 |
| `T.copy(src, dst, coalesced_width=...)` | 同步数据搬运，支持任意内存层级 |
| `T.async_copy(src, dst)` | 显式异步搬运（需手动等待） |
| `T.clear(buf)` / `T.fill(buf, val)` | 缓冲区清零/填充 |
| `T.annotate_layout({buf: layout})` | 声明缓冲区内存布局 |
| `tl.layout.make_hcu_swizzled_layout(buf, major_pack=N)` | DCU Swizzle 布局生成 |

### 计算
| 原语 | 用途 |
|------|------|
| `T.gemm(A, B, C, k_pack=N, transpose_B=True, policy=...)` | 矩阵乘法 |
| `T.GemmWarpPolicy.Square/FullRow/FullCol/FullColK` | GEMM Warp 分区策略 |
| `T.reduce_sum/max/min(buf)` | 归约操作 |
| `T.reduce_sum_warp(...)` | Warp 内求和归约 |
| `T.max(a, b)` / `T.exp/log/exp2/log2(x)` | 逐元素数学运算 |
| `T.rsqrt(x)` / `T.sigmoid(x)` | 逐元素数学运算 |
| `T.cast(x, dtype)` | 类型转换 |
| `T.infinity(dtype)` | 该类型的无穷大值 |
| `T.clamp(x, lo, hi)` | 限制在 [lo, hi] 范围 |
| `T.dp4a(A, B, C)` | 4 元素点积累加 |
| `T.if_then_else(cond, t, f)` | 条件选择表达式 |

### 控制流
| 原语 | 执行模式 | 使用场景 |
|------|---------|---------|
| `T.serial(n)` | 串行 | 迭代间有依赖 |
| `T.unroll(n)` | 串行+编译展开 | 迭代次数少且固定 |
| `T.Parallel(n)` | 线程并行 | 迭代间无依赖，Element-wise |
| `T.Pipelined(n, num_stages)` | 串行+预取重叠 | 加载与计算重叠，隐藏访存延迟 |
| `T.Persistent(dims, grid, id, group_size)` | 持久化 Block 调度 | Block 数 < tile 数，Block 轮询处理 tile |

### 同步与线程索引
| 原语 | 用途 |
|------|------|
| `T.sync_threads()` | Block 内全同步 |
| `T.sync_warp()` | Wave 内同步（DCU: 64 线程） |
| `T.get_thread_binding(dim)` | 获取当前线程索引（threadIdx） |

### 调试
| 原语 | 用途 |
|------|------|
| `T.print(buf, msg='...')` | 打印缓冲区/标量（单线程） |
| `T.device_assert(cond, msg='...')` | 设备端断言 |

### 编译配置
| 配置键 | 作用 |
|-------|------|
| `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE` | 激进合并 Shared Memory 分配 |
| `TL_DISABLE_THREAD_STORAGE_SYNC` | 禁用线程存储同步 |
