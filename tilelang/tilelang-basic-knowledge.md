# TileLang 核心关键字与概念

## 文档定位

本文档是 TileLang 关键字的速查手册，逐个解释每个原语的语法、语义、与 GPU 硬件的对应关系，以及使用场景。阅读前请先理解 [GPU 基础概念](./gpu-basic-knowledge.md)（Grid、Block、Wave、Shared Memory/LDS、寄存器、Tiling 等）。本文档以海光 DCU 为主要目标硬件，LDS 默认 64 KB/CU，Wave 为 64 线程。

---

## 一、函数定义：`@T.prim_func` 与 `@tl.jit`

### `@T.prim_func` — 定义 TileLang 原始函数

```python
@T.prim_func
def my_kernel(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((N, K), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=128) as (bx, by):
        ...
```

`@T.prim_func` 是定义 TileLang Kernel 函数的装饰器。被它装饰的函数：
- 参数必须用 `T.Tensor(shape, dtype)` 或 `T.Buffer` 标注类型和形状
- 函数体内使用 TileLang DSL 语法（`T.Kernel`、`T.alloc_shared`、`T.copy` 等）
- 返回的是一个 IR 函数对象，不直接执行，需要交给 `tl.jit` 编译后才能调用

它和 `@tl.jit` 的关系：
- `@T.prim_func` 定义的是**纯 IR 层**函数，不包含任何 Python 运行时逻辑
- `@tl.jit` 包裹一个**返回 prim_func 的 Python 工厂函数**，负责 JIT 编译和缓存

### `@tl.jit` — JIT 编译装饰器

```python
@tl.jit(out_idx=[-1], pass_configs={
    tl.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
})
def my_gemm(M, N, K, block_M, block_N, block_K, ...):
    # 这里可以写 Python 逻辑：计算 grid_size、定义符号变量等
    @T.prim_func
    def kernel(A: T.Tensor(...), B: T.Tensor(...), C: T.Tensor(...)):
        ...
    return kernel
```

`@tl.jit` 将一个**返回 prim_func 的 Python 函数**包装成可调用的 JIT 编译函数：

- `out_idx=[-1]`：指定函数的第几个参数是"输出"。`[-1]` 表示最后一个参数 C 是输出，编译器会据此做优化（如自动清零输出 buffer）
- `pass_configs={...}`：传递编译 pass 选项，控制编译优化行为
- 工厂函数的 Python 参数（如 `block_M`、`block_N`）会在 JIT 编译时被"冻结"为常量，生成特化版本的 Kernel
- 第一次调用时触发编译并缓存，后续同参数调用直接命中缓存

**典型用法**：工厂函数里做 Python 级的计算（如 `T.ceildiv`、`T.min`），把结果作为常量传给 prim_func，prim_func 里只写纯 GPU 逻辑。

### `T.min` — 取最小值

```python
grid_size = T.min(m_blocks * n_blocks, wgs_per_cu * cu_num)
```

`T.min` 在 JIT 工厂函数中计算两个值的最小值，结果作为常量传入 prim_func。这和 `T.ceildiv` 一样，是在**编译时**求值的 Python 级辅助函数，不是 GPU 指令。

---

## 二、Kernel 定义与执行上下文

### `T.Kernel` — 定义 GPU 启动配置

```python
with T.Kernel(grid_size, threads=128) as (block_id):
```

除了之前介绍的二维 Grid `T.Kernel(grid_x, grid_y, threads=...) as (bx, by)`，`T.Kernel` 还支持**一维 Grid**：

- `grid_size`：一维 Grid 的 Block 总数
- `threads=128`：每个 Block 内的线程数，**DCU 上应为 64 的倍数，最大 1024**
- `block_id`：当前 Block 在一维 Grid 中的索引（0 到 grid_size-1）

一维 Grid 常用于**持久化 Kernel**（Persistent Kernel）——grid_size 远小于实际 tile 数量，每个 Block 循环处理多个 tile（见 `T.Persistent`）。

### `T.ceildiv` — 向上取整除

```python
grid_x = T.ceildiv(N, block_N)  # 等价于 ceil(N / block_N)
```

当矩阵尺寸不能被分块大小整除时，边缘 Block 需要处理"不足一块"的数据（通过边界 `if` 守卫）。

---

## 三、内存分配

### `T.alloc_shared` — 分配 Shared Memory 缓冲区

```python
A_shared = T.alloc_shared((block_M, block_K), dtype="float16")
```

在 GPU 的 LDS 上分配二维缓冲区：

- **生命周期**：当前 Block 内有效，Block 执行完毕自动释放
- **可见范围**：Block 内所有线程共享，其他 Block 不可访问
- **性能意义**：LDS 比 Global Memory 快约 10-20 倍，是分块算法的核心加速手段
- **容量约束**：大小直接决定单 Block 占用多少 LDS，结合 `num_stages` 会倍数放大

### `T.alloc_fragment` — 分配寄存器级缓冲区

```python
C_local = T.alloc_fragment((block_M, block_N), accum_dtype="float32")
```

在寄存器文件上分配缓冲区，用于存储计算中间结果：

- `fragment` 不是单个线程的寄存器，而是**整个 Thread Block 的寄存器文件**中的一块
- TileLang 的 Layout Inference 会自动推导每个线程持有 fragment 的哪个子矩阵
- 寄存器是最快的内存（~1 cycle 延迟），累加器应放在 fragment 中
- 使用前必须用 `T.clear` 初始化为零
- **fragment 也可用于存储输入数据**（如 A_local、B_local），当需要手动控制数据在 Shared Memory 和寄存器之间的流转时，把输入也放到 fragment 中

### `T.clear` / `T.fill` — 缓冲区初始化

```python
T.clear(C_local)       # 所有元素置零
T.fill(buf, value)     # 所有元素填充为 value
```

---

## 四、数据搬运

### `T.copy` — 同步数据搬运

```python
# Global → Fragment（直接加载到寄存器，绕开 Shared Memory）
T.copy(A[by * block_M, k * block_K], A_local, coalesced_width=8)

# Fragment → Shared（寄存器写入 LDS，常用于 swizzle 转置）
T.copy(A_local, A_shared)

# Shared → Fragment（从 LDS 读取到寄存器）
T.copy(A_shared, A_local_)

# Fragment → Global（寄存器写回全局内存）
T.copy(C_local, C[by * block_M, bx * block_N])

# Fragment → Shared → Global（通过 LDS 中转写回，确保 coalesced）
T.copy(C_local, C_shared)
T.copy(C_shared, C[by * block_M, bx * block_N])
```

`T.copy` 是 TileLang 中**统一的数据搬运原语**，支持任意内存层级之间的传输：

- **语义上是同步的**：`T.copy` 执行完毕后，目标缓冲区数据保证可用
- 编译器自动优化：根据源/目标类型和 GPU 架构，自动选择最优底层指令
- 自动 coalesce：保证 Global Memory 读写是合并访问

#### `coalesced_width` 参数

```python
T.copy(src, dst, coalesced_width=8)
```

指定 Global Memory 访问时的合并宽度（以元素为单位）。例如 `coalesced_width=8` 表示编译器会尝试用 128-bit（8×16bit）向量化加载指令一次性读取 8 个 fp16 元素，提高带宽利用率。这个值需要根据数据类型和目标硬件调整。

### `T.async_copy` — 显式异步搬运（进阶）

```python
T.async_copy(src, dst)      # 发起异步拷贝，不等完成就继续
T.ptx_wait_group(N)         # 等待还剩 N 组未完成的异步拷贝
```

需要手动管理同步，适合需要精细控制异步预取时机的场景。如果 `T.Pipelined` 已经满足需求，不需要用这个。

---

## 五、计算

### `T.gemm` — 矩阵乘法原语

```python
# 基础用法：Shared Memory 输入
T.gemm(A_shared, B_shared, C_local)

# 进阶用法：Fragment 输入 + 参数控制
T.gemm(A_local, B_local, C_local, k_pack=2, transpose_B=True)
```

在 Shared Memory 或 Fragment 上的两个输入矩阵做矩阵乘法，结果累加到 fragment 累加器（`C += A @ B`）：

- 输入 A、B 可以位于 **Shared Memory**（`T.alloc_shared`）或 **Fragment**（`T.alloc_fragment`）
- 累加器 C 必须位于 **Fragment**（`T.alloc_fragment`）
- 编译器根据 GPU 架构自动降级为最优硬件指令：DCU 的 M-FMA 指令、NVIDIA Tensor Core 等

#### `k_pack` 参数

```python
T.gemm(A, B, C, k_pack=2)
```

控制 K 维度的打包因子。`k_pack=2` 表示每次迭代处理 K 维度的 2 个元素，利用向量化指令一次完成多个乘加运算。这个值需要与数据类型和目标架构的指令能力匹配。

#### `transpose_B` 参数

```python
T.gemm(A, B, C, transpose_B=True)
```

当 B 矩阵的存储格式为 **N×K（列主序）** 而非 K×N（行主序）时，设置 `transpose_B=True` 告诉编译器 B 已经在逻辑上被转置了。这对应 DCU 上 B 矩阵通常按 N×K 存储的惯例，避免手动转置的开销。

### 其他常用计算原语

| 原语 | 用途 |
|------|------|
| `T.gemm_sp(...)` | 2:4 稀疏矩阵乘法 |
| `T.reduce_sum/max/min(buf, axis)` | 归约操作 |
| `T.exp/log/rsqrt/sigmoid(x)` | 逐元素数学运算 |

---

## 六、控制流

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

### `T.Persistent` — 持久化 Block 调度（新增）

```python
for bx, by in T.Persistent(
    [T.ceildiv(N, block_N), T.ceildiv(M, block_M)],  # tile 网格尺寸
    wgs_per_cu * cu_num,                               # 实际 Block 数量
    block_id,                                          # 当前 Block 的一维 ID
    group_size=1                                       # tile 分组大小
):
    # bx: N 方向 tile 坐标, by: M 方向 tile 坐标
    ...
```

`T.Persistent` 是持久化 Kernel 的核心原语，解决的问题是：

> **当 Grid 中的 Block 数量少于实际 tile 数量时，让每个 Block 自动轮询处理多个 tile。**

**为什么需要持久化 Kernel**：

普通 GEMM 中 Grid 的 Block 数量 = `m_blocks × n_blocks`，有多少个 tile 就启动多少个 Block。当矩阵很大时，Block 数量可能达到数万个，GPU 硬件调度器需要大量时间创建、分配、回收 Block。

持久化 Kernel 的做法是：**只启动少量 Block（等于 CU 数量 × wgs_per_cu），每个 Block 循环处理多个 tile**。Block 通过 `T.Persistent` 自动获取下一个要处理的 tile 坐标 `(bx, by)`，处理完后继续获取下一个，直到所有 tile 处理完毕。

**参数说明**：

- 第一个参数 `[tile_x, tile_y]`：二维 tile 网格的尺寸（总共 tile_x × tile_y 个 tile）
- 第二个参数：实际启动的 Block 总数（通常 = `wgs_per_cu × cu_num`，即每个 CU 上驻留 wgs_per_cu 个 Block）
- 第三个参数 `block_id`：当前 Block 在一维 Grid 中的 ID（来自 `T.Kernel(grid_size, ...) as (block_id)`）
- `group_size`：tile 分组大小，控制相邻 tile 是否合并处理。`group_size=1` 表示每个 Block 每次只拿一个 tile

**与手动 waves 循环的对比**：

```python
# 手动 waves 方式（需要自己算 tile_id、坐标映射）
for w in T.serial(waves):
    tile_id = grid_size * w + block_id
    bx = (tile_id // group_size) % m_blocks
    by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size
    ...

# T.Persistent 方式（编译器自动处理调度逻辑）
for bx, by in T.Persistent([n_blocks, m_blocks], grid_size, block_id, group_size=1):
    ...
```

`T.Persistent` 让编译器自动生成最优的 tile 分配逻辑，避免了手动计算坐标的复杂性，并且可以利用硬件提供的 persistent 调度优化。

---

## 七、布局注解（Layout Annotation）

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

## 八、编译 Pass 配置

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

## 九、同步

```python
T.sync_threads()   # Block 内所有线程同步
T.sync_warp()      # Wave 内线程同步（DCU 上 64 线程）
```

Shared Memory 写入后、读取前通常需要同步。但 TileLang 的 `T.copy` 和 `T.gemm` 在大多数场景下已自动插入必要的同步。**当手动做 Fragment ↔ Shared 的数据流转时**，需要显式插入同步：

```python
T.copy(C_local_0, C_shared_0)
T.sync_threads()                                 # 确保所有线程写入完成
T.copy(C_shared_0, C[by * block_M, bx * block_N])
T.sync_threads()                                 # 确保 C_shared_0 可以被下一轮复用
T.copy(C_local_1, C_shared_0)
```

---

## 十、完整 GEMM 示例

### 10.1 基础 GEMM（每个 Block 处理一个 tile）

```python
import tilelang.language as T

M, N, K = T.define_symbol("M N K")
block_M = 128
block_N = 128
block_K = 32
threads = 128

def Matmul(A: T.Buffer, B: T.Buffer, C: T.Buffer):
    with T.Kernel(
        T.ceildiv(N, block_N),   # grid_x
        T.ceildiv(M, block_M),   # grid_y
        threads=threads
    ) as (bx, by):

        A_shared = T.alloc_shared((block_M, block_K))
        B_shared = T.alloc_shared((block_K, block_N))
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype=T.float32)
        T.clear(C_local)

        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * block_M, bx * block_N])
```

### 10.2 持久化 GEMM（少量 Block 循环处理大量 tile，含 Swizzle 优化）

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
            C_local_0 = T.alloc_fragment((block_M, sub_block_N), accum_dtype)
            C_local_1 = T.alloc_fragment((block_M, sub_block_N), accum_dtype)

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
| `T.min(a, b)` | 编译时取最小值（工厂函数中用） |
| `T.ceildiv(a, b)` | 编译时向上取整除 |

### 内存与搬运
| 原语 | 用途 |
|------|------|
| `T.alloc_shared(shape, dtype)` | 分配 Shared Memory 缓冲区 |
| `T.alloc_fragment(shape, dtype)` | 分配寄存器级缓冲区 |
| `T.copy(src, dst, coalesced_width=...)` | 同步数据搬运，支持任意内存层级 |
| `T.async_copy(src, dst)` | 显式异步搬运（需手动等待） |
| `T.clear(buf)` / `T.fill(buf, val)` | 缓冲区清零/填充 |
| `T.annotate_layout({buf: layout})` | 声明缓冲区内存布局 |
| `tl.layout.make_hcu_swizzled_layout(buf, major_pack=N)` | DCU Swizzle 布局生成 |

### 控制流
| 原语 | 执行模式 | 使用场景 |
|------|---------|---------|
| `T.serial(n)` | 串行 | 迭代间有依赖 |
| `T.unroll(n)` | 串行+编译展开 | 迭代次数少且固定 |
| `T.Parallel(n)` | 线程并行 | 迭代间无依赖，Element-wise |
| `T.Pipelined(n, num_stages)` | 串行+预取重叠 | 加载与计算重叠，隐藏访存延迟 |
| `T.Persistent(dims, grid, id, group_size)` | 持久化 Block 调度 | Block 数 < tile 数，Block 轮询处理 tile |

### 计算
| 原语 | 用途 |
|------|------|
| `T.gemm(A, B, C, k_pack=N, transpose_B=True)` | 矩阵乘法，支持 fragment 输入和参数控制 |
| `T.reduce_sum/max/min(buf)` | 归约操作 |
| `T.exp/log/rsqrt/sigmoid(x)` | 逐元素数学运算 |

### 同步
| 原语 | 用途 |
|------|------|
| `T.sync_threads()` | Block 内全同步 |
| `T.sync_warp()` | Wave 内同步（DCU: 64 线程） |

### 编译配置
| 配置键 | 作用 |
|-------|------|
| `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE` | 激进合并 Shared Memory 分配 |
| `TL_DISABLE_THREAD_STORAGE_SYNC` | 禁用线程存储同步 |
