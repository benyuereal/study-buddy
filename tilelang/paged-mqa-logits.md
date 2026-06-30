# Paged MQA Logits 算子技术分享

## 一、背景与公式

### 1.1 算子定义

**输入**（每个 batch）：

$$
\begin{aligned}
Q &\in \mathbb{R}^{H \times D} \quad &\text{（query，} H \text{ 头，每头 } D \text{ 维）} \\
K &\in \mathbb{R}^{L \times D} \quad &\text{（历史 keys，} L \text{ 为 context 长度，MQA 下单头）} \\
w &\in \mathbb{R}^{H} \quad &\text{（门控权重，每头一个标量）}
\end{aligned}
$$

**输出**：

$$
\text{logits} \in \mathbb{R}^{L}
$$

**计算**：

$$
\boxed{\;\text{logits}[k] = \sum_{h=0}^{H-1} \operatorname{relu}\!\big(Q[h] \cdot K[k]\big) \times w[h]\;, \quad k = 0, \dots, L-1\;}
$$

**等价矩阵形式**：

$$
\text{logits} = \operatorname{rowsum}\!\Big(\;\operatorname{relu}\!\big(K \, Q^{\top}\big) \odot w\;\Big)
$$

其中 $K Q^{\top} \in \mathbb{R}^{L \times H}$（内积），$\operatorname{relu}$ 和 $\odot \, w$（广播）逐元素作用，$\operatorname{rowsum}$ 沿 heads 维度归约回 $\mathbb{R}^{L}$。

> 对比标准 attention $\operatorname{softmax}(QK^{\top}/\sqrt{d})\,V$：这里**不做 softmax、不乘 V**，而是用 relu + 门控权重 + 多头求和替代，输出 logits 供下游模块使用。

### 1.2 分块计算

K 按固定大小 $B = 64$ 行切分为逻辑块。对第 $t$ 个逻辑块（$K_t \in \mathbb{R}^{B \times D}$）：

$$
\text{logits}_t = \operatorname{rowsum}\!\Big(\;\operatorname{relu}\!\big(K_t \, Q^{\top}\big) \odot w\;\Big) \;\in\; \mathbb{R}^{B}
$$

每个 Block 处理一个逻辑块，通过 `block_table` 查表找到 $K_t$ 所在的物理地址。

### 1.3 为什么是 "Paged"

KV cache 不是连续存储的，而是按固定大小 block（$B = 64$ 行/block）分页存储。每个 batch 的 `block_table` 记录逻辑位置到物理 block 的映射，和操作系统的虚拟内存分页类似：

- 消除显存碎片（所有 block 尺寸统一）
- 支持变长序列，无需预分配最大长度
- 支持 prefix caching（多个请求共享同一物理 block）

计算时先查表：`phys_block = block_table[b][logical_block]`，再从 `KV[phys_block]` 读取。

### 1.4 与标准 GEMM 的区别

| | 标准 GEMM | Paged MQA Logits |
|---|---|---|
| 输出 | `[M, N]` 矩阵 | `[batch, max_context_len]`（heads 被归约掉） |
| K 维度遍历 | 连续 K 维度切 tile | 按 **logical block** 切 tile，查表映射物理 block |
| 后处理 | 无 | relu + 门控加权 + reduce_sum |
| Grid 组织 | `(M, N)` 每 tile 一个 Block | `(max_block_len, batch)` 每个 logical KV block 一个 Block |

### 1.5 伪代码

```
# 输入: Q [H, D], K [L, D], w [H]
# 输出: logits [L]
# 常量: BLOCK_KV = 64

for each logical_block in 0 .. ceil(L / BLOCK_KV):
    phys_block = block_table[logical_block]          # 查表：逻辑→物理地址
    K_tile = KV_cache[phys_block]                    # 加载 K 块 [BLOCK_KV, D]
    S = K_tile × Q^T                                 # GEMM: [BLOCK_KV, H]
    S = relu(S) ⊙ w                                  # 融合后处理
    logits[logical_block * BLOCK_KV : ...] = rowsum(S, dim=heads)   # 规约写回
```

每个 Block 独立处理一个 logical KV block，Block 之间无数据依赖。

---

## 二、性能数据

DCU 上实测，以手写 HIP 汇编的 lightop 为 baseline：

```
Case                             B    H    D  avg_ctx  tilelang(ms)   lightop(ms)   speedup
------------------------------------------------------------------------------------------------
bs1_H32_D128_4k                  1   32  128     4096         0.132         0.075     1.76x
bs64_H32_D128_4k                64   32  128     4096         0.187         0.192     0.97x
bs128_H32_D128_4k              128   32  128     4096         0.268         0.286     0.94x
bs1_H64_D128_4k                  1   64  128     4096         0.122         0.074     1.66x
bs64_H64_D128_4k                64   64  128     4096         0.301         0.224     1.34x
bs1_H32_D128_72k                 1   32  128    72000         0.121         0.076     1.60x
bs1_H64_D128_72k                 1   64  128    72000         0.120         0.084     1.42x
```

结论：

- **大 batch（≥64）**：与手写汇编基本持平（0.94x ~ 1.34x），Python DSL 能做到这个性能是编译器自动向量化 + 流水线优化的直接体现。
- **小 batch（=1）**：比手写汇编慢 1.4x ~ 1.8x。小 batch 时 Grid 太小（max_block_len 个 Block 分散到 80 个 CU，多数 CU 闲置），手写汇编可以做 warp-specialized 的细粒度优化，TileLang 的通用流水线在这个场景下开销占比放大。

在 vLLM 中的 profiler 延迟对比：

![优化前](assets/paged_mqa_logits_pre.png)

![优化后](assets/paged_mqa_logits_after.png)

优化后延迟从 1ms 降至 10μs，**提升数十倍**。

---

## 三、数据布局

```
Q:            [batch_size, heads, D]           bfloat16   （view from [B, next_n, H, D]，next_n=1）
KV cache:     [num_blocks, BLOCK_KV=64, 1, D]  bfloat16   （K 无 heads 维度，block 内 64 行连续）
Weights:      [batch_size, heads]              float32    （与 gemm accum_dtype 对齐）
Block table:  [batch_size, max_block_len]      int32      （逻辑 block → 物理 block 映射）
Logits:       [batch_size, max_context_len]    float32    （直接供下游 softmax 使用）
Context lens: [batch_size]                     int32
```

几个设计决策：

- **KV 存为 `[N, D]` 而非 `[K, N]`**：每个物理 block 内 64 行连续存储，读取时一行一次加载。`T.gemm(k_smem, q_smem, s, transpose_B=True)` 告诉编译器 Q 被隐式转置为 `[D, heads]`，实现 `K × Q^T`。
- **Weights 和 Logits 都是 float32**：后处理 `relu × weight` 与 gemm 累加器同精度，避免多次类型转换的精度损失。

---

## 四、完整代码

以下是从 `examples/paged_mqa_logits.py` 提取的核心 kernel（去掉了 benchmark 和接口包装层）：

```python
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def paged_mqa_logits_kernel(
    heads: int,
    index_dim: int,
    block_N: int = 64,
    num_stages: int = 1,
    threads: int = 256,
    policy: str = "square",
):
    D = index_dim
    dtype = T.bfloat16
    accum_dtype = T.float32
    index_dtype = T.int32
    K_PACK = 1
    _policy = T.GemmWarpPolicy.FullRow if policy == "full_row" else T.GemmWarpPolicy.Square

    batch_size = T.dynamic("batch_size")
    num_blocks = T.dynamic("num_blocks")
    max_block_len = T.dynamic("max_block_len")
    max_context_len = T.dynamic("max_context_len")

    @T.prim_func
    def kernel(
        Q: T.Tensor([batch_size, heads, D], dtype),                            # ①
        KV: T.Tensor([num_blocks, BLOCK_KV, 1, D], dtype),                     # ②
        Logits: T.Tensor([batch_size, max_context_len], accum_dtype),          # ③
        Weights: T.Tensor([batch_size, heads], accum_dtype),                   # ④
        BlockTable: T.Tensor([batch_size, max_block_len], index_dtype),        # ⑤
    ):
        with T.Kernel(max_block_len, batch_size, threads=threads) as (logical_block, b_idx):  # ⑥
            phys_block = BlockTable[b_idx, logical_block]                       # ⑦
            kv_offset_global = logical_block * BLOCK_KV

            q_smem = T.alloc_shared([heads, D], dtype)                          # ⑧
            k_smem = T.alloc_shared([block_N, D], dtype)
            s = T.alloc_fragment([block_N, heads], accum_dtype)
            logits_tile = T.alloc_fragment([block_N], accum_dtype)
            w_frag = T.alloc_fragment([heads], accum_dtype)

            T.copy(Q[b_idx, 0:heads, 0:D], q_smem)                             # ⑨
            T.copy(Weights[b_idx, 0:heads], w_frag)

            for nbn_i in T.Pipelined(T.ceildiv(BLOCK_KV, block_N),             # ⑩
                                     num_stages=num_stages):
                kv_row = nbn_i * block_N
                T.copy(KV[phys_block, kv_row:kv_row + block_N, 0, 0:D], k_smem)  # ⑪

                T.clear(s)                                                       # ⑫
                T.gemm(k_smem, q_smem, s,                                        # ⑬
                       k_pack=K_PACK, transpose_B=True, policy=_policy)

                for bn_i, h_i in T.Parallel(block_N, heads):                     # ⑭
                    s[bn_i, h_i] = T.max(s[bn_i, h_i], T.cast(0, accum_dtype)) * w_frag[h_i]

                T.reduce_sum(s, logits_tile, dim=1, clear=True)                  # ⑮

                for bn_i in T.Parallel(block_N):                                 # ⑯
                    gkv = kv_offset_global + kv_row + bn_i
                    if gkv < max_context_len:
                        Logits[b_idx, gkv] = logits_tile[bn_i]

    return kernel
```

---

## 五、逐行解读（聚焦关键决策）

### ⑥ Grid 组织

```python
with T.Kernel(max_block_len, batch_size, threads=threads) as (logical_block, b_idx):
```

每个 Block 处理 **一个 batch 的一个 logical KV block（最多 64 行）**。Grid 是 `(max_block_len, batch_size)` 的二维网格。不同 batch 完全独立，同一 batch 的不同 logical block 之间没有数据依赖（读不同的物理 block、写 logits 的不同位置）。

**为什么不用 `T.Persistent`**：每个 Block 已经处理了 64 行 KV，计算量足够大（64×heads×D 的 GEMM + 后处理）。如果让一个 Block 持久化处理多个 logical block，寄存器压力会过大（需在循环间保留 Q 和 weights 并不断更新 logits 写回位置），收益不抵开销。

### ⑦ 地址映射

```python
phys_block = BlockTable[b_idx, logical_block]
kv_offset_global = logical_block * BLOCK_KV
```

分页的核心：从 block table 查物理 block ID。`kv_offset_global` 是当前 logical block 在 `max_context_len` 空间的起始偏移，用于写回时的全局坐标计算。逻辑→物理映射由内存分配器（如 vLLM BlockAllocator）维护，算子只读表。

### ⑧ 内存分配策略

```python
q_smem = T.alloc_shared([heads, D], dtype)       # LDS
k_smem = T.alloc_shared([block_N, D], dtype)     # LDS
s = T.alloc_fragment([block_N, heads], accum_dtype)  # 寄存器
logits_tile = T.alloc_fragment([block_N], accum_dtype)
w_frag = T.alloc_fragment([heads], accum_dtype)
```

**Q 放 Shared Memory 而非每次从 Global 读**：Q 在整个 Block 处理期间不变（一个 Block 只处理一个 batch 的一个 logical block），放 LDS 后 `T.gemm` 从 LDS 读取 Q 的延迟远低于从 HBM 读取。这是最直接的 LDS 复用优化。

**K 放 Shared Memory 是流水线的基础**：`T.Pipelined` 自动做双缓冲，下一轮迭代的 K 加载与当前轮的计算重叠，隐藏 HBM 访存延迟。

**`s` 累加器用 float32**：`T.gemm` 做的是 `C += A × B`，bfloat16 输入、float32 累加，避免多次乘加后精度丢失。对应 [tilelang-basic-knowledge.md §二](tilelang-basic-knowledge.md#talloc_fragment--分配寄存器级缓冲区)。

### ⑨ Q 和 Weights 的加载方式

```python
T.copy(Q[b_idx, 0:heads, 0:D], q_smem)
T.copy(Weights[b_idx, 0:heads], w_frag)
```

Q 和 Weights 在循环外一次性加载。`T.copy` 语义是同步的——调用后数据保证可用（编译器自动插入 `commit + wait`）。Weights 直接加载到 Fragment（寄存器），因为后处理 `relu × weight` 在每个线程上访问 weights，不需要 Shared Memory 广播。

参考 [tilelang-basic-knowledge.md §三 `T.copy`](tilelang-basic-knowledge.md#tcopy--同步数据搬运)。

### ⑩~⑪ 流水线加载 KV

```python
for nbn_i in T.Pipelined(T.ceildiv(BLOCK_KV, block_N), num_stages=num_stages):
    kv_row = nbn_i * block_N
    T.copy(KV[phys_block, kv_row:kv_row + block_N, 0, 0:D], k_smem)
```

一个 logical block 有 `BLOCK_KV=64` 行：

- `block_N=64` 时：1 次迭代，不需流水线（`num_stages=0`）
- `block_N=32` 时：2 次迭代，`num_stages=1` 让第二轮 KV 加载与第一轮 GEMM 重叠

注意这里的流水线和标准 GEMM 不同——只有 **K 在流水线加载**（Q 不变），LDS 总占用 = `q_smem + (num_stages + 1) × k_smem`，而不是标准 GEMM 的 `num_stages × (A_smem + B_smem)`。因为 A、B 两个缓冲区都参与流水线时编译器会为两者各分配 num_stages 份，但这里只有 K 参与。

参考 [tilelang-basic-knowledge.md §五 `T.Pipelined`](tilelang-basic-knowledge.md#tpipelined--软件流水线)。

### ⑫~⑬ GEMM

```python
T.clear(s)
T.gemm(k_smem, q_smem, s, k_pack=K_PACK, transpose_B=True, policy=_policy)
```

`T.clear(s)` 必须——`T.gemm` 是累加操作（`C += A×B`），寄存器未初始化内容不可预测。对应 [tilelang-basic-knowledge.md §二 clear/fill](tilelang-basic-knowledge.md#tclear--tfill--缓冲区初始化)。

`transpose_B=True`：Q 存储为 `[heads, D]`，计算需要 `[D, heads]`，编译器隐式转置。

`k_pack=1`：RDNA 架构约束。CDNA 架构可用 `k_pack=2`，一次处理 K 维度的 2 个元素。

参考 [tilelang-basic-knowledge.md §四 `T.gemm`](tilelang-basic-knowledge.md#tgemm--矩阵乘法原语)。

### ⑭ 融合后处理

```python
for bn_i, h_i in T.Parallel(block_N, heads):
    s[bn_i, h_i] = T.max(s[bn_i, h_i], T.cast(0, accum_dtype)) * w_frag[h_i]
```

三个操作融合在一个表达式里：

- `T.max(x, 0)` — ReLU（TileLang 没有单独的 relu 原语，用 max 代替）
- `× w_frag[h_i]` — 门控加权
- `T.Parallel` — 将 `block_N × heads` 次逐元素操作分配到所有线程并行执行

编译器会将 `max + mul` 生成连续的指令序列，没有中间写回，这是**算子融合**在 TileLang 中的自然表达——不需要单独写一个 "fused relu-mul" 原语，正常的逐元素表达式即可。

### ⑮ reduce_sum

```python
T.reduce_sum(s, logits_tile, dim=1, clear=True)
```

沿 heads 维度归约：`[block_N, heads] → [block_N]`。`clear=True` 自动将 `logits_tile` 初始化为 0。参考 [tilelang-basic-knowledge.md §四 归约操作](tilelang-basic-knowledge.md#归约操作)。

### ⑯ 写回 + 边界检查

```python
for bn_i in T.Parallel(block_N):
    gkv = kv_offset_global + kv_row + bn_i
    if gkv < max_context_len:
        Logits[b_idx, gkv] = logits_tile[bn_i]
```

每个线程只写入不越界的行。超出 `max_context_len` 的位置由 clean kernel 统一处理。`if` 条件中的 `max_context_len` 是编译时 `T.dynamic` 符号，编译器将其作为运行时常量优化，不产生分支发散。

---

## 六、用到的优化点总结

与 [tilelang-basic-knowledge.md](tilelang-basic-knowledge.md) 中介绍的原语一一对应：

| 优化点 | 原语 | 在这个算子中的体现 |
|--------|------|-------------------|
| **LDS 复用** | `T.alloc_shared` | Q 放 LDS 跨多轮 GEMM 复用，避免反复从 HBM 读取 |
| **流水线加载** | `T.Pipelined` | K 的 HBM→LDS 加载与当前轮 GEMM 重叠，隐藏访存延迟 |
| **寄存器累加** | `T.alloc_fragment` + float32 | GEMM 累加器和后处理中间结果都在寄存器中，避免写回 LDS/Global |
| **算子融合** | `T.Parallel` + 逐元素表达式 | ReLU、门控加权在同一个并行循环中完成，无中间写回 |
| **Warp 分区策略** | `GemmWarpPolicy.Square / FullRow` | 小 batch 用 FullRow 减少 warp 间通信开销 |
| **动态形状 JIT** | `T.dynamic` + `@tilelang.jit` | batch_size、max_context_len 等运行时可变维度在 kernel 启动时确定，编译时做特化优化 |
| **Fast Math** | `TL_ENABLE_FAST_MATH` | 允许编译器用近似指令替换精确数学函数（如 reciprocal），提升吞吐 |

### 这个算子没有用到的优化（为什么？）

| 未使用 | 原因 |
|--------|------|
| Swizzle 布局 (`T.annotate_layout`) | `T.gemm` 的输入是 Shared Memory 而非 Fragment 中转，编译器自动处理 bank conflict，不需要手动 swizzle |
| `T.Persistent` | 每个 Block 处理 64 行 KV，计算量已足够大，持久化调度反而增加寄存器压力 |
| `T.async_copy` | `T.Pipelined` 的自动流水线已满足需求，不需要手动控制异步预取时机 |
| D 维度拆分 | `D=128` 不大，拆分反而增加 Shared Memory 分配碎片和额外同步 |

---

## 七、LDS 容量与配置选择

DCU LDS 为 64 KB，核心策略：**分块占用控制在 32 KB 以内**，这样 LDS 至少能容纳 2 个 block，通过 `T.Pipelined` 双缓冲隐藏访存延迟。

### LDS 预算公式

```
LDS 总占用 = q_smem + K_LDS
           = heads × D × 2 + (num_stages + 1) × block_N × D × 2 bytes
```

Q 不参与流水线（只有 1 份），K 有 `num_stages + 1` 份用于双缓冲。

### 配置选择

在 `block_N ∈ {32, 64}` × `num_stages ∈ {2, 1, 0}` 中，优先 `num_stages` 大（更多缓冲级数隐藏延迟），其次 `block_N` 小（每轮计算量小，与加载更容易重叠）。

**`heads=32, D=128`**（q_smem = 8 KB）：

| block_N | num_stages | 总占用 | 说明 |
|---------|-----------|--------|------|
| 32 | 2 | 32 KB | 最优：刚好一半 LDS，2 级流水线 |
| 32 | 1 | 24 KB | 备选 |
| 64 | 1 | 40 KB | 备选 |
| 64 | 0 | 24 KB | 无流水线，退而求其次 |

**`heads=64, D=128`**（q_smem = 16 KB）：

| block_N | num_stages | 总占用 | 说明 |
|---------|-----------|--------|------|
| 32 | 1 | 32 KB | 最优：刚好一半 LDS |
| 32 | 0 | 24 KB | 备选 |
| 64 | 1 | 48 KB | 超过一半但未超总量 |
| 64 | 0 | 32 KB | 刚好一半，无流水线 |

### 小 batch 优化（batch_size ≤ 4）

```python
if batch_size <= 4 and block_N == 32:
    if q_smem + 64 * D * 2 <= LDS_LIMIT:
        block_N = 64
        num_stages = 0
        policy = "full_row"
```

小 batch 下 Grid 小、CU 利用率低，牺牲流水线换取单个 Block 处理整块 64 行 KV，配合 `FullRow` 策略消除 warp 间通信——计算换同步开销。
