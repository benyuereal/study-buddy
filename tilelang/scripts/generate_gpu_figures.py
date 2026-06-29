"""
Generate GPU concept figures for gpu-basic-knowledge.md.
Output: assets/ directory with 8 PNG figures.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style constants
DARK_BG = "#1a1a2e"
CARD_BG = "#16213e"
ACCENT = "#0f3460"
HIGHLIGHT = "#e94560"
GOLD = "#f5c518"
GREEN = "#4ecca3"
BLUE = "#4a9eff"
PURPLE = "#a855f7"
ORANGE = "#f97316"
LIGHT = "#e2e8f0"
GRAY = "#94a3b8"
WHITE = "#ffffff"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Heiti TC", "Hei", "STHeiti", "SimHei", "DejaVu Sans"],
    "font.size": 11,
    "text.color": LIGHT,
    "axes.unicode_minus": False,
    "axes.facecolor": DARK_BG,
    "figure.facecolor": DARK_BG,
    "savefig.facecolor": DARK_BG,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# Figure 1: Memory Hierarchy — funnel/triangle layout
# ============================================================
def fig_memory_hierarchy():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(6, 9.5, "GPU 内存层次结构", fontsize=20, fontweight="bold",
            color=WHITE, ha="center")
    ax.text(6, 9.0, "容量越大越慢，离计算单元越近越快", fontsize=10, color=GRAY, ha="center")

    # Funnel layers — wide at top (HBM), narrow at bottom (Reg)
    layers = [
        # name, desc_line1, desc_line2, y, height, width_ratio, color, alpha
        ("HBM (Global Memory)", "~数十 GB", "~数百 cycles | 所有 Block 共享",
         6.8, 1.4, 1.0, "#ef4444", 0.30),
        ("L2 Cache", "~数 MB", "~百 cycles | 所有 CU 共享",
         5.0, 1.0, 0.72, "#f97316", 0.30),
        ("LDS / Shared Memory", "64 KB / CU", "~数十 cycles | Block 内共享",
         3.5, 0.85, 0.44, GOLD, 0.30),
        ("Register File", "~256 KB / CU", "~1 cycle | 线程私有",
         2.2, 0.7, 0.20, GREEN, 0.35),
    ]

    for name, d1, d2, y, h, wr, color, alpha in layers:
        w = 10.0 * wr
        x = (12 - w) / 2
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor=color, alpha=alpha, linewidth=2.5)
        ax.add_patch(box)
        ax.text(6, y + h - 0.3, name, fontsize=14, fontweight="bold", color=color, ha="center")
        ax.text(6, y + h - 0.75, f"{d1}    {d2}", fontsize=9, color=GRAY, ha="center")

    # Arrows between layers
    for i in range(len(layers) - 1):
        _, _, _, y_top, h_top, _, _, _ = layers[i]
        _, _, _, y_bot, h_bot, _, _, _ = layers[i+1]
        ax.annotate("", xy=(6, y_bot + h_bot), xytext=(6, y_top),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=2.5))

    # Side labels
    ax.text(0.3, 8.3, "大\n\n容\n量\n\n小", fontsize=9, color=GRAY, ha="center", va="center",
            linespacing=1.8)
    ax.annotate("", xy=(0.5, 8.8), xytext=(0.5, 1.8),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5))

    ax.text(11.7, 8.3, "慢\n\n\n速\n\n\n快", fontsize=9, color=GRAY, ha="center", va="center",
            linespacing=1.8)
    ax.annotate("", xy=(11.5, 8.8), xytext=(11.5, 1.8),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.5))

    save(fig, "memory-hierarchy.png")


# ============================================================
# Figure 2: Scheduling Hierarchy — pyramid layout
# ============================================================
def fig_scheduling_hierarchy():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(6, 9.6, "GPU 调度层级结构", fontsize=20, fontweight="bold", color=WHITE, ha="center")
    ax.text(6, 9.1, "DCU: CU / Wave(64)   |   NVIDIA: SM / Warp(32)", fontsize=9, color=GRAY, ha="center")

    # Pyramid: wide at top, narrow at bottom
    nodes = [
        # name, desc, y, width_ratio, color, count_label
        ("Kernel", "GPU 函数，CPU 端发起调用", 7.6, 0.55, "#ef4444", "1 个"),
        ("Grid", "逻辑坐标系，分配 (bx, by) 坐标", 6.0, 0.70, ORANGE, "grid_x × grid_y 个"),
        ("Thread Block", "调度到单个 CU，独占 LDS，≤1024 线程", 4.2, 0.85, GOLD, "≤ CU 数 × 驻留上限"),
        ("Wave (64线程)", "CU 内部执行粒度，64 线程同指令", 2.4, 0.90, GREEN, "threads / 64 个"),
        ("Thread", "执行指令的最小单元，私有寄存器", 0.8, 0.55, BLUE, "threads 个"),
    ]

    for name, desc, y, wr, color, count_label in nodes:
        w = 11.0 * wr
        x = (12 - w) / 2
        h = 1.2
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor=color, alpha=0.22, linewidth=2.5)
        ax.add_patch(box)
        ax.text(6, y + 0.75, name, fontsize=14, fontweight="bold", color=color, ha="center")
        ax.text(6, y + 0.30, desc, fontsize=9, color=GRAY, ha="center", va="center")
        # Count on the right
        ax.text(11.3, y + 0.60, count_label, fontsize=8.5, color=GRAY, ha="right",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=DARK_BG, edgecolor=GRAY, alpha=0.5))

    # Connecting arrows
    for i in range(len(nodes) - 1):
        _, _, y_top, _, _, _ = nodes[i]
        _, _, y_bot, _, _, _ = nodes[i+1]
        ax.annotate("", xy=(6, y_bot + 1.2), xytext=(6, y_top),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=2))

    save(fig, "scheduling-hierarchy.png")


# ============================================================
# Figure 3: Bank Conflict
# ============================================================
def fig_bank_conflict():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("LDS Bank Conflict 原理", fontsize=16, fontweight="bold", color=WHITE, y=0.98)

    titles = ["无冲突（理想情况）", "2-way Bank Conflict"]
    data_list = []

    # Left: no conflict
    banks1 = np.arange(32)
    threads1 = np.arange(32)
    data1 = np.zeros((2, 32))
    for i in range(32):
        data1[0, i] = i  # row 0: bank accessed
        data1[1, i] = 1  # row 1: count
    data_list.append(data1)

    # Right: 2-way conflict
    data2 = np.zeros((2, 32))
    for i in range(32):
        bank = i % 16
        data2[0, i] = bank
        data2[1, i] = 2 if i >= 16 else 1
    data_list.append(data2)

    for idx, ax in enumerate(axes):
        ax.set_facecolor(DARK_BG)
        data = data_list[idx]
        colors_arr = [GREEN if data[1, i] == 1 else HIGHLIGHT for i in range(32)]
        bars = ax.bar(range(32), data[1, :], color=colors_arr, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(0, 32, 4))
        ax.set_xticklabels([f"B{i}" for i in range(0, 32, 4)], fontsize=8, color=GRAY)
        ax.set_ylim(0, 3)
        ax.set_ylabel("访问次数", fontsize=10, color=GRAY)
        ax.set_xlabel("Bank 编号", fontsize=10, color=GRAY)
        ax.set_title(titles[idx], fontsize=12, color=WHITE)
        ax.tick_params(colors=GRAY)
        ax.spines["bottom"].set_color(GRAY)
        ax.spines["left"].set_color(GRAY)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        if idx == 0:
            ax.text(16, 2.2, "32 线程 -> 32 个不同 Bank\n一个周期完成",
                    fontsize=9, color=GREEN, ha="center")
        else:
            ax.text(16, 2.5, "32 线程 -> 16 个 Bank\n每个 Bank 被访问 2 次\n需要 2 个周期",
                    fontsize=9, color=HIGHLIGHT, ha="center")

    plt.tight_layout()
    save(fig, "bank-conflict.png")


# ============================================================
# Figure 4: Swizzle
# ============================================================
def fig_swizzle():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Swizzle 地址变换", fontsize=16, fontweight="bold", color=WHITE, y=0.98)

    # Left: linear addressing showing conflict pattern
    ax = axes[0]
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("线性地址 → Bank Conflict", fontsize=12, color=WHITE)

    # Draw a grid: 4 rows of 8 elements each
    for row in range(4):
        for col in range(8):
            bank = (row * 8 + col) % 4  # simplified: 4 banks
            colors = [GREEN, BLUE, ORANGE, PURPLE]
            rect = Rectangle((col, 3 - row), 0.85, 0.85, facecolor=colors[bank],
                             edgecolor="white", linewidth=0.5, alpha=0.7)
            ax.add_patch(rect)
            ax.text(col + 0.42, 3 - row + 0.42, f"{bank}", fontsize=7,
                    color="white", ha="center", va="center", fontweight="bold")

    ax.text(4, -0.3, "按列读取：同列元素落到同一 Bank → 冲突！", fontsize=10,
            color=HIGHLIGHT, ha="center")

    # Right: swizzled
    ax = axes[1]
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Swizzle 地址 → 冲突消除", fontsize=12, color=WHITE)

    for row in range(4):
        for col in range(8):
            # XOR swizzle: bank = (row*8 + col) XOR ((row*8+col)//16 * 2)
            addr = row * 8 + col
            swizzled = addr ^ ((addr // 16) * 2)
            bank = swizzled % 4
            colors = [GREEN, BLUE, ORANGE, PURPLE]
            rect = Rectangle((col, 3 - row), 0.85, 0.85, facecolor=colors[bank],
                             edgecolor="white", linewidth=0.5, alpha=0.7)
            ax.add_patch(rect)
            ax.text(col + 0.42, 3 - row + 0.42, f"{bank}", fontsize=7,
                    color="white", ha="center", va="center", fontweight="bold")

    ax.text(4, -0.3, "按列读取：同列元素分散到不同 Bank → 无冲突", fontsize=10,
            color=GREEN, ha="center")

    plt.tight_layout()
    save(fig, "swizzle.png")


# ============================================================
# Figure 5: LDS Constraint
# ============================================================
def fig_lds_constraint():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("LDS 容量约束：单 Block LDS 占用决定 CU 驻留数", fontsize=14,
                 fontweight="bold", color=WHITE, y=1.02)

    scenarios = [
        ("单 Block 32 KB\n→ CU 驻留 2 Block", [32, 32], [GREEN, GREEN]),
        ("单 Block 48 KB\n→ CU 只能驻留 1 Block", [48, 16], [ORANGE, "#333"]),
        ("单 Block 96 KB\n→ 超出 LDS 容量！", [96, -1], [HIGHLIGHT]),
    ]

    for idx, (title, sizes, colors) in enumerate(scenarios):
        ax = axes[idx]
        ax.set_facecolor(DARK_BG)
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3.5)
        ax.axis("off")
        ax.set_title(title, fontsize=11, color=WHITE)

        # Draw LDS container
        lds_box = FancyBboxPatch((0.3, 0.2), 2.4, 3.0, boxstyle="round,pad=0.08",
                                 facecolor="none", edgecolor=GRAY, linewidth=2, linestyle="--")
        ax.add_patch(lds_box)
        ax.text(1.5, 3.3, "LDS (64 KB)", fontsize=9, color=GRAY, ha="center")

        y_pos = 0.35
        for i, (size, color) in enumerate(zip(sizes, colors)):
            if size < 0:
                ax.text(1.5, 1.8, "编译报错", fontsize=12, color=HIGHLIGHT,
                        ha="center", fontweight="bold")
                break
            h = size / 64 * 2.8
            block_box = FancyBboxPatch((0.5, y_pos), 2.0, h, boxstyle="round,pad=0.05",
                                       facecolor=color, edgecolor=color, alpha=0.35, linewidth=2)
            ax.add_patch(block_box)
            ax.text(1.5, y_pos + h/2, f"Block {i}\n{size} KB", fontsize=9,
                    color=color, ha="center", va="center", fontweight="bold")
            y_pos += h + 0.08

    plt.tight_layout()
    save(fig, "lds-constraint.png")


# ============================================================
# Figure 6: CU Utilization
# ============================================================
def fig_cu_utilization():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle("Block 数量对 CU 利用率的影响（64 CU GPU）", fontsize=14,
                 fontweight="bold", color=WHITE, y=1.02)

    scenarios = [
        ("Block 数 = 32\n一半 CU 闲置", 32, 32),
        ("Block 数 = 128\n所有 CU 满负荷", 128, 0),
        ("Block 数 = 4096\n利用率高，调度开销大", 4096, 0),
    ]

    for idx, (title, total_blocks, idle) in enumerate(scenarios):
        ax = axes[idx]
        ax.set_facecolor(DARK_BG)
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.axis("off")
        ax.set_title(title, fontsize=11, color=WHITE)

        ncu = 64
        cols = 8
        rows = 8

        active_count = total_blocks if total_blocks <= ncu else ncu

        for r in range(rows):
            for c in range(cols):
                cu_id = r * cols + c
                if cu_id < active_count:
                    color = GREEN
                    alpha = 0.7
                elif cu_id < ncu:
                    color = GRAY
                    alpha = 0.2
                else:
                    continue

                rect = Rectangle((c, 7 - r), 0.8, 0.8, facecolor=color,
                                 edgecolor="white", linewidth=0.3, alpha=alpha)
                ax.add_patch(rect)

        if total_blocks > ncu:
            ax.text(4, -0.4, f"共 {total_blocks} Block，需排队 {total_blocks // ncu} 轮",
                    fontsize=9, color=ORANGE, ha="center")

        if idle > 0:
            ax.text(4, -0.4, f"{idle} 个 CU 空闲", fontsize=9, color=HIGHLIGHT, ha="center")

    plt.tight_layout()
    save(fig, "cu-utilization.png")


# ============================================================
# Figure 7: Two-Level Pipeline
# ============================================================
def fig_two_level_pipeline():
    fig, axes = plt.subplots(2, 1, figsize=(12, 4.5))
    fig.suptitle("两层流水线：Block 内部 + CU 上 Block 间切换", fontsize=14,
                 fontweight="bold", color=WHITE, y=1.02)

    # Level 1: Block internal pipeline
    ax = axes[0]
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    ax.set_title("层级 1：Block 内部流水线（T.Pipelined, num_stages=2）", fontsize=11, color=BLUE)

    stages = [
        [(0, 2, "加载A0", GREEN), (0, 4, "计算A0", BLUE)],
        [(2, 4, "加载A1", GREEN), (4, 6, "计算A1", BLUE)],
        [(4, 6, "加载A2", GREEN), (6, 8, "计算A2", BLUE)],
    ]

    for stage_idx, [(ld_x, ld_w, ld_label, ld_c), (cp_x, cp_w, cp_label, cp_c)] in enumerate(stages):
        y = 1.2 - stage_idx * 0.6
        ax.barh(y, ld_w, height=0.5, left=ld_x, color=ld_c, alpha=0.5, edgecolor=ld_c, linewidth=1.5)
        ax.text(ld_x + ld_w/2, y, ld_label, fontsize=8, color="white", ha="center", va="center", fontweight="bold")
        ax.barh(y-0.7, cp_w, height=0.5, left=cp_x, color=cp_c, alpha=0.5, edgecolor=cp_c, linewidth=1.5)
        ax.text(cp_x + cp_w/2, y-0.7, cp_label, fontsize=8, color="white", ha="center", va="center", fontweight="bold")

    ax.text(11.5, 1.5, "→ 加载与计算\n   重叠执行", fontsize=8, color=GREEN, va="center")

    # Level 2: CU multi-block switching
    ax = axes[1]
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2.5)
    ax.axis("off")
    ax.set_title("层级 2：CU 上 Block 间切换（2 个 Block 驻留）", fontsize=11, color=PURPLE)

    # Block 0
    segments_b0 = [(0, 1.5, "加载", GRAY), (1.5, 2.5, "等内存...", "#555"),
                   (2.5, 4, "计算", BLUE), (4, 5, "加载", GRAY), (5, 5.5, "等", "#555"),
                   (5.5, 7, "计算", BLUE), (7, 8, "加载", GRAY), (8, 9, "等", "#555"),
                   (9, 10.5, "计算", BLUE)]
    for x, w, label, c in segments_b0:
        ax.barh(1.5, w, height=0.6, left=x, color=c, alpha=0.6 if c != GRAY else 0.3,
                edgecolor=c, linewidth=1)

    # Block 1
    segments_b1 = [(0.5, 1.5, "计算", BLUE), (1.5, 3, "加载", GRAY), (3, 3.5, "等", "#555"),
                   (3.5, 5, "计算", BLUE), (5, 6.5, "加载", GRAY), (6.5, 7.5, "等", "#555"),
                   (7.5, 9, "计算", BLUE), (9, 10, "加载", GRAY)]
    for x, w, label, c in segments_b1:
        ax.barh(0.5, w, height=0.6, left=x, color=c, alpha=0.6 if c != GRAY else 0.3,
                edgecolor=c, linewidth=1)

    ax.text(11.5, 1.5, "→ CU 始终\n   有活干", fontsize=8, color=GREEN, va="center")
    ax.text(0.2, 2.2, "Block 0", fontsize=9, color=BLUE, fontweight="bold")
    ax.text(0.2, 1.2, "Block 1", fontsize=9, color=PURPLE, fontweight="bold")

    # Legend at bottom
    ax.text(0.2, -0.1, "图例：", fontsize=8, color=GRAY)
    ax.add_patch(Rectangle((1.2, -0.25), 1, 0.2, facecolor=BLUE, alpha=0.6))
    ax.text(2.3, -0.15, "计算", fontsize=7, color=BLUE)
    ax.add_patch(Rectangle((3.5, -0.25), 1, 0.2, facecolor=GRAY, alpha=0.3))
    ax.text(4.6, -0.15, "加载", fontsize=7, color=GRAY)
    ax.add_patch(Rectangle((5.8, -0.25), 1, 0.2, facecolor="#555", alpha=0.4))
    ax.text(6.9, -0.15, "等内存", fontsize=7, color=GRAY)

    plt.tight_layout()
    save(fig, "two-level-pipeline.png")


# ============================================================
# Figure 8: Tuning Decision Flow
# ============================================================
def fig_tuning_decision():
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14)
    ax.axis("off")

    ax.text(7, 13.5, "分块参数调优决策流程", fontsize=18, fontweight="bold", color=WHITE, ha="center")
    ax.text(7, 13.0, "以 DCU (64 KB LDS / CU) 为例", fontsize=10, color=GRAY, ha="center")

    def draw_box(x, y, w, h, text, color, fontsize=9, alpha=0.25):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor=color, alpha=alpha, linewidth=2.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, fontsize=fontsize, color=color,
                ha="center", va="center", fontweight="bold", linespacing=1.3)

    def arrow(x1, y1, x2, y2, label="", color=GRAY):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx + 0.2, my, label, fontsize=8, color=color, fontweight="bold")

    # Step 1: Start
    draw_box(4, 11.5, 6, 0.9, "1. 设置分块参数\nblock_M, block_N, block_K", BLUE, 10, 0.25)

    # Step 2: Calculate LDS
    arrow(7, 11.5, 7, 10.7)
    draw_box(3, 9.5, 8, 1.2,
             "2. 计算单 Block LDS 占用\n"
             "num_stages x (BMxBK + BKxBN) x dtype_bytes\n"
             "含 num_stages 多缓冲开销",
             PURPLE, 9, 0.25)

    # Step 3: Branch -- LDS check
    arrow(7, 9.5, 7, 8.8)

    # Left branch: over limit
    draw_box(1, 7.5, 4.5, 1.3,
             "LDS 占用 > 64 KB\n编译/运行报错\n-> 减小 BM/BN/BK\n-> 减小 num_stages",
             HIGHLIGHT, 9, 0.30)
    arrow(7, 8.8, 3.25, 8.8, "超限", HIGHLIGHT)

    # Right branch: OK
    draw_box(8, 7.5, 5, 1.3,
             "LDS 占用 <= 64 KB\n-> 配置合法，继续检查",
             GREEN, 9, 0.25)
    arrow(7, 8.8, 10.5, 8.8, "OK", GREEN)

    # Step 4: CU residency
    arrow(10.5, 7.5, 10.5, 6.8)
    draw_box(7.5, 5.5, 5.5, 1.3,
             "LDS 占用 <= 32 KB?\n"
             "是 -> CU 驻留 >= 2 Block [OK]\n"
             "否 -> CU 驻留 1 Block\n"
             "      延迟无法隐藏",
             GREEN, 9, 0.25)

    # Step 5: Block count
    arrow(10.5, 5.5, 10.5, 4.8)
    draw_box(7.5, 3.3, 5.5, 1.5,
             "3. 计算 Grid Block 总数\n"
             "ceil(M/BM) x ceil(N/BN)\n"
             "\n"
             "Block 总数 >= CU 数?\n"
             "是 -> 所有 CU 有活干 [OK]\n"
             "否 -> 部分 CU 闲置，减小 BM/BN",
             GREEN, 9, 0.25)

    # Step 6: Final
    arrow(10.5, 3.3, 10.5, 2.3)
    draw_box(6, 1.2, 5.5, 1.1,
             "4. 通过 autotuning\n微调至最优配置",
             GREEN, 10, 0.30)

    # Quick reference on the left
    ax.text(0.5, 4.5, "关键边界\n( DCU )", fontsize=10, color=GRAY, ha="left",
            fontweight="bold")
    refs = [
        "LDS: 64 KB / CU",
        "Register: 256 KB / CU",
        "Wave: 64 threads",
        "Block: <= 1024 threads",
        "LDS <= 32KB -> 2+ Block",
    ]
    for i, ref in enumerate(refs):
        ax.text(0.5, 3.8 - i * 0.55, f"  . {ref}", fontsize=8, color=GRAY, ha="left")

    save(fig, "tuning-decision.png")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Generating GPU concept figures...")
    fig_memory_hierarchy()
    fig_scheduling_hierarchy()
    fig_bank_conflict()
    fig_swizzle()
    fig_lds_constraint()
    fig_cu_utilization()
    fig_two_level_pipeline()
    fig_tuning_decision()
    print(f"\nDone! {len(os.listdir(OUTPUT_DIR))} figures in {OUTPUT_DIR}")
