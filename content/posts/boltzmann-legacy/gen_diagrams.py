#!/usr/bin/env python3
"""
Generate diagrams for 玻尔兹曼的遗产 (boltzmann-legacy) blog post.
6 PNGs + 1 GIF

Usage:
    source ~/ai-lab-venv/bin/activate
    cd ~/ai-blog/content/posts/boltzmann-legacy
    python3 gen_diagrams.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec

# ── Font Setup ──────────────────────────────────────────────
_cjk_font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if os.path.exists(_cjk_font_path):
    _cjk_font = fm.FontProperties(fname=_cjk_font_path)
    fm.fontManager.addfont(_cjk_font_path)
    plt.rcParams['font.family'] = _cjk_font.get_name()
plt.rcParams['font.size'] = 13
plt.rcParams['axes.unicode_minus'] = False

# ── Color Palette (Material Design) ────────────────────────
C = {
    'orange': '#FF9800', 'blue': '#2196F3', 'green': '#4CAF50',
    'purple': '#9C27B0', 'red': '#F44336', 'pink': '#E91E63',
    'teal': '#009688',   'grey': '#607D8B', 'amber': '#FFC107',
    'bg': '#FFFFFF',     'dark': '#333333', 'mid': '#666666',
    'light_bg': '#F5F5F5',
}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ════════════════════════════════════════════════════════════
# 1. boltzmann_distribution.png
#    玻尔兹曼分布在不同温度下的曲线
# ════════════════════════════════════════════════════════════
def gen_boltzmann_distribution():
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=C['bg'])

    E = np.linspace(0, 10, 300)
    temps = [0.5, 1.0, 2.0, 5.0]
    colors = [C['blue'], C['green'], C['orange'], C['red']]
    labels = ['T = 0.5 (极冷)', 'T = 1.0 (冷)', 'T = 2.0 (暖)', 'T = 5.0 (热)']

    for T, color, label in zip(temps, colors, labels):
        P = np.exp(-E / T)
        P = P / (np.sum(P) * (E[1] - E[0]))  # normalize
        ax.plot(E, P, color=color, linewidth=2.5, label=label)
        ax.fill_between(E, P, alpha=0.08, color=color)

    ax.set_xlabel('能量 E', fontsize=14, color=C['dark'])
    ax.set_ylabel('概率密度 P(E)', fontsize=14, color=C['dark'])
    ax.set_title('玻尔兹曼分布：温度越高，分布越平坦', fontsize=16,
                 fontweight='bold', color=C['dark'], pad=15)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2)

    # Annotation
    ax.annotate('温度越低 → 粒子挤在低能态\n温度越高 → 粒子分散到高能态',
                xy=(6, 0.15), fontsize=11, color=C['mid'],
                bbox=dict(boxstyle='round,pad=0.5', facecolor=C['light_bg'],
                          edgecolor=C['grey'], alpha=0.8))

    path = os.path.join(OUT_DIR, 'boltzmann_distribution.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=C['bg'])
    plt.close(fig)
    print(f"OK: {path}")


# ════════════════════════════════════════════════════════════
# 2. softmax_temperature.png
#    Softmax 在不同 Temperature 下的输出对比
# ════════════════════════════════════════════════════════════
def gen_softmax_temperature():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=C['bg'])
    fig.suptitle('Softmax 温度效应：同一组 logits，不同 Temperature',
                 fontsize=16, fontweight='bold', color=C['dark'], y=1.02)

    logits = np.array([2.0, 1.5, 0.8, 0.3, -0.5])
    words = ['的', '了', '是', '在', '有']
    temps = [0.3, 1.0, 3.0]
    titles = ['T = 0.3（冰冷·确定）', 'T = 1.0（标准）', 'T = 3.0（炽热·随机）']
    colors_list = [C['blue'], C['green'], C['red']]

    for ax, T, title, color in zip(axes, temps, titles, colors_list):
        exp_vals = np.exp(logits / T)
        probs = exp_vals / exp_vals.sum()

        bars = ax.bar(words, probs, color=color, alpha=0.8, edgecolor=color, linewidth=1.5)
        ax.set_title(title, fontsize=13, fontweight='bold', color=color, pad=10)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('概率' if ax == axes[0] else '', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.2)

        # Add value labels
        for bar, p in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{p:.1%}', ha='center', va='bottom', fontsize=10,
                    fontweight='bold', color=color)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'softmax_temperature.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=C['bg'])
    plt.close(fig)
    print(f"OK: {path}")


# ════════════════════════════════════════════════════════════
# 3. entropy_formula.png
#    S = k ln W 的视觉拆解
# ════════════════════════════════════════════════════════════
def gen_entropy_formula():
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=C['bg'])
    ax.axis('off')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)

    # Title
    ax.text(6, 6.5, 'S = k ln W  —— 逐字翻译', fontsize=22,
            fontweight='bold', ha='center', color=C['dark'])

    # Main formula in center
    ax.text(6, 5.2, 'S  =  k  ln  W', fontsize=36, fontweight='bold',
            ha='center', va='center', color=C['pink'],
            fontfamily='serif')

    # Breakdown boxes
    items = [
        (1.5, 3.5, 'S', '熵\nEntropy', '系统的"混乱度"\n越大 = 越无序', C['orange']),
        (3.8, 3.5, '=', '', '', C['mid']),
        (5.0, 3.5, 'k', '玻尔兹曼常数', '1.38e-23 J/K\n把微观和宏观连接的桥', C['blue']),
        (7.8, 3.5, 'ln', '自然对数', '为什么用 ln？\n因为要让熵具有\n"可加性"', C['green']),
        (10.5, 3.5, 'W', '微观状态数', '有多少种微观排列\n对应同一个宏观状态\n（越多 = 越可能）', C['purple']),
    ]

    for x, y, symbol, name, desc, color in items:
        if symbol == '=':
            ax.text(x, y, symbol, fontsize=28, ha='center', va='center',
                    color=C['mid'], fontfamily='serif')
            continue

        # Box
        box_w, box_h = 2.2, 2.8
        box = FancyBboxPatch((x - box_w/2, y - box_h/2), box_w, box_h,
                             boxstyle="round,pad=0.15", facecolor=color,
                             alpha=0.08, edgecolor=color, linewidth=2)
        ax.add_patch(box)

        # Symbol
        ax.text(x, y + 0.8, symbol, fontsize=24, fontweight='bold',
                ha='center', va='center', color=color, fontfamily='serif')
        # Name
        ax.text(x, y + 0.1, name, fontsize=11, fontweight='bold',
                ha='center', va='center', color=C['dark'])
        # Description
        ax.text(x, y - 0.7, desc, fontsize=9, ha='center', va='center',
                color=C['mid'], linespacing=1.4)

    # Bottom insight
    insight_box = FancyBboxPatch((1.5, 0.2), 9, 1.0,
                                boxstyle="round,pad=0.2",
                                facecolor=C['orange'], alpha=0.08,
                                edgecolor=C['orange'], linewidth=2)
    ax.add_patch(insight_box)
    ax.text(6, 0.7, '核心洞察：熵 = 对"有多少种微观可能性"取对数。可能性越多，熵越大，系统越可能处于那个状态。',
            fontsize=11, ha='center', va='center', color=C['dark'],
            style='italic')

    path = os.path.join(OUT_DIR, 'entropy_formula.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=C['bg'])
    plt.close(fig)
    print(f"OK: {path}")


# ════════════════════════════════════════════════════════════
# 4. seven_faces.png
#    e^(-E/kT) 在七个领域中的对照
# ════════════════════════════════════════════════════════════
def gen_seven_faces():
    fig, ax = plt.subplots(figsize=(14, 9), facecolor=C['bg'])
    ax.axis('off')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)

    ax.text(7, 8.5, r'$e^{-E/kT}$ 的七张面孔', fontsize=22,
            fontweight='bold', ha='center', color=C['dark'])

    # Center formula
    center_box = FancyBboxPatch((4.5, 4.8), 5, 1.2,
                                boxstyle="round,pad=0.2",
                                facecolor=C['pink'], alpha=0.1,
                                edgecolor=C['pink'], linewidth=3)
    ax.add_patch(center_box)
    ax.text(7, 5.4, r'$e^{-E/kT}$', fontsize=28, fontweight='bold',
            ha='center', va='center', color=C['pink'])

    # Seven faces arranged around center
    faces = [
        (2.0, 7.5, '统计力学', '粒子能量分布', C['blue']),
        (7.0, 7.5, '化学反应', '阿伦尼乌斯方程', C['green']),
        (12.0, 7.5, '半导体', '载流子浓度', C['purple']),
        (1.0, 3.5, '神经网络', 'Softmax 分类', C['orange']),
        (5.0, 2.0, '生成模型', '扩散·去噪', C['teal']),
        (9.0, 2.0, '优化算法', '模拟退火', C['red']),
        (13.0, 3.5, '信息论', '最大熵分布', C['grey']),
    ]

    for x, y, name, desc, color in faces:
        box_w, box_h = 2.5, 1.3
        box = FancyBboxPatch((x - box_w/2, y - box_h/2), box_w, box_h,
                             boxstyle="round,pad=0.15",
                             facecolor=color, alpha=0.1,
                             edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y + 0.2, name, fontsize=12, fontweight='bold',
                ha='center', va='center', color=color)
        ax.text(x, y - 0.25, desc, fontsize=9, ha='center', va='center',
                color=C['mid'], linespacing=1.3)

        # Arrow to center
        cx, cy = 7, 5.4
        # Compute arrow start (from face box edge toward center)
        dx = cx - x
        dy = cy - y
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0:
            # Start from box edge
            sx = x + dx / dist * (box_w / 2 + 0.1)
            sy = y + dy / dist * (box_h / 2 + 0.1)
            # End near center box edge
            ex = cx - dx / dist * 2.7
            ey = cy - dy / dist * 0.7
            ax.annotate('', xy=(ex, ey), xytext=(sx, sy),
                       arrowprops=dict(arrowstyle='->', color=color,
                                       lw=1.5, alpha=0.5,
                                       connectionstyle='arc3,rad=0'))

    # Bottom text
    ax.text(7, 0.5, '同一个数学结构，七个不同领域。因为它不是物理定律——它是"约束下最大化不确定性"的数学必然。',
            fontsize=11, ha='center', va='center', color=C['dark'],
            style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=C['orange'],
                      alpha=0.08, edgecolor=C['orange'], linewidth=2))

    path = os.path.join(OUT_DIR, 'seven_faces.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=C['bg'])
    plt.close(fig)
    print(f"OK: {path}")


# ════════════════════════════════════════════════════════════
# 5. shannon_boltzmann.png
#    热力学熵与信息熵的对照
# ════════════════════════════════════════════════════════════
def gen_shannon_boltzmann():
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=C['bg'])
    ax.axis('off')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)

    ax.text(6, 7.5, '两种"熵"——同一个数学灵魂', fontsize=20,
            fontweight='bold', ha='center', color=C['dark'])

    # Left: Boltzmann
    left_box = FancyBboxPatch((0.5, 2.5), 5, 4.5,
                               boxstyle="round,pad=0.2",
                               facecolor=C['blue'], alpha=0.06,
                               edgecolor=C['blue'], linewidth=2)
    ax.add_patch(left_box)

    ax.text(3, 6.5, '热力学熵', fontsize=16, fontweight='bold',
            ha='center', color=C['blue'])
    ax.text(3, 5.9, 'Ludwig Boltzmann · 1870s', fontsize=11,
            ha='center', color=C['mid'])
    ax.text(3, 5.1, r'$S = -k \sum p_i \ln p_i$', fontsize=18,
            ha='center', color=C['blue'],
            fontweight='bold')

    left_items = [
        ('度量对象', '分子排列的无序度'),
        ('概率 pi', '粒子处于状态 i 的概率'),
        ('单位', '焦耳/开尔文 (J/K)'),
        ('常数', 'k = 1.38e-23'),
    ]
    for i, (key, val) in enumerate(left_items):
        y = 4.2 - i * 0.5
        ax.text(1.0, y, f'> {key}', fontsize=10, fontweight='bold', color=C['dark'])
        ax.text(2.8, y, val, fontsize=10, color=C['mid'])

    # Right: Shannon
    right_box = FancyBboxPatch((6.5, 2.5), 5, 4.5,
                                boxstyle="round,pad=0.2",
                                facecolor=C['orange'], alpha=0.06,
                                edgecolor=C['orange'], linewidth=2)
    ax.add_patch(right_box)

    ax.text(9, 6.5, '信息熵', fontsize=16, fontweight='bold',
            ha='center', color=C['orange'])
    ax.text(9, 5.9, 'Claude Shannon · 1948', fontsize=11,
            ha='center', color=C['mid'])
    ax.text(9, 5.1, r'$H = -\sum p_i \log_2 p_i$', fontsize=18,
            ha='center', color=C['orange'],
            fontweight='bold')

    right_items = [
        ('度量对象', '信息的不确定性'),
        ('概率 pi', '符号 i 出现的概率'),
        ('单位', '比特 (bit)'),
        ('常数', '1 (无量纲)'),
    ]
    for i, (key, val) in enumerate(right_items):
        y = 4.2 - i * 0.5
        ax.text(7.0, y, f'> {key}', fontsize=10, fontweight='bold', color=C['dark'])
        ax.text(8.8, y, val, fontsize=10, color=C['mid'])

    # Center: equals sign
    eq_box = FancyBboxPatch((5.3, 4.2), 1.4, 1.0,
                             boxstyle="round,pad=0.15",
                             facecolor=C['pink'], alpha=0.15,
                             edgecolor=C['pink'], linewidth=2)
    ax.add_patch(eq_box)
    ax.text(6, 4.7, '≡', fontsize=32, ha='center', va='center',
            color=C['pink'], fontweight='bold')

    # Bottom connection
    conn_box = FancyBboxPatch((1.5, 0.3), 9, 1.8,
                               boxstyle="round,pad=0.2",
                               facecolor=C['green'], alpha=0.06,
                               edgecolor=C['green'], linewidth=2)
    ax.add_patch(conn_box)
    ax.text(6, 1.6, '在 AI 中汇合', fontsize=14, fontweight='bold',
            ha='center', color=C['green'])
    ax.text(6, 1.0, '交叉熵损失（训练）← 香农     玻尔兹曼 → Softmax（推理）',
            fontsize=11, ha='center', color=C['dark'])
    ax.text(6, 0.6, '一个定义了 AI 如何学习，一个定义了 AI 如何选择',
            fontsize=10, ha='center', color=C['mid'], style='italic')

    path = os.path.join(OUT_DIR, 'shannon_boltzmann.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=C['bg'])
    plt.close(fig)
    print(f"OK: {path}")


# ════════════════════════════════════════════════════════════
# 6. tombstone.png
#    玻尔兹曼墓碑 S = k ln W 示意
# ════════════════════════════════════════════════════════════
def gen_tombstone():
    fig, ax = plt.subplots(figsize=(8, 10), facecolor='#F0EDE4')
    ax.axis('off')
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 10)

    # Tombstone shape - arch top
    from matplotlib.patches import Arc, Rectangle

    # Base rectangle
    stone_color = '#D4CFC5'
    stone_edge = '#8B8578'
    rect = Rectangle((1.5, 1), 5, 6, facecolor=stone_color,
                      edgecolor=stone_edge, linewidth=3)
    ax.add_patch(rect)

    # Arch top
    theta = np.linspace(0, np.pi, 100)
    arch_x = 4 + 2.5 * np.cos(theta)
    arch_y = 7 + 2.5 * np.sin(theta)
    # Fill arch
    arch_fill_x = np.concatenate([[1.5], arch_x, [6.5]])
    arch_fill_y = np.concatenate([[7], arch_y, [7]])
    ax.fill(arch_fill_x, arch_fill_y, color=stone_color)
    ax.plot(arch_x, arch_y, color=stone_edge, linewidth=3)
    ax.plot([1.5, 1.5], [1, 7], color=stone_edge, linewidth=3)
    ax.plot([6.5, 6.5], [1, 7], color=stone_edge, linewidth=3)
    ax.plot([1.5, 6.5], [1, 1], color=stone_edge, linewidth=3)

    # Name
    ax.text(4, 8.2, 'LUDWIG', fontsize=16, ha='center', va='center',
            color='#4A4540', fontweight='bold', fontfamily='serif',
            style='italic')
    ax.text(4, 7.5, 'BOLTZMANN', fontsize=18, ha='center', va='center',
            color='#4A4540', fontweight='bold', fontfamily='serif')

    # Dates
    ax.text(4, 6.6, '1844 — 1906', fontsize=13, ha='center', va='center',
            color='#6B6560', fontfamily='serif')

    # Divider line
    ax.plot([2.3, 5.7], [6.2, 6.2], color=stone_edge, linewidth=1, alpha=0.5)

    # THE FORMULA - the centerpiece
    ax.text(4, 5.0, 'S = k ln W', fontsize=32, ha='center', va='center',
            color='#3A3530', fontweight='bold', fontfamily='serif',
            style='italic')

    # Decorative line below formula
    ax.plot([2.3, 5.7], [4.0, 4.0], color=stone_edge, linewidth=1, alpha=0.5)

    # Subtitle
    ax.text(4, 3.2, '他的墓碑上没有生卒年月的赞词\n只有一个公式', fontsize=11,
            ha='center', va='center', color='#6B6560', linespacing=1.5)

    # Ground
    ax.fill_between([0, 8], [1, 1], [0, 0], color='#8B9E6B', alpha=0.4)
    ax.fill_between([0, 8], [0.5, 0.5], [0, 0], color='#6B8E4B', alpha=0.3)

    # Caption below
    ax.text(4, 0.2, '维也纳中央公墓 · Zentralfriedhof Wien',
            fontsize=10, ha='center', va='center', color='#6B6560',
            style='italic')

    path = os.path.join(OUT_DIR, 'tombstone.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='#F0EDE4')
    plt.close(fig)
    print(f"OK: {path}")


# ════════════════════════════════════════════════════════════
# 7. temperature_animation.gif
#    温度从高到低，分布从平坦到尖锐的动画
# ════════════════════════════════════════════════════════════
def gen_temperature_gif():
    from matplotlib.animation import FuncAnimation, PillowWriter

    words = ['的', '了', '是', '在', '有', '不', '人', '这']
    logits = np.array([3.0, 2.2, 1.8, 1.2, 0.5, 0.2, -0.3, -1.0])

    # Temperature schedule: high → low
    n_frames = 60
    temps = np.concatenate([
        np.linspace(5.0, 0.2, 50),   # cooling
        np.full(10, 0.2),              # hold at cold
    ])

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=C['bg'])

    def update(frame):
        ax.clear()
        T = temps[frame]

        exp_vals = np.exp(logits / T)
        probs = exp_vals / exp_vals.sum()

        # Color intensity based on temperature
        if T > 2.0:
            bar_color = C['red']
            temp_label = '高温 · 随机'
        elif T > 0.8:
            bar_color = C['orange']
            temp_label = '中温 · 平衡'
        else:
            bar_color = C['blue']
            temp_label = '低温 · 确定'

        bars = ax.bar(words, probs, color=bar_color, alpha=0.8,
                      edgecolor=bar_color, linewidth=1.5)

        ax.set_ylim(0, 1.05)
        ax.set_title(f'Temperature = {T:.1f}  ({temp_label})',
                     fontsize=16, fontweight='bold', color=C['dark'], pad=10)
        ax.set_ylabel('选择概率', fontsize=13, color=C['dark'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.2)

        # Value labels on top bars
        for bar, p in zip(bars, probs):
            if p > 0.03:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f'{p:.0%}', ha='center', va='bottom',
                        fontsize=9, color=bar_color, fontweight='bold')

        # Physics analogy at bottom
        ax.text(0.98, 0.95, f'T={T:.1f}',
                transform=ax.transAxes, fontsize=20,
                ha='right', va='top', color=bar_color,
                fontweight='bold', alpha=0.3)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=100)

    path = os.path.join(OUT_DIR, 'temperature_animation.gif')
    anim.save(path, writer=PillowWriter(fps=10), dpi=100)
    plt.close(fig)
    print(f"OK: {path}")


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating diagrams for 玻尔兹曼的遗产...")
    print("=" * 50)

    gen_boltzmann_distribution()
    gen_softmax_temperature()
    gen_entropy_formula()
    gen_seven_faces()
    gen_shannon_boltzmann()
    gen_tombstone()
    gen_temperature_gif()

    print("=" * 50)
    print("All diagrams generated successfully!")
