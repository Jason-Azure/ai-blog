"""残差连接配图 v2 — 色彩优化 + 残差流动画"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
import os

plt.rcParams['font.family'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
OUT = os.path.expanduser('~/ai-blog/content/posts/residual-connection')
DPI = 100
SLOW = 250  # 慢速动画

# 柔和色板
C_BLUE = '#6A9EC7'
C_RED = '#C47A6E'
C_GREEN = '#6AAF78'
C_PURPLE = '#9A7DBF'
C_ORANGE = '#D4A55A'
C_GRAY = '#A0A0A0'

# ============================================================
# 图1: 梯度高速公路（缩小，色彩柔和）
# ============================================================
def gen_01():
    fig, axes = plt.subplots(2, 1, figsize=(7, 4))

    # 上：普通网络
    ax = axes[0]
    ax.set_xlim(-0.3, 7.5); ax.set_ylim(-0.4, 1.3)
    ax.set_title('普通网络：梯度逐层衰减', fontsize=9, fontweight='bold', color=C_RED)
    ax.axis('off')
    layers = 6; grad = 1.0
    for i in range(layers):
        x = i * 1.2
        alpha = 0.4 + 0.5 * (1 - i/layers)
        rect = mpatches.FancyBboxPatch((x, 0.3), 0.7, 0.6,
            boxstyle="round,pad=0.08", facecolor=C_RED, alpha=alpha)
        ax.add_patch(rect)
        ax.text(x+0.35, 0.6, f'L{i+1}', ha='center', va='center', fontsize=7, color='white', fontweight='bold')
        grad *= 0.65
        ax.bar(x+0.35, grad*0.5, 0.25, bottom=-0.3, color=C_RED, alpha=0.4)
        if i < layers-1:
            ax.annotate('', xy=(x+0.9, 0.6), xytext=(x+0.8, 0.6),
                arrowprops=dict(arrowstyle='->', color='#ddd', lw=1))
    ax.text(7.2, 0.2, '梯度\n消失', fontsize=7, color=C_RED, ha='center')

    # 下：残差网络
    ax = axes[1]
    ax.set_xlim(-0.3, 7.5); ax.set_ylim(-0.4, 1.5)
    ax.set_title('残差网络：梯度有"高速公路"', fontsize=9, fontweight='bold', color=C_GREEN)
    ax.axis('off')
    for i in range(layers):
        x = i * 1.2
        rect = mpatches.FancyBboxPatch((x, 0.3), 0.7, 0.6,
            boxstyle="round,pad=0.08", facecolor=C_GREEN, alpha=0.6)
        ax.add_patch(rect)
        ax.text(x+0.35, 0.6, f'L{i+1}', ha='center', va='center', fontsize=7, color='white', fontweight='bold')
        ax.bar(x+0.35, 0.45, 0.25, bottom=-0.3, color=C_GREEN, alpha=0.4)
    ax.annotate('', xy=(0.35, 1.2), xytext=(6.35, 1.2),
        arrowprops=dict(arrowstyle='<-', color=C_ORANGE, lw=2.5))
    ax.text(3.3, 1.35, '梯度高速公路 (skip connection)', ha='center', fontsize=8, color=C_ORANGE, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUT}/01_gradient_highway.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v2 01")

# ============================================================
# 图2: 残差流静态图（缩小，色彩柔和）
# ============================================================
def gen_02():
    fig, ax = plt.subplots(figsize=(5, 7))
    ax.set_xlim(0, 10); ax.set_ylim(0, 11)
    ax.axis('off')

    river_x = 4.5
    ax.plot([river_x, river_x], [0.8, 10], color=C_BLUE, lw=5, alpha=0.25, solid_capstyle='round')
    ax.text(river_x, 10.3, '残差流', ha='center', fontsize=10, color=C_BLUE, fontweight='bold')

    ax.text(river_x, 9.5, 'tok + pos', ha='center', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.25', facecolor=C_ORANGE, alpha=0.8, edgecolor='white'),
            color='white', fontweight='bold')

    factories = [
        (8.2, 'Attn 1', C_PURPLE),
        (6.8, 'MLP 1', C_ORANGE),
        (5.4, 'Attn 2', C_PURPLE),
        (4.0, 'MLP 2', C_ORANGE),
        (2.6, '... x N', C_GRAY),
    ]
    for y, text, color in factories:
        ax.text(7.5, y, text, ha='center', va='center', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.25', facecolor=color, alpha=0.75, edgecolor='white'),
                color='white', fontweight='bold')
        ax.annotate('', xy=(5.8, y), xytext=(5.0, y),
            arrowprops=dict(arrowstyle='<->', color=color, lw=1.2, alpha=0.6))

    ax.text(river_x, 1.5, '输出', ha='center', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.25', facecolor=C_GREEN, alpha=0.8, edgecolor='white'),
            color='white', fontweight='bold')

    ax.text(1.8, 5.5, '河流不中断\n只叠加\n不覆盖', ha='center', fontsize=8, color=C_BLUE, fontstyle='italic')

    plt.tight_layout()
    plt.savefig(f'{OUT}/02_residual_stream.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v2 02")

# ============================================================
# 图3: y = F(x) + x 示意图（缩小，柔和）
# ============================================================
def gen_03():
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    ax.axis('off')

    # 输入
    ax.text(1, 2, 'x', ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor=C_BLUE, alpha=0.8, edgecolor='white'), color='white')

    ax.plot([1.55, 2.3], [2, 2], color='#bbb', lw=1.5)
    ax.plot([2.3, 2.3], [2, 3.1], color='#bbb', lw=1.5)
    ax.plot([2.3, 2.3], [2, 0.9], color='#bbb', lw=1.5)

    # skip
    ax.annotate('', xy=(6.8, 3.1), xytext=(2.3, 3.1),
        arrowprops=dict(arrowstyle='->', color=C_ORANGE, lw=2))
    ax.text(4.5, 3.35, 'skip（直接传递）', ha='center', fontsize=7, color=C_ORANGE)

    # F(x)
    ax.text(4.5, 0.9, 'F(x)', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.35', facecolor=C_RED, alpha=0.75, edgecolor='white'),
            color='white', fontweight='bold')
    ax.plot([2.3, 3.4], [0.9, 0.9], color='#bbb', lw=1.5)
    ax.annotate('', xy=(6.8, 0.9), xytext=(5.6, 0.9),
        arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.5))

    # +
    ax.text(7.1, 2, '+', ha='center', va='center', fontsize=16, fontweight='bold', color=C_GREEN,
            bbox=dict(boxstyle='circle,pad=0.15', facecolor='#E8F5E9', edgecolor=C_GREEN, lw=1.5))
    ax.plot([6.85, 6.95], [3.1, 2.35], color=C_ORANGE, lw=1.5)
    ax.plot([6.85, 6.95], [0.9, 1.65], color=C_RED, lw=1.5)

    # 输出
    ax.annotate('', xy=(8.7, 2), xytext=(7.5, 2),
        arrowprops=dict(arrowstyle='->', color='#bbb', lw=1.5))
    ax.text(9, 2, 'y', ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor=C_GREEN, alpha=0.8, edgecolor='white'), color='white')

    ax.text(5, -0.05, 'y = F(x) + x', ha='center', fontsize=11, fontweight='bold', color='#555')

    plt.tight_layout()
    plt.savefig(f'{OUT}/03_residual_block.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v2 03")

# ============================================================
# 图4: 2^n 条路径（缩小）
# ============================================================
def gen_04():
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('3 个残差块 = 2³ = 8 条路径', fontsize=10, fontweight='bold')

    paths = [
        ('skip → skip → skip', C_ORANGE, 3.8, '0 层深'),
        ('F₁ → skip → skip', C_BLUE, 2.8, '1 层深'),
        ('F₁ → F₂ → skip', C_PURPLE, 1.8, '2 层深'),
        ('F₁ → F₂ → F₃', C_RED, 0.8, '3 层深'),
    ]
    for text, color, y, depth in paths:
        ax.text(0.3, y, text, fontsize=8, color=color, fontweight='bold', va='center')
        bar_w = float(depth[0]) * 1.2 + 0.3
        ax.barh(y, bar_w, 0.35, left=5.5, color=color, alpha=0.45)
        ax.text(5.5 + bar_w + 0.15, y, depth, fontsize=7, va='center', color=color)

    ax.text(5, 4.5, '大部分梯度来自较浅的路径', ha='center', fontsize=8,
            color=C_RED, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFF8E1', edgecolor=C_ORANGE))
    plt.tight_layout()
    plt.savefig(f'{OUT}/04_paths.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v2 04")

# ============================================================
# 图5: 残差流动画（新增！慢速）
# ============================================================
def gen_05_animation():
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_xlim(0, 10); ax.set_ylim(-0.5, 11)
    ax.axis('off')
    ax.set_title('数据在残差流中的旅程', fontsize=11, fontweight='bold')

    # 固定元素：河流背景
    river_x = 4.5
    ax.plot([river_x, river_x], [0.3, 10.2], color=C_BLUE, lw=5, alpha=0.15, solid_capstyle='round')

    # 节点位置和标签
    nodes = [
        (9.8, 'tok + pos', C_ORANGE),
        (8.4, 'Attention 1', C_PURPLE),
        (7.0, 'MLP 1', C_ORANGE),
        (5.6, 'Attention 2', C_PURPLE),
        (4.2, 'MLP 2', C_ORANGE),
        (2.8, 'Attention N', C_PURPLE),
        (1.4, 'MLP N', C_ORANGE),
        (0.3, '输出', C_GREEN),
    ]

    # 画固定的工厂标签
    for y, text, color in nodes:
        if text in ('tok + pos', '输出'):
            ax.text(river_x, y, text, ha='center', va='center', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor=color, alpha=0.8, edgecolor='white'),
                    color='white', fontweight='bold')
        elif text == 'Attention N':
            ax.text(7.5, y+0.35, '...', ha='center', fontsize=12, color='#aaa')
            ax.text(7.5, y, text, ha='center', va='center', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7, edgecolor='white'),
                    color='white', fontweight='bold')
        else:
            ax.text(7.5, y, text, ha='center', va='center', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7, edgecolor='white'),
                    color='white', fontweight='bold')

    # 动画元素
    # 数据球
    ball, = ax.plot([], [], 'o', color=C_BLUE, markersize=12, zorder=10, alpha=0.9)
    ball_glow, = ax.plot([], [], 'o', color=C_BLUE, markersize=18, zorder=9, alpha=0.3)
    # 累积信息的文字
    info_text = ax.text(1.5, 5, '', fontsize=7, color=C_BLUE, fontstyle='italic',
                        va='top', linespacing=1.4)
    # 当前操作标签
    action_text = ax.text(river_x, -0.3, '', ha='center', fontsize=9, fontweight='bold', color='#555')

    # 动画阶段
    stages = [
        (9.8, '初始化残差流', '语义 + 位置'),
        (9.1, '→ 流向 Attention 1', '语义 + 位置'),
        (8.4, 'Attention 1 读取 & 写入', '语义 + 位置\n+ 上下文关系₁'),
        (7.7, '→ 流向 MLP 1', '语义 + 位置\n+ 上下文关系₁'),
        (7.0, 'MLP 1 读取 & 写入', '语义 + 位置\n+ 上下文关系₁\n+ 世界知识₁'),
        (6.3, '→ 流向 Attention 2', '语义 + 位置\n+ 上下文关系₁\n+ 世界知识₁'),
        (5.6, 'Attention 2 读取 & 写入', '语义 + 位置\n+ 上下文关系₁₂\n+ 世界知识₁'),
        (4.9, '→ 流向 MLP 2', '语义 + 位置\n+ 上下文关系₁₂\n+ 世界知识₁'),
        (4.2, 'MLP 2 读取 & 写入', '语义 + 位置\n+ 上下文关系₁₂\n+ 世界知识₁₂'),
        (3.5, '→ ... 继续流过 N 层 ...', '语义 + 位置\n+ 上下文关系₁₂...N\n+ 世界知识₁₂...N'),
        (2.8, 'Attention N', '(信息越来越丰富)'),
        (2.1, '→ 流向 MLP N', ''),
        (1.4, 'MLP N', ''),
        (0.8, '→ 准备输出', ''),
        (0.3, '输出下一个词！', '残差流中的全部信息\n→ 投影到词表'),
    ]

    total_frames = len(stages) * 4 + 10  # 每阶段停留 4 帧，结尾停留 10 帧

    def animate(frame):
        stage_idx = min(frame // 4, len(stages) - 1)
        y, action, info = stages[stage_idx]

        ball.set_data([river_x], [y])
        ball_glow.set_data([river_x], [y])
        action_text.set_text(action)
        if info:
            info_text.set_text(info)
            info_text.set_position((1.0, y + 0.3))

        return ball, ball_glow, action_text, info_text

    anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                   interval=SLOW, blit=False, repeat=True)
    plt.tight_layout()
    anim.save(f'{OUT}/05_residual_flow.gif', writer='pillow', fps=4, dpi=DPI)
    plt.close()
    print("v2 05 animation")

if __name__ == '__main__':
    gen_01(); gen_02(); gen_03(); gen_04(); gen_05_animation()
    print("全部完成！")
