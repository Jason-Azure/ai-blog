"""残差连接文章配图"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

plt.rcParams['font.family'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
OUT = os.path.expanduser('~/ai-blog/content/posts/residual-connection')
DPI = 100

# 图1: 梯度高速公路
def gen_01():
    fig, axes = plt.subplots(2, 1, figsize=(8, 5))

    # 上：普通网络梯度衰减
    ax = axes[0]
    ax.set_xlim(-0.5, 8); ax.set_ylim(-0.3, 1.5)
    ax.set_title('普通网络：梯度逐层衰减', fontsize=10, fontweight='bold', color='#C06050')
    ax.axis('off')
    layers = 7
    grad = 1.0
    for i in range(layers):
        x = i * 1.1
        c = plt.cm.Reds(0.3 + 0.7 * (1 - i/layers))
        rect = mpatches.FancyBboxPatch((x, 0.3), 0.8, 0.8,
            boxstyle="round,pad=0.1", facecolor=c, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x+0.4, 0.7, f'L{i+1}', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        grad *= 0.7
        bar_h = grad * 0.8
        ax.bar(x+0.4, bar_h, 0.3, bottom=-0.25, color='#C06050', alpha=0.6)
        if i < layers-1:
            ax.annotate('', xy=(x+1.0, 0.7), xytext=(x+0.9, 0.7),
                arrowprops=dict(arrowstyle='->', color='#ccc', lw=1))
    ax.text(8, 0.7, '梯度\n几乎\n消失', fontsize=8, color='#C06050', ha='center', fontweight='bold')

    # 下：残差网络梯度直达
    ax = axes[1]
    ax.set_xlim(-0.5, 8); ax.set_ylim(-0.3, 1.8)
    ax.set_title('残差网络：梯度有"高速公路"直达', fontsize=10, fontweight='bold', color='#50A068')
    ax.axis('off')
    for i in range(layers):
        x = i * 1.1
        c = plt.cm.Greens(0.3 + 0.5 * (1 - i/layers))
        rect = mpatches.FancyBboxPatch((x, 0.3), 0.8, 0.8,
            boxstyle="round,pad=0.1", facecolor=c, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x+0.4, 0.7, f'L{i+1}', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        bar_h = 0.7  # 梯度保持稳定
        ax.bar(x+0.4, bar_h, 0.3, bottom=-0.25, color='#50A068', alpha=0.6)
    # 高速公路箭头
    ax.annotate('', xy=(0.4, 1.5), xytext=(7.1, 1.5),
        arrowprops=dict(arrowstyle='<-', color='#E0A040', lw=3, linestyle='-'))
    ax.text(3.8, 1.65, '梯度高速公路 (skip connection)', ha='center', fontsize=9,
            color='#E0A040', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUT}/01_gradient_highway.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v 01")

# 图2: 残差流（河流 + 工厂）
def gen_02():
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_title('残差流：一条河流 + 沿河工厂', fontsize=12, fontweight='bold')

    # 河流（中间的粗线）
    river_x = 5
    ax.plot([river_x, river_x], [0.5, 11], color='#5080B0', lw=6, alpha=0.3, solid_capstyle='round')
    ax.text(river_x, 11.3, '残差流（Residual Stream）', ha='center', fontsize=10,
            color='#5080B0', fontweight='bold')

    # 起点
    ax.text(river_x, 10.5, 'tok_emb + pos_emb', ha='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E0A040', alpha=0.85, edgecolor='white'),
            color='white', fontweight='bold')

    # 工厂
    factories = [
        (9.0, 'Attention 1\n读取 → 处理 → 写入', '#8060A0'),
        (7.5, 'MLP 1\n读取 → 处理 → 写入', '#D09040'),
        (6.0, 'Attention 2\n读取 → 处理 → 写入', '#8060A0'),
        (4.5, 'MLP 2\n读取 → 处理 → 写入', '#D09040'),
        (3.0, '...× N 层...', '#999'),
    ]
    for y, text, color in factories:
        # 工厂方框
        ax.text(8.0, y, text, ha='center', va='center', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8, edgecolor='white'),
                color='white', fontweight='bold')
        # 双向箭头（读取 & 写入）
        ax.annotate('', xy=(6.5, y), xytext=(5.3, y),
            arrowprops=dict(arrowstyle='<->', color=color, lw=1.5, alpha=0.7))

    # 终点
    ax.text(river_x, 1.5, 'LayerNorm → 输出', ha='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#50A068', alpha=0.85, edgecolor='white'),
            color='white', fontweight='bold')

    # 关键标注
    ax.text(2.0, 6, '河流从不\n中断\n\n只有叠加\n没有覆盖', ha='center', fontsize=9,
            color='#5080B0', fontstyle='italic', linespacing=1.5)

    plt.tight_layout()
    plt.savefig(f'{OUT}/02_residual_stream.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v 02")

# 图3: y = F(x) + x 示意图
def gen_03():
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title('y = F(x) + x：残差连接的核心', fontsize=11, fontweight='bold')

    # 输入 x
    ax.text(1, 2, 'x', ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#5080B0', alpha=0.85, edgecolor='white'),
            color='white')

    # 分叉点
    ax.plot([1.6, 2.5], [2, 2], color='#888', lw=2)
    ax.plot([2.5, 2.5], [2, 3.2], color='#888', lw=2)  # 上：skip
    ax.plot([2.5, 2.5], [2, 0.8], color='#888', lw=2)  # 下：F(x)

    # 上路：skip connection
    ax.annotate('', xy=(7, 3.2), xytext=(2.5, 3.2),
        arrowprops=dict(arrowstyle='->', color='#E0A040', lw=2.5))
    ax.text(4.7, 3.5, 'skip connection（直接传递）', ha='center', fontsize=8, color='#E0A040')

    # 下路：F(x) 网络
    ax.text(4.7, 0.8, 'F(x)\n网络层', ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#C06050', alpha=0.8, edgecolor='white'),
            color='white', fontweight='bold')
    ax.plot([2.5, 3.5], [0.8, 0.8], color='#888', lw=2)
    ax.annotate('', xy=(7, 0.8), xytext=(5.9, 0.8),
        arrowprops=dict(arrowstyle='->', color='#C06050', lw=2))

    # 加号
    ax.text(7.3, 2, '+', ha='center', va='center', fontsize=20, fontweight='bold',
            color='#50A068',
            bbox=dict(boxstyle='circle,pad=0.2', facecolor='#E8F5E9', edgecolor='#50A068', lw=2))
    ax.plot([7, 7.1], [3.2, 2.4], color='#E0A040', lw=2)
    ax.plot([7, 7.1], [0.8, 1.6], color='#C06050', lw=2)

    # 输出 y
    ax.annotate('', xy=(9, 2), xytext=(7.7, 2),
        arrowprops=dict(arrowstyle='->', color='#888', lw=2))
    ax.text(9.3, 2, 'y', ha='center', va='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#50A068', alpha=0.85, edgecolor='white'),
            color='white')

    ax.text(5, -0.1, 'y = F(x) + x', ha='center', fontsize=12, fontweight='bold', color='#333')

    plt.tight_layout()
    plt.savefig(f'{OUT}/03_residual_block.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v 03")

# 图4: 2^n 条路径
def gen_04():
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    ax.axis('off')
    ax.set_title('3 个残差块 = 2³ = 8 条路径', fontsize=11, fontweight='bold')

    # 画 8 条路径（简化为 4 条有代表性的）
    paths = [
        ('skip → skip → skip', '#E0A040', 3.8, '0 层深'),
        ('F₁ → skip → skip', '#5080B0', 2.8, '1 层深'),
        ('F₁ → F₂ → skip', '#8060A0', 1.8, '2 层深'),
        ('F₁ → F₂ → F₃', '#C06050', 0.8, '3 层深'),
    ]

    for text, color, y, depth in paths:
        ax.text(0.5, y, text, fontsize=9, color=color, fontweight='bold', va='center')
        # 深度条
        bar_w = float(depth[0]) * 1.5 + 0.3
        ax.barh(y, bar_w, 0.4, left=6, color=color, alpha=0.5)
        ax.text(6 + bar_w + 0.2, y, depth, fontsize=8, va='center', color=color)

    ax.text(5, 4.5, '大部分梯度来自\n10-34 层深的路径', ha='center', fontsize=9,
            color='#C06050', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8E1', edgecolor='#E0A040'))

    plt.tight_layout()
    plt.savefig(f'{OUT}/04_paths.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v 04")

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    gen_01(); gen_02(); gen_03(); gen_04()
    print("完成！")
