#!/usr/bin/env python3
"""Figures for 世界模型之争.
Outputs:
  - arch_compare.png  (Transformer vs JEPA architecture contrast)
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.expanduser("~/ai-blog/content/posts/world-model-debate")
os.makedirs(OUT, exist_ok=True)


def _box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=11, fontweight='normal', textcolor='#222'):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.05",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=1.6)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, color=textcolor)


def _arrow(ax, x1, y1, x2, y2, color='#555', style='->'):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle=style, color=color,
                          mutation_scale=16, linewidth=1.6)
    ax.add_patch(arr)


def make_arch_compare():
    print("→ arch_compare.png ...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    fig.patch.set_facecolor('white')
    fig.suptitle('两种架构的根本分歧：要不要预测每一个细节？',
                 fontsize=15, fontweight='bold', y=1.00)

    # --- Left: Transformer ---
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Transformer（GPT / Sora 路线）\n目标：预测下一个 token / pixel',
                 fontsize=12, color='#1565c0', pad=10)

    _box(ax, 1.0, 8.0, 8.0, 1.0, '输入序列：「今天 天气 很」',
         '#e3f2fd', '#1565c0', fontsize=11)
    _arrow(ax, 5.0, 8.0, 5.0, 7.2)
    _box(ax, 1.5, 5.5, 7.0, 1.6, 'Transformer\n（自注意力 + FFN）',
         '#bbdefb', '#1565c0', fontsize=12, fontweight='bold')
    _arrow(ax, 5.0, 5.5, 5.0, 4.7)
    _box(ax, 0.6, 3.3, 8.8, 1.2,
         '预测下一个 token 的完整概率分布\nP("好"|...) P("热"|...) P("冷"|...) ...',
         '#fff8e1', '#e0b030', fontsize=10.5)

    _box(ax, 0.5, 0.6, 9.0, 1.8,
         '代价：模型必须精确预测每一个 token / 像素。\n'
         '下一帧叶子的每一片、镜头边角的噪点、杯子的每一道反光——都要对。\n'
         'LeCun 批评：海量模型容量花在不重要的细节上。',
         '#ffebee', '#c0392b', fontsize=10)

    # --- Right: JEPA ---
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('JEPA（LeCun 的替代方案）\n目标：在抽象表示空间里预测',
                 fontsize=12, color='#2e7d32', pad=10)

    _box(ax, 1.0, 8.0, 8.0, 1.0, '输入：图像 / 视频片段',
         '#e8f5e9', '#2e7d32', fontsize=11)
    _arrow(ax, 5.0, 8.0, 5.0, 7.2)
    _box(ax, 1.5, 5.7, 7.0, 1.4, 'Encoder（抽象压缩）',
         '#c8e6c9', '#2e7d32', fontsize=12, fontweight='bold')
    _arrow(ax, 5.0, 5.7, 5.0, 5.0)
    _box(ax, 1.5, 3.5, 7.0, 1.4, '预测器\n→ 预测下一段的 embedding',
         '#a5d6a7', '#2e7d32', fontsize=11, fontweight='bold')

    _box(ax, 0.5, 0.6, 9.0, 2.4,
         '只预测抽象状态：物体在哪、在做什么、接下来往哪去。\n'
         '不预测每一片叶子、每一道反光。\n'
         'LeCun 的类比：\n'
         '  Transformer = 记住书里每一个逗号的位置\n'
         '  JEPA = 记住这本书在讲什么',
         '#e8f5e9', '#2e7d32', fontsize=10)

    plt.tight_layout()
    out = os.path.join(OUT, 'arch_compare.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {out}  ({os.path.getsize(out)//1024} KB)")


if __name__ == '__main__':
    make_arch_compare()
    print("✓ Done.")
