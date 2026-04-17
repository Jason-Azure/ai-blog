#!/usr/bin/env python3
"""Generate figures for the hallucination article.
Output: ~/ai-blog/content/posts/llm-hallucination-bullshit/entropy_floor.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.expanduser("~/ai-blog/content/posts/llm-hallucination-bullshit")
os.makedirs(OUT, exist_ok=True)


def make_entropy_floor():
    """Two-panel: known question (sharp peak) vs unknown question (flat).
    Both: model MUST pick one. That's the entropy floor."""
    print("→ Generating entropy_floor.png ...")

    tokens_known = ['乌尔姆', '柏林', '慕尼黑', '汉堡', '法兰克福', '科隆', '其他']
    probs_known  = [0.78,    0.05,   0.04,   0.03,  0.03,     0.02,  0.05]

    tokens_unknown = ['牛肉面', '番茄炒蛋', '三明治', '寿司', '米饭', '包子', '其他']
    probs_unknown  = [0.14,     0.13,      0.12,    0.11,  0.11,  0.11,  0.28]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.patch.set_facecolor('white')
    fig.suptitle('熵的地板：模型知道 vs 不知道，但它必须挑一个',
                 fontsize=15, fontweight='bold', y=1.02)

    # Left: known question
    ax = axes[0]
    ax.set_facecolor('#fafafa')
    colors_k = ['#2e7d32'] + ['#8ab5b3'] * (len(tokens_known) - 1)
    bars_k = ax.bar(range(len(tokens_known)), probs_known, color=colors_k,
                    edgecolor='white', linewidth=1.5)
    ax.set_xticks(range(len(tokens_known)))
    ax.set_xticklabels(tokens_known, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('概率', fontsize=11)
    ax.set_title('提示：「爱因斯坦出生在」\n→ 模型有知识：分布尖锐，熵低',
                 fontsize=12, pad=12, color='#2e7d32')
    for i, (b, p) in enumerate(zip(bars_k, probs_known)):
        ax.text(b.get_x() + b.get_width()/2, p + 0.02, f'{p:.2f}',
                ha='center', fontsize=9, color='#333')
    H_k = -sum(p * np.log(p) for p in probs_known if p > 0)
    ax.text(0.98, 0.94, f'熵 H = {H_k:.2f} nats',
            transform=ax.transAxes, ha='right', fontsize=11,
            color='#2e7d32', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#2e7d32'))
    ax.grid(axis='y', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Right: unknown question
    ax = axes[1]
    ax.set_facecolor('#fafafa')
    colors_u = ['#c0392b'] * len(tokens_unknown)
    bars_u = ax.bar(range(len(tokens_unknown)), probs_unknown, color=colors_u,
                    edgecolor='white', linewidth=1.5, alpha=0.75)
    ax.set_xticks(range(len(tokens_unknown)))
    ax.set_xticklabels(tokens_unknown, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('概率', fontsize=11)
    ax.set_title('提示：「小明 2019 年 3 月 17 日中午吃了」\n→ 模型无知：分布摊平，熵高 —— 但仍必须挑一个',
                 fontsize=12, pad=12, color='#c0392b')
    for i, (b, p) in enumerate(zip(bars_u, probs_unknown)):
        ax.text(b.get_x() + b.get_width()/2, p + 0.02, f'{p:.2f}',
                ha='center', fontsize=9, color='#333')
    H_u = -sum(p * np.log(p) for p in probs_unknown if p > 0)
    ax.text(0.98, 0.94, f'熵 H = {H_u:.2f} nats',
            transform=ax.transAxes, ha='right', fontsize=11,
            color='#c0392b', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#c0392b'))
    ax.grid(axis='y', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Bottom annotation
    fig.text(0.5, -0.06,
             '关键：右侧概率加起来仍 = 1.00。词表里没有「沉默」token，模型必须从分布里抽一个 —— 这就是幻觉的第一性原理。',
             ha='center', fontsize=11, color='#444', style='italic',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#fff8e1',
                       edgecolor='#e0b030', linewidth=1.2))

    plt.tight_layout()
    out = os.path.join(OUT, "entropy_floor.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    kb = os.path.getsize(out) // 1024
    print(f"  ✓ {out}  ({kb} KB)")


if __name__ == '__main__':
    make_entropy_floor()
    print("\n✓ Done.")
