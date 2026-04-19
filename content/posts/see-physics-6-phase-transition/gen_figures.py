#!/usr/bin/env python3
"""Figures for 看见物理（六）：相变.
Outputs:
  - emergence.png  (emergence curve: flat flat flat then jump)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.expanduser("~/ai-blog/content/posts/see-physics-6-phase-transition")
os.makedirs(OUT, exist_ok=True)


def make_emergence():
    print("→ emergence.png ...")
    x = np.linspace(8, 12, 200)

    def sigmoid(x, x0, k=4):
        return 1.0 / (1.0 + np.exp(-k * (x - x0)))

    y_add = sigmoid(x, 10.2) * 0.95
    y_cot = sigmoid(x, 10.8) * 0.80
    y_math = sigmoid(x, 11.3) * 0.60

    rng = np.random.default_rng(42)
    y_add += rng.normal(0, 0.005, size=x.size)
    y_cot += rng.normal(0, 0.005, size=x.size)
    y_math += rng.normal(0, 0.005, size=x.size)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor('white')

    ax.plot(x, y_add * 100, color='#2196F3', linewidth=2.5, label='三位数加法')
    ax.plot(x, y_cot * 100, color='#4CAF50', linewidth=2.5, label='思维链推理')
    ax.plot(x, y_math * 100, color='#9C27B0', linewidth=2.5, label='大学数学')

    for x0, color in [(10.2, '#2196F3'), (10.8, '#4CAF50'), (11.3, '#9C27B0')]:
        ax.axvline(x0, color=color, linestyle=':', alpha=0.5, linewidth=1.2)

    ax.annotate('平，平，平 ...',
                xy=(9.2, 8), xytext=(9.2, 25),
                fontsize=12, color='#555', ha='center',
                arrowprops=dict(arrowstyle='->', color='#888', lw=1.2))
    ax.annotate('然后突然起飞',
                xy=(11.0, 60), xytext=(11.4, 85),
                fontsize=12, color='#c0392b', ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.4))

    ax.set_xlabel('模型参数量 (log10)', fontsize=12)
    ax.set_ylabel('任务准确率 (%)', fontsize=12)
    ax.set_title('涌现能力：像相变一样锐利',
                 fontsize=15, fontweight='bold', pad=14)
    ax.set_xticks([8, 9, 10, 11, 12])
    ax.set_xticklabels(['100M', '1B', '10B', '100B', '1T'])
    ax.set_ylim(-3, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', fontsize=11, frameon=True)

    fig.text(0.5, -0.02,
             '每条曲线都像居里温度附近的磁化曲线：低于临界点接近零，过了临界点集体起飞。',
             ha='center', fontsize=11, color='#444', style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff8e1',
                       edgecolor='#e0b030', linewidth=1.0))

    plt.tight_layout()
    out = os.path.join(OUT, 'emergence.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {out}  ({os.path.getsize(out)//1024} KB)")


if __name__ == '__main__':
    make_emergence()
    print("✓ Done.")
