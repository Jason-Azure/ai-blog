#!/usr/bin/env python3
"""
Generate all figures for the "MoE 架构" blog post.
4 GIFs total.

Usage:
    source ~/ai-lab-venv/bin/activate
    cd ~/ai-blog/content/posts/moe-architecture/
    python generate_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Wedge
import matplotlib.patheffects as pe
import imageio
import io
import os

plt.rcParams['font.family'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

COLORS = {
    'orange': '#FF9800', 'blue': '#2196F3', 'green': '#4CAF50',
    'purple': '#9C27B0', 'red': '#E91E63', 'gray': '#607D8B',
    'dark': '#333333', 'light_bg': '#f6f8fa',
    'expert_colors': ['#1565C0', '#2E7D32', '#E65100', '#6A1B9A',
                      '#00838F', '#AD1457', '#4E342E', '#37474F'],
}

def fig_to_frame(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor(), pad_inches=0.3)
    buf.seek(0)
    frame = imageio.v3.imread(buf)
    buf.close()
    return frame

def save_gif(frames, name, fps=2):
    path = os.path.join(OUTPUT_DIR, name)
    # Pad all frames to same size
    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)
    padded = []
    for f in frames:
        h, w = f.shape[:2]
        pad_h = max_h - h
        pad_w = max_w - w
        if pad_h > 0 or pad_w > 0:
            f = np.pad(f, ((0, pad_h), (0, pad_w), (0, 0)),
                      mode='constant', constant_values=255)
        padded.append(f)
    imageio.mimsave(path, padded, fps=fps, loop=0)
    size = os.path.getsize(path)
    print(f"  OK {name} ({size/1024:.0f} KB)")


# ============================================================
# GIF 1: Dense vs MoE comparison
# Shows token flowing through Dense (all params) vs MoE (selected experts)
# ============================================================
def generate_dense_vs_moe():
    print("Generating dense-vs-moe.gif...")
    frames = []

    for phase in range(8):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
        fig.set_facecolor('#ffffff')
        fig.suptitle('Dense Model vs MoE Model', fontsize=16,
                    fontweight='bold', color=COLORS['dark'], y=0.97)

        for idx, ax in enumerate(axes := [ax1, ax2]):
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')

            if idx == 0:
                ax.set_title('Dense (e.g. LLaMA 405B)', fontsize=13,
                           fontweight='bold', color=COLORS['gray'], pad=10)

                # Single large FFN block - all neurons active
                box = FancyBboxPatch((1, 2.5), 8, 4.5, boxstyle="round,pad=0.2",
                                    facecolor='#FFCDD2' if phase >= 2 else '#E3F2FD',
                                    edgecolor=COLORS['red'] if phase >= 2 else COLORS['blue'],
                                    linewidth=2)
                ax.add_patch(box)

                # Grid of neurons - ALL lit up
                for r in range(4):
                    for c in range(6):
                        nx = 2 + c * 1.15
                        ny = 3.2 + r * 1.0
                        color = '#EF5350' if phase >= 2 else '#42A5F5'
                        alpha = 0.8 if phase >= 2 else 0.5
                        circle = Circle((nx, ny), 0.3, facecolor=color,
                                       edgecolor='white', linewidth=1, alpha=alpha)
                        ax.add_patch(circle)

                ax.text(5, 7.5, 'FFN Layer', fontsize=12, ha='center',
                        fontweight='bold', color=COLORS['dark'])

                if phase >= 2:
                    ax.text(5, 1.5, 'ALL 405B params active', fontsize=11,
                            ha='center', color=COLORS['red'], fontweight='bold')
                    ax.text(5, 0.8, 'for every single token', fontsize=10,
                            ha='center', color='#999')

                # Token arrow
                if phase >= 1:
                    ax.annotate('', xy=(5, 2.5), xytext=(5, 1.2),
                               arrowprops=dict(arrowstyle='->', color=COLORS['dark'],
                                              lw=2))
                    ax.text(5, 0.5, 'token', fontsize=11, ha='center',
                            fontweight='bold', color=COLORS['dark'],
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4',
                                     edgecolor='#FBC02D'))

            else:
                ax.set_title('MoE (DeepSeek-V3 671B)', fontsize=13,
                           fontweight='bold', color=COLORS['blue'], pad=10)

                # Router at top
                if phase >= 3:
                    router_box = FancyBboxPatch((3, 7.8), 4, 0.8, boxstyle="round,pad=0.1",
                                               facecolor='#FFF3E0',
                                               edgecolor=COLORS['orange'], linewidth=1.5)
                    ax.add_patch(router_box)
                    ax.text(5, 8.2, 'Router', fontsize=11, ha='center',
                            fontweight='bold', color=COLORS['orange'])

                # 8 expert boxes (2 rows of 4)
                expert_labels = [f'E{i+1}' for i in range(8)]
                selected = [0, 2, 5, 7] if phase >= 4 else []  # 4 selected experts

                for i in range(8):
                    row = i // 4
                    col = i % 4
                    x = 1.0 + col * 2.2
                    y = 4.5 - row * 2.5

                    is_selected = i in selected
                    if phase >= 4:
                        fc = COLORS['expert_colors'][i] if is_selected else '#F5F5F5'
                        ec = COLORS['expert_colors'][i] if is_selected else '#E0E0E0'
                        alpha = 0.85 if is_selected else 0.3
                    else:
                        fc = '#E3F2FD'
                        ec = COLORS['blue']
                        alpha = 0.5

                    box = FancyBboxPatch((x, y), 1.6, 1.8, boxstyle="round,pad=0.1",
                                        facecolor=fc, edgecolor=ec,
                                        linewidth=2 if is_selected else 1,
                                        alpha=alpha)
                    ax.add_patch(box)

                    tc = 'white' if (is_selected and phase >= 4) else '#999'
                    ax.text(x + 0.8, y + 0.9, expert_labels[i], fontsize=12,
                            fontweight='bold', ha='center', va='center', color=tc)

                if phase >= 5:
                    ax.text(5, 1.0, 'Only 8 of 256 experts active', fontsize=11,
                            ha='center', color=COLORS['green'], fontweight='bold')
                    ax.text(5, 0.3, '37B of 671B params = 5.5%', fontsize=10,
                            ha='center', color='#999')

                # Token arrow
                if phase >= 3:
                    ax.annotate('', xy=(5, 7.8), xytext=(5, 7.0),
                               arrowprops=dict(arrowstyle='->', color=COLORS['dark'],
                                              lw=2))
                    ax.text(5, 6.7, 'token', fontsize=11, ha='center',
                            fontweight='bold', color=COLORS['dark'],
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4',
                                     edgecolor='#FBC02D'))

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        frames.append(fig_to_frame(fig))
        plt.close(fig)

    # Hold final
    for _ in range(4):
        frames.append(frames[-1])

    save_gif(frames, 'dense-vs-moe.gif', fps=1)


# ============================================================
# GIF 2: Router mechanism - token routing to experts
# ============================================================
def generate_router():
    print("Generating router-mechanism.gif...")
    frames = []

    # Scores for 8 experts
    scores = [0.82, 0.15, 0.71, 0.08, 0.45, 0.91, 0.33, 0.68]
    expert_names = [f'Expert {i+1}' for i in range(8)]
    # Top-3 selected (indices sorted by score descending)
    sorted_idx = sorted(range(8), key=lambda i: scores[i], reverse=True)
    top_k = sorted_idx[:3]  # indices: 5, 0, 2

    for phase in range(7):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.set_facecolor('#ffffff')
        fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.08)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')

        ax.text(5, 7.5, 'How the Router Works (simplified to Top-3)', fontsize=16,
                fontweight='bold', ha='center', color=COLORS['dark'])

        # Step labels
        steps = ['Step 1: Score each expert', 'Step 2: Select Top-K',
                 'Step 3: Normalize & compute']
        if phase < 3:
            step_text = steps[0]
        elif phase < 5:
            step_text = steps[1]
        else:
            step_text = steps[2]
        ax.text(5, 6.8, step_text, fontsize=12, ha='center',
                color=COLORS['orange'], fontweight='bold')

        # Token box at left
        token_box = FancyBboxPatch((0.3, 3.2), 1.4, 1.0, boxstyle="round,pad=0.1",
                                   facecolor='#FFF9C4', edgecolor='#FBC02D', linewidth=1.5)
        ax.add_patch(token_box)
        ax.text(1.0, 3.7, 'token', fontsize=11, ha='center', va='center',
                fontweight='bold', color=COLORS['dark'])

        # Expert bars with scores
        bar_x = 2.5
        bar_width = 0.7
        for i in range(8):
            y = 6.0 - i * 0.7
            score = scores[i]
            is_top = i in top_k

            if phase >= 1:
                # Show score bars
                bar_len = score * 4.5
                if phase >= 3:
                    # Phase 3+: highlight top-K
                    color = COLORS['expert_colors'][i] if is_top else '#E0E0E0'
                    alpha = 0.9 if is_top else 0.3
                else:
                    color = COLORS['blue']
                    alpha = 0.6

                ax.barh(y, bar_len, height=0.5, left=bar_x, color=color,
                        alpha=alpha, edgecolor='white', linewidth=0.5)

                # Score text
                if phase >= 2:
                    tc = COLORS['dark'] if (phase < 3 or is_top) else '#CCC'
                    ax.text(bar_x + bar_len + 0.15, y, f'{score:.2f}',
                            fontsize=10, va='center', color=tc, fontweight='bold')

            # Expert label
            label_color = COLORS['dark'] if (phase < 3 or is_top) else '#CCC'
            ax.text(bar_x - 0.15, y, expert_names[i], fontsize=9,
                    va='center', ha='right', color=label_color)

            # Top-K markers
            if phase >= 4 and is_top:
                rank = top_k.index(i) + 1
                ax.text(bar_x + scores[i] * 4.5 + 0.6, y, f'Top-{rank}',
                        fontsize=9, va='center', color=COLORS['green'],
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F5E9',
                                 edgecolor=COLORS['green'], linewidth=0.5))

        # Normalized weights
        if phase >= 5:
            total = sum(scores[i] for i in top_k)
            weights = [scores[i] / total for i in top_k]
            weight_text = '  +  '.join([f'E{top_k[j]+1}: {weights[j]:.1%}'
                                        for j in range(3)])
            ax.text(5, 0.8, f'Normalized weights: {weight_text}', fontsize=11,
                    ha='center', color=COLORS['purple'], fontweight='bold')
            ax.text(5, 0.2, 'output = w1*E6(x) + w2*E1(x) + w3*E3(x)',
                    fontsize=10, ha='center', color='#666', family='monospace')

        frames.append(fig_to_frame(fig))
        plt.close(fig)

    for _ in range(4):
        frames.append(frames[-1])

    save_gif(frames, 'router-mechanism.gif', fps=1)


# ============================================================
# GIF 3: Load balancing problem & solution
# ============================================================
def generate_load_balance():
    print("Generating load-balance.gif...")
    frames = []

    n_experts = 8
    np.random.seed(42)

    # Phase 1: Unbalanced (collapse to 2 experts)
    # Phase 2: Auxiliary loss (uniform but quality drops)
    # Phase 3: DeepSeek bias-term (balanced + quality maintained)

    phases = [
        ('No balancing: Expert Collapse', 'red',
         [85, 5, 3, 2, 1, 1, 2, 1]),
        ('Auxiliary Loss: Balanced but quality drops', 'orange',
         [13, 12, 13, 12, 13, 12, 13, 12]),
        ("DeepSeek's Bias Terms: Balanced + Quality", 'green',
         [14, 11, 13, 12, 14, 10, 13, 13]),
    ]

    for hold_repeat in range(2):
        for pi, (title, scheme, loads) in enumerate(phases):
            for sub_frame in range(3 if hold_repeat == 0 else 1):
                fig, ax = plt.subplots(figsize=(10, 5.5))
                fig.set_facecolor('#ffffff')
                fig.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.15)

                ax.set_title(title, fontsize=14, fontweight='bold',
                            color={'red': COLORS['red'], 'orange': COLORS['orange'],
                                   'green': COLORS['green']}[scheme], pad=12)

                colors_map = {
                    'red': ['#EF5350' if l > 20 else '#FFCDD2' for l in loads],
                    'orange': ['#FFB74D'] * 8,
                    'green': [COLORS['expert_colors'][i] for i in range(8)],
                }

                bars = ax.bar(range(n_experts), loads, color=colors_map[scheme],
                             edgecolor='white', linewidth=1, width=0.6)

                # Ideal line
                ideal = 100 / n_experts
                ax.axhline(y=ideal, color='#666', linestyle='--', alpha=0.5,
                          linewidth=1)
                ax.text(7.8, ideal + 1.5, f'Ideal: {ideal:.0f}%', fontsize=9,
                        color='#666', ha='right')

                # Labels
                for i, v in enumerate(loads):
                    ax.text(i, v + 1, f'{v}%', ha='center', fontsize=10,
                            fontweight='bold', color=COLORS['dark'])

                ax.set_xlabel('Experts', fontsize=11)
                ax.set_ylabel('Token Load (%)', fontsize=11)
                ax.set_xticks(range(n_experts))
                ax.set_xticklabels([f'E{i+1}' for i in range(n_experts)], fontsize=9)
                ax.set_ylim(0, 100)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(True, axis='y', alpha=0.2)

                # Annotations
                if pi == 0:
                    ax.text(0, 92, 'E1 gets 85% of tokens!\nOther experts starve',
                            fontsize=10, color=COLORS['red'], fontweight='bold')
                elif pi == 1:
                    ax.text(1, 60, 'Balanced, but auxiliary loss\ncompetes with main objective\n'
                            + '=> quality degrades', fontsize=10,
                            color=COLORS['orange'], fontweight='bold')
                elif pi == 2:
                    ax.text(1, 60, 'Bias term adjusts routing\nwithout touching loss function\n'
                            + '=> balanced + quality preserved', fontsize=10,
                            color=COLORS['green'], fontweight='bold')

                frames.append(fig_to_frame(fig))
                plt.close(fig)

    save_gif(frames, 'load-balance.gif', fps=1)


# ============================================================
# GIF 4: MoE cost efficiency comparison
# ============================================================
def generate_cost_comparison():
    print("Generating cost-comparison.gif...")
    frames = []

    models = [
        ('GPT-4\n(est.)', 1800, 1800, '~$100M+', '#EF5350'),
        ('LLaMA-3.1\n405B', 405, 405, '~$30M+', '#FF7043'),
        ('DeepSeek-V3\n671B MoE', 671, 37, '$5.6M', '#42A5F5'),
        ('Mixtral\n8x7B MoE', 47, 13, 'N/A', '#66BB6A'),
    ]

    for show_count in range(len(models) + 2):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
        fig.set_facecolor('#ffffff')
        fig.suptitle('MoE Efficiency: Total vs Active Parameters', fontsize=15,
                    fontweight='bold', color=COLORS['dark'], y=0.97)

        n = min(show_count, len(models))

        # Left: Total params
        ax1.set_title('Total Parameters', fontsize=12, fontweight='bold',
                     color=COLORS['gray'])
        ax1.set_ylabel('Parameters (Billions)', fontsize=10)

        # Right: Active params per token
        ax2.set_title('Active per Token (= Actual Cost)', fontsize=12,
                     fontweight='bold', color=COLORS['blue'])
        ax2.set_ylabel('Active Parameters (Billions)', fontsize=10)

        for ax in [ax1, ax2]:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, axis='y', alpha=0.2)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels([m[0] for m in models], fontsize=8)

        ax1.set_ylim(0, 2200)
        ax2.set_ylim(0, 2200)

        for i in range(n):
            name, total, active, cost, color = models[i]

            # Total params bar
            ax1.bar(i, total, color=color, alpha=0.7, width=0.5,
                   edgecolor='white')
            ax1.text(i, total + 40, f'{total}B', ha='center', fontsize=9,
                    fontweight='bold', color=COLORS['dark'])

            # Active params bar
            ax2.bar(i, active, color=color, alpha=0.9, width=0.5,
                   edgecolor='white')
            ax2.text(i, active + 40, f'{active}B', ha='center', fontsize=9,
                    fontweight='bold', color=COLORS['dark'])

            # Cost label
            if show_count >= len(models):
                ax2.text(i, active + 120, cost, ha='center', fontsize=8,
                        color=COLORS['orange'], fontweight='bold')

        # Highlight DeepSeek
        if n >= 3 and show_count >= len(models):
            ax2.annotate('Same quality,\n1/20th cost!', xy=(2, 37),
                        xytext=(2.8, 800),
                        fontsize=10, fontweight='bold', color=COLORS['green'],
                        arrowprops=dict(arrowstyle='->', color=COLORS['green'],
                                       lw=1.5))

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        frames.append(fig_to_frame(fig))
        plt.close(fig)

    for _ in range(4):
        frames.append(frames[-1])

    save_gif(frames, 'cost-comparison.gif', fps=1)


if __name__ == '__main__':
    print("=" * 50)
    print("Generating figures for MoE article")
    print("=" * 50)

    generate_dense_vs_moe()
    generate_router()
    generate_load_balance()
    generate_cost_comparison()

    print("\nAll figures generated!")
