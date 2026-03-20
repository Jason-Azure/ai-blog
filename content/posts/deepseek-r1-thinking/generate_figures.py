#!/usr/bin/env python3
"""
Generate all figures for the "DeepSeek-R1：一个模型如何学会思考" blog post.
4 GIFs total.

Usage:
    source ~/ai-lab-venv/bin/activate
    cd ~/ai-blog/content/posts/deepseek-r1-thinking/
    python generate_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
import imageio
import io
import os

# ============================================================
# Global settings
# ============================================================
plt.rcParams['font.family'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def fig_to_frame(fig, dpi=100):
    """Convert matplotlib figure to numpy array for GIF frame."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    frame = imageio.v3.imread(buf)
    buf.close()
    return frame

def save_gif(frames, name, fps=2):
    path = os.path.join(OUTPUT_DIR, name)
    imageio.mimsave(path, frames, fps=fps, loop=0)
    size = os.path.getsize(path)
    print(f"  OK {name} ({size/1024:.0f} KB)")

def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    size = os.path.getsize(path)
    print(f"  OK {name} ({size/1024:.0f} KB)")

# Color palette
COLORS = {
    'orange': '#FF9800',
    'blue': '#2196F3',
    'green': '#4CAF50',
    'purple': '#9C27B0',
    'red': '#E91E63',
    'gray': '#607D8B',
    'dark': '#333333',
    'light_bg': '#f6f8fa',
    'correct': '#4CAF50',
    'wrong': '#f44336',
    'think': '#FF9800',
    'answer': '#2196F3',
}

# ============================================================
# GIF 1: GRPO Training Process
# Shows: One question → Generate G answers → Score → Rank → Update
# ============================================================
def generate_grpo_process():
    print("Generating grpo-process.gif...")

    # Problem: "What is 7 × 8?"
    # Generate 8 answers with scores
    answers = [
        ("56", True, 1.0),
        ("54", False, 0.0),
        ("56", True, 1.0),
        ("58", False, 0.0),
        ("56", True, 1.0),
        ("48", False, 0.0),
        ("56", True, 1.0),
        ("55", False, 0.0),
    ]

    frames = []

    # --- Frame 1: Show the question ---
    for hold in range(3):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.set_facecolor('#ffffff')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.axis('off')

        ax.text(5, 6.3, 'GRPO Training: One Step', fontsize=18, fontweight='bold',
                ha='center', color=COLORS['dark'])

        # Question box
        box = FancyBboxPatch((2.5, 4.5), 5, 1.2, boxstyle="round,pad=0.15",
                            facecolor='#E3F2FD', edgecolor=COLORS['blue'], linewidth=2)
        ax.add_patch(box)
        ax.text(5, 5.1, 'Question: 7 x 8 = ?', fontsize=16, fontweight='bold',
                ha='center', va='center', color=COLORS['blue'])

        ax.text(5, 3.5, 'Step 1: Ask the model to generate 8 answers...',
                fontsize=13, ha='center', color=COLORS['gray'], style='italic')

        frames.append(fig_to_frame(fig))
        plt.close(fig)

    # --- Frame 2-3: Show all answers appearing ---
    for show_count in [4, 8]:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.set_facecolor('#ffffff')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.axis('off')

        ax.text(5, 6.3, 'GRPO Training: One Step', fontsize=18, fontweight='bold',
                ha='center', color=COLORS['dark'])

        # Question box (smaller)
        box = FancyBboxPatch((3, 5.3), 4, 0.8, boxstyle="round,pad=0.1",
                            facecolor='#E3F2FD', edgecolor=COLORS['blue'], linewidth=1.5)
        ax.add_patch(box)
        ax.text(5, 5.7, '7 x 8 = ?', fontsize=14, fontweight='bold',
                ha='center', va='center', color=COLORS['blue'])

        ax.text(5, 4.7, 'Step 1: Generate 8 answers', fontsize=12,
                ha='center', color=COLORS['gray'])

        # Answer boxes in 2 rows of 4
        for i in range(show_count):
            row = i // 4
            col = i % 4
            x = 1.2 + col * 2.2
            y = 3.2 - row * 1.5

            ans, correct, score = answers[i]
            box = FancyBboxPatch((x, y), 1.6, 0.9, boxstyle="round,pad=0.1",
                                facecolor='#FAFAFA', edgecolor='#BDBDBD', linewidth=1)
            ax.add_patch(box)
            ax.text(x + 0.8, y + 0.45, ans, fontsize=16, fontweight='bold',
                    ha='center', va='center', color=COLORS['dark'])

        frames.append(fig_to_frame(fig))
        plt.close(fig)

    # --- Frame 4-5: Score with checkmarks/crosses ---
    for hold in range(2):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.set_facecolor('#ffffff')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.axis('off')

        ax.text(5, 6.3, 'GRPO Training: One Step', fontsize=18, fontweight='bold',
                ha='center', color=COLORS['dark'])

        box = FancyBboxPatch((3, 5.3), 4, 0.8, boxstyle="round,pad=0.1",
                            facecolor='#E3F2FD', edgecolor=COLORS['blue'], linewidth=1.5)
        ax.add_patch(box)
        ax.text(5, 5.7, '7 x 8 = ?   (correct: 56)', fontsize=13, fontweight='bold',
                ha='center', va='center', color=COLORS['blue'])

        ax.text(5, 4.7, 'Step 2: Score each answer (rule-based, no human needed!)',
                fontsize=12, ha='center', color=COLORS['orange'], fontweight='bold')

        for i in range(8):
            row = i // 4
            col = i % 4
            x = 1.2 + col * 2.2
            y = 3.2 - row * 1.5

            ans, correct, score = answers[i]
            bg = '#E8F5E9' if correct else '#FFEBEE'
            ec = COLORS['correct'] if correct else COLORS['wrong']
            box = FancyBboxPatch((x, y), 1.6, 0.9, boxstyle="round,pad=0.1",
                                facecolor=bg, edgecolor=ec, linewidth=2)
            ax.add_patch(box)
            ax.text(x + 0.8, y + 0.55, ans, fontsize=16, fontweight='bold',
                    ha='center', va='center', color=COLORS['dark'])

            mark = 'OK +1' if correct else 'X  0'
            mc = COLORS['correct'] if correct else COLORS['wrong']
            ax.text(x + 0.8, y + 0.15, mark, fontsize=10, fontweight='bold',
                    ha='center', va='center', color=mc)

        frames.append(fig_to_frame(fig))
        plt.close(fig)

    # --- Frame 6-7: Group ranking & advantage ---
    for hold in range(2):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.set_facecolor('#ffffff')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.axis('off')

        ax.text(5, 6.3, 'GRPO Training: One Step', fontsize=18, fontweight='bold',
                ha='center', color=COLORS['dark'])

        ax.text(5, 5.5, 'Step 3: Rank within group', fontsize=13,
                ha='center', color=COLORS['purple'], fontweight='bold')

        ax.text(5, 4.9, 'Group mean = (1+0+1+0+1+0+1+0) / 8 = 0.5',
                fontsize=12, ha='center', color=COLORS['gray'])

        # Show advantages
        labels_pos = ['56 (+)', '56 (+)', '56 (+)', '56 (+)']
        labels_neg = ['54 (-)', '58 (-)', '48 (-)', '55 (-)']

        # Positive group
        ax.text(2.8, 4.1, 'Above average (advantage > 0):', fontsize=11,
                color=COLORS['correct'], fontweight='bold')
        for i, lbl in enumerate(labels_pos):
            x = 1.5 + i * 2.0
            box = FancyBboxPatch((x, 3.2), 1.4, 0.7, boxstyle="round,pad=0.1",
                                facecolor='#E8F5E9', edgecolor=COLORS['correct'], linewidth=1.5)
            ax.add_patch(box)
            ax.text(x + 0.7, 3.55, lbl, fontsize=12, fontweight='bold',
                    ha='center', va='center', color=COLORS['correct'])

        # Negative group
        ax.text(2.8, 2.5, 'Below average (advantage < 0):', fontsize=11,
                color=COLORS['wrong'], fontweight='bold')
        for i, lbl in enumerate(labels_neg):
            x = 1.5 + i * 2.0
            box = FancyBboxPatch((x, 1.6), 1.4, 0.7, boxstyle="round,pad=0.1",
                                facecolor='#FFEBEE', edgecolor=COLORS['wrong'], linewidth=1.5)
            ax.add_patch(box)
            ax.text(x + 0.7, 1.95, lbl, fontsize=12, fontweight='bold',
                    ha='center', va='center', color=COLORS['wrong'])

        # Update arrow
        ax.annotate('', xy=(5, 0.6), xytext=(5, 1.4),
                    arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=2))
        ax.text(5, 0.3, 'Step 4: Update model  -  generate more "56", fewer wrong answers',
                fontsize=12, ha='center', color=COLORS['orange'], fontweight='bold')

        frames.append(fig_to_frame(fig))
        plt.close(fig)

    save_gif(frames, 'grpo-process.gif', fps=1)


# ============================================================
# GIF 2: Thinking vs Direct comparison
# Shows: Same question, two models side-by-side
# ============================================================
def generate_thinking_vs_direct():
    print("Generating thinking-vs-direct.gif...")

    frames = []

    # The thinking steps that appear one by one
    think_steps = [
        "Comparing 9.11 and 9.8...",
        "Integer parts: both are 9. Equal.",
        "Align decimals: 9.110 vs 9.800",
        "Tenths: 1 < 8",
        "Therefore: 9.11 < 9.8",
    ]

    direct_answer = "9.11 > 9.8"   # wrong!
    think_answer = "9.11 < 9.8"    # correct!

    # --- Frames showing thinking process step by step ---
    for step in range(len(think_steps) + 3):  # +3 for intro, final answer, hold
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        fig.set_facecolor('#ffffff')
        fig.suptitle('Same question, two approaches', fontsize=16, fontweight='bold',
                    color=COLORS['dark'], y=0.97)

        for idx, ax in enumerate(axes):
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')

            if idx == 0:
                # Direct model (left)
                ax.set_title('Direct Model', fontsize=14, fontweight='bold',
                           color=COLORS['gray'], pad=10)

                # Question
                box = FancyBboxPatch((0.5, 8.0), 9, 1.2, boxstyle="round,pad=0.15",
                                    facecolor='#F5F5F5', edgecolor='#BDBDBD', linewidth=1)
                ax.add_patch(box)
                ax.text(5, 8.6, '"9.11 vs 9.8, which is larger?"', fontsize=11,
                        ha='center', va='center', color=COLORS['dark'])

                if step >= 1:
                    # Instant answer (wrong)
                    box = FancyBboxPatch((1, 4.5), 8, 2.5, boxstyle="round,pad=0.15",
                                        facecolor='#FFEBEE', edgecolor=COLORS['wrong'], linewidth=2)
                    ax.add_patch(box)
                    ax.text(5, 6.2, 'Instant output:', fontsize=11,
                            ha='center', va='center', color=COLORS['gray'])
                    ax.text(5, 5.3, direct_answer, fontsize=20, fontweight='bold',
                            ha='center', va='center', color=COLORS['wrong'])
                    ax.text(5, 4.8, 'X WRONG', fontsize=12, fontweight='bold',
                            ha='center', va='center', color=COLORS['wrong'])

                    ax.text(5, 3.0, 'No thinking process\nJust pattern matching',
                            fontsize=11, ha='center', va='center', color='#999',
                            style='italic')
                else:
                    ax.text(5, 5.5, '...', fontsize=24, ha='center', color='#CCC')

            else:
                # Thinking model (right)
                ax.set_title('Reasoning Model (R1)', fontsize=14, fontweight='bold',
                           color=COLORS['blue'], pad=10)

                # Question
                box = FancyBboxPatch((0.5, 8.0), 9, 1.2, boxstyle="round,pad=0.15",
                                    facecolor='#E3F2FD', edgecolor=COLORS['blue'], linewidth=1)
                ax.add_patch(box)
                ax.text(5, 8.6, '"9.11 vs 9.8, which is larger?"', fontsize=11,
                        ha='center', va='center', color=COLORS['dark'])

                if step >= 1:
                    # Think box
                    n_steps = min(step - 1, len(think_steps))
                    if n_steps > 0:
                        box_h = 0.6 + n_steps * 0.65
                        box_y = 7.5 - box_h
                        box = FancyBboxPatch((0.5, box_y), 9, box_h, boxstyle="round,pad=0.1",
                                            facecolor='#FFF8E1', edgecolor=COLORS['think'],
                                            linewidth=1.5, linestyle='--')
                        ax.add_patch(box)
                        ax.text(1.2, 7.3, '<think>', fontsize=10, color=COLORS['think'],
                                fontweight='bold', family='monospace')

                        for j in range(n_steps):
                            y_pos = 6.8 - j * 0.65
                            ax.text(1.5, y_pos, think_steps[j], fontsize=10,
                                    color='#666', va='center')

                        if n_steps == len(think_steps):
                            ax.text(1.2, 6.8 - n_steps * 0.65 + 0.15, '</think>',
                                    fontsize=10, color=COLORS['think'], fontweight='bold',
                                    family='monospace')

                # Final answer
                if step >= len(think_steps) + 1:
                    box = FancyBboxPatch((1.5, 0.8), 7, 1.5, boxstyle="round,pad=0.15",
                                        facecolor='#E8F5E9', edgecolor=COLORS['correct'], linewidth=2)
                    ax.add_patch(box)
                    ax.text(5, 1.8, think_answer, fontsize=20, fontweight='bold',
                            ha='center', va='center', color=COLORS['correct'])
                    ax.text(5, 1.2, 'OK CORRECT', fontsize=12, fontweight='bold',
                            ha='center', va='center', color=COLORS['correct'])
                elif step >= 1:
                    ax.text(5, 1.5, 'thinking...', fontsize=12, ha='center',
                            color=COLORS['think'], style='italic')

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        frames.append(fig_to_frame(fig))
        plt.close(fig)

    # Hold final frame
    for _ in range(3):
        frames.append(frames[-1])

    save_gif(frames, 'thinking-vs-direct.gif', fps=1)


# ============================================================
# GIF 3: Aha Moment - Emergence of reasoning behaviors
# Shows: Training progress → reasoning behaviors appearing
# ============================================================
def generate_aha_moment():
    print("Generating aha-moment.gif...")

    frames = []
    np.random.seed(42)

    total_frames = 20

    for frame in range(total_frames):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                        gridspec_kw={'width_ratios': [1.3, 1]})
        fig.set_facecolor('#ffffff')
        fig.suptitle('R1-Zero Training: Emergence of Reasoning', fontsize=16,
                    fontweight='bold', color=COLORS['dark'], y=0.97)

        # Left: Accuracy curve
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 100)
        ax1.set_xlabel('Training Steps (thousands)', fontsize=11)
        ax1.set_ylabel('AIME Accuracy (%)', fontsize=11)
        ax1.set_title('Math Reasoning Accuracy', fontsize=12, fontweight='bold',
                      color=COLORS['blue'])
        ax1.grid(True, alpha=0.3)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Generate smooth accuracy curve
        progress = (frame + 1) / total_frames
        x_max = int(progress * 100)
        if x_max > 0:
            x = np.linspace(0, x_max, max(x_max, 2))
            # S-curve growth with noise
            y_base = 75 * (1 - np.exp(-x / 25)) + 5
            noise = np.random.randn(len(x)) * 2
            y = np.clip(y_base + noise, 0, 85)

            ax1.plot(x, y, color=COLORS['blue'], linewidth=2, alpha=0.8)
            ax1.fill_between(x, 0, y, alpha=0.1, color=COLORS['blue'])

            # Current accuracy marker
            ax1.plot(x[-1], y[-1], 'o', color=COLORS['red'], markersize=8, zorder=5)
            ax1.text(x[-1] + 2, y[-1], f'{y[-1]:.0f}%', fontsize=11,
                    color=COLORS['red'], fontweight='bold')

        # Milestone markers
        milestones = [
            (15, 'Self-check\nemerges'),
            (40, '"Wait, let me\nreconsider..."'),
            (65, 'Strategy\nswitching'),
            (85, 'Full\nreasoning'),
        ]
        for mx, mlabel in milestones:
            if x_max >= mx:
                my = 75 * (1 - np.exp(-mx / 25)) + 5
                ax1.axvline(x=mx, color=COLORS['orange'], alpha=0.3, linestyle='--')
                ax1.text(mx, 90, mlabel, fontsize=8, ha='center', color=COLORS['orange'],
                        fontweight='bold', va='bottom')

        # Right: Behavior emergence panel
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.axis('off')
        ax2.set_title('Emergent Behaviors', fontsize=12, fontweight='bold',
                      color=COLORS['orange'])

        behaviors = [
            (15, 'Simple step-by-step', '#90CAF9'),
            (30, 'Self-verification', '#81C784'),
            (45, '"Wait, let me rethink..."', '#FFB74D'),
            (60, 'Error correction', '#CE93D8'),
            (75, 'Strategy switching', '#EF9A9A'),
            (90, 'Full chain-of-thought', '#80CBC4'),
        ]

        for i, (threshold, label, color) in enumerate(behaviors):
            y_pos = 8.5 - i * 1.4
            active = x_max >= threshold

            if active:
                box = FancyBboxPatch((0.5, y_pos - 0.4), 9, 0.9,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, edgecolor='#666',
                                    linewidth=1, alpha=0.8)
                ax2.add_patch(box)
                ax2.text(0.8, y_pos, 'OK', fontsize=11, fontweight='bold',
                        color='#333', va='center')
                ax2.text(2.0, y_pos, label, fontsize=11, color='#333',
                        va='center', fontweight='bold')
            else:
                box = FancyBboxPatch((0.5, y_pos - 0.4), 9, 0.9,
                                    boxstyle="round,pad=0.1",
                                    facecolor='#F5F5F5', edgecolor='#E0E0E0',
                                    linewidth=1, alpha=0.5)
                ax2.add_patch(box)
                ax2.text(0.8, y_pos, '...', fontsize=11, color='#CCC', va='center')
                ax2.text(2.0, y_pos, label, fontsize=11, color='#CCC',
                        va='center')

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        frames.append(fig_to_frame(fig))
        plt.close(fig)

    # Hold final frame
    for _ in range(4):
        frames.append(frames[-1])

    save_gif(frames, 'aha-moment.gif', fps=2)


# ============================================================
# GIF 4: Distillation Funnel
# Shows: 671B → smaller models with capability preservation
# ============================================================
def generate_distillation():
    print("Generating distillation.gif...")

    frames = []

    models = [
        ('R1 Full', '671B', 79.8, '#1565C0'),
        ('Distill-70B', '70B', 70.0, '#1976D2'),
        ('Distill-32B', '32B', 72.6, '#1E88E5'),
        ('Distill-14B', '14B', 69.7, '#42A5F5'),
        ('Distill-7B', '7B', 55.5, '#64B5F6'),
        ('Distill-1.5B', '1.5B', 28.9, '#90CAF9'),
    ]

    # OpenAI reference line
    o1_mini_score = 63.6

    # All tick labels (pre-set for consistent frame size)
    all_xlabels = [f'{m[0]}\n({m[1]})' for m in models]

    for show_count in range(len(models) + 3):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.set_facecolor('#ffffff')
        fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.18)
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel('AIME 2024 Accuracy (%)', fontsize=12)
        ax.set_title('Distillation: 671B Knowledge Compressed', fontsize=16,
                    fontweight='bold', color=COLORS['dark'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', alpha=0.3)

        # Always set all x ticks for consistent frame size
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(all_xlabels, fontsize=9)

        # Draw o1-mini reference line
        if show_count >= 2:
            ax.axhline(y=o1_mini_score, color=COLORS['wrong'], linestyle='--',
                      alpha=0.6, linewidth=1.5)
            ax.text(5.8, o1_mini_score + 2, f'OpenAI o1-mini ({o1_mini_score}%)',
                    fontsize=10, color=COLORS['wrong'], ha='right', fontweight='bold')

        n = min(show_count, len(models))

        for i in range(n):
            name, params, score, color = models[i]

            ax.bar(i, score, width=0.6, color=color, edgecolor='white',
                        linewidth=1, alpha=0.9)

            ax.text(i, score + 1.5, f'{score}%', ha='center', fontsize=11,
                    fontweight='bold', color=COLORS['dark'])

            # Highlight 32B beating o1-mini
            if i == 2 and show_count >= 4:
                ax.annotate('Beats o1-mini!', xy=(i, score),
                           xytext=(i + 1.2, score + 8),
                           fontsize=10, fontweight='bold', color=COLORS['correct'],
                           arrowprops=dict(arrowstyle='->', color=COLORS['correct'],
                                         lw=1.5))

        # Bottom annotation
        if show_count >= len(models):
            ax.text(2.5, 5, '671B to 1.5B: 447x smaller, still can reason!',
                    fontsize=12, ha='center', color=COLORS['orange'], fontweight='bold')

        frames.append(fig_to_frame(fig, dpi=100))
        plt.close(fig)

    # Hold final
    for _ in range(4):
        frames.append(frames[-1])

    save_gif(frames, 'distillation.gif', fps=1)


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 50)
    print("Generating figures for DeepSeek-R1 article")
    print("=" * 50)

    generate_grpo_process()
    generate_thinking_vs_direct()
    generate_aha_moment()
    generate_distillation()

    print("\nAll figures generated!")
