#!/usr/bin/env python3
"""
Generate polished figures for the Agent/MCP Reality article.
3 GIFs: elegant, clean, thought-provoking visuals.

Usage:
    source ~/ai-lab-venv/bin/activate
    cd ~/ai-blog/content/posts/agent-mcp-reality/
    python generate_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
import matplotlib.patheffects as pe
import imageio
import io
import os

plt.rcParams['font.family'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Elegant color palette
C = {
    'bg': '#FAFBFC',
    'dark': '#1a1a2e',
    'mid': '#16213e',
    'accent1': '#e94560',   # warm red
    'accent2': '#0f3460',   # deep blue
    'accent3': '#533483',   # purple
    'gold': '#f0a500',
    'green': '#4CAF50',
    'gray': '#8896AB',
    'light': '#e8edf3',
    'white': '#ffffff',
}

def fig_to_frame(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor(), pad_inches=0.4)
    buf.seek(0)
    frame = imageio.v3.imread(buf)
    buf.close()
    return frame

def save_gif(frames, name, fps=1.5):
    path = os.path.join(OUTPUT_DIR, name)
    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)
    padded = []
    for f in frames:
        h, w = f.shape[:2]
        if h < max_h or w < max_w:
            f = np.pad(f, ((0, max_h-h), (0, max_w-w), (0, 0)),
                      mode='constant', constant_values=255)
        padded.append(f)
    imageio.mimsave(path, padded, fps=fps, loop=0)
    size = os.path.getsize(path)
    print(f"  OK {name} ({size/1024:.0f} KB)")


# ============================================================
# GIF 1: The Layers — what's real vs what's packaging
# An elegant "peeling" animation showing layers of AI stack
# ============================================================
def generate_layers():
    print("Generating layers.gif...")
    frames = []

    layers = [
        ('AI Products You See', 'ChatGPT / Claude / Kimi / Manus', C['accent1'], 0.15),
        ('Agent / MCP / RAG / Skills', 'The "hot words" layer', C['accent3'], 0.20),
        ('LLM (Large Language Model)', 'The real engine', C['accent2'], 0.30),
        ('Transformer Architecture', 'The foundation', C['mid'], 0.25),
    ]

    for reveal in range(len(layers) + 2):
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.set_facecolor(C['bg'])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8.5)
        ax.axis('off')

        ax.text(5, 8.0, 'What is real? What is packaging?',
                fontsize=17, fontweight='bold', ha='center', color=C['dark'],
                style='italic')

        n = min(reveal, len(layers))
        total_h = 1.2 * n + 0.3 * (n - 1) if n > 0 else 0
        start_y = 3.5 + total_h / 2

        for i in range(n):
            y = start_y - i * 1.5
            w = 7.5 - i * 0.6
            x = 5 - w / 2

            name, desc, color, alpha_val = layers[i]

            # Highlight layer 1 (hot words) specially
            if i == 1 and reveal >= len(layers):
                ec = C['accent1']
                lw = 2.5
                ls = '--'
            else:
                ec = color
                lw = 1.5
                ls = '-'

            box = FancyBboxPatch((x, y - 0.5), w, 1.0,
                                boxstyle="round,pad=0.15",
                                facecolor=color, edgecolor=ec,
                                linewidth=lw, linestyle=ls,
                                alpha=0.85 if i != 1 or reveal < len(layers) else 0.4)
            ax.add_patch(box)

            ax.text(5, y + 0.15, name, fontsize=13, fontweight='bold',
                    ha='center', va='center', color=C['white'])
            ax.text(5, y - 0.2, desc, fontsize=9,
                    ha='center', va='center', color='#d0d8e8', style='italic')

        # Arrow and annotation on final frames
        if reveal >= len(layers):
            # Arrow pointing to layer 1 (hot words)
            y_hot = start_y - 1 * 1.5
            ax.annotate('', xy=(1.2, y_hot), xytext=(0.5, y_hot),
                        arrowprops=dict(arrowstyle='->', color=C['accent1'],
                                       lw=2))
            ax.text(0.3, y_hot + 0.6, 'This layer\nchanges fastest',
                    fontsize=9, color=C['accent1'], fontweight='bold',
                    ha='left')

            y_llm = start_y - 2 * 1.5
            ax.annotate('', xy=(8.5, y_llm), xytext=(9.2, y_llm),
                        arrowprops=dict(arrowstyle='->', color=C['green'],
                                       lw=2))
            ax.text(9.4, y_llm + 0.6, 'This layer\nis the real power',
                    fontsize=9, color=C['green'], fontweight='bold',
                    ha='right')

        if reveal >= len(layers) + 1:
            ax.text(5, 0.6, 'Tools come and go. The engine stays.',
                    fontsize=13, ha='center', color=C['gray'],
                    style='italic', fontweight='bold')

        frames.append(fig_to_frame(fig))
        plt.close(fig)

    for _ in range(4):
        frames.append(frames[-1])

    save_gif(frames, 'layers.gif', fps=1)


# ============================================================
# GIF 2: The Middleware Cycle — history repeats
# Elegant timeline showing rise and fall of middleware
# ============================================================
def generate_middleware_cycle():
    print("Generating middleware-cycle.gif...")
    frames = []

    eras = [
        ('2000s', 'ESB', 'Enterprise\nService Bus', '#E57373',
         'Promised to\nconnect everything', 'Replaced by\nmicroservices'),
        ('2010s', 'SOA', 'Service-Oriented\nArchitecture', '#FFB74D',
         'Promised to\nunify services', 'Replaced by\nREST APIs'),
        ('2020s', 'Agent\nFrameworks', 'LangChain / MCP\n/ Coze / Dify', '#64B5F6',
         'Promising to\nconnect AI + tools', '???'),
    ]

    for show in range(len(eras) + 2):
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.set_facecolor(C['bg'])
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(0, 6)
        ax.axis('off')

        ax.text(5, 5.5, 'The Middleware Cycle: History Repeats',
                fontsize=16, fontweight='bold', ha='center', color=C['dark'])

        # Timeline base line
        ax.plot([0.5, 9.5], [2.8, 2.8], color=C['gray'], linewidth=1.5, alpha=0.3)

        n = min(show, len(eras))
        for i in range(n):
            era, name, full_name, color, promise, fate = eras[i]
            x = 1.5 + i * 3.5

            # Era dot
            circle = Circle((x, 2.8), 0.15, facecolor=color, edgecolor='white',
                           linewidth=2, zorder=5)
            ax.add_patch(circle)

            # Era label
            ax.text(x, 2.3, era, fontsize=10, ha='center', color=C['gray'],
                    fontweight='bold')

            # Name box above
            box = FancyBboxPatch((x - 1.2, 3.2), 2.4, 1.5,
                                boxstyle="round,pad=0.15",
                                facecolor=color, edgecolor='white',
                                linewidth=1, alpha=0.85)
            ax.add_patch(box)
            ax.text(x, 4.25, name, fontsize=12, fontweight='bold',
                    ha='center', va='center', color='white')
            ax.text(x, 3.7, full_name, fontsize=8, ha='center', va='center',
                    color='#ffffffcc')

            # Promise below
            ax.text(x, 1.7, promise, fontsize=8, ha='center', va='center',
                    color=color, style='italic')

            # Fate (crossed out for past eras)
            if i < 2 and show >= len(eras):
                ax.text(x, 0.9, fate, fontsize=8, ha='center', va='center',
                        color=C['accent1'], fontweight='bold')
                # Strikethrough the name box
                ax.plot([x - 1.0, x + 1.0], [4.1, 3.8], color='white',
                        linewidth=2, alpha=0.6)

            if i == 2 and show >= len(eras) + 1:
                ax.text(x, 0.9, fate, fontsize=14, ha='center', va='center',
                        color=C['accent1'], fontweight='bold')

        # Bottom quote on final frame
        if show >= len(eras) + 1:
            ax.text(5, 0.15, 'Every generation believes its middleware is different.',
                    fontsize=11, ha='center', color=C['gray'], style='italic')

        frames.append(fig_to_frame(fig))
        plt.close(fig)

    for _ in range(5):
        frames.append(frames[-1])

    save_gif(frames, 'middleware-cycle.gif', fps=1)


# ============================================================
# GIF 3: What Remains — tools change, questions stay
# A beautiful concentric circle diagram
# ============================================================
def generate_what_remains():
    print("Generating what-remains.gif...")
    frames = []

    rings = [
        (3.8, 'Tools (MCP, Agent, RAG...)', C['light'], C['gray'], 10),
        (2.8, 'Models (GPT, Claude, DeepSeek...)', '#d4e6f9', C['accent2'], 9),
        (1.8, 'Algorithms (Transformer, RL, MoE...)', '#e8d5f5', C['accent3'], 9),
        (1.0, 'The Question:\n"What should we do?"', '#fff3e0', C['gold'], 11),
    ]

    labels_outer = [
        (0, 'MCP'), (45, 'LangChain'), (90, 'Agent'), (135, 'Coze'),
        (180, 'RAG'), (225, 'Dify'), (270, 'Skills'), (315, 'Manus'),
    ]

    for phase in range(6):
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.set_facecolor(C['bg'])
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.axis('off')

        ax.text(0, 4.6, 'What Changes. What Remains.',
                fontsize=16, fontweight='bold', ha='center', color=C['dark'])

        n_rings = min(phase, len(rings))
        for i in range(n_rings):
            r, label, fc, ec, fs = rings[i]

            if i == 0 and phase >= 5:
                # Outer ring fading
                alpha = 0.3
                ls = '--'
            else:
                alpha = 0.7 if i == 0 else 0.8
                ls = '-'

            circle = Circle((0, 0), r, facecolor=fc, edgecolor=ec,
                           linewidth=2, alpha=alpha, linestyle=ls, zorder=i+1)
            ax.add_patch(circle)

            if i < 3:
                ax.text(0, -r + 0.3, label, fontsize=fs, ha='center',
                        va='center', color=ec, fontweight='bold',
                        zorder=10, alpha=0.9 if not (i==0 and phase>=5) else 0.4)
            else:
                ax.text(0, 0, label, fontsize=fs, ha='center',
                        va='center', color=ec, fontweight='bold',
                        zorder=10)

        # Outer labels for tools
        if phase >= 1:
            for angle_deg, name in labels_outer:
                angle_rad = np.radians(angle_deg)
                x = 4.0 * np.cos(angle_rad)
                y = 4.0 * np.sin(angle_rad) - 0.2

                alpha_val = 0.3 if phase >= 5 else 0.6
                ax.text(x, y, name, fontsize=8, ha='center', va='center',
                        color=C['gray'], alpha=alpha_val, fontweight='bold')

        # Arrows showing change
        if phase >= 5:
            ax.text(0, -4.5, 'The outer ring changes every 2-3 years.\nThe center never changes.',
                    fontsize=10, ha='center', color=C['gray'], style='italic')

            # Circular arrow suggesting rotation on outer ring
            for angle in [30, 120, 210, 300]:
                rad = np.radians(angle)
                x1 = 3.9 * np.cos(rad)
                y1 = 3.9 * np.sin(rad) - 0.2
                x2 = 3.9 * np.cos(rad + 0.3)
                y2 = 3.9 * np.sin(rad + 0.3) - 0.2
                ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                           arrowprops=dict(arrowstyle='->', color=C['accent1'],
                                          lw=1.5, alpha=0.4))

        frames.append(fig_to_frame(fig))
        plt.close(fig)

    for _ in range(5):
        frames.append(frames[-1])

    save_gif(frames, 'what-remains.gif', fps=1)


if __name__ == '__main__':
    print("=" * 50)
    print("Generating figures for Agent/MCP Reality article")
    print("=" * 50)

    generate_layers()
    generate_middleware_cycle()
    generate_what_remains()

    print("\nAll figures generated!")
