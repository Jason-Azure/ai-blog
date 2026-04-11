#!/usr/bin/env python3
"""
Generate figures for the multimodal LLM architecture blog post.
4 figures: patch_embedding, clip_contrastive, multimodal_architecture, training_stages

Usage:
    source ~/ai-lab-venv/bin/activate
    cd ~/ai-blog/content/posts/multimodal-llm-architecture/
    python3 generate_figures.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# -- Style constants --
BG_COLOR = '#FAFAFA'
ACCENT_ORANGE = '#FF9800'
ACCENT_BLUE = '#2196F3'
ACCENT_GREEN = '#4CAF50'
ACCENT_PURPLE = '#9C27B0'
ACCENT_RED = '#E91E63'
DARK_TEXT = '#333333'
LIGHT_TEXT = '#666666'
GRID_COLOR = '#E0E0E0'

plt.rcParams['font.family'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def draw_rounded_box(ax, x, y, w, h, color, text, fontsize=11, text_color='white', alpha=0.9):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='none', alpha=alpha,
                          transform=ax.transData)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight='bold')


def draw_arrow(ax, x1, y1, x2, y2, color='#888888'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))


# ============================================================
# Figure 1: Patch Embedding (ViT)
# ============================================================
def gen_patch_embedding():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=BG_COLOR)
    fig.suptitle('Vision Transformer: Image → Patch → Embedding',
                 fontsize=16, fontweight='bold', color=DARK_TEXT, y=0.98)

    # Panel 1: Original image (simulated as colored grid)
    ax1 = axes[0]
    ax1.set_facecolor(BG_COLOR)
    np.random.seed(42)
    img = np.random.rand(14, 14, 3) * 0.5 + 0.3
    # Make a cat-like pattern
    img[3:11, 4:10, :] = [0.85, 0.65, 0.4]  # body
    img[3:7, 3:6, :] = [0.8, 0.6, 0.35]     # head
    img[3:7, 8:11, :] = [0.8, 0.6, 0.35]     # head
    img[4, 4, :] = [0.2, 0.2, 0.2]           # eye
    img[4, 9, :] = [0.2, 0.2, 0.2]           # eye

    ax1.imshow(img, interpolation='nearest')
    ax1.set_title('224×224 Image', fontsize=13, color=DARK_TEXT, pad=10)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.spines[:].set_visible(False)

    # Panel 2: Patches with grid
    ax2 = axes[1]
    ax2.set_facecolor(BG_COLOR)
    ax2.imshow(img, interpolation='nearest')

    # Draw grid lines
    for i in range(15):
        ax2.axhline(y=i-0.5, color='white', linewidth=2)
        ax2.axvline(x=i-0.5, color='white', linewidth=2)

    # Highlight one patch
    rect = mpatches.FancyBboxPatch((1.5, 3.5), 1, 1, linewidth=3,
                                    edgecolor=ACCENT_ORANGE, facecolor='none',
                                    boxstyle="square,pad=0")
    ax2.add_patch(rect)

    ax2.set_title('14×14 = 196 Patches\n(each 16×16 pixels)', fontsize=13, color=DARK_TEXT, pad=10)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.spines[:].set_visible(False)

    # Panel 3: Embedding vectors
    ax3 = axes[2]
    ax3.set_facecolor(BG_COLOR)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')

    # Draw patch tokens as colored rows
    colors = [ACCENT_BLUE, ACCENT_GREEN, ACCENT_ORANGE, ACCENT_PURPLE,
              ACCENT_RED, '#607D8B', ACCENT_BLUE, ACCENT_GREEN]
    labels = ['patch₁', 'patch₂', 'patch₃', '...', 'patch₁₉₅', 'patch₁₉₆', '[CLS]', '[POS]']

    for i, (label, color) in enumerate(zip(labels, colors)):
        y_pos = 9.0 - i * 1.1
        if label == '...':
            ax3.text(5, y_pos + 0.2, '⋮', ha='center', va='center',
                     fontsize=20, color=LIGHT_TEXT)
            continue

        # Draw a thin colored bar representing 1024-dim vector
        bar = FancyBboxPatch((1.5, y_pos), 7, 0.5,
                              boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='none', alpha=0.7)
        ax3.add_patch(bar)
        ax3.text(0.5, y_pos + 0.25, label, ha='center', va='center',
                 fontsize=10, color=DARK_TEXT, fontweight='bold')

        # Add dimension markers
        for j in range(20):
            x_pos = 1.7 + j * 0.33
            val = np.random.randn() * 0.3
            ax3.plot([x_pos, x_pos], [y_pos + 0.1, y_pos + 0.4],
                     color='white', alpha=0.3, linewidth=1)

    ax3.set_title('196 Embedding Vectors\n(each 1024-dim)', fontsize=13, color=DARK_TEXT, pad=10)

    # Add arrows between panels
    fig.text(0.35, 0.5, '→', fontsize=30, color=ACCENT_ORANGE,
             ha='center', va='center', fontweight='bold')
    fig.text(0.66, 0.5, '→', fontsize=30, color=ACCENT_ORANGE,
             ha='center', va='center', fontweight='bold')

    # Bottom annotation
    fig.text(0.18, 0.02, 'Pixels', ha='center', fontsize=11, color=LIGHT_TEXT, style='italic')
    fig.text(0.50, 0.02, 'Split into patches', ha='center', fontsize=11, color=LIGHT_TEXT, style='italic')
    fig.text(0.82, 0.02, 'Linear Projection', ha='center', fontsize=11, color=LIGHT_TEXT, style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig('patch_embedding.png', dpi=150, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close()
    print("✓ patch_embedding.png")


# ============================================================
# Figure 2: CLIP Contrastive Learning
# ============================================================
def gen_clip_contrastive():
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    fig.suptitle('CLIP: Contrastive Learning\nMaximize similarity of matched pairs, minimize others',
                 fontsize=15, fontweight='bold', color=DARK_TEXT, y=0.98)

    N = 5
    images = ['🐱 cat photo', '🏔 mountain', '🚗 red car', '🌻 sunflower', '🏠 house']
    texts = ['"a cat sitting"', '"mountain view"', '"red sports car"', '"yellow flower"', '"a cozy house"']

    # Draw the similarity matrix
    mat_x, mat_y = 2.5, 1.0
    cell_w, cell_h = 1.2, 0.8

    # Column headers (text)
    for j, t in enumerate(texts):
        ax.text(mat_x + j * cell_w + cell_w/2, mat_y + N * cell_h + 0.5,
                t, ha='center', va='center', fontsize=8, color=DARK_TEXT,
                rotation=30, fontweight='bold')

    # Row headers (images)
    for i, img in enumerate(images):
        ax.text(mat_x - 0.3, mat_y + (N - 1 - i) * cell_h + cell_h/2,
                img, ha='right', va='center', fontsize=9, color=DARK_TEXT)

    # Draw cells
    for i in range(N):
        for j in range(N):
            x = mat_x + j * cell_w
            y = mat_y + (N - 1 - i) * cell_h

            if i == j:
                # Diagonal = matched pairs → high similarity
                color = ACCENT_GREEN
                val = f'{0.85 + np.random.rand()*0.1:.2f}'
                text_c = 'white'
            else:
                # Off-diagonal = unmatched → low similarity
                sim = 0.05 + np.random.rand() * 0.15
                color = '#E8E8E8'
                val = f'{sim:.2f}'
                text_c = LIGHT_TEXT

            rect = FancyBboxPatch((x + 0.05, y + 0.05), cell_w - 0.1, cell_h - 0.1,
                                   boxstyle="round,pad=0.02",
                                   facecolor=color, edgecolor='#DDD', linewidth=0.5)
            ax.add_patch(rect)
            ax.text(x + cell_w/2, y + cell_h/2, val,
                    ha='center', va='center', fontsize=10,
                    color=text_c, fontweight='bold' if i == j else 'normal')

    # Labels
    ax.text(mat_x + N * cell_w / 2, mat_y + N * cell_h + 1.8,
            'Text Encoder Output', ha='center', fontsize=12,
            color=ACCENT_BLUE, fontweight='bold')
    ax.text(mat_x - 1.5, mat_y + N * cell_h / 2,
            'Image Encoder\nOutput', ha='center', va='center', fontsize=12,
            color=ACCENT_ORANGE, fontweight='bold', rotation=90)

    # Legend
    legend_y = 0.3
    rect1 = FancyBboxPatch((mat_x, legend_y), 0.4, 0.3,
                             boxstyle="round,pad=0.02",
                             facecolor=ACCENT_GREEN, edgecolor='none')
    ax.add_patch(rect1)
    ax.text(mat_x + 0.6, legend_y + 0.15, '= Matched pair (maximize)',
            va='center', fontsize=10, color=DARK_TEXT)

    rect2 = FancyBboxPatch((mat_x + 4.0, legend_y), 0.4, 0.3,
                             boxstyle="round,pad=0.02",
                             facecolor='#E8E8E8', edgecolor='#DDD')
    ax.add_patch(rect2)
    ax.text(mat_x + 4.6, legend_y + 0.15, '= Unmatched pair (minimize)',
            va='center', fontsize=10, color=DARK_TEXT)

    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.3, 7.5)

    fig.savefig('clip_contrastive.png', dpi=150, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close()
    print("✓ clip_contrastive.png")


# ============================================================
# Figure 3: Multimodal LLM Architecture (Three-Stage)
# ============================================================
def gen_multimodal_architecture():
    fig, ax = plt.subplots(figsize=(14, 9), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)

    fig.suptitle('Multimodal LLM Architecture',
                 fontsize=18, fontweight='bold', color=DARK_TEXT, y=0.97)

    # ---- Image input path (top) ----
    # Image icon
    draw_rounded_box(ax, 0.3, 6.5, 2.0, 1.5, '#BBDEFB', 'Input\nImage', 12, DARK_TEXT)

    # Vision Encoder
    draw_rounded_box(ax, 3.5, 6.5, 2.5, 1.5, ACCENT_ORANGE, 'Vision Encoder\n(ViT / CLIP)', 11, 'white')
    ax.text(4.75, 6.2, '196 visual tokens\n(each 1024-dim)', ha='center', fontsize=8, color=LIGHT_TEXT)

    # Arrow: image -> vision encoder
    draw_arrow(ax, 2.3, 7.25, 3.5, 7.25, ACCENT_ORANGE)

    # Projection / Alignment
    draw_rounded_box(ax, 7.0, 6.5, 2.5, 1.5, ACCENT_PURPLE, 'Alignment\nModule', 12, 'white')
    ax.text(8.25, 6.2, 'Linear / Q-Former\n1024→4096 dim', ha='center', fontsize=8, color=LIGHT_TEXT)

    # Arrow: vision encoder -> projection
    draw_arrow(ax, 6.0, 7.25, 7.0, 7.25, ACCENT_PURPLE)

    # ---- Text input path (bottom) ----
    draw_rounded_box(ax, 0.3, 2.5, 2.0, 1.5, '#C8E6C9', 'Input\nText', 12, DARK_TEXT)

    # Tokenizer + Embedding
    draw_rounded_box(ax, 3.5, 2.5, 2.5, 1.5, ACCENT_GREEN, 'Tokenizer +\nEmbedding', 11, 'white')
    ax.text(4.75, 2.2, 'N text tokens\n(each 4096-dim)', ha='center', fontsize=8, color=LIGHT_TEXT)

    # Arrow: text -> tokenizer
    draw_arrow(ax, 2.3, 3.25, 3.5, 3.25, ACCENT_GREEN)

    # ---- Merge point ----
    # Visual tokens + text tokens merge
    merge_x, merge_y = 9.8, 5.0
    ax.text(merge_x + 0.5, merge_y + 0.3, '+', ha='center', va='center',
            fontsize=30, color=ACCENT_BLUE, fontweight='bold')

    # Arrows to merge point
    draw_arrow(ax, 9.5, 7.0, merge_x + 0.2, merge_y + 0.8, '#888')
    draw_arrow(ax, 6.0, 3.25, merge_x, merge_y - 0.2, '#888')

    # Annotation: merged sequence
    ax.text(merge_x + 0.5, merge_y - 0.6,
            '[vis₁][vis₂]...[vis₁₉₆][tok₁][tok₂]...[tokₙ]',
            ha='center', fontsize=8, color=LIGHT_TEXT, family='monospace')

    # ---- LLM ----
    draw_rounded_box(ax, 11.0, 3.8, 2.5, 2.8, ACCENT_BLUE, 'LLM\n(Transformer)\n\nMulti-Head\nAttention\n×N layers', 11, 'white')

    # Arrow: merge -> LLM
    draw_arrow(ax, merge_x + 1.0, merge_y + 0.3, 11.0, 5.2, ACCENT_BLUE)

    # ---- Output ----
    ax.text(12.25, 3.2, '↓', fontsize=24, ha='center', color=ACCENT_BLUE)
    draw_rounded_box(ax, 11.0, 1.5, 2.5, 1.2, '#E3F2FD', 'Text Output\n(Generated Response)', 10, DARK_TEXT)

    # ---- Labels for the three stages ----
    # Stage 1
    ax.add_patch(FancyBboxPatch((3.2, 8.5), 3.0, 0.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor=ACCENT_ORANGE, alpha=0.15, edgecolor=ACCENT_ORANGE, linewidth=1))
    ax.text(4.7, 8.7, '① Visual Encoder', ha='center', fontsize=10,
            color=ACCENT_ORANGE, fontweight='bold')

    # Stage 2
    ax.add_patch(FancyBboxPatch((6.7, 8.5), 3.0, 0.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor=ACCENT_PURPLE, alpha=0.15, edgecolor=ACCENT_PURPLE, linewidth=1))
    ax.text(8.2, 8.7, '② Alignment', ha='center', fontsize=10,
            color=ACCENT_PURPLE, fontweight='bold')

    # Stage 3
    ax.add_patch(FancyBboxPatch((10.7, 8.5), 3.0, 0.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor=ACCENT_BLUE, alpha=0.15, edgecolor=ACCENT_BLUE, linewidth=1))
    ax.text(12.2, 8.7, '③ Language Model', ha='center', fontsize=10,
            color=ACCENT_BLUE, fontweight='bold')

    fig.savefig('multimodal_architecture.png', dpi=150, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close()
    print("✓ multimodal_architecture.png")


# ============================================================
# Figure 4: Training Stages (Freeze/Unfreeze)
# ============================================================
def gen_training_stages():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG_COLOR)
    fig.suptitle('Two-Stage Training: What to Freeze, What to Train',
                 fontsize=16, fontweight='bold', color=DARK_TEXT, y=0.98)

    for idx, ax in enumerate(axes):
        ax.set_facecolor(BG_COLOR)
        ax.axis('off')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)

        if idx == 0:
            title = 'Stage 1: Alignment Pre-training'
            subtitle = 'Learn to bridge vision and language'
            states = [
                ('Vision Encoder', ACCENT_ORANGE, True),   # frozen
                ('Alignment Module', ACCENT_PURPLE, False), # trained
                ('LLM', ACCENT_BLUE, True),                 # frozen
            ]
            data_text = 'Data: 558K image-text pairs'
        else:
            title = 'Stage 2: Visual Instruction Tuning'
            subtitle = 'Learn to follow visual instructions'
            states = [
                ('Vision Encoder', ACCENT_ORANGE, True),   # frozen
                ('Alignment Module', ACCENT_PURPLE, False), # trained
                ('LLM (LoRA)', ACCENT_BLUE, False),        # trained (LoRA)
            ]
            data_text = 'Data: 158K visual Q&A pairs'

        ax.text(5, 7.5, title, ha='center', fontsize=13, fontweight='bold', color=DARK_TEXT)
        ax.text(5, 7.0, subtitle, ha='center', fontsize=10, color=LIGHT_TEXT, style='italic')

        for i, (name, color, frozen) in enumerate(states):
            y = 5.0 - i * 1.8
            w = 7.0
            h = 1.2
            x = 1.5

            if frozen:
                # Frozen: gray with ice pattern
                box_color = '#E0E0E0'
                edge_color = '#BDBDBD'
                text_color = '#888'
                status = '❄ FROZEN'
                status_color = '#90CAF9'
            else:
                # Training: bright color with fire
                box_color = color
                edge_color = color
                text_color = 'white'
                status = '🔥 TRAINING'
                status_color = ACCENT_RED

            box = FancyBboxPatch((x, y), w, h,
                                  boxstyle="round,pad=0.05",
                                  facecolor=box_color, edgecolor=edge_color,
                                  linewidth=2, alpha=0.85)
            ax.add_patch(box)

            ax.text(x + w/2 - 1.0, y + h/2, name,
                    ha='center', va='center', fontsize=12,
                    color=text_color, fontweight='bold')
            ax.text(x + w - 0.8, y + h/2, status,
                    ha='center', va='center', fontsize=9,
                    color=status_color if frozen else 'white', fontweight='bold')

        ax.text(5, 0.8, data_text, ha='center', fontsize=10,
                color=LIGHT_TEXT, style='italic')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig('training_stages.png', dpi=150, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none')
    plt.close()
    print("✓ training_stages.png")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Generating figures for multimodal LLM article...")
    gen_patch_embedding()
    gen_clip_contrastive()
    gen_multimodal_architecture()
    gen_training_stages()
    print("\nAll figures generated successfully!")
