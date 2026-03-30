#!/usr/bin/env python3
"""
Regenerate GIF animations for the ascii-to-token article.
Fixes: text overlap, speed (doubled frame duration), clarity.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
import io
import os
import numpy as np

# ── Font config ──
plt.rcParams['font.family'] = ['WenQuanYi Zen Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def fig_to_pil(fig, dpi=150):
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.3)
    buf.seek(0)
    img = Image.open(buf).convert('RGBA')
    return img


def draw_rounded_box(ax, x, y, w, h, color, text, text_color='white',
                     fontsize=22, border_color=None, border_width=2, alpha=1.0):
    """Draw a rounded rectangle with centered text."""
    if border_color is None:
        # Darken the fill color for border
        import matplotlib.colors as mcolors
        rgb = mcolors.to_rgb(color)
        border_color = tuple(max(0, c - 0.15) for c in rgb)

    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor=border_color,
        linewidth=border_width, alpha=alpha
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight='bold')


# ══════════════════════════════════════════════════════════════════════════════
#  GIF 1: UTF-8 Encoding of "你好"
# ══════════════════════════════════════════════════════════════════════════════

def create_utf8_frame0():
    """Frame 0: Title - '你好' introduction."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Main title
    ax.text(5, 5.2, '你好', ha='center', va='center',
            fontsize=72, color='#2c3e50', fontweight='bold')

    # Subtitle
    ax.text(5, 3.4, '这两个字，计算机是怎么存储的？',
            ha='center', va='center', fontsize=24, color='#e74c3c')

    # Sub-subtitle
    ax.text(5, 2.2, '让我们一步步拆解...',
            ha='center', va='center', fontsize=18, color='#999999')

    return fig_to_pil(fig)


def create_utf8_frame1():
    """Frame 1: Step 1 - Unicode code point lookup."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(5, 6.3, 'Step 1: Unicode 码点查找', ha='center', va='center',
            fontsize=28, color='#2c3e50', fontweight='bold')

    # "你" row
    ax.text(2.0, 4.8, '"你"', ha='center', va='center',
            fontsize=42, color='#3498db', fontweight='bold')
    # Arrow
    ax.annotate('', xy=(4.5, 4.8), xytext=(3.2, 4.8),
                arrowprops=dict(arrowstyle='->', color='#666', lw=2))
    ax.text(3.85, 5.15, '查Unicode表', ha='center', va='center',
            fontsize=13, color='#888')
    # Box
    draw_rounded_box(ax, 4.8, 4.2, 3.2, 1.2, '#4CAF50', 'U+4F60',
                     fontsize=32, text_color='white')

    # "好" row
    ax.text(2.0, 2.6, '"好"', ha='center', va='center',
            fontsize=42, color='#3498db', fontweight='bold')
    # Arrow
    ax.annotate('', xy=(4.5, 2.6), xytext=(3.2, 2.6),
                arrowprops=dict(arrowstyle='->', color='#666', lw=2))
    ax.text(3.85, 2.95, '查Unicode表', ha='center', va='center',
            fontsize=13, color='#888')
    # Box
    draw_rounded_box(ax, 4.8, 2.0, 3.2, 1.2, '#4CAF50', 'U+597D',
                     fontsize=32, text_color='white')

    # Footer note
    ax.text(5, 0.7, 'Unicode: 给全球每个字符一个唯一编号',
            ha='center', va='center', fontsize=16, color='#666')

    return fig_to_pil(fig)


def create_utf8_frame2():
    """Frame 2: Step 2 - UTF-8 encode '你'."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(5, 6.3, 'Step 2: UTF-8 编码 "你"', ha='center', va='center',
            fontsize=28, color='#2c3e50', fontweight='bold')

    # Source: U+4F60
    ax.text(1.5, 4.6, 'U+4F60', ha='center', va='center',
            fontsize=30, color='#4CAF50', fontweight='bold')
    ax.text(1.5, 4.0, '(0x4F60)', ha='center', va='center',
            fontsize=14, color='#999')

    # Arrow with label - FIXED: moved label higher, more spacing
    ax.annotate('', xy=(4.0, 4.5), xytext=(2.7, 4.5),
                arrowprops=dict(arrowstyle='->', color='#666', lw=2))
    ax.text(3.35, 5.0, '三字节编码', ha='center', va='center',
            fontsize=14, color='#888')

    # 3 byte boxes
    byte_colors = ['#e74c3c', '#3498db', '#f39c12']
    byte_hex = ['E4', 'BD', 'A0']
    byte_bin = ['11100100', '10111101', '10100000']
    box_w = 1.5
    start_x = 4.2
    for i in range(3):
        x = start_x + i * (box_w + 0.3)
        draw_rounded_box(ax, x, 4.0, box_w, 1.2, byte_colors[i], byte_hex[i],
                         fontsize=28, text_color='white')
        ax.text(x + box_w/2, 3.7, byte_bin[i], ha='center', va='center',
                fontsize=11, color=byte_colors[i])

    # UTF-8 template explanation
    # Box for template
    template_box = FancyBboxPatch(
        (2.8, 1.8), 4.4, 0.6,
        boxstyle="round,pad=0.1",
        facecolor='#f5f5f5', edgecolor='#ccc', linewidth=1.5
    )
    ax.add_patch(template_box)
    ax.text(5, 2.1, 'UTF-8 三字节模板:', ha='center', va='center',
            fontsize=14, color='#555', fontweight='bold')

    ax.text(5, 1.3, '1110xxxx  10xxxxxx  10xxxxxx',
            ha='center', va='center', fontsize=16, color='#e74c3c',
            fontfamily='monospace')

    ax.text(5, 0.6, '将 U+4F60 的二进制位填入 x 的位置',
            ha='center', va='center', fontsize=15, color='#666')

    return fig_to_pil(fig)


def create_utf8_frame3():
    """Frame 3: Step 3 - UTF-8 encode '好'."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(5, 6.3, 'Step 3: UTF-8 编码 "好"', ha='center', va='center',
            fontsize=28, color='#2c3e50', fontweight='bold')

    # Source: U+597D
    ax.text(1.5, 4.6, 'U+597D', ha='center', va='center',
            fontsize=30, color='#4CAF50', fontweight='bold')
    ax.text(1.5, 4.0, '(0x597D)', ha='center', va='center',
            fontsize=14, color='#999')

    # Arrow with label - FIXED
    ax.annotate('', xy=(4.0, 4.5), xytext=(2.7, 4.5),
                arrowprops=dict(arrowstyle='->', color='#666', lw=2))
    ax.text(3.35, 5.0, '三字节编码', ha='center', va='center',
            fontsize=14, color='#888')

    # 3 byte boxes
    byte_colors = ['#9b59b6', '#1abc9c', '#e67e22']
    byte_hex = ['E5', 'A5', 'BD']
    byte_bin = ['11100101', '10100101', '10111101']
    box_w = 1.5
    start_x = 4.2
    for i in range(3):
        x = start_x + i * (box_w + 0.3)
        draw_rounded_box(ax, x, 4.0, box_w, 1.2, byte_colors[i], byte_hex[i],
                         fontsize=28, text_color='white')
        ax.text(x + box_w/2, 3.7, byte_bin[i], ha='center', va='center',
                fontsize=11, color=byte_colors[i])

    # Result box
    result_box = FancyBboxPatch(
        (2.2, 1.8), 5.6, 0.7,
        boxstyle="round,pad=0.1",
        facecolor='#f5f5f5', edgecolor='#ccc', linewidth=1.5
    )
    ax.add_patch(result_box)
    ax.text(5, 2.15, '"好" = E5 A5 BD  (3 bytes)',
            ha='center', va='center', fontsize=18, color='#333',
            fontweight='bold')

    return fig_to_pil(fig)


def create_utf8_frame4():
    """Frame 4: Final result with comparison."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(5, 6.5, '最终结果', ha='center', va='center',
            fontsize=32, color='#9b59b6', fontweight='bold')

    # "你好" text
    ax.text(1.2, 5.3, '"你好"', ha='center', va='center',
            fontsize=36, color='#3498db', fontweight='bold')
    ax.text(1.8, 4.7, '=', ha='center', va='center',
            fontsize=28, color='#333')

    # 6 byte boxes in a row
    ni_bytes = [('E4', '#e74c3c'), ('BD', '#e74c3c'), ('A0', '#e74c3c')]
    hao_bytes = [('E5', '#9b59b6'), ('A5', '#9b59b6'), ('BD', '#9b59b6')]
    all_bytes = ni_bytes + hao_bytes
    box_w = 0.95
    start_x = 2.3
    for i, (hex_val, color) in enumerate(all_bytes):
        x = start_x + i * (box_w + 0.15)
        draw_rounded_box(ax, x, 4.3, box_w, 0.8, color, hex_val,
                         fontsize=18, text_color='white')

    # Labels under byte groups
    ax.text(start_x + 1.5 * (box_w + 0.15) - 0.08, 3.9,
            '"你" = 3 bytes', ha='center', va='center',
            fontsize=13, color='#e74c3c')
    ax.text(start_x + 4.5 * (box_w + 0.15) - 0.08, 3.9,
            '"好" = 3 bytes', ha='center', va='center',
            fontsize=13, color='#9b59b6')

    # Total
    ax.text(5, 3.3, '共 6 bytes', ha='center', va='center',
            fontsize=22, color='#333', fontweight='bold')

    # Divider
    ax.plot([1, 9], [2.7, 2.7], '--', color='#ccc', lw=1)

    # Comparison section
    ax.text(5, 2.3, '对比: 变长编码的智慧', ha='center', va='center',
            fontsize=18, color='#555', fontweight='bold')

    ax.text(3.0, 1.6, '"A" = 41  (1 byte)', ha='center', va='center',
            fontsize=17, color='#4CAF50', fontweight='bold')
    ax.text(5, 1.6, 'vs', ha='center', va='center',
            fontsize=15, color='#999')
    ax.text(7.2, 1.6, '"你" = E4 BD A0  (3 bytes)', ha='center', va='center',
            fontsize=17, color='#e74c3c', fontweight='bold')

    ax.text(5, 0.7, '高频字符(英文)短编码, 低频字符(中文)长编码',
            ha='center', va='center', fontsize=15, color='#666')

    return fig_to_pil(fig)


def generate_utf8_gif():
    """Generate the UTF-8 encoding GIF."""
    print("Generating 03_unicode_utf8.gif ...")

    frames = [
        create_utf8_frame0(),
        create_utf8_frame1(),
        create_utf8_frame2(),
        create_utf8_frame3(),
        create_utf8_frame4(),
    ]

    # Resize all frames to match
    target_size = frames[0].size
    frames = [f.resize(target_size, Image.LANCZOS) if f.size != target_size else f
              for f in frames]

    # Convert to RGB (GIF doesn't support RGBA well)
    rgb_frames = []
    for f in frames:
        bg = Image.new('RGB', f.size, (255, 255, 255))
        bg.paste(f, mask=f.split()[3] if f.mode == 'RGBA' else None)
        rgb_frames.append(bg)

    output_path = os.path.join(OUTPUT_DIR, '03_unicode_utf8.gif')
    rgb_frames[0].save(
        output_path,
        save_all=True,
        append_images=rgb_frames[1:],
        duration=1600,  # 1600ms per frame (doubled from 800)
        loop=0,
        optimize=True
    )

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Saved: {output_path} ({size_kb:.0f} KB, {len(frames)} frames, 1600ms/frame)")
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
#  GIF 2: BPE Process
# ══════════════════════════════════════════════════════════════════════════════

# Colors
BLUE = '#4AADE8'
GRAY = '#B0B0B0'
RED_MERGE = '#E74C3C'
PURPLE = '#9B59B6'
TEAL = '#1ABC9C'
ORANGE = '#E67E22'
LIGHT_BLUE_BG = '#E0F2FE'
MERGE_BG = '#FFF8E1'


def draw_token_row(ax, tokens, y_center, colors, box_height=0.9,
                   fontsize=20, min_width=0.8, spacing=0.06):
    """Draw a row of token boxes, sizing each box to its content."""
    total_width = sum(max(min_width, len(t) * 0.55 + 0.3) for t in tokens) + \
                  spacing * (len(tokens) - 1)
    start_x = 5.5 - total_width / 2  # Center on 5.5 (wider canvas)

    x = start_x
    positions = []
    for i, token in enumerate(tokens):
        w = max(min_width, len(token) * 0.55 + 0.3)
        color = colors[i] if i < len(colors) else BLUE
        draw_rounded_box(ax, x, y_center - box_height/2, w, box_height,
                         color, token, fontsize=fontsize, text_color='white',
                         border_color=None, border_width=2)
        positions.append((x, w))
        x += w + spacing

    return positions


def draw_vocab_row(ax, vocab, y_start, initial_count=8):
    """Draw vocabulary items in rows."""
    box_w = 0.7
    box_h = 0.45
    spacing = 0.15
    items_per_row = 9
    x_start = 1.0

    for i, token in enumerate(vocab):
        row = i // items_per_row
        col = i % items_per_row
        x = x_start + col * (box_w + spacing)
        y = y_start - row * (box_h + 0.15)

        if i < initial_count:
            bg_color = LIGHT_BLUE_BG
            border_color = '#B0D4E8'
        else:
            bg_color = MERGE_BG
            border_color = '#F5C842'

        box = FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.04",
            facecolor=bg_color, edgecolor=border_color, linewidth=1.5
        )
        ax.add_patch(box)
        text_color = '#555' if i < initial_count else '#B8860B'
        ax.text(x + box_w/2, y + box_h/2, token, ha='center', va='center',
                fontsize=13, color=text_color, fontweight='bold')


def create_bpe_frame(step, title_sub, merge_text, tokens, token_colors,
                     vocab, highlight_merge=None, show_ids=False, ids=None,
                     summary_text=None):
    """Create a single BPE frame."""
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    # Main title
    ax.text(6.0, 7.0, 'BPE 分词：' + title_sub, ha='center', va='center',
            fontsize=26, color='#2c3e50', fontweight='bold')

    # Subtitle
    ax.text(6.0, 6.4, merge_text, ha='center', va='center',
            fontsize=16, color='#e74c3c')

    # Merge rule box (if applicable)
    if highlight_merge:
        rule_box = FancyBboxPatch(
            (4.5, 5.7), 3.0, 0.5,
            boxstyle="round,pad=0.08",
            facecolor='white', edgecolor='#4CAF50', linewidth=2
        )
        ax.add_patch(rule_box)
        ax.text(6.0, 5.95, highlight_merge, ha='center', va='center',
                fontsize=17, color='#2E7D32', fontweight='bold')

    # Token row
    y_tokens = 4.5
    positions = draw_token_row(ax, tokens, y_tokens, token_colors,
                               fontsize=18, min_width=0.75)

    # Token IDs
    if show_ids and ids:
        for i, (pos, token_id) in enumerate(zip(positions, ids)):
            x, w = pos
            ax.text(x + w/2, y_tokens - 0.7, f'ID: {token_id}',
                    ha='center', va='center', fontsize=12, color='#888')

    # Vocab section
    ax.text(1.0, 3.0, '词表:', ha='left', va='center',
            fontsize=18, color='#555', fontweight='bold')
    draw_vocab_row(ax, vocab, 2.4)

    # Vocab size
    ax.text(1.0, 1.0, f'词表大小: {len(vocab)}', ha='left', va='center',
            fontsize=14, color='#888')

    # Legend
    # Initial
    legend_box1 = FancyBboxPatch(
        (4.5, 0.85), 0.4, 0.3,
        boxstyle="round,pad=0.03",
        facecolor=LIGHT_BLUE_BG, edgecolor='#B0D4E8', linewidth=1.5
    )
    ax.add_patch(legend_box1)
    ax.text(5.1, 1.0, '= 初始字符', ha='left', va='center',
            fontsize=13, color='#666')

    # Merged
    legend_box2 = FancyBboxPatch(
        (7.2, 0.85), 0.4, 0.3,
        boxstyle="round,pad=0.03",
        facecolor=MERGE_BG, edgecolor='#F5C842', linewidth=1.5
    )
    ax.add_patch(legend_box2)
    ax.text(7.8, 1.0, '= 合并新增', ha='left', va='center',
            fontsize=13, color='#E67E22')

    # Summary text at bottom
    if summary_text:
        summary_box = FancyBboxPatch(
            (2.0, 0.1), 8.0, 0.55,
            boxstyle="round,pad=0.08",
            facecolor='#f5f5f5', edgecolor='#ccc', linewidth=1.5
        )
        ax.add_patch(summary_box)
        ax.text(6.0, 0.38, summary_text, ha='center', va='center',
                fontsize=15, color='#333', fontweight='bold')

    return fig_to_pil(fig)


def generate_bpe_gif():
    """Generate the BPE process GIF."""
    print("Generating 06_bpe_process.gif ...")

    initial_vocab = ['l', 'o', 'w', 'e', 'r', '_', 's', 't']

    frames = []

    # Frame 0: Initial state
    tokens0 = ['l', 'o', 'w', 'e', 'r', '_', 'l', 'o', 'w', 'e', 's', 't']
    colors0 = [BLUE] * 12
    colors0[5] = GRAY  # underscore
    frames.append(create_bpe_frame(
        0, '从字符到子词',
        '初始状态：每个字符是一个 token',
        tokens0, colors0,
        initial_vocab
    ))

    # Frame 1: Merge #1: (l, o) → lo
    tokens1 = ['lo', 'w', 'e', 'r', '_', 'lo', 'w', 'e', 's', 't']
    colors1 = [RED_MERGE, BLUE, BLUE, BLUE, GRAY, RED_MERGE, BLUE, BLUE, BLUE, BLUE]
    vocab1 = initial_vocab + ['lo']
    frames.append(create_bpe_frame(
        1, '从字符到子词',
        'Merge #1: 最高频pair (l, o) 出现2次',
        tokens1, colors1, vocab1,
        highlight_merge='(l, o) → lo'
    ))

    # Frame 2: Merge #2: (lo, w) → low
    tokens2 = ['low', 'e', 'r', '_', 'low', 'e', 's', 't']
    colors2 = [PURPLE, BLUE, BLUE, GRAY, PURPLE, BLUE, BLUE, BLUE]
    vocab2 = initial_vocab + ['lo', 'low']
    frames.append(create_bpe_frame(
        2, '从字符到子词',
        'Merge #2: 最高频pair (lo, w) 出现2次',
        tokens2, colors2, vocab2,
        highlight_merge='(lo, w) → low'
    ))

    # Frame 3: Merge #3: (e, r) → er
    tokens3 = ['low', 'er', '_', 'low', 'e', 's', 't']
    colors3 = [PURPLE, TEAL, GRAY, PURPLE, BLUE, BLUE, BLUE]
    vocab3 = initial_vocab + ['lo', 'low', 'er']
    frames.append(create_bpe_frame(
        3, '从字符到子词',
        'Merge #3: 最高频pair (e, r)',
        tokens3, colors3, vocab3,
        highlight_merge='(e, r) → er'
    ))

    # Frame 4: Merge #4: (e, s) → es
    tokens4 = ['low', 'er', '_', 'low', 'es', 't']
    colors4 = [PURPLE, TEAL, GRAY, PURPLE, ORANGE, BLUE]
    vocab4 = initial_vocab + ['lo', 'low', 'er', 'es']
    frames.append(create_bpe_frame(
        4, '从字符到子词',
        'Merge #4: 最高频pair (e, s)',
        tokens4, colors4, vocab4,
        highlight_merge='(e, s) → es'
    ))

    # Frame 5: Final result
    tokens5 = ['low', 'er', '_', 'low', 'es', 't']
    colors5 = [PURPLE, TEAL, GRAY, PURPLE, ORANGE, BLUE]
    vocab5 = initial_vocab + ['lo', 'low', 'er', 'es']
    ids5 = [9, 10, 5, 9, 11, 7]
    frames.append(create_bpe_frame(
        5, '最终结果',
        '从12个字符压缩到6个 token',
        tokens5, colors5, vocab5,
        show_ids=True, ids=ids5,
        summary_text='原始: "lower_lowest" (12字符)  →  BPE: 6个token'
    ))

    # Resize all frames to match
    target_size = frames[0].size
    frames = [f.resize(target_size, Image.LANCZOS) if f.size != target_size else f
              for f in frames]

    # Convert to RGB
    rgb_frames = []
    for f in frames:
        bg = Image.new('RGB', f.size, (255, 255, 255))
        bg.paste(f, mask=f.split()[3] if f.mode == 'RGBA' else None)
        rgb_frames.append(bg)

    output_path = os.path.join(OUTPUT_DIR, '06_bpe_process.gif')
    rgb_frames[0].save(
        output_path,
        save_all=True,
        append_images=rgb_frames[1:],
        duration=2000,  # 2000ms per frame (doubled from 1000)
        loop=0,
        optimize=True
    )

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Saved: {output_path} ({size_kb:.0f} KB, {len(frames)} frames, 2000ms/frame)")
    return output_path


if __name__ == '__main__':
    plt.close('all')
    generate_utf8_gif()
    plt.close('all')
    generate_bpe_gif()
    plt.close('all')
    print("\nDone! Both GIFs regenerated.")
