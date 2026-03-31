#!/usr/bin/env python3
"""
Generate the full pipeline animation GIF:
User types text → chars → binary → token IDs → embedding vectors →
neural network → output token → decoded text → display to user.

9 frames showing the complete journey of "你好" through an LLM.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
import io
import os
import numpy as np

plt.rcParams['font.family'] = ['WenQuanYi Zen Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT = "/home/azureuser/ai-blog/content/posts/ascii-to-token/09_full_pipeline.gif"


def fig_to_pil(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.3)
    buf.seek(0)
    return Image.open(buf).convert('RGBA')


def draw_box(ax, x, y, w, h, color, text, text_color='white',
             fontsize=18, border_color=None, lw=2, alpha=1.0):
    import matplotlib.colors as mcolors
    if border_color is None:
        rgb = mcolors.to_rgb(color)
        border_color = tuple(max(0, c - 0.15) for c in rgb)
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                         facecolor=color, edgecolor=border_color,
                         linewidth=lw, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=text_color, fontweight='bold')


def draw_arrow(ax, x1, y1, x2, y2, color='#888', lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))


def make_base(title_main='', title_sub='', step_text=''):
    """Create base figure with title area."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Step indicator
    ax.text(6, 6.5, title_main, ha='center', va='center',
            fontsize=26, color='#2c3e50', fontweight='bold')
    if title_sub:
        ax.text(6, 5.9, title_sub, ha='center', va='center',
                fontsize=15, color='#e74c3c')
    if step_text:
        ax.text(6, 5.9, step_text, ha='center', va='center',
                fontsize=14, color='#888')

    return fig, ax


# ════════════════════════════════════════════════════
#  Frame 0: User input
# ════════════════════════════════════════════════════
def frame0():
    fig, ax = make_base('从输入到输出：一段文字的完整旅程',
                         '"你好世界" 在 LLM 中经历了什么？')

    # User icon
    draw_box(ax, 1.8, 3.6, 1.5, 1.2, '#95a5a6', '用户',
             fontsize=22, text_color='white')
    ax.text(2.55, 3.0, '输入文字', ha='center', va='center',
            fontsize=14, color='#666')

    # Arrow
    draw_arrow(ax, 3.8, 4.0, 5.0, 4.0, '#3498db', 2.5)

    # Input text box
    draw_box(ax, 5.2, 3.4, 4.5, 1.2, '#3498db', '"你好世界"',
             fontsize=28, text_color='white')

    # Note at bottom
    ax.text(6, 1.5, '计算机不认识文字\n它需要把文字翻译成数字才能处理',
            ha='center', va='center', fontsize=16, color='#666',
            linespacing=1.8)

    return fig_to_pil(fig)


# ════════════════════════════════════════════════════
#  Frame 1: Character encoding (what the computer first sees)
# ════════════════════════════════════════════════════
def frame1():
    fig, ax = make_base('Step 1: 字符编码 (UTF-8)',
                         '每个字符先变成字节')

    chars = ['"', '你', '好', '世', '界', '"']
    bytes_per = ['22', 'E4 BD A0', 'E5 A5 BD', 'E4 B8 96', 'E7 95 8C', '22']
    byte_counts = ['1B', '3B', '3B', '3B', '3B', '1B']

    # Top row: characters
    box_w = 1.3
    start_x = 1.0
    spacing = 0.5
    y_char = 4.6
    y_bytes = 3.0

    for i, (ch, bts, cnt) in enumerate(zip(chars, bytes_per, byte_counts)):
        x = start_x + i * (box_w + spacing)
        # Character box
        char_color = '#3498db' if ch not in ('"', '"') else '#95a5a6'
        draw_box(ax, x, y_char, box_w, 0.8, char_color, ch,
                 fontsize=22, text_color='white')
        # Arrow down
        draw_arrow(ax, x + box_w/2, y_char, x + box_w/2, y_bytes + 0.8,
                   '#aaa', 1.5)
        # Bytes box
        draw_box(ax, x - 0.15, y_bytes - 0.15, box_w + 0.3, 0.8, '#f5f5f5',
                 bts, fontsize=10 if len(bts) > 4 else 13,
                 text_color='#e74c3c', border_color='#ddd')
        # Byte count
        ax.text(x + box_w/2, y_bytes - 0.55, cnt,
                ha='center', va='center', fontsize=11, color='#999')

    ax.text(6, 1.3, '14 个字节 (bytes) = 112 个 0 和 1',
            ha='center', va='center', fontsize=17, color='#e74c3c',
            fontweight='bold')
    ax.text(6, 0.7, '这一步和发微信、存文件完全一样',
            ha='center', va='center', fontsize=14, color='#888')

    return fig_to_pil(fig)


# ════════════════════════════════════════════════════
#  Frame 2: Tokenization
# ════════════════════════════════════════════════════
def frame2():
    fig, ax = make_base('Step 2: 分词 (Tokenization)',
                         'BPE 分词器把字节切成 token')

    # Input text
    ax.text(6, 4.8, '"你好世界"', ha='center', va='center',
            fontsize=28, color='#3498db', fontweight='bold')

    # Arrow
    draw_arrow(ax, 6, 4.3, 6, 3.6, '#888', 2)
    ax.text(7.5, 3.95, 'BPE 分词', ha='center', va='center',
            fontsize=14, color='#888')

    # Token boxes
    tokens = ['你好', '世界']
    token_colors = ['#9b59b6', '#1abc9c']
    box_w = 2.5
    start_x = 2.0
    spacing = 1.0

    for i, (tok, col) in enumerate(zip(tokens, token_colors)):
        x = start_x + i * (box_w + spacing)
        draw_box(ax, x, 2.6, box_w, 1.0, col, tok,
                 fontsize=26, text_color='white')

    ax.text(6, 1.5, '4个字 → 2个 token（高频组合被合并）',
            ha='center', va='center', fontsize=16, color='#e74c3c',
            fontweight='bold')
    ax.text(6, 0.8, '和 Morse "高频短编码" 同一原理',
            ha='center', va='center', fontsize=14, color='#888')

    return fig_to_pil(fig)


# ════════════════════════════════════════════════════
#  Frame 3: Token IDs
# ════════════════════════════════════════════════════
def frame3():
    fig, ax = make_base('Step 3: Token → ID (查词表)',
                         '每个 token 变成一个整数编号')

    tokens = ['你好', '世界']
    ids = ['19526', '31809']
    token_colors = ['#9b59b6', '#1abc9c']

    box_w = 2.2
    start_x = 1.5

    for i in range(2):
        x = start_x + i * 5.0
        # Token
        draw_box(ax, x, 4.3, box_w, 0.9, token_colors[i], tokens[i],
                 fontsize=24, text_color='white')
        # Arrow
        draw_arrow(ax, x + box_w/2, 4.3, x + box_w/2, 3.5, '#888', 2)
        ax.text(x + box_w + 0.3, 3.9, '查表', ha='left', va='center',
                fontsize=13, color='#888')
        # ID box
        draw_box(ax, x, 2.4, box_w, 0.9, '#e67e22', ids[i],
                 fontsize=24, text_color='white')

    ax.text(6, 1.3, '和 ASCII 查表完全一样：输入符号 → 输出数字',
            ha='center', va='center', fontsize=16, color='#e74c3c',
            fontweight='bold')
    ax.text(6, 0.6, 'GPT 词表 ≈ 100,000 行  vs  ASCII 表 128 行',
            ha='center', va='center', fontsize=14, color='#888')

    return fig_to_pil(fig)


# ════════════════════════════════════════════════════
#  Frame 4: Embedding lookup
# ════════════════════════════════════════════════════
def frame4():
    fig, ax = make_base('Step 4: Embedding (查嵌入表)',
                         '整数 ID → 高维向量')

    # ID boxes
    ids = ['19526', '31809']
    id_colors = ['#e67e22', '#e67e22']

    for i in range(2):
        x = 1.0 + i * 5.5
        draw_box(ax, x, 4.5, 1.8, 0.7, '#e67e22', ids[i],
                 fontsize=18, text_color='white')
        draw_arrow(ax, x + 0.9, 4.5, x + 0.9, 3.9, '#888', 1.5)

    # Embedding vectors - show as colored blocks
    vec_labels = ['[0.12, -0.34, 0.56, ..., 0.89]', '[0.45, 0.23, -0.67, ..., 0.11]']
    vec_colors = ['#9b59b6', '#1abc9c']
    for i in range(2):
        x = 0.3 + i * 5.5
        # Vector visualization: a row of small colored cells
        for j in range(12):
            val = np.random.uniform(-1, 1)
            r = max(0.5, min(1.0, 0.75 + val * 0.25))
            g = max(0.5, min(1.0, 0.75 - abs(val) * 0.2))
            b = max(0.5, min(1.0, 0.75 + val * 0.15))
            cell_x = x + j * 0.35
            cell = FancyBboxPatch(
                (cell_x, 3.0), 0.32, 0.7,
                boxstyle="square,pad=0",
                facecolor=vec_colors[i], alpha=0.3 + abs(val) * 0.5,
                edgecolor='white', linewidth=0.5
            )
            ax.add_patch(cell)
        ax.text(x + 2.1, 2.5, f'768 维向量', ha='center', va='center',
                fontsize=13, color=vec_colors[i], fontweight='bold')

    ax.text(6, 1.5, '从"一个数字"到"768个数字"',
            ha='center', va='center', fontsize=17, color='#e74c3c',
            fontweight='bold')
    ax.text(6, 0.8, '每个维度都编码了语义信息',
            ha='center', va='center', fontsize=14, color='#888')

    return fig_to_pil(fig)


# ════════════════════════════════════════════════════
#  Frame 5: Neural network processing
# ════════════════════════════════════════════════════
def frame5():
    fig, ax = make_base('Step 5: Transformer 计算',
                         '向量在神经网络中经历矩阵运算')

    # Input vectors
    for i in range(2):
        x = 1.0 + i * 2.5
        draw_box(ax, x, 4.5, 2.0, 0.6, '#9b59b6' if i == 0 else '#1abc9c',
                 f'向量{i+1}', fontsize=14, text_color='white')

    # Arrow into transformer
    draw_arrow(ax, 5.0, 4.7, 5.8, 4.7, '#888', 2)

    # Transformer block
    draw_box(ax, 5.8, 3.3, 3.5, 2.8, '#2c3e50', '', fontsize=1)
    ax.text(7.55, 5.2, 'Transformer', ha='center', va='center',
            fontsize=16, color='white', fontweight='bold')
    # Internal layers
    layers = ['Self-Attention', 'Feed-Forward', 'Layer Norm']
    layer_colors = ['#e74c3c', '#f39c12', '#3498db']
    for j, (layer, lc) in enumerate(zip(layers, layer_colors)):
        draw_box(ax, 6.1, 4.4 - j * 0.7, 2.9, 0.5, lc, layer,
                 fontsize=12, text_color='white', lw=1)

    # Arrow out
    draw_arrow(ax, 9.5, 4.7, 10.5, 4.7, '#888', 2)

    # Output vector
    draw_box(ax, 10.0, 4.3, 1.5, 0.8, '#e67e22', '新向量',
             fontsize=14, text_color='white')

    ax.text(6, 1.5, '× 96 层（GPT-4）= 数十亿次矩阵乘法',
            ha='center', va='center', fontsize=16, color='#e74c3c',
            fontweight='bold')
    ax.text(6, 0.8, '向量在高维空间中被旋转、拉伸、投影',
            ha='center', va='center', fontsize=14, color='#888')

    return fig_to_pil(fig)


# ════════════════════════════════════════════════════
#  Frame 6: Output token selection
# ════════════════════════════════════════════════════
def frame6():
    fig, ax = make_base('Step 6: 解码 (Decode)',
                         '输出向量 → 概率分布 → 选 token')

    # Output vector
    draw_box(ax, 1.0, 4.3, 2.0, 0.8, '#e67e22', '输出向量',
             fontsize=14, text_color='white')

    draw_arrow(ax, 3.2, 4.7, 4.2, 4.7, '#888', 2)
    ax.text(3.7, 5.1, 'Softmax', ha='center', va='center',
            fontsize=13, color='#888')

    # Probability bars
    probs = [('！', 0.45), ('吗', 0.20), ('啊', 0.15), ('。', 0.10), ('...', 0.10)]
    bar_x = 4.5
    bar_max_w = 4.0

    for i, (tok, prob) in enumerate(probs):
        y = 4.8 - i * 0.6
        w = prob * bar_max_w
        color = '#4CAF50' if i == 0 else '#ccc'
        border = '#388E3C' if i == 0 else '#bbb'
        draw_box(ax, bar_x, y - 0.2, w, 0.4, color, '',
                 border_color=border, lw=1)
        ax.text(bar_x - 0.15, y, tok, ha='right', va='center',
                fontsize=15, color='#333', fontweight='bold')
        ax.text(bar_x + w + 0.15, y, f'{prob:.0%}', ha='left', va='center',
                fontsize=13, color='#666')

    # Selected token
    draw_arrow(ax, 8.8, 4.6, 9.8, 4.6, '#4CAF50', 2.5)
    draw_box(ax, 9.5, 4.1, 1.5, 0.9, '#4CAF50', '！',
             fontsize=28, text_color='white')

    ax.text(6, 1.3, '从 100,000 个候选 token 中选出概率最高的',
            ha='center', va='center', fontsize=16, color='#e74c3c',
            fontweight='bold')
    ax.text(6, 0.6, '反查词表：Token ID → 文字',
            ha='center', va='center', fontsize=14, color='#888')

    return fig_to_pil(fig)


# ════════════════════════════════════════════════════
#  Frame 7: User sees result
# ════════════════════════════════════════════════════
def frame7():
    fig, ax = make_base('Step 7: 用户看到结果')

    # Screen mockup
    screen_x, screen_y = 2.5, 2.0
    screen_w, screen_h = 7.0, 3.5

    # Screen border
    screen = FancyBboxPatch(
        (screen_x, screen_y), screen_w, screen_h,
        boxstyle="round,pad=0.1",
        facecolor='white', edgecolor='#333', linewidth=3
    )
    ax.add_patch(screen)

    # Chat bubbles
    # User message
    draw_box(ax, 6.0, 4.5, 2.8, 0.6, '#3498db', '你好世界',
             fontsize=16, text_color='white')

    # AI response
    draw_box(ax, 3.2, 3.3, 4.5, 0.6, '#4CAF50', '你好世界！',
             fontsize=16, text_color='white')

    # Labels
    ax.text(9.2, 4.8, '[用户]', fontsize=12, ha='center', va='center', color='#888')
    ax.text(2.8, 3.6, '[AI]', fontsize=12, ha='center', va='center', color='#888')

    ax.text(6, 1.0, '数字 → 文字：解码完成\n用户看到的永远是文字，计算机处理的永远是数字',
            ha='center', va='center', fontsize=15, color='#666',
            linespacing=1.8)

    return fig_to_pil(fig)


# ════════════════════════════════════════════════════
#  Frame 8: Summary - the full loop
# ════════════════════════════════════════════════════
def frame8():
    fig, ax = make_base('完整流程：200 年不变的范式')

    # Pipeline boxes
    steps = [
        ('文字', '#3498db'),
        ('字节', '#9b59b6'),
        ('Token ID', '#e67e22'),
        ('向量', '#1abc9c'),
        ('计算', '#2c3e50'),
        ('向量', '#1abc9c'),
        ('Token ID', '#e67e22'),
        ('文字', '#4CAF50'),
    ]

    labels_top = ['输入', 'UTF-8', 'BPE\n分词', 'Embedding\n查表',
                  'Transformer', 'Softmax', '反查\n词表', '输出']

    y = 4.0
    box_w = 1.1
    start_x = 0.4
    spacing = 0.2

    for i, ((text, color), label) in enumerate(zip(steps, labels_top)):
        x = start_x + i * (box_w + spacing)
        draw_box(ax, x, y, box_w, 0.9, color, text,
                 fontsize=10 if len(text) > 4 else 12,
                 text_color='white', lw=1.5)
        ax.text(x + box_w/2, y + 1.2, label, ha='center', va='center',
                fontsize=9, color='#666', linespacing=1.3)

        if i < len(steps) - 1:
            ax.annotate('', xy=(x + box_w + spacing - 0.05, y + 0.45),
                        xytext=(x + box_w + 0.05, y + 0.45),
                        arrowprops=dict(arrowstyle='->', color='#aaa', lw=1.5))

    # Bottom: the eternal pattern
    ax.text(6, 2.2, '符号 → 数字 → 处理 → 数字 → 符号',
            ha='center', va='center', fontsize=22, color='#2c3e50',
            fontweight='bold')

    ax.text(6, 1.3, '从莫尔斯电码到 GPT，200 年来同一个模式',
            ha='center', va='center', fontsize=16, color='#e74c3c')

    ax.text(6, 0.5, 'Shannon (1948): 信源 → 编码器 → 信道 → 解码器 → 信宿',
            ha='center', va='center', fontsize=14, color='#888')

    return fig_to_pil(fig)


def generate():
    print("Generating 09_full_pipeline.gif ...")

    frames = [frame0(), frame1(), frame2(), frame3(),
              frame4(), frame5(), frame6(), frame7(), frame8()]

    # Normalize sizes
    target = frames[0].size
    frames = [f.resize(target, Image.LANCZOS) if f.size != target else f for f in frames]

    rgb_frames = []
    for f in frames:
        bg = Image.new('RGB', f.size, (255, 255, 255))
        bg.paste(f, mask=f.split()[3] if f.mode == 'RGBA' else None)
        rgb_frames.append(bg)

    # Custom durations: title/summary get longer
    durations = [3000, 3000, 2500, 2500, 2500, 3000, 3000, 3000, 4000]

    rgb_frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=rgb_frames[1:],
        duration=durations,
        loop=0,
        optimize=True
    )

    size_kb = os.path.getsize(OUTPUT) / 1024
    print(f"  Saved: {OUTPUT} ({size_kb:.0f} KB, {len(frames)} frames)")


if __name__ == '__main__':
    plt.close('all')
    generate()
    plt.close('all')
