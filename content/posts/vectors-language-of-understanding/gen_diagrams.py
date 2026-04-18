"""Generate diagrams for 'Vectors as Language of Understanding' article."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.font_manager as fm
import numpy as np

_cjk_font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(_cjk_font_path)
_cjk_font = fm.FontProperties(fname=_cjk_font_path)
plt.rcParams['font.family'] = _cjk_font.get_name()
plt.rcParams['font.size'] = 13
plt.rcParams['axes.unicode_minus'] = False

COLORS = {
    'green': '#4CAF50',
    'blue': '#2196F3',
    'orange': '#FF9800',
    'red': '#F44336',
    'purple': '#9C27B0',
    'grey': '#607D8B',
    'teal': '#009688',
    'white': '#FFFFFF',
    'dark_text': '#333333',
    'mid_text': '#666666',
}


# ============================================================
# Diagram 1: Convergence Funnel — different modalities → one space
# ============================================================
def make_convergence():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis('off')
    fig.patch.set_facecolor(COLORS['white'])

    ax.text(6, 8.5, '万物归一：不同的感知，同一个向量空间',
            ha='center', va='center', fontsize=18, fontweight='bold', color=COLORS['dark_text'])

    # Left side: different modalities
    modalities = [
        (1.5, 6.5, '文字', '"一只橘猫"', COLORS['blue']),
        (1.5, 4.5, '图像', '🖼 猫的照片', COLORS['green']),
        (1.5, 2.5, '声音', '🔊 猫叫声', COLORS['purple']),
    ]

    for x, y, name, example, color in modalities:
        ax.add_patch(FancyBboxPatch((x-1.2, y-0.6), 2.4, 1.2,
                     boxstyle="round,pad=0.2", facecolor=color, alpha=0.12,
                     edgecolor=color, linewidth=2))
        ax.text(x, y + 0.2, name, ha='center', va='center', fontsize=13, fontweight='bold', color=color)
        ax.text(x, y - 0.25, example, ha='center', va='center', fontsize=10, color=COLORS['mid_text'])

    # Middle: encoder arrows
    for _, y, _, _, color in modalities:
        ax.annotate('', xy=(5.3, 4.5), xytext=(2.8, y),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.6,
                                   connectionstyle="arc3,rad=0.1"))

    # Middle: encoder label
    ax.text(4.2, 5.5, '各自的编码器', ha='center', va='center', fontsize=11,
            color=COLORS['mid_text'], style='italic', rotation=20)

    # Center: convergence point (shared vector space)
    circle = plt.Circle((6.5, 4.5), 1.3, facecolor=COLORS['orange'], alpha=0.1,
                         edgecolor=COLORS['orange'], linewidth=2.5)
    ax.add_patch(circle)
    ax.text(6.5, 4.9, '共享', ha='center', va='center', fontsize=14, fontweight='bold', color=COLORS['orange'])
    ax.text(6.5, 4.5, '向量空间', ha='center', va='center', fontsize=14, fontweight='bold', color=COLORS['orange'])
    ax.text(6.5, 3.9, '[0.82, 0.15, -0.31, ...]', ha='center', va='center', fontsize=9, color=COLORS['mid_text'],
            family='monospace')

    # Right: what emerges
    ax.annotate('', xy=(9.5, 4.5), xytext=(7.9, 4.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['dark_text'], lw=2.5))

    ax.add_patch(FancyBboxPatch((9.3, 3.5), 2.4, 2.0,
                 boxstyle="round,pad=0.2", facecolor=COLORS['teal'], alpha=0.1,
                 edgecolor=COLORS['teal'], linewidth=2))
    ax.text(10.5, 4.9, 'Transformer', ha='center', va='center', fontsize=12, fontweight='bold', color=COLORS['teal'])
    ax.text(10.5, 4.4, '统一推理', ha='center', va='center', fontsize=12, fontweight='bold', color=COLORS['teal'])
    ax.text(10.5, 3.9, '→ 理解 + 生成', ha='center', va='center', fontsize=10, color=COLORS['mid_text'])

    # Bottom insight
    ax.text(6, 1.0, '不同的入口，同一种表示，同一个推理引擎\n'
                     '向量是 AI 的"通用语言"——思考发生在向量空间中',
            ha='center', va='center', fontsize=12, color=COLORS['mid_text'], style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['orange'], alpha=0.06,
                     edgecolor=COLORS['orange'], linewidth=1))

    fig.tight_layout()
    fig.savefig('/home/azureuser/ai-blog/content/posts/vectors-language-of-understanding/convergence.png',
                dpi=150, bbox_inches='tight', facecolor=COLORS['white'])
    plt.close()
    print("OK: convergence.png")


# ============================================================
# Diagram 2: Human vs AI understanding comparison
# ============================================================
def make_human_vs_ai():
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis('off')
    fig.patch.set_facecolor(COLORS['white'])

    ax.text(6.5, 8.5, '人类理解 vs AI 理解：殊途同归？',
            ha='center', va='center', fontsize=18, fontweight='bold', color=COLORS['dark_text'])

    # Left: Human
    hx = 3.5
    ax.add_patch(FancyBboxPatch((hx-2.8, 0.6), 5.6, 7.2,
                 boxstyle="round,pad=0.3", facecolor=COLORS['blue'], alpha=0.04,
                 edgecolor=COLORS['blue'], linewidth=2))
    ax.text(hx, 7.5, '人类', ha='center', va='center', fontsize=16, fontweight='bold', color=COLORS['blue'])

    human_layers = [
        (6.5, '感官输入', '视觉 / 听觉 / 触觉 / 嗅觉', COLORS['grey']),
        (5.5, '神经编码', '视网膜 → 视觉皮层\n耳蜗 → 听觉皮层', COLORS['blue']),
        (4.2, '概念融合', '多模态神经元\n"猫"的概念 = 看+听+摸', COLORS['purple']),
        (3.0, '语言表达', '"我看到了一只猫"', COLORS['green']),
        (1.8, '但语言 ≠ 理解', '无法言说的感受\n依然被"理解"', COLORS['orange']),
    ]
    for y, title, desc, color in human_layers:
        ax.add_patch(FancyBboxPatch((hx-2.3, y-0.45), 4.6, 0.9,
                     boxstyle="round,pad=0.1", facecolor=color, alpha=0.08,
                     edgecolor=color, linewidth=1))
        ax.text(hx - 1.1, y, title, ha='center', va='center', fontsize=10.5, fontweight='bold', color=color)
        ax.text(hx + 1.2, y, desc, ha='center', va='center', fontsize=9, color=COLORS['mid_text'])

    # Right: AI
    ax2 = 9.5
    ax.add_patch(FancyBboxPatch((ax2-2.8, 0.6), 5.6, 7.2,
                 boxstyle="round,pad=0.3", facecolor=COLORS['green'], alpha=0.04,
                 edgecolor=COLORS['green'], linewidth=2))
    ax.text(ax2, 7.5, 'AI (多模态 LLM)', ha='center', va='center', fontsize=16, fontweight='bold', color=COLORS['green'])

    ai_layers = [
        (6.5, '模态输入', '文字 / 图像 / 音频', COLORS['grey']),
        (5.5, '编码器', 'Tokenizer / ViT / 音频编码器', COLORS['blue']),
        (4.2, '向量空间', '所有 token 在\n同一个高维空间', COLORS['purple']),
        (3.0, 'Attention', '计算 token 间关系\n→ 生成回答', COLORS['green']),
        (1.8, '向量 ≠ 理解？', '能算不能感受\n但产生了"涌现"', COLORS['orange']),
    ]
    for y, title, desc, color in ai_layers:
        ax.add_patch(FancyBboxPatch((ax2-2.3, y-0.45), 4.6, 0.9,
                     boxstyle="round,pad=0.1", facecolor=color, alpha=0.08,
                     edgecolor=color, linewidth=1))
        ax.text(ax2 - 1.1, y, title, ha='center', va='center', fontsize=10.5, fontweight='bold', color=color)
        ax.text(ax2 + 1.2, y, desc, ha='center', va='center', fontsize=9, color=COLORS['mid_text'])

    # Connecting arrows between corresponding layers
    for y, _, _, _ in human_layers:
        ax.annotate('', xy=(ax2 - 2.9, y), xytext=(hx + 2.4, y),
                   arrowprops=dict(arrowstyle='<->', color=COLORS['mid_text'], lw=0.8,
                                   alpha=0.3, linestyle='dashed'))

    fig.tight_layout()
    fig.savefig('/home/azureuser/ai-blog/content/posts/vectors-language-of-understanding/human_vs_ai.png',
                dpi=150, bbox_inches='tight', facecolor=COLORS['white'])
    plt.close()
    print("OK: human_vs_ai.png")


# ============================================================
# Diagram 3: Language vs Understanding - the gap
# ============================================================
def make_language_gap():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    fig.patch.set_facecolor(COLORS['white'])

    ax.text(6, 7.5, '语言是理解的工具，还是理解的牢笼？',
            ha='center', va='center', fontsize=18, fontweight='bold', color=COLORS['dark_text'])

    # Left: Things language can capture
    ax.add_patch(FancyBboxPatch((0.5, 1.0), 4.5, 5.5,
                 boxstyle="round,pad=0.3", facecolor=COLORS['green'], alpha=0.06,
                 edgecolor=COLORS['green'], linewidth=2))
    ax.text(2.75, 6.1, '语言能抓住的', ha='center', va='center', fontsize=14, fontweight='bold', color=COLORS['green'])

    can_say = [
        '"这是一只橘色的猫"',
        '"E = mc²"',
        '"她很开心"',
        '"西红柿炒蛋的做法"',
    ]
    for i, t in enumerate(can_say):
        ax.text(2.75, 5.2 - i*1.0, t, ha='center', va='center', fontsize=11, color=COLORS['dark_text'])

    # Right: Things language cannot capture
    ax.add_patch(FancyBboxPatch((7.0, 1.0), 4.5, 5.5,
                 boxstyle="round,pad=0.3", facecolor=COLORS['orange'], alpha=0.06,
                 edgecolor=COLORS['orange'], linewidth=2))
    ax.text(9.25, 6.1, '语言抓不住的', ha='center', va='center', fontsize=14, fontweight='bold', color=COLORS['orange'])

    cant_say = [
        '那朵云的具体形状',
        '妈妈的味道',
        '第一次坠入爱河的感觉',
        '莫扎特 G 小调的"悲伤"',
    ]
    for i, t in enumerate(cant_say):
        ax.text(9.25, 5.2 - i*1.0, t, ha='center', va='center', fontsize=11, color=COLORS['dark_text'])

    # Middle: gap
    ax.text(6, 4.0, '?', ha='center', va='center', fontsize=40, fontweight='bold',
            color=COLORS['mid_text'], alpha=0.3)
    ax.text(6, 2.6, '理解 > 语言', ha='center', va='center', fontsize=13, fontweight='bold',
            color=COLORS['purple'])
    ax.text(6, 2.0, '但 AI 用向量\n弥合了这个差距', ha='center', va='center', fontsize=11,
            color=COLORS['mid_text'], style='italic')

    # Bottom note
    ax.text(6, 0.4, '维特根斯坦说"语言的边界就是世界的边界"——但向量空间没有这个边界',
            ha='center', va='center', fontsize=11.5, color=COLORS['purple'], style='italic')

    fig.tight_layout()
    fig.savefig('/home/azureuser/ai-blog/content/posts/vectors-language-of-understanding/language_gap.png',
                dpi=150, bbox_inches='tight', facecolor=COLORS['white'])
    plt.close()
    print("OK: language_gap.png")


if __name__ == '__main__':
    make_convergence()
    make_human_vs_ai()
    make_language_gap()
    print("\nAll vectors article diagrams generated!")
