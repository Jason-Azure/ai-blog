"""Generate diagrams for 'CLIP open source story' article."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Global style
import matplotlib.font_manager as fm
_cjk_font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
_cjk_font = fm.FontProperties(fname=_cjk_font_path)
fm.fontManager.addfont(_cjk_font_path)
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
    'light_bg': '#F5F5F5',
    'white': '#FFFFFF',
    'dark_text': '#333333',
    'mid_text': '#666666',
}

OUT_DIR = '/home/azureuser/ai-blog/content/posts/clip-open-source-story/'
DPI = 130  # slightly smaller images


# ============================================================
# Diagram 1: CLIP Timeline / Evolution
# ============================================================
def make_timeline():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(0, 9)
    ax.axis('off')
    fig.patch.set_facecolor(COLORS['white'])

    ax.text(5, 8.5, 'CLIP 的涟漪：一个模型如何改变整个视觉 AI 生态',
            ha='center', va='center', fontsize=17, fontweight='bold', color=COLORS['dark_text'])

    # Timeline line — moved up to y=6.5
    tl_y = 6.5
    ax.plot([1, 9], [tl_y, tl_y], color=COLORS['grey'], linewidth=2, zorder=1)

    # Year markers
    years = [(1.5, '2021.1'), (3.5, '2021.3'), (5, '2022'), (6.5, '2023'), (8.5, '2024')]
    for x, label in years:
        ax.plot(x, tl_y, 'o', color=COLORS['grey'], markersize=10, zorder=2)
        ax.text(x, tl_y + 0.3, label, ha='center', va='bottom', fontsize=10,
                color=COLORS['mid_text'], fontweight='bold')

    # Events above timeline (OpenAI / closed)
    events_above = [
        (1.5, 'OpenAI 发布\nCLIP + DALL·E', COLORS['blue']),
        (6.5, 'GPT-4V\n原生多模态', COLORS['blue']),
    ]
    for x, text, color in events_above:
        ax.annotate(text, xy=(x, tl_y + 0.1), xytext=(x, 7.8),
                    ha='center', va='bottom', fontsize=10, color=color, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # Events below timeline (open source / community)
    events_below = [
        (1.5, '公开权重+代码\n但数据未公开', COLORS['orange'], 5.3),
        (3.5, 'LAION-400M\n开源 4 亿对数据', COLORS['green'], 4.9),
        (5, 'LAION-5B\n50 亿对！\nOpenCLIP 复现', COLORS['green'], 4.2),
        (6.5, 'SigLIP (Google)\nDINOv2 (Meta)\nMetaCLIP (Meta)', COLORS['purple'], 3.8),
        (8.5, 'InternViT (清华)\nQwen-VL (阿里)\n各家自研眼睛', COLORS['teal'], 4.2),
    ]
    for x, text, color, y in events_below:
        ax.annotate(text, xy=(x, tl_y - 0.1), xytext=(x, y),
                    ha='center', va='top', fontsize=9.5, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    # Legend boxes — more compact
    legend_y = 1.2
    for i, (label, color) in enumerate([
        ('OpenAI（闭源）', COLORS['blue']),
        ('社区开源', COLORS['green']),
        ('大厂自研', COLORS['purple']),
    ]):
        ax.add_patch(FancyBboxPatch((1 + i*3, legend_y - 0.25), 2.5, 0.5,
                     boxstyle="round,pad=0.15", facecolor=color, alpha=0.15, edgecolor=color, linewidth=1.5))
        ax.text(2.25 + i*3, legend_y, label, ha='center', va='center', fontsize=11, color=color, fontweight='bold')

    # Bottom note
    ax.text(5, 0.3, '一个模型的开源引发了整个生态的"造眼睛"军备竞赛',
            ha='center', va='center', fontsize=12, color=COLORS['mid_text'], style='italic')

    fig.tight_layout()
    fig.savefig(OUT_DIR + 'clip_timeline.png',
                dpi=DPI, bbox_inches='tight', facecolor=COLORS['white'])
    plt.close()
    print("OK: clip_timeline.png")


# ============================================================
# Diagram 2: Open Source Anatomy — what's opened, what's hidden
# ============================================================
def make_open_source_anatomy():
    fig, ax = plt.subplots(figsize=(12, 9.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 11)
    ax.axis('off')
    fig.patch.set_facecolor(COLORS['white'])

    # Title — pushed higher
    ax.text(6, 10.5, '"开源"的解剖：到底开了什么？',
            ha='center', va='center', fontsize=18, fontweight='bold', color=COLORS['dark_text'])

    # Three columns — pushed down so they don't overlap title
    columns = [
        ('OpenAI\nCLIP', 2, COLORS['blue']),
        ('LAION\nOpenCLIP', 6, COLORS['green']),
        ('Google/Meta\nSigLIP/DINOv2', 10, COLORS['purple']),
    ]

    items = [
        ('论文 / 方法', 8.5),
        ('模型权重', 7.3),
        ('训练代码', 6.1),
        ('训练数据', 4.9),
        ('数据筛选算法', 3.7),
        ('训练超参数', 2.5),
    ]

    # Column headers — at y=9.5
    header_y = 9.5
    for label, x, color in columns:
        ax.add_patch(FancyBboxPatch((x-1.5, header_y - 0.45), 3, 0.9,
                     boxstyle="round,pad=0.15", facecolor=color, alpha=0.2, edgecolor=color, linewidth=2))
        ax.text(x, header_y, label, ha='center', va='center', fontsize=11, fontweight='bold', color=color)

    # Status for each item in each column
    statuses = [
        ['open', 'open', 'open'],        # Paper
        ['open', 'open', 'open'],        # Weights
        ['open', 'open', 'partial'],     # Code
        ['closed', 'open', 'closed'],    # Data
        ['closed', 'open', 'partial'],   # Algorithm
        ['partial', 'open', 'partial'],  # Hyperparams
    ]

    status_display = {
        'open': ('✓ 公开', COLORS['green'], COLORS['green']),
        'closed': ('✗ 未公开', COLORS['red'], COLORS['red']),
        'partial': ('△ 部分', COLORS['orange'], COLORS['orange']),
    }

    for i, (item_label, y) in enumerate(items):
        ax.text(0.1, y, item_label, ha='left', va='center', fontsize=12, fontweight='bold', color=COLORS['dark_text'])
        ax.plot([0, 12], [y - 0.5, y - 0.5], color='#E0E0E0', linewidth=0.5)

        for j, (_, x, _) in enumerate(columns):
            status = statuses[i][j]
            symbol, fg_color, _ = status_display[status]
            bg_alpha = 0.1
            bg_color = fg_color

            ax.add_patch(FancyBboxPatch((x-1.2, y-0.3), 2.4, 0.6,
                         boxstyle="round,pad=0.1", facecolor=bg_color, alpha=bg_alpha,
                         edgecolor=bg_color, linewidth=1))
            ax.text(x, y, symbol, ha='center', va='center', fontsize=11, color=fg_color, fontweight='bold')

    # Bottom insight
    ax.add_patch(FancyBboxPatch((0.5, 0.6), 11, 1.2,
                 boxstyle="round,pad=0.2", facecolor=COLORS['orange'], alpha=0.08,
                 edgecolor=COLORS['orange'], linewidth=1.5))
    ax.text(6, 1.2, '关键洞察："开源"不是一个二元概念——它是一个光谱。\n'
                     '只有 LAION/OpenCLIP 做到了从论文到数据的完全开源，实现了真正的可复现。',
            ha='center', va='center', fontsize=11, color=COLORS['dark_text'], style='italic')

    fig.tight_layout()
    fig.savefig(OUT_DIR + 'open_source_anatomy.png',
                dpi=DPI, bbox_inches='tight', facecolor=COLORS['white'])
    plt.close()
    print("OK: open_source_anatomy.png")


# ============================================================
# Diagram 3: Vision Encoder Ecosystem — various "eyes"
# ============================================================
def make_eyes_comparison():
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor(COLORS['white'])

    # Title — moved higher
    ax.text(6.5, 9.5, '各家的"眼睛"：视觉编码器全景',
            ha='center', va='center', fontsize=18, fontweight='bold', color=COLORS['dark_text'])

    # Center: shared concept — stays at 5
    cx, cy = 6.5, 5.0
    circle = plt.Circle((cx, cy), 1.5, facecolor=COLORS['orange'], alpha=0.1,
                         edgecolor=COLORS['orange'], linewidth=2)
    ax.add_patch(circle)
    ax.text(cx, cy + 0.3, '共同目标', ha='center', va='center', fontsize=13, fontweight='bold', color=COLORS['orange'])
    ax.text(cx, cy - 0.3, '图像 → 语义向量', ha='center', va='center', fontsize=11, color=COLORS['mid_text'])

    # Surrounding entries — spread out more, OpenCLIP moved down from title
    entries = [
        (1.8, 8.2, 'CLIP (OpenAI)', '对比学习\n图+文配对', COLORS['blue'], '2021'),
        (11.2, 8.2, 'SigLIP (Google)', 'Sigmoid 损失\n无需 softmax', COLORS['green'], '2023'),
        (1.8, 1.8, 'DINOv2 (Meta)', '自监督\n不需要文字！', COLORS['purple'], '2023'),
        (11.2, 1.8, 'InternViT (清华)', '60 亿参数\n中文优化', COLORS['teal'], '2024'),
        (6.5, 8.5, 'OpenCLIP (社区)', '开源复现 LAION-5B', COLORS['green'], '2022'),
        (6.5, 1.5, 'EVA-CLIP', '蒸馏优化 更高效率', COLORS['grey'], '2023'),
    ]

    for x, y, name, desc, color, year in entries:
        bw, bh = 3.0, 1.3
        ax.add_patch(FancyBboxPatch((x - bw/2, y - bh/2), bw, bh,
                     boxstyle="round,pad=0.2", facecolor=color, alpha=0.1,
                     edgecolor=color, linewidth=1.5))
        ax.text(x, y + 0.25, name, ha='center', va='center', fontsize=10, fontweight='bold', color=color)
        ax.text(x, y - 0.25, desc, ha='center', va='center', fontsize=8.5, color=COLORS['mid_text'])
        ax.text(x + bw/2 - 0.15, y + bh/2 - 0.15, year, ha='right', va='top', fontsize=8,
                color=COLORS['mid_text'], style='italic', alpha=0.7)

        # Arrow to center
        dx = cx - x
        dy = cy - y
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 2.5:
            ndx, ndy = dx/dist, dy/dist
            ax.annotate('', xy=(cx - ndx*1.6, cy - ndy*1.6),
                       xytext=(x + ndx*1.6, y + ndy*0.7),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.2, alpha=0.5))

    fig.tight_layout()
    fig.savefig(OUT_DIR + 'eyes_comparison.png',
                dpi=DPI, bbox_inches='tight', facecolor=COLORS['white'])
    plt.close()
    print("OK: eyes_comparison.png")


# ============================================================
# Diagram 4: LAION data pipeline
# ============================================================
def make_laion_pipeline():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.5)
    ax.axis('off')
    fig.patch.set_facecolor(COLORS['white'])

    ax.text(7, 6.1, 'LAION 如何"逆向工程"训练数据',
            ha='center', va='center', fontsize=17, fontweight='bold', color=COLORS['dark_text'])

    # Pipeline steps
    steps = [
        (1.5, 3.5, 'Common Crawl\n互联网爬虫', '数十亿网页\n公开数据', COLORS['grey']),
        (4.5, 3.5, '提取 <img> 标签\n+ alt 文本', '图片 URL\n+ 描述文字', COLORS['blue']),
        (7.5, 3.5, '用 CLIP 打分\n计算图文相似度', '过滤低质量\n图文对', COLORS['orange']),
        (10.5, 3.5, 'LAION-5B\n50 亿对', '完全开源\n可下载', COLORS['green']),
    ]

    for i, (x, y, title, desc, color) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((x-1.3, y-0.9), 2.6, 1.8,
                     boxstyle="round,pad=0.2", facecolor=color, alpha=0.12,
                     edgecolor=color, linewidth=2))
        ax.text(x, y + 0.25, title, ha='center', va='center', fontsize=10.5, fontweight='bold', color=color)
        ax.text(x, y - 0.45, desc, ha='center', va='center', fontsize=9, color=COLORS['mid_text'])

        if i < len(steps) - 1:
            next_x = steps[i+1][0]
            ax.annotate('', xy=(next_x - 1.4, y), xytext=(x + 1.4, y),
                       arrowprops=dict(arrowstyle='->', color=COLORS['dark_text'], lw=2))

    # Irony note
    ax.add_patch(FancyBboxPatch((2, 0.8), 10, 1.0,
                 boxstyle="round,pad=0.15", facecolor=COLORS['purple'], alpha=0.06,
                 edgecolor=COLORS['purple'], linewidth=1, linestyle='--'))
    ax.text(7, 1.3, '递归的味道：用 OpenAI 的 CLIP 来过滤数据 → 训练开源的 CLIP 替代品\n'
                     '就像用别人的秤来校准自己的秤——最终你可以不再依赖那把秤',
            ha='center', va='center', fontsize=10.5, color=COLORS['purple'], style='italic')

    fig.tight_layout()
    fig.savefig(OUT_DIR + 'laion_pipeline.png',
                dpi=DPI, bbox_inches='tight', facecolor=COLORS['white'])
    plt.close()
    print("OK: laion_pipeline.png")


if __name__ == '__main__':
    make_timeline()
    make_open_source_anatomy()
    make_eyes_comparison()
    make_laion_pipeline()
    print("\nAll CLIP article diagrams generated!")
