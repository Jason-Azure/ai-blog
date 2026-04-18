"""
位置编码文章配图 v2 — 修复版
- 尺寸缩小到 700-800px 宽
- 文字不重叠
- 颜色柔和
- 动画慢速
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
import os

plt.rcParams['font.family'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.expanduser('~/ai-blog/content/posts/positional-encoding')
DPI = 100  # 降低 DPI，控制输出尺寸
SLOW = 200

# ============================================================
# 图1: 置换不变性（修复文字重叠）
# ============================================================
def gen_01():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax in axes:
        ax.set_xlim(-0.3, 3.3)
        ax.set_ylim(-0.2, 3.2)
        ax.set_aspect('equal')
        ax.axis('off')

    colors = ['#E8837C', '#7DC4B8', '#7BB8D0']  # 柔和色调

    def draw_panel(ax, labels, title, out_labels, out_colors):
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        positions = [(0.5, 2.5), (1.5, 2.5), (2.5, 2.5)]
        for pos, label, color in zip(positions, labels, out_colors):
            circle = plt.Circle(pos, 0.3, color=color, alpha=0.85)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], label, ha='center', va='center',
                    fontsize=13, fontweight='bold', color='white')
        # Attention 连线
        for i, p1 in enumerate(positions):
            for j, p2 in enumerate(positions):
                if i != j:
                    ax.annotate('', xy=p2, xytext=p1,
                        arrowprops=dict(arrowstyle='->', color='#aaa', alpha=0.3, lw=1))
        ax.text(1.5, 1.5, 'Attention', ha='center', fontsize=9, color='#888')
        ax.annotate('', xy=(1.5, 1.1), xytext=(1.5, 1.85),
            arrowprops=dict(arrowstyle='->', color='#888', lw=1.5))
        # 输出
        out_pos = [(0.5, 0.4), (1.5, 0.4), (2.5, 0.4)]
        for pos, label, color in zip(out_pos, out_labels, out_colors):
            rect = mpatches.FancyBboxPatch((pos[0]-0.3, pos[1]-0.2), 0.6, 0.4,
                boxstyle="round,pad=0.08", facecolor=color, alpha=0.4)
            ax.add_patch(rect)
            ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=9)

    draw_panel(axes[0], ['cat', 'eat', 'fish'], '"cat eat fish"',
               ['cat*', 'eat*', 'fish*'], colors)
    draw_panel(axes[1], ['fish', 'eat', 'cat'], '"fish eat cat"',
               ['fish*', 'eat*', 'cat*'], [colors[2], colors[1], colors[0]])

    fig.text(0.5, 0.02, '= Attention sees the same set {cat, eat, fish}',
             ha='center', fontsize=10, fontweight='bold', color='#C0392B',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8E1', edgecolor='#E0A040', alpha=0.9))
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(f'{OUT}/01_permutation_invariance.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v2 01")

# ============================================================
# 图2: 正弦波（动画，缩小）
# ============================================================
def gen_02():
    fig, axes = plt.subplots(4, 1, figsize=(7, 5), sharex=True)
    fig.suptitle('Positional Encoding: different dims = different frequencies',
                 fontsize=11, fontweight='bold', y=0.99)
    d_model = 512
    max_pos = 80
    positions = np.arange(max_pos)
    dims = [0, 10, 50, 200]
    labels = ['dim 0 (high freq)', 'dim 20 (mid-high)', 'dim 100 (mid-low)', 'dim 400 (low freq)']
    clrs = ['#E07060', '#E0A040', '#50A060', '#4080B0']
    lines = []
    for ax, di, lab, c in zip(axes, dims, labels, clrs):
        freq = 1.0 / (10000 ** (2*di/d_model))
        ax.set_ylabel(lab, fontsize=7, rotation=0, ha='right', va='center', labelpad=70)
        ax.set_ylim(-1.3, 1.3)
        ax.axhline(0, color='#ddd', lw=0.5)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, max_pos)
        l, = ax.plot([], [], color=c, lw=1.5)
        lines.append((l, freq))
    axes[-1].set_xlabel('Position', fontsize=9)

    def animate(frame):
        n = min(frame+1, max_pos)
        x = positions[:n]
        for l, f in lines:
            l.set_data(x, np.sin(x*f))
        return [l for l,_ in lines]

    anim = animation.FuncAnimation(fig, animate, frames=max_pos, interval=SLOW, blit=True, repeat=True)
    plt.tight_layout(rect=[0.12, 0, 1, 0.96])
    anim.save(f'{OUT}/02_sinusoidal_waves.gif', writer='pillow', fps=5, dpi=DPI)
    plt.close()
    print("v2 02")

# ============================================================
# 图3: 旋转动画（缩小）
# ============================================================
def gen_03():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.set_xlim(-1.3, 1.3); ax1.set_ylim(-1.3, 1.3); ax1.set_aspect('equal')
    ax1.set_xlabel('sin(w*pos)', fontsize=9); ax1.set_ylabel('cos(w*pos)', fontsize=9)
    ax1.set_title('(sin, cos) traces a circle', fontsize=10, fontweight='bold')
    ax1.axhline(0, color='#ddd'); ax1.axvline(0, color='#ddd')
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.sin(theta), np.cos(theta), '#eee', lw=1)

    trail, = ax1.plot([], [], '#7090C0', alpha=0.4, lw=1)
    point, = ax1.plot([], [], 'o', color='#D05050', markersize=7, zorder=5)
    ptxt = ax1.text(0, -1.15, '', ha='center', fontsize=9, fontweight='bold')

    ax2.set_xlim(-1.8, 1.8); ax2.set_ylim(-1.8, 1.8); ax2.set_aspect('equal')
    ax2.set_title('Position offset = rotation angle', fontsize=10, fontweight='bold')
    ax2.axhline(0, color='#ddd'); ax2.axvline(0, color='#ddd')
    otxt = ax2.text(0, -1.55, '', ha='center', fontsize=9, fontweight='bold', color='#C05050')

    omega = 0.3
    n_pos = 35
    arrows = [None, None]

    def animate(frame):
        pos = frame
        xs = np.sin(omega*np.arange(pos+1))
        ys = np.cos(omega*np.arange(pos+1))
        trail.set_data(xs, ys)
        point.set_data([xs[-1]], [ys[-1]])
        ptxt.set_text(f'pos = {pos}')

        for a in arrows:
            if a: a.remove()
        x0, y0 = np.sin(0), np.cos(0)
        xp, yp = np.sin(omega*pos), np.cos(omega*pos)
        arrows[0] = ax2.annotate('', xy=(x0*1.2, y0*1.2), xytext=(0,0),
            arrowprops=dict(arrowstyle='->', color='#5080B0', lw=2))
        arrows[1] = ax2.annotate('', xy=(xp*1.2, yp*1.2), xytext=(0,0),
            arrowprops=dict(arrowstyle='->', color='#D05050', lw=2))
        otxt.set_text(f'rotation = {np.degrees(omega*pos):.0f} deg')
        return trail, point, ptxt, otxt

    anim = animation.FuncAnimation(fig, animate, frames=n_pos, interval=SLOW, blit=False, repeat=True)
    plt.tight_layout()
    anim.save(f'{OUT}/03_rotation_matrix.gif', writer='pillow', fps=5, dpi=DPI)
    plt.close()
    print("v2 03")

# ============================================================
# 图4: 高维正交性（缩小）
# ============================================================
def gen_04():
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    for ax in axes:
        ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
        ax.set_aspect('equal'); ax.axhline(0, color='#eee'); ax.axvline(0, color='#eee')

    axes[0].set_title('2D', fontsize=11, fontweight='bold')
    axes[0].annotate('', xy=(1,0), xytext=(0,0), arrowprops=dict(arrowstyle='->', color='#D06050', lw=2))
    axes[0].annotate('', xy=(0,1), xytext=(0,0), arrowprops=dict(arrowstyle='->', color='#5080B0', lw=2))
    axes[0].annotate('', xy=(0.6,0.8), xytext=(0,0), arrowprops=dict(arrowstyle='->', color='#E0A040', lw=1.5, alpha=0.5))
    axes[0].text(0, -1.25, '2 orthogonal dirs\n3rd interferes', ha='center', fontsize=7, color='#888')

    axes[1].set_title('3D', fontsize=11, fontweight='bold')
    for ang, c in [(0,'#D06050'), (np.pi/2,'#5080B0'), (np.pi*5/6,'#50A060')]:
        axes[1].annotate('', xy=(np.cos(ang), np.sin(ang)), xytext=(0,0),
            arrowprops=dict(arrowstyle='->', color=c, lw=2))
    axes[1].text(0, -1.25, '3 orthogonal dirs', ha='center', fontsize=7, color='#888')

    axes[2].set_title('768D', fontsize=11, fontweight='bold')
    np.random.seed(42)
    cmap = plt.cm.coolwarm(np.linspace(0.15, 0.85, 25))
    for i in range(25):
        ang = np.random.uniform(0, 2*np.pi)
        length = 0.7 + np.random.uniform(0, 0.3)
        axes[2].annotate('', xy=(length*np.cos(ang), length*np.sin(ang)), xytext=(0,0),
            arrowprops=dict(arrowstyle='->', color=cmap[i], lw=1.2, alpha=0.7))
    axes[2].text(0, -1.25, 'Many near-orthogonal\nvectors coexist', ha='center', fontsize=7,
                 color='#50A060', fontweight='bold')

    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{OUT}/04_high_dim_orthogonal.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v2 04")

# ============================================================
# 图5: 向量加法=平移（修复文字重叠，缩小）
# ============================================================
def gen_05():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-0.5, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Position Encoding = Translation in Embedding Space', fontsize=11, fontweight='bold')

    e_cat = np.array([1.5, 2.0])
    p_vecs = [np.array([0, 0]), np.array([1.5, 0.8]), np.array([3.0, 1.6]), np.array([4.5, 2.4])]
    clrs = ['#D06050', '#E0A040', '#50A060', '#5080B0']
    lbls = ['pos=0', 'pos=1', 'pos=2', 'pos=3']

    # 语义向量
    ax.annotate('', xy=e_cat, xytext=(0,0),
        arrowprops=dict(arrowstyle='->', color='#aaa', lw=2, linestyle='dashed'))
    ax.text(e_cat[0]-0.6, e_cat[1]+0.3, 'semantic\nvector e', fontsize=8, color='#999', fontstyle='italic')

    for i, (p, c, lb) in enumerate(zip(p_vecs, clrs, lbls)):
        result = e_cat + p
        # 位置编码箭头
        ax.annotate('', xy=result, xytext=e_cat,
            arrowprops=dict(arrowstyle='->', color=c, lw=1.8, alpha=0.7))
        ax.plot(result[0], result[1], 'o', color=c, markersize=9, zorder=5)
        # 标签放在点的右上方，错开位置避免重叠
        offset_x = 0.2
        offset_y = 0.3 if i % 2 == 0 else -0.4
        ax.text(result[0]+offset_x, result[1]+offset_y, f'"cat"@{lb}',
                fontsize=8, color=c, fontweight='bold')

    ax.text(6.5, 1.0, 'Same word,\ndifferent positions\n= different points',
            fontsize=9, color='#C05050', ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8E1', edgecolor='#E0A040'))

    plt.tight_layout()
    plt.savefig(f'{OUT}/05_vector_addition.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v2 05")

# ============================================================
# 图6: 数据流（缩小）
# ============================================================
def gen_06():
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_title('Token Journey through Transformer', fontsize=12, fontweight='bold', pad=8)

    steps = [
        (5, 13,  '"cat" (ID=3721)',        '#95a5a6'),
        (5, 11.5, 'Token Embedding\ne = wte[3721]',  '#5080B0'),
        (5, 10,  '+ Position Embedding\nx = e + p (translation!)', '#D06050'),
        (5, 8.5, 'Q, K, V projections\n(rotation + scaling)', '#8060A0'),
        (5, 7,   'Attention\n(weighted average)', '#40A0A0'),
        (5, 5.5, 'FFN / MLP\n(inject knowledge)', '#E09040'),
        (5, 4,   'x N layers', '#95a5a6'),
        (5, 2.5, 'lm_head: x * wte^T\n(project to vocab)', '#50A060'),
        (5, 1,   'softmax -> next token',  '#D06050'),
    ]
    for i, (x, y, text, color) in enumerate(steps):
        bbox = dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.85, edgecolor='white')
        ax.text(x, y, text, ha='center', va='center', fontsize=8,
                fontweight='bold', color='white', bbox=bbox)
        if i < len(steps) - 1:
            ax.annotate('', xy=(x, steps[i+1][1]+0.45), xytext=(x, y-0.45),
                arrowprops=dict(arrowstyle='->', color='#bbb', lw=1.2))

    plt.tight_layout()
    plt.savefig(f'{OUT}/05_data_flow.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v2 06")

# ============================================================
# 图7: 矩阵变换（缩小）
# ============================================================
def gen_07():
    fig, axes = plt.subplots(1, 3, figsize=(8, 3.2))
    for ax in axes:
        ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.axhline(0, color='#eee'); ax.axvline(0, color='#eee')
        ax.grid(True, alpha=0.08)
        ax.tick_params(labelsize=6)

    axes[0].set_title('x = e + p', fontsize=9, fontweight='bold')
    axes[1].set_title('x * W_Q (query)', fontsize=9, fontweight='bold')
    axes[2].set_title('x * W_K (key)', fontsize=9, fontweight='bold')

    vecs = [(1.2, 0.8, '#D06050', 'cat@0'), (0.9, 1.8, '#5080B0', 'cat@3'),
            (-0.8, 1.2, '#50A060', 'eat@1'), (0.4, -1.0, '#E0A040', 'fish@2')]

    th_q, th_k = np.radians(30), np.radians(-45)
    W_Q = np.array([[np.cos(th_q)*0.8, -np.sin(th_q)*1.2],
                     [np.sin(th_q)*0.8, np.cos(th_q)*1.2]])
    W_K = np.array([[np.cos(th_k)*1.1, -np.sin(th_k)*0.7],
                     [np.sin(th_k)*1.1, np.cos(th_k)*0.7]])

    for (x, y, c, lb) in vecs:
        v = np.array([x, y])
        vq, vk = W_Q @ v, W_K @ v
        for ax, vec in [(axes[0], v), (axes[1], vq), (axes[2], vk)]:
            ax.annotate('', xy=vec, xytext=(0,0), arrowprops=dict(arrowstyle='->', color=c, lw=1.5))
        axes[0].text(x+0.1, y+0.1, lb, fontsize=6, color=c)

    plt.tight_layout()
    plt.savefig(f'{OUT}/06_matrix_transform.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v2 07")

# ============================================================
# 图8: 因果掩码（柔和颜色）
# ============================================================
def gen_08():
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    words = ['cat', 'eat', 'the', 'fish']
    n = len(words)
    mask = np.tril(np.ones((n, n)))

    # 柔和的蓝白配色代替大红大绿
    cmap = matplotlib.colors.ListedColormap(['#F5F0F0', '#D6EAF8'])  # 浅灰 vs 浅蓝
    ax.imshow(mask, cmap=cmap, aspect='equal')

    for i in range(n):
        for j in range(n):
            if mask[i, j] == 1:
                ax.text(j, i, 'o', ha='center', va='center', fontsize=14,
                        color='#2E86C1', fontweight='bold')
            else:
                ax.text(j, i, 'x', ha='center', va='center', fontsize=14,
                        color='#C0C0C0')

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(words, fontsize=10)
    ax.set_yticklabels(words, fontsize=10)
    ax.set_xlabel('Key', fontsize=10)
    ax.set_ylabel('Query', fontsize=10)
    ax.set_title('Causal Mask\nEach token can only attend to itself and earlier tokens',
                 fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUT}/07_causal_mask.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v2 08")

# ============================================================
# 图9: PE 热力图（缩小）
# ============================================================
def gen_09():
    d_model = 128
    max_pos = 80
    pe = np.zeros((max_pos, d_model))
    for pos in range(max_pos):
        for i in range(0, d_model, 2):
            freq = 1.0 / (10000 ** (i / d_model))
            pe[pos, i] = np.sin(pos * freq)
            pe[pos, i+1] = np.cos(pos * freq)

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pe, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    ax.set_xlabel('Dimension (i)', fontsize=10)
    ax.set_ylabel('Position (pos)', fontsize=10)
    ax.set_title('Sinusoidal PE Matrix (d=128)', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(f'{OUT}/08_pe_heatmap.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v2 09")

# ============================================================
# 图10: PE 点积热力图（缩小）
# ============================================================
def gen_10():
    d_model = 128
    max_pos = 80
    pe = np.zeros((max_pos, d_model))
    for pos in range(max_pos):
        for i in range(0, d_model, 2):
            freq = 1.0 / (10000 ** (i / d_model))
            pe[pos, i] = np.sin(pos * freq)
            pe[pos, i+1] = np.cos(pos * freq)

    dot = pe @ pe.T
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(dot, cmap='viridis', aspect='equal')
    ax.set_xlabel('Position j', fontsize=10)
    ax.set_ylabel('Position i', fontsize=10)
    ax.set_title('PE Dot Product: dot(PE(i), PE(j))\nDepends only on |i-j|', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(f'{OUT}/09_pe_dot_product.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v2 10")

if __name__ == '__main__':
    print("Regenerating all figures (v2)...")
    gen_01()
    gen_02()
    gen_03()
    gen_04()
    gen_05()
    gen_06()
    gen_07()
    gen_08()
    gen_09()
    gen_10()
    print("All done!")
