"""
位置编码文章配图 v3 — 最终修复版
- 箭头不重叠，间距充足
- 所有文字用中文（关键术语附英文）
- 图片居中显示
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
DPI = 100
SLOW = 200

# ============================================================
# 图1: 置换不变性
# ============================================================
def gen_01():
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for ax in axes:
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-1.0, 4.0)
        ax.set_aspect('equal')
        ax.axis('off')

    clrs = ['#E8837C', '#7DC4B8', '#7BB8D0']

    def draw(ax, labels, title, out_labels, out_clrs):
        ax.text(1.5, 3.6, title, ha='center', fontsize=11, fontweight='bold')
        pos_top = [(0.5, 2.8), (1.5, 2.8), (2.5, 2.8)]
        for p, lb, c in zip(pos_top, labels, out_clrs):
            circle = plt.Circle(p, 0.32, color=c, alpha=0.85)
            ax.add_patch(circle)
            ax.text(p[0], p[1], lb, ha='center', va='center', fontsize=12, fontweight='bold', color='white')

        # Attention 标注（不画连线，避免重叠）
        ax.text(1.5, 1.9, 'Attention', ha='center', fontsize=9, color='#999',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#f8f8f8', edgecolor='#ddd'))
        ax.annotate('', xy=(1.5, 1.55), xytext=(1.5, 2.25),
            arrowprops=dict(arrowstyle='->', color='#bbb', lw=1.5))

        pos_bot = [(0.5, 0.5), (1.5, 0.5), (2.5, 0.5)]
        for p, lb, c in zip(pos_bot, out_labels, out_clrs):
            rect = mpatches.FancyBboxPatch((p[0]-0.32, p[1]-0.22), 0.64, 0.44,
                boxstyle="round,pad=0.08", facecolor=c, alpha=0.35)
            ax.add_patch(rect)
            ax.text(p[0], p[1], lb, ha='center', va='center', fontsize=9, color='#555')

    draw(axes[0], ['猫', '吃', '鱼'], '输入："猫 吃 鱼"',
         ['猫*', '吃*', '鱼*'], clrs)
    draw(axes[1], ['鱼', '吃', '猫'], '输入："鱼 吃 猫"',
         ['鱼*', '吃*', '猫*'], [clrs[2], clrs[1], clrs[0]])

    fig.text(0.5, 0.02,
             '没有位置编码 → Attention 看到的是同一个集合 {猫, 吃, 鱼} → 无法区分！',
             ha='center', fontsize=10, fontweight='bold', color='#C0392B',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8E1', edgecolor='#E0A040'))
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(f'{OUT}/01_permutation_invariance.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v3 01")


# ============================================================
# 图2: 正弦波动画
# ============================================================
def gen_02():
    fig, axes = plt.subplots(4, 1, figsize=(7, 5.5), sharex=True)
    fig.suptitle('正弦位置编码：不同维度 = 不同频率的波', fontsize=11, fontweight='bold', y=0.99)
    d = 512
    max_pos = 80
    positions = np.arange(max_pos)
    dims = [0, 10, 50, 200]
    labels = ['维度 0（高频）', '维度 20（中高频）', '维度 100（中低频）', '维度 400（低频）']
    clrs = ['#D07060', '#D0A050', '#50A068', '#5080B0']
    lns = []
    for ax, di, lab, c in zip(axes, dims, labels, clrs):
        freq = 1.0 / (10000 ** (2*di/d))
        ax.set_ylabel(lab, fontsize=7, rotation=0, ha='right', va='center', labelpad=75)
        ax.set_ylim(-1.3, 1.3)
        ax.axhline(0, color='#eee', lw=0.5)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, max_pos)
        l, = ax.plot([], [], color=c, lw=1.5)
        lns.append((l, freq))
    axes[-1].set_xlabel('位置 (pos)', fontsize=9)

    def animate(frame):
        n = min(frame+1, max_pos)
        x = positions[:n]
        for l, f in lns:
            l.set_data(x, np.sin(x*f))
        return [l for l,_ in lns]

    anim = animation.FuncAnimation(fig, animate, frames=max_pos, interval=SLOW, blit=True, repeat=True)
    plt.tight_layout(rect=[0.14, 0, 1, 0.96])
    anim.save(f'{OUT}/02_sinusoidal_waves.gif', writer='pillow', fps=5, dpi=DPI)
    plt.close()
    print("v3 02")


# ============================================================
# 图3: 旋转动画
# ============================================================
def gen_03():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.set_xlim(-1.4, 1.4); ax1.set_ylim(-1.4, 1.4); ax1.set_aspect('equal')
    ax1.set_xlabel('sin(ω·pos)', fontsize=9); ax1.set_ylabel('cos(ω·pos)', fontsize=9)
    ax1.set_title('(sin, cos) 在圆上运动', fontsize=10, fontweight='bold')
    ax1.axhline(0, color='#eee'); ax1.axvline(0, color='#eee')
    th = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.sin(th), np.cos(th), '#eee', lw=1)
    trail, = ax1.plot([], [], '#7090C0', alpha=0.4, lw=1)
    point, = ax1.plot([], [], 'o', color='#C06050', markersize=7, zorder=5)
    ptxt = ax1.text(0, -1.25, '', ha='center', fontsize=9, fontweight='bold')

    ax2.set_xlim(-1.8, 1.8); ax2.set_ylim(-1.8, 1.8); ax2.set_aspect('equal')
    ax2.set_title('位置偏移 = 旋转角度', fontsize=10, fontweight='bold')
    ax2.axhline(0, color='#eee'); ax2.axvline(0, color='#eee')
    otxt = ax2.text(0, -1.6, '', ha='center', fontsize=9, fontweight='bold', color='#C06050')

    omega = 0.3
    n_pos = 35
    arrows = [None, None]

    def animate(frame):
        pos = frame
        xs = np.sin(omega*np.arange(pos+1))
        ys = np.cos(omega*np.arange(pos+1))
        trail.set_data(xs, ys)
        point.set_data([xs[-1]], [ys[-1]])
        ptxt.set_text(f'位置 = {pos}')
        for a in arrows:
            if a: a.remove()
        x0, y0 = np.sin(0), np.cos(0)
        xp, yp = np.sin(omega*pos), np.cos(omega*pos)
        arrows[0] = ax2.annotate('', xy=(x0*1.2, y0*1.2), xytext=(0,0),
            arrowprops=dict(arrowstyle='->', color='#5080B0', lw=2))
        arrows[1] = ax2.annotate('', xy=(xp*1.2, yp*1.2), xytext=(0,0),
            arrowprops=dict(arrowstyle='->', color='#C06050', lw=2))
        otxt.set_text(f'旋转角度 = {np.degrees(omega*pos):.0f}°')
        return trail, point, ptxt, otxt

    anim = animation.FuncAnimation(fig, animate, frames=n_pos, interval=SLOW, blit=False, repeat=True)
    plt.tight_layout()
    anim.save(f'{OUT}/03_rotation_matrix.gif', writer='pillow', fps=5, dpi=DPI)
    plt.close()
    print("v3 03")


# ============================================================
# 图4: 高维正交性
# ============================================================
def gen_04():
    fig, axes = plt.subplots(1, 3, figsize=(8, 3.2))
    titles = ['2 维空间', '3 维空间', '768 维空间']
    subtitles = ['只有 2 个正交方向\n第 3 个必然干扰', '3 个正交方向', '大量向量近似正交共存\n语义 + 位置不干扰']
    sub_colors = ['#999', '#999', '#50A060']

    for ax, t, st, sc in zip(axes, titles, subtitles, sub_colors):
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(t, fontsize=10, fontweight='bold')
        ax.text(0, -1.35, st, ha='center', fontsize=7, color=sc)

    # 2D
    axes[0].annotate('', xy=(1,0), xytext=(0,0), arrowprops=dict(arrowstyle='->', color='#D07060', lw=2))
    axes[0].annotate('', xy=(0,1), xytext=(0,0), arrowprops=dict(arrowstyle='->', color='#5080B0', lw=2))
    axes[0].annotate('', xy=(0.6,0.8), xytext=(0,0), arrowprops=dict(arrowstyle='->', color='#E0A040', lw=1.5, alpha=0.5))
    axes[0].text(0.65, 0.85, '?', fontsize=12, color='#E0A040', fontweight='bold')

    # 3D
    for ang, c in [(0,'#D07060'), (np.pi/2,'#5080B0'), (np.pi*5/6,'#50A060')]:
        axes[1].annotate('', xy=(np.cos(ang), np.sin(ang)), xytext=(0,0),
            arrowprops=dict(arrowstyle='->', color=c, lw=2))

    # 768D
    np.random.seed(42)
    cmap = plt.cm.coolwarm(np.linspace(0.15, 0.85, 20))
    for i in range(20):
        ang = np.random.uniform(0, 2*np.pi)
        length = 0.6 + np.random.uniform(0, 0.35)
        axes[2].annotate('', xy=(length*np.cos(ang), length*np.sin(ang)), xytext=(0,0),
            arrowprops=dict(arrowstyle='->', color=cmap[i], lw=1.2, alpha=0.65))

    plt.tight_layout()
    plt.savefig(f'{OUT}/04_high_dim_orthogonal.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v3 04")


# ============================================================
# 图5: 向量加法 = 平移（修复重叠，充足间距）
# ============================================================
def gen_05():
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.set_xlim(-1, 9)
    ax.set_ylim(-1, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('位置编码 = 空间中的平移', fontsize=12, fontweight='bold', pad=10)

    e = np.array([1.5, 1.5])  # 语义坐标

    p_vecs = [np.array([0, 0]), np.array([1.2, 1.5]), np.array([2.5, 2.8]), np.array([4.0, 3.8])]
    clrs = ['#D07060', '#E0A040', '#50A068', '#5080B0']
    lbls = ['位置 0', '位置 1', '位置 2', '位置 3']

    # 原点
    ax.plot(0, 0, 'o', color='#ccc', markersize=4)

    # 语义向量（灰色虚线）
    ax.annotate('', xy=e, xytext=(0,0),
        arrowprops=dict(arrowstyle='->', color='#bbb', lw=2, linestyle='dashed'))
    ax.text(0.2, 1.8, '语义向量 e\n("猫"是什么)', fontsize=8, color='#999', fontstyle='italic')

    # 每个位置的平移结果
    label_offsets = [(0.25, 0.35), (0.25, -0.55), (0.25, 0.35), (0.25, -0.55)]
    for i, (p, c, lb, lo) in enumerate(zip(p_vecs, clrs, lbls, label_offsets)):
        result = e + p
        # 平移箭头（从语义点到结果点）
        ax.annotate('', xy=result, xytext=e,
            arrowprops=dict(arrowstyle='->', color=c, lw=1.8, alpha=0.7))
        # 结果点
        ax.plot(result[0], result[1], 'o', color=c, markersize=10, zorder=5)
        # 标签（交错上下，避免重叠）
        ax.text(result[0]+lo[0], result[1]+lo[1],
                f'"猫"@{lb}', fontsize=8, color=c, fontweight='bold')

    # 说明框（放右下角，不与箭头重叠）
    ax.text(7, 1.5, '同一个词\n不同位置\n= 不同的点\n\n模型可以区分！',
            fontsize=9, color='#C06050', ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF8E1', edgecolor='#E0A040'))

    plt.tight_layout()
    plt.savefig(f'{OUT}/05_vector_addition.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v3 05")


# ============================================================
# 图6: 数据流全景
# ============================================================
def gen_06():
    fig, ax = plt.subplots(figsize=(5.5, 9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 14)
    ax.axis('off')
    ax.set_title('Token 在 Transformer 中的旅程', fontsize=11, fontweight='bold', pad=8)

    steps = [
        (5, 13,  '"猫" (ID=3721)',         '#999'),
        (5, 11.5, 'Token Embedding 查表\ne = wte[3721]', '#5080B0'),
        (5, 10,  '+ 位置编码\nx = e + p（平移）',      '#C06050'),
        (5, 8.5, 'Q, K, V 投影\n（旋转 + 缩放）',      '#8060A0'),
        (5, 7,   'Attention\n（加权平均）',             '#40A0A0'),
        (5, 5.5, 'MLP / FFN\n（注入知识）',             '#D09040'),
        (5, 4,   '× N 层',                             '#999'),
        (5, 2.5, 'lm_head\n（投影回词表）',             '#50A060'),
        (5, 1,   'softmax → 下一个词',                  '#C06050'),
    ]
    for i, (x, y, text, color) in enumerate(steps):
        bbox = dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.85, edgecolor='white')
        ax.text(x, y, text, ha='center', va='center', fontsize=8,
                fontweight='bold', color='white', bbox=bbox)
        if i < len(steps) - 1:
            ax.annotate('', xy=(x, steps[i+1][1]+0.5), xytext=(x, y-0.5),
                arrowprops=dict(arrowstyle='->', color='#ccc', lw=1.2))

    plt.tight_layout()
    plt.savefig(f'{OUT}/05_data_flow.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v3 06")


# ============================================================
# 图7: 矩阵变换
# ============================================================
def gen_07():
    fig, axes = plt.subplots(1, 3, figsize=(8, 3.5))
    for ax in axes:
        ax.set_xlim(-2.8, 2.8); ax.set_ylim(-2.8, 2.8)
        ax.set_aspect('equal')
        ax.axhline(0, color='#f0f0f0'); ax.axvline(0, color='#f0f0f0')
        ax.tick_params(labelsize=6)

    axes[0].set_title('原始向量 x = e + p', fontsize=9, fontweight='bold')
    axes[1].set_title('× W_Q（查询投影）', fontsize=9, fontweight='bold')
    axes[2].set_title('× W_K（键投影）', fontsize=9, fontweight='bold')

    vecs = [(1.2, 0.6, '#D07060', '猫@0'), (0.8, 1.8, '#5080B0', '猫@3'),
            (-0.9, 1.0, '#50A068', '吃@1'), (0.3, -1.1, '#E0A040', '鱼@2')]

    th_q, th_k = np.radians(35), np.radians(-40)
    W_Q = np.array([[np.cos(th_q)*0.9, -np.sin(th_q)*1.1],
                     [np.sin(th_q)*0.9, np.cos(th_q)*1.1]])
    W_K = np.array([[np.cos(th_k)*1.0, -np.sin(th_k)*0.8],
                     [np.sin(th_k)*1.0, np.cos(th_k)*0.8]])

    for (x, y, c, lb) in vecs:
        v = np.array([x, y])
        vq, vk = W_Q @ v, W_K @ v
        for ax, vec in [(axes[0], v), (axes[1], vq), (axes[2], vk)]:
            ax.annotate('', xy=vec, xytext=(0,0), arrowprops=dict(arrowstyle='->', color=c, lw=1.5))
        # 只在原始图上标文字，避免重叠
        axes[0].text(x*1.15, y*1.15, lb, fontsize=7, color=c, fontweight='bold')

    fig.text(0.5, 0.01, '不同的矩阵 = 不同的"视角"。W_Q 决定"怎么提问"，W_K 决定"怎么回答"',
             ha='center', fontsize=8, color='#888')
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(f'{OUT}/06_matrix_transform.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v3 07")


# ============================================================
# 图8: 因果掩码（柔和色调）
# ============================================================
def gen_08():
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    words = ['猫', '吃', '了', '鱼']
    n = len(words)
    mask = np.tril(np.ones((n, n)))

    # 柔和的蓝白配色
    cmap = matplotlib.colors.ListedColormap(['#F5F0F0', '#D6EAF8'])
    ax.imshow(mask, cmap=cmap, aspect='equal')

    for i in range(n):
        for j in range(n):
            if mask[i, j] == 1:
                ax.text(j, i, '可见', ha='center', va='center', fontsize=10,
                        color='#2E86C1', fontweight='bold')
            else:
                ax.text(j, i, '遮挡', ha='center', va='center', fontsize=10,
                        color='#C0C0C0')

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(words, fontsize=11)
    ax.set_yticklabels(words, fontsize=11)
    ax.set_xlabel('被关注的词 (Key)', fontsize=10)
    ax.set_ylabel('当前词 (Query)', fontsize=10)
    ax.set_title('因果掩码（Causal Mask）\n每个词只能看到自己和前面的词', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUT}/07_causal_mask.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v3 08")


# ============================================================
# 图9: PE 热力图
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
    ax.set_xlabel('嵌入维度 (i)', fontsize=10)
    ax.set_ylabel('位置 (pos)', fontsize=10)
    ax.set_title('正弦位置编码矩阵 (d=128)', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='PE 值')
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(f'{OUT}/08_pe_heatmap.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v3 09")


# ============================================================
# 图10: PE 点积热力图
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
    ax.set_xlabel('位置 j', fontsize=10)
    ax.set_ylabel('位置 i', fontsize=10)
    ax.set_title('PE 点积：dot(PE(i), PE(j))\n颜色只取决于距离 |i-j|', fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='点积值')
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(f'{OUT}/09_pe_dot_product.png', dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print("v3 10")


if __name__ == '__main__':
    print("v3 全部重新生成...")
    gen_01(); gen_02(); gen_03(); gen_04(); gen_05()
    gen_06(); gen_07(); gen_08(); gen_09(); gen_10()
    print("完成！")
