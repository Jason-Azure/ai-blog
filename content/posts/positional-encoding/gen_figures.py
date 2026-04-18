"""
位置编码文章配图生成脚本
所有动画 interval=200ms（默认的1/4速度）
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
import os

# 中文字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.dirname(os.path.abspath(__file__))
if 'positional-encoding' not in OUT:
    OUT = os.path.expanduser('~/ai-blog/content/posts/positional-encoding')
os.makedirs(OUT, exist_ok=True)

SLOW_INTERVAL = 200  # 动画帧间隔 ms（慢速）

# ============================================================
# 图1: 置换不变性 — "猫吃鱼" = "鱼吃猫"
# ============================================================
def gen_01_permutation():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax in axes:
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 3.5)
        ax.set_aspect('equal')
        ax.axis('off')

    # 左图：猫吃鱼
    words_left = ['cat', 'eat', 'fish']
    labels_left = ['猫', '吃', '鱼']
    positions_left = [(0.5, 2.5), (1.5, 2.5), (2.5, 2.5)]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    axes[0].set_title('输入: "猫 吃 鱼"', fontsize=14, fontweight='bold', pad=15)
    for pos, label, color in zip(positions_left, labels_left, colors):
        circle = plt.Circle(pos, 0.35, color=color, alpha=0.8)
        axes[0].add_patch(circle)
        axes[0].text(pos[0], pos[1], label, ha='center', va='center',
                     fontsize=16, fontweight='bold', color='white')

    # Attention 连线（全连接）
    for i, p1 in enumerate(positions_left):
        for j, p2 in enumerate(positions_left):
            if i != j:
                axes[0].annotate('', xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3, lw=1.5))

    # 输出
    out_positions = [(0.5, 0.5), (1.5, 0.5), (2.5, 0.5)]
    out_labels = ['猫*', '吃*', '鱼*']
    for pos, label, color in zip(out_positions, out_labels, colors):
        rect = mpatches.FancyBboxPatch((pos[0]-0.35, pos[1]-0.25), 0.7, 0.5,
            boxstyle="round,pad=0.1", facecolor=color, alpha=0.5)
        axes[0].add_patch(rect)
        axes[0].text(pos[0], pos[1], label, ha='center', va='center', fontsize=12)

    axes[0].annotate('', xy=(1.5, 1.2), xytext=(1.5, 1.9),
        arrowprops=dict(arrowstyle='->', color='black', lw=2))
    axes[0].text(1.5, 1.55, 'Attention', ha='center', va='center', fontsize=10, color='black')

    # 右图：鱼吃猫（打乱顺序）
    labels_right = ['鱼', '吃', '猫']
    colors_right = ['#45B7D1', '#4ECDC4', '#FF6B6B']

    axes[1].set_title('输入: "鱼 吃 猫"', fontsize=14, fontweight='bold', pad=15)
    for pos, label, color in zip(positions_left, labels_right, colors_right):
        circle = plt.Circle(pos, 0.35, color=color, alpha=0.8)
        axes[1].add_patch(circle)
        axes[1].text(pos[0], pos[1], label, ha='center', va='center',
                     fontsize=16, fontweight='bold', color='white')

    for i, p1 in enumerate(positions_left):
        for j, p2 in enumerate(positions_left):
            if i != j:
                axes[1].annotate('', xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.3, lw=1.5))

    out_labels_right = ['鱼*', '吃*', '猫*']
    for pos, label, color in zip(out_positions, out_labels_right, colors_right):
        rect = mpatches.FancyBboxPatch((pos[0]-0.35, pos[1]-0.25), 0.7, 0.5,
            boxstyle="round,pad=0.1", facecolor=color, alpha=0.5)
        axes[1].add_patch(rect)
        axes[1].text(pos[0], pos[1], label, ha='center', va='center', fontsize=12)

    axes[1].annotate('', xy=(1.5, 1.2), xytext=(1.5, 1.9),
        arrowprops=dict(arrowstyle='->', color='black', lw=2))
    axes[1].text(1.5, 1.55, 'Attention', ha='center', va='center', fontsize=10, color='black')

    # 中间等号
    fig.text(0.5, 0.15, '=  Attention 看到的是同一个 {猫, 吃, 鱼} 集合！',
             ha='center', fontsize=13, fontweight='bold', color='#E74C3C',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF3CD', edgecolor='#E74C3C'))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(f'{OUT}/01_permutation_invariance.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("✓ 01_permutation_invariance.png")


# ============================================================
# 图2: 正弦位置编码的多频率波形（动画）
# ============================================================
def gen_02_sinusoidal_waves():
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    fig.suptitle('正弦位置编码: 不同维度 = 不同频率的波', fontsize=14, fontweight='bold', y=0.98)

    d_model = 512
    max_pos = 100
    positions = np.arange(max_pos)

    dims = [0, 10, 50, 200]
    dim_labels = ['维度 0 (i=0)\n高频 — 区分相邻位置',
                  '维度 20 (i=10)\n中高频',
                  '维度 100 (i=50)\n中低频',
                  '维度 400 (i=200)\n低频 — 区分远距离']
    colors_wave = ['#E74C3C', '#F39C12', '#27AE60', '#2980B9']

    lines = []
    for ax, dim_i, label, color in zip(axes, dims, dim_labels, colors_wave):
        freq = 1.0 / (10000 ** (2 * dim_i / d_model))
        ax.set_ylabel(label, fontsize=8, rotation=0, ha='right', va='center', labelpad=90)
        ax.set_ylim(-1.3, 1.3)
        ax.axhline(y=0, color='gray', alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=8)
        line, = ax.plot([], [], color=color, lw=2)
        lines.append((line, freq))
        ax.set_xlim(0, max_pos)

    axes[-1].set_xlabel('位置 (pos)', fontsize=11)

    def init():
        for line, _ in lines:
            line.set_data([], [])
        return [l for l, _ in lines]

    def animate(frame):
        n = min(frame + 1, max_pos)
        x = positions[:n]
        for line, freq in lines:
            y = np.sin(x * freq)
            line.set_data(x, y)
        return [l for l, _ in lines]

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=max_pos, interval=SLOW_INTERVAL,
                                   blit=True, repeat=True)

    plt.tight_layout(rect=[0.15, 0, 1, 0.96])
    anim.save(f'{OUT}/02_sinusoidal_waves.gif', writer='pillow', fps=5, dpi=100)
    plt.close()
    print("✓ 02_sinusoidal_waves.gif")


# ============================================================
# 图3: sin+cos 配对形成旋转（动画）
# ============================================================
def gen_03_rotation():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # 左图：一对 (sin, cos) 在二维平面上的轨迹
    ax1.set_xlim(-1.4, 1.4)
    ax1.set_ylim(-1.4, 1.4)
    ax1.set_aspect('equal')
    ax1.set_xlabel('sin(ω·pos)', fontsize=11)
    ax1.set_ylabel('cos(ω·pos)', fontsize=11)
    ax1.set_title('每对 (sin, cos) 在圆上运动', fontsize=13, fontweight='bold')
    ax1.axhline(y=0, color='gray', alpha=0.3)
    ax1.axvline(x=0, color='gray', alpha=0.3)

    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.sin(theta), np.cos(theta), 'lightgray', lw=1, alpha=0.5)

    trail, = ax1.plot([], [], 'b-', alpha=0.3, lw=1)
    point, = ax1.plot([], [], 'ro', markersize=10, zorder=5)
    pos_text = ax1.text(0, -1.25, '', ha='center', fontsize=11, fontweight='bold')

    # 右图：旋转矩阵可视化
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_aspect('equal')
    ax2.set_title('位置偏移 = 旋转角度', fontsize=13, fontweight='bold')
    ax2.axhline(y=0, color='gray', alpha=0.3)
    ax2.axvline(x=0, color='gray', alpha=0.3)

    omega = 0.3
    n_positions = 40

    arrow_base = None
    arrow_current = None
    offset_text = ax2.text(0, -1.7, '', ha='center', fontsize=11, fontweight='bold', color='#E74C3C')

    def init():
        trail.set_data([], [])
        point.set_data([], [])
        pos_text.set_text('')
        offset_text.set_text('')
        return trail, point, pos_text, offset_text

    def animate(frame):
        nonlocal arrow_base, arrow_current
        pos = frame
        x_vals = np.sin(omega * np.arange(pos+1))
        y_vals = np.cos(omega * np.arange(pos+1))

        trail.set_data(x_vals, y_vals)
        point.set_data([x_vals[-1]], [y_vals[-1]])
        pos_text.set_text(f'pos = {pos}')

        # 右图：显示 pos=0 的向量和当前向量
        if arrow_base:
            arrow_base.remove()
        if arrow_current:
            arrow_current.remove()

        x0, y0 = np.sin(0), np.cos(0)
        xp, yp = np.sin(omega * pos), np.cos(omega * pos)

        arrow_base = ax2.annotate('', xy=(x0*1.3, y0*1.3), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#3498DB', lw=2.5))
        arrow_current = ax2.annotate('', xy=(xp*1.3, yp*1.3), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2.5))

        angle_deg = np.degrees(omega * pos)
        offset_text.set_text(f'旋转角度 = ω×{pos} = {angle_deg:.0f}°')

        return trail, point, pos_text, offset_text

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=n_positions, interval=SLOW_INTERVAL,
                                   blit=False, repeat=True)

    plt.tight_layout()
    anim.save(f'{OUT}/03_rotation_matrix.gif', writer='pillow', fps=5, dpi=100)
    plt.close()
    print("✓ 03_rotation_matrix.gif")


# ============================================================
# 图4: 高维空间的近似正交性
# ============================================================
def gen_04_high_dim():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # 2D: 只有2个正交方向
    ax = axes[0]
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('2 维空间', fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='gray', alpha=0.2)
    ax.axvline(x=0, color='gray', alpha=0.2)

    ax.annotate('', xy=(1, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2.5))
    ax.annotate('', xy=(0, 1), xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color='#3498DB', lw=2.5))
    ax.annotate('', xy=(0.6, 0.8), xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color='#F39C12', lw=2, alpha=0.6))
    ax.text(1.1, -0.15, 'e₁', fontsize=12, color='#E74C3C', fontweight='bold')
    ax.text(-0.25, 1.1, 'e₂', fontsize=12, color='#3498DB', fontweight='bold')
    ax.text(0.7, 0.85, '?', fontsize=14, color='#F39C12', fontweight='bold')
    ax.text(0, -1.3, '只有 2 个正交方向\n第 3 个必然干扰', ha='center', fontsize=9, color='#666')

    # 3D
    ax = axes[1]
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('3 维空间', fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='gray', alpha=0.2)
    ax.axvline(x=0, color='gray', alpha=0.2)

    angles_3d = [0, np.pi/2, np.pi*5/6]
    colors_3d = ['#E74C3C', '#3498DB', '#27AE60']
    for angle, color in zip(angles_3d, colors_3d):
        ax.annotate('', xy=(np.cos(angle), np.sin(angle)), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
    ax.text(0, -1.3, '3 个正交方向\n(投影到 2D 显示)', ha='center', fontsize=9, color='#666')

    # 768D
    ax = axes[2]
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title('768 维空间', fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='gray', alpha=0.2)
    ax.axvline(x=0, color='gray', alpha=0.2)

    np.random.seed(42)
    n_arrows = 30
    angles_hd = np.random.uniform(0, 2*np.pi, n_arrows)
    cmap = plt.cm.rainbow(np.linspace(0, 1, n_arrows))
    for angle, color in zip(angles_hd, cmap):
        length = 0.8 + np.random.uniform(0, 0.3)
        ax.annotate('', xy=(length*np.cos(angle), length*np.sin(angle)), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.7))

    ax.text(0, -1.3, '大量向量近似正交共存\n语义 + 位置 互不干扰', ha='center', fontsize=9,
            color='#27AE60', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUT}/04_high_dim_orthogonal.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("✓ 04_high_dim_orthogonal.png")


# ============================================================
# 图5: 向量加法 = 空间平移（动画）
# ============================================================
def gen_05_vector_addition():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, 7)
    ax.set_aspect('equal')
    ax.set_title('位置编码 = 空间平移: 同一个词在不同位置被移到不同区域', fontsize=13, fontweight='bold')
    ax.axis('off')

    # 原始语义向量（不加位置）
    e_cat = np.array([2.0, 3.0])  # "猫"的语义坐标

    # 不同位置的 position embedding（平移向量）
    p_vecs = [
        np.array([0.0, 0.0]),     # pos=0
        np.array([1.5, 0.5]),     # pos=1
        np.array([3.0, 1.0]),     # pos=2
        np.array([4.5, 1.5]),     # pos=3
    ]

    colors_pos = ['#E74C3C', '#F39C12', '#27AE60', '#2980B9']
    pos_labels = ['pos=0', 'pos=1', 'pos=2', 'pos=3']

    # 画原点
    ax.plot(0, 0, 'ko', markersize=5)
    ax.text(0.1, -0.3, '原点', fontsize=9, color='gray')

    # 语义向量（灰色虚线）
    ax.annotate('', xy=e_cat, xytext=(0, 0),
        arrowprops=dict(arrowstyle='->', color='gray', lw=2, linestyle='dashed'))
    ax.text(e_cat[0]+0.1, e_cat[1]+0.2, '"猫" 的语义向量 e',
            fontsize=10, color='gray', fontstyle='italic')

    # 每个位置的结果
    for i, (p, color, label) in enumerate(zip(p_vecs, colors_pos, pos_labels)):
        result = e_cat + p
        # 位置编码向量（从 e_cat 出发的箭头）
        ax.annotate('', xy=result, xytext=e_cat,
            arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.7))
        # 结果点
        ax.plot(result[0], result[1], 'o', color=color, markersize=12, zorder=5)
        ax.text(result[0]+0.15, result[1]+0.15, f'"猫"@{label}\ne + p{i}',
                fontsize=9, color=color, fontweight='bold')

    # 标注平移说明
    ax.text(5.5, 0.5, '每个箭头 = 一个\nposition embedding\n(平移向量)',
            fontsize=10, color='#666', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='#ccc'))

    ax.text(4, 5.5, '同一个词, 不同位置\n→ 空间中不同的点\n→ 模型可以区分!',
            fontsize=11, color='#E74C3C', ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3CD', edgecolor='#E74C3C'))

    plt.tight_layout()
    plt.savefig(f'{OUT}/05_vector_addition.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("✓ 05_vector_addition.png")


# ============================================================
# 图6: Transformer 数据流全景（静态）
# ============================================================
def gen_06_data_flow():
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')
    ax.set_title('一个 Token 在 Transformer 中的旅程', fontsize=14, fontweight='bold', pad=10)

    steps = [
        (8, 15,  '"猫" (token ID=3721)',     '#95a5a6', 0.9),
        (8, 13.5, 'Token Embedding 查表\ne = wte[3721]',  '#3498DB', 0.85),
        (8, 11.8, '+ Position Embedding\nx = e + p    ← 平移!', '#E74C3C', 0.85),
        (8, 10.1, 'Layer Norm 归一化',       '#F39C12', 0.8),
        (8, 8.5,  'Q, K, V = x·W_Q, x·W_K, x·W_V\n← 线性投影（旋转+缩放）', '#9B59B6', 0.85),
        (8, 6.8,  'Attention: softmax(QK^T/√d)·V\n← 加权平均（与其他词交互）', '#1ABC9C', 0.85),
        (8, 5.1,  'MLP / FFN\n← 非线性变换（注入知识）', '#E67E22', 0.85),
        (8, 3.5,  '× N 层 (重复 Attention + MLP)', '#95a5a6', 0.7),
        (8, 2.0,  'lm_head: x · wte^T\n← 投影回词表', '#2ECC71', 0.85),
        (8, 0.5,  'softmax → 下一个词的概率',  '#E74C3C', 0.9),
    ]

    for i, (x, y, text, color, alpha) in enumerate(steps):
        bbox = dict(boxstyle='round,pad=0.5', facecolor=color, alpha=alpha, edgecolor='white')
        ax.text(x, y, text, ha='center', va='center', fontsize=10,
                fontweight='bold', color='white', bbox=bbox)
        if i < len(steps) - 1:
            ax.annotate('', xy=(x, steps[i+1][1] + 0.5), xytext=(x, y - 0.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    # 左侧标注
    labels = [
        (13.5, '离散 → 连续', '#3498DB'),
        (11.8, '叠加位置信息', '#E74C3C'),
        (8.5,  '空间变换\n(矩阵乘法)', '#9B59B6'),
        (6.8,  '上下文混合', '#1ABC9C'),
        (5.1,  '知识注入', '#E67E22'),
        (2.0,  '连续 → 离散', '#2ECC71'),
    ]
    for y, text, color in labels:
        ax.text(2.5, y, text, ha='center', va='center', fontsize=9,
                color=color, fontstyle='italic')

    plt.tight_layout()
    plt.savefig(f'{OUT}/05_data_flow.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("✓ 05_data_flow.png")


# ============================================================
# 图7: 矩阵变换可视化（动画）— 向量经过 W_Q 投影
# ============================================================
def gen_07_matrix_transform():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax in axes:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='gray', alpha=0.2)
        ax.axvline(x=0, color='gray', alpha=0.2)
        ax.grid(True, alpha=0.1)

    axes[0].set_title('原始向量 x = e + p\n(语义 + 位置)', fontsize=11, fontweight='bold')
    axes[1].set_title('经过 W_Q 投影\n(旋转 + 缩放)', fontsize=11, fontweight='bold')
    axes[2].set_title('经过 W_K 投影\n(不同的旋转 + 缩放)', fontsize=11, fontweight='bold')

    # 原始向量
    vectors = [
        (1.5, 1.0, '#E74C3C', '"猫"@pos0'),
        (1.2, 2.0, '#3498DB', '"猫"@pos3'),
        (-1.0, 1.5, '#27AE60', '"吃"@pos1'),
        (0.5, -1.2, '#F39C12', '"鱼"@pos2'),
    ]

    # W_Q 矩阵 (旋转30° + 缩放)
    theta_q = np.radians(30)
    W_Q = np.array([[np.cos(theta_q)*0.8, -np.sin(theta_q)*1.2],
                     [np.sin(theta_q)*0.8, np.cos(theta_q)*1.2]])

    # W_K 矩阵 (旋转-45° + 缩放)
    theta_k = np.radians(-45)
    W_K = np.array([[np.cos(theta_k)*1.1, -np.sin(theta_k)*0.7],
                     [np.sin(theta_k)*1.1, np.cos(theta_k)*0.7]])

    for (x, y, color, label) in vectors:
        v = np.array([x, y])
        vq = W_Q @ v
        vk = W_K @ v

        # 原始
        axes[0].annotate('', xy=(x, y), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=color, lw=2))
        axes[0].text(x+0.1, y+0.15, label, fontsize=8, color=color)

        # W_Q 投影
        axes[1].annotate('', xy=(vq[0], vq[1]), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=color, lw=2))
        axes[1].text(vq[0]+0.1, vq[1]+0.15, label, fontsize=8, color=color)

        # W_K 投影
        axes[2].annotate('', xy=(vk[0], vk[1]), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=color, lw=2))
        axes[2].text(vk[0]+0.1, vk[1]+0.15, label, fontsize=8, color=color)

    fig.text(0.5, 0.02, '同一组向量，不同的矩阵 → 不同的"视角"。W_Q 决定"怎么提问"，W_K 决定"怎么回答"',
             ha='center', fontsize=10, fontweight='bold', color='#666')

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(f'{OUT}/06_matrix_transform.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("✓ 06_matrix_transform.png")


# ============================================================
# 图8: 因果掩码可视化
# ============================================================
def gen_08_causal_mask():
    fig, ax = plt.subplots(figsize=(6, 6))

    words = ['猫', '吃', '了', '鱼']
    n = len(words)
    mask = np.tril(np.ones((n, n)))

    ax.imshow(mask, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')

    for i in range(n):
        for j in range(n):
            if mask[i, j] == 1:
                ax.text(j, i, '✓', ha='center', va='center', fontsize=18, color='white', fontweight='bold')
            else:
                ax.text(j, i, '✗', ha='center', va='center', fontsize=18, color='#E74C3C', fontweight='bold')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(words, fontsize=13)
    ax.set_yticklabels(words, fontsize=13)
    ax.set_xlabel('被关注的词 (Key)', fontsize=12)
    ax.set_ylabel('当前词 (Query)', fontsize=12)
    ax.set_title('因果掩码 (Causal Mask)\n每个词只能看到自己和前面的词', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{OUT}/07_causal_mask.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print("✓ 07_causal_mask.png")


# ============================================================
# 运行所有
# ============================================================
if __name__ == '__main__':
    print("开始生成位置编码文章配图...")
    gen_01_permutation()
    gen_02_sinusoidal_waves()
    gen_03_rotation()
    gen_04_high_dim()
    gen_05_vector_addition()
    gen_06_data_flow()
    gen_07_matrix_transform()
    gen_08_causal_mask()
    print("\n全部完成！")
