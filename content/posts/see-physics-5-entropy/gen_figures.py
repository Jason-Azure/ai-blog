#!/usr/bin/env python3
"""Generate visuals for 看见物理（五）：熵
Outputs to ~/ai-blog/content/posts/see-physics-5-entropy/
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

# Support Chinese
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.expanduser("~/ai-blog/content/posts/see-physics-5-entropy")
os.makedirs(OUT, exist_ok=True)


# ============================================================
# 1. INK DIFFUSION GIF — 粒子从中心扩散，熵值实时递增
# ============================================================
def make_ink_diffusion():
    print("→ Generating ink_diffusion.gif ...")
    np.random.seed(42)
    N_PARTICLES = 800
    N_CELLS = 100
    N_FRAMES = 160
    STEPS_PER_FRAME = 3

    positions = np.full(N_PARTICLES, N_CELLS // 2, dtype=int)
    history_positions = []
    history_entropy = []

    def entropy(pos):
        counts = np.bincount(pos, minlength=N_CELLS).astype(float)
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    for f in range(N_FRAMES):
        for _ in range(STEPS_PER_FRAME):
            positions = np.clip(positions + np.random.choice([-1, 1], N_PARTICLES), 0, N_CELLS - 1)
        history_positions.append(positions.copy())
        history_entropy.append(entropy(positions))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.2),
                                    gridspec_kw={'height_ratios': [2.2, 1]})
    fig.patch.set_facecolor('white')

    # Top: particle distribution
    ax1.set_xlim(0, N_CELLS)
    ax1.set_ylim(-1, 1)
    ax1.set_yticks([])
    ax1.set_facecolor('#fafafa')
    ax1.set_title('墨水扩散：熵随时间增加（时间之箭）', fontsize=13, pad=10)
    scat = ax1.scatter([], [], s=8, c='#1a5490', alpha=0.55)
    time_text = ax1.text(0.02, 0.93, '', transform=ax1.transAxes, fontsize=11,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                   edgecolor='#ccc'))
    entropy_text = ax1.text(0.98, 0.93, '', transform=ax1.transAxes, fontsize=11,
                            ha='right', color='#c0392b', fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                      edgecolor='#ccc'))

    # Bottom: entropy curve
    ax2.set_xlim(0, N_FRAMES)
    max_H = np.log(N_CELLS)
    ax2.set_ylim(0, max_H * 1.05)
    ax2.set_xlabel('时间步', fontsize=10)
    ax2.set_ylabel('熵 H', fontsize=10)
    ax2.axhline(max_H, color='#888', linestyle='--', linewidth=0.8,
                label=f'最大熵 log({N_CELLS}) ≈ {max_H:.2f}')
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax2.set_facecolor('#fafafa')
    ax2.grid(alpha=0.25)
    line, = ax2.plot([], [], color='#c0392b', linewidth=2.2)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        line.set_data([], [])
        return scat, line, time_text, entropy_text

    def update(frame):
        pos = history_positions[frame]
        y_jitter = np.random.uniform(-0.8, 0.8, size=len(pos))
        scat.set_offsets(np.column_stack([pos, y_jitter]))
        line.set_data(range(frame + 1), history_entropy[:frame + 1])
        time_text.set_text(f't = {frame * STEPS_PER_FRAME:>4d}')
        entropy_text.set_text(f'H = {history_entropy[frame]:.3f}')
        return scat, line, time_text, entropy_text

    anim = animation.FuncAnimation(fig, update, frames=N_FRAMES, init_func=init,
                                    interval=60, blit=True)
    plt.tight_layout()
    out_path = os.path.join(OUT, "ink_diffusion.gif")
    anim.save(out_path, writer='pillow', fps=12, dpi=90)
    plt.close(fig)
    size_kb = os.path.getsize(out_path) // 1024
    print(f"  ✓ {out_path}  ({size_kb} KB)")


# ============================================================
# 2. MAXWELL DEMON — 双腔盒子 + 门 + 小妖精 + 快慢分子
# ============================================================
def make_maxwell_demon():
    print("→ Generating maxwell_demon.png ...")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 55)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(50, 52, '麦克斯韦的妖精（Maxwell\'s Demon）',
            fontsize=15, fontweight='bold', ha='center')
    ax.text(50, 48.3, '挑拣快分子入右室、慢分子入左室 → 自动制造温差？',
            fontsize=10.5, ha='center', color='#555', style='italic')

    # Box
    box = FancyBboxPatch((5, 8), 90, 34, boxstyle="round,pad=0.2",
                         edgecolor='#333', facecolor='#fcfcfc', linewidth=2)
    ax.add_patch(box)

    # Middle wall with door
    wall_x = 50
    ax.plot([wall_x, wall_x], [8, 22], color='#333', linewidth=2.2)
    ax.plot([wall_x, wall_x], [28, 42], color='#333', linewidth=2.2)
    # Door slot indicator
    ax.plot([wall_x - 0.8, wall_x + 0.8], [22, 22], color='#888', linewidth=1, linestyle=':')
    ax.plot([wall_x - 0.8, wall_x + 0.8], [28, 28], color='#888', linewidth=1, linestyle=':')

    # Demon — stylized
    demon_x, demon_y = wall_x, 25
    demon = Circle((demon_x, demon_y), 2.2, facecolor='#8e44ad',
                   edgecolor='#4a235a', linewidth=1.5, zorder=5)
    ax.add_patch(demon)
    # Eyes
    ax.plot(demon_x - 0.7, demon_y + 0.5, 'o', color='white', markersize=3, zorder=6)
    ax.plot(demon_x + 0.7, demon_y + 0.5, 'o', color='white', markersize=3, zorder=6)
    ax.text(demon_x, demon_y - 4.3, '妖精', fontsize=9.5, ha='center',
            color='#4a235a', fontweight='bold')

    # Labels for chambers
    ax.text(25, 44.5, '左室 (慢分子聚集 → 变冷)', fontsize=11,
            ha='center', color='#2874a6', fontweight='bold')
    ax.text(75, 44.5, '右室 (快分子聚集 → 变热)', fontsize=11,
            ha='center', color='#c0392b', fontweight='bold')

    # Molecules — slow (blue) mostly left, fast (red) mostly right
    np.random.seed(7)
    # Slow in left
    for _ in range(14):
        x = np.random.uniform(9, 47)
        y = np.random.uniform(11, 39)
        ax.plot(x, y, 'o', color='#3498db', markersize=7, alpha=0.7)
    # A few slow in right (not yet sorted)
    for _ in range(3):
        x = np.random.uniform(53, 95)
        y = np.random.uniform(11, 39)
        ax.plot(x, y, 'o', color='#3498db', markersize=7, alpha=0.45)
    # Fast in right
    for _ in range(14):
        x = np.random.uniform(53, 95)
        y = np.random.uniform(11, 39)
        ax.plot(x, y, 'o', color='#e74c3c', markersize=7, alpha=0.75)
    # A few fast in left
    for _ in range(3):
        x = np.random.uniform(9, 47)
        y = np.random.uniform(11, 39)
        ax.plot(x, y, 'o', color='#e74c3c', markersize=7, alpha=0.45)

    # Arrow: fast molecule going through to right
    arr1 = FancyArrowPatch((40, 33), (58, 33), arrowstyle='->',
                            mutation_scale=18, color='#c0392b', linewidth=1.8)
    ax.add_patch(arr1)
    # Arrow: slow molecule going to left
    arr2 = FancyArrowPatch((60, 18), (42, 18), arrowstyle='->',
                            mutation_scale=18, color='#2874a6', linewidth=1.8)
    ax.add_patch(arr2)

    # Legend
    ax.plot(15, 4, 'o', color='#3498db', markersize=8)
    ax.text(17.5, 3.7, '慢分子（低能量）', fontsize=9.5, va='center')
    ax.plot(50, 4, 'o', color='#e74c3c', markersize=8)
    ax.text(52.5, 3.7, '快分子（高能量）', fontsize=9.5, va='center')
    ax.plot(82, 4, 's', color='#8e44ad', markersize=8)
    ax.text(84.5, 3.7, '有选择能力的妖精', fontsize=9.5, va='center')

    plt.tight_layout()
    out_path = os.path.join(OUT, "maxwell_demon.png")
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    size_kb = os.path.getsize(out_path) // 1024
    print(f"  ✓ {out_path}  ({size_kb} KB)")


# ============================================================
# 3. ENTROPY TIMELINE — 四列时间轴
# ============================================================
def make_entropy_timeline():
    print("→ Generating entropy_timeline.png ...")
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 55)
    ax.axis('off')

    ax.text(50, 51.5, '熵的 160 年：三个公式，同一个灵魂',
            fontsize=16, fontweight='bold', ha='center')
    ax.text(50, 47.5, 'Clausius → Boltzmann → Shannon → Jaynes',
            fontsize=11, ha='center', color='#888', style='italic')

    # Main timeline horizontal line
    ax.plot([6, 94], [32, 32], color='#333', linewidth=2, zorder=1)

    entries = [
        dict(x=15, year='1865', name='克劳修斯\nClausius', color='#FF9800',
             formula=r'$dS = \frac{dQ}{T}$',
             tag='热力学',
             caption='熵是热量与温度的比值。\n关心蒸汽机能不能造出来\n的工程师的语言。'),
        dict(x=38, year='1877', name='玻尔兹曼\nBoltzmann', color='#4CAF50',
             formula=r'$S = k \ln W$',
             tag='统计力学',
             caption='熵是微观状态数的对数。\n熵 = 我们对系统微观状态\n的无知程度。'),
        dict(x=62, year='1948', name='Shannon', color='#2196F3',
             formula=r'$H = -\sum p_i \log p_i$',
             tag='信息论',
             caption='熵是概率分布的不确定性。\n关心一封电报多贵的\n工程师的语言。'),
        dict(x=85, year='1957', name='贾因斯\nJaynes', color='#9C27B0',
             formula=r'它们是同一个公式',
             tag='最大熵原理',
             caption='"Entropy is a measure of\nour knowledge, not a\nproperty of the system."'),
    ]

    for e in entries:
        x = e['x']
        color = e['color']
        # Year dot
        circ = Circle((x, 32), 1.8, facecolor=color, edgecolor='white',
                      linewidth=2.5, zorder=3)
        ax.add_patch(circ)
        # Year label above dot
        ax.text(x, 37.5, e['year'], fontsize=13, fontweight='bold',
                ha='center', color=color)
        # Name
        ax.text(x, 41.5, e['name'], fontsize=10, ha='center', color='#333')
        # Tag
        ax.text(x, 44.8, e['tag'], fontsize=8.5, ha='center', color='#888',
                style='italic')
        # Formula box below
        box = FancyBboxPatch((x - 10, 22), 20, 7, boxstyle="round,pad=0.3",
                              facecolor=color, alpha=0.12,
                              edgecolor=color, linewidth=1.4)
        ax.add_patch(box)
        ax.text(x, 25.5, e['formula'], fontsize=13, ha='center',
                va='center', color=color, fontweight='bold')
        # Caption
        ax.text(x, 13, e['caption'], fontsize=8.5, ha='center', va='top',
                color='#444', linespacing=1.4)

    # Bottom unifying arrow
    arr = FancyArrowPatch((8, 5), (92, 5), arrowstyle='->',
                          mutation_scale=22, color='#c0392b', linewidth=2)
    ax.add_patch(arr)
    ax.text(50, 2.5, '熵，是我们对世界的无知的量化',
            fontsize=11, ha='center', color='#c0392b', fontweight='bold',
            style='italic')

    plt.tight_layout()
    out_path = os.path.join(OUT, "entropy_timeline.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    size_kb = os.path.getsize(out_path) // 1024
    print(f"  ✓ {out_path}  ({size_kb} KB)")


if __name__ == '__main__':
    make_ink_diffusion()
    make_maxwell_demon()
    make_entropy_timeline()
    print("\n✓ All 3 visuals generated.")
