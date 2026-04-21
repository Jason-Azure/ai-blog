#!/usr/bin/env python3
"""Figures for 看见物理（七）：量子——观察者与被观察.
Outputs:
  - double_slit.gif       单电子双缝干涉的逐步累积
  - wave_collapse.gif     波函数测量坍缩
  - qbism_update.png      贝叶斯更新 vs 量子测量：同一条公式
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.expanduser("~/ai-blog/content/posts/see-physics-7-quantum")
os.makedirs(OUT, exist_ok=True)


def make_double_slit_gif():
    """单电子双缝：每一个点看似随机，累积起来却是干涉条纹。"""
    print("→ double_slit.gif ...")

    rng = np.random.default_rng(2026)

    # Screen x coordinate
    x = np.linspace(-6, 6, 600)
    # Two-slit interference intensity pattern: cos^2 * Gaussian envelope
    envelope = np.exp(-x**2 / 14.0)
    pattern = envelope * np.cos(1.8 * x)**2 + 0.02
    pattern /= pattern.sum()

    # Sample 1200 electron hit positions from the pattern
    N_TOTAL = 1200
    hits = rng.choice(x, size=N_TOTAL, p=pattern)
    hits_y = rng.uniform(0, 1, size=N_TOTAL)  # vertical scatter for scatter plot

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.2),
                                    gridspec_kw={'width_ratios': [1.3, 1]})
    fig.patch.set_facecolor('white')

    # Left: scatter of electron hits
    scat = axL.scatter([], [], s=9, c='#1E88E5', alpha=0.6, edgecolors='none')
    axL.set_xlim(-6, 6)
    axL.set_ylim(-0.05, 1.05)
    axL.set_yticks([])
    axL.set_xlabel('屏幕位置', fontsize=11)
    axL.set_title('屏幕上的电子落点', fontsize=13, fontweight='bold', pad=10)
    axL.spines['top'].set_visible(False)
    axL.spines['right'].set_visible(False)
    count_text = axL.text(0.02, 0.96, '', transform=axL.transAxes,
                          fontsize=12, color='#444', fontweight='bold',
                          verticalalignment='top',
                          bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor='#FFF8E1', edgecolor='#e0b030'))

    # Right: histogram building up, overlay theoretical curve
    axR.set_xlim(-6, 6)
    axR.set_ylim(0, pattern.max() * N_TOTAL * 0.5 + 5)
    axR.set_xlabel('屏幕位置', fontsize=11)
    axR.set_ylabel('落点数', fontsize=11)
    axR.set_title('累积分布 → 干涉条纹', fontsize=13, fontweight='bold', pad=10)
    axR.spines['top'].set_visible(False)
    axR.spines['right'].set_visible(False)
    theory_line, = axR.plot(x, pattern * N_TOTAL * 0.33, color='#E53935',
                            linewidth=1.8, alpha=0.0, linestyle='--',
                            label='理论 |ψ|²')
    bins = np.linspace(-6, 6, 60)
    current_bars = []  # list of Rectangle artists for cleanup

    total_frames = 60

    def frame_count(frame):
        # Smooth ramp from 1 to N_TOTAL
        t = frame / (total_frames - 1)
        return max(1, int(N_TOTAL * (t ** 1.3)))

    def update(frame):
        n = frame_count(frame)
        scat.set_offsets(np.c_[hits[:n], hits_y[:n]])

        # Redraw histogram
        for b in current_bars:
            b.remove()
        current_bars.clear()
        counts, edges = np.histogram(hits[:n], bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]
        bars = axR.bar(centers, counts, width=width * 0.95,
                       color='#1E88E5', alpha=0.7, edgecolor='none')
        current_bars.extend(bars)

        if n >= 300:
            theory_line.set_alpha(0.85)

        count_text.set_text(f'已打中：{n} 个电子')

        return scat, count_text

    anim = FuncAnimation(fig, update, frames=total_frames, interval=120, blit=False)

    plt.tight_layout()
    fig.text(0.5, -0.02,
             '每个电子落点都是"随机的"。但只要落得够多，|ψ|² 画出的干涉条纹就会自己浮现——'
             '它是全体未来的统计，不是任何一颗粒子的"真实路径"。',
             ha='center', fontsize=10.5, color='#444', style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F3E5F5',
                       edgecolor='#9C27B0', linewidth=1.0))

    out = os.path.join(OUT, 'double_slit.gif')
    anim.save(out, writer=PillowWriter(fps=10), dpi=110)
    plt.close(fig)
    print(f"  ✓ {out}  ({os.path.getsize(out)//1024} KB)")


def make_wave_collapse_gif():
    """波函数坍缩：|ψ|² 的平滑分布，被一次测量变成一个尖峰。"""
    print("→ wave_collapse.gif ...")

    x = np.linspace(-5, 5, 600)

    def psi_broad(x):
        g1 = np.exp(-(x + 1.5)**2 / 1.8)
        g2 = 0.85 * np.exp(-(x - 1.8)**2 / 1.6)
        g3 = 0.6 * np.exp(-(x - 0.1)**2 / 4.0)
        return g1 + g2 + g3

    broad = psi_broad(x)
    broad /= np.trapz(broad, x)

    # Measurement outcome (sampled from |psi|^2)
    rng = np.random.default_rng(7)
    outcome = rng.choice(x, p=broad / broad.sum())

    # Post-collapse: narrow Gaussian around outcome
    sigma_collapsed = 0.15
    collapsed = np.exp(-(x - outcome)**2 / (2 * sigma_collapsed**2))
    collapsed /= np.trapz(collapsed, x)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor('white')

    line, = ax.plot(x, broad, color='#1E88E5', linewidth=2.6, label='|ψ(x)|²')
    fill = ax.fill_between(x, 0, broad, color='#1E88E5', alpha=0.22)
    marker = ax.axvline(outcome, color='#E53935', linestyle='--',
                        linewidth=1.8, alpha=0.0)
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, collapsed.max() * 1.08)
    ax.set_xlabel('位置 x', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title('测量前 vs 测量后：波函数坍缩',
                 fontsize=14, fontweight='bold', pad=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize=11)

    stage_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                         fontsize=13, color='#333', fontweight='bold',
                         verticalalignment='top',
                         bbox=dict(boxstyle='round,pad=0.4',
                                   facecolor='#FFF8E1', edgecolor='#e0b030'))

    total_frames = 70

    def update(frame):
        nonlocal fill
        # 0-19: hold broad distribution
        # 20-24: flash "measurement!"
        # 25-49: smooth interpolate broad -> collapsed
        # 50-69: hold collapsed
        if frame < 20:
            curr = broad.copy()
            stage_text.set_text('测量前：所有可能性共存')
            marker.set_alpha(0.0)
        elif frame < 25:
            curr = broad.copy()
            stage_text.set_text('测量！')
            stage_text.set_color('#C62828')
            marker.set_alpha(min(1.0, (frame - 19) * 0.25))
        elif frame < 50:
            t = (frame - 25) / 24.0
            t = t ** 0.7
            curr = (1 - t) * broad + t * collapsed
            stage_text.set_text('坍缩中…')
            stage_text.set_color('#E65100')
            marker.set_alpha(1.0)
        else:
            curr = collapsed.copy()
            stage_text.set_text(f'测量后：x ≈ {outcome:.2f} 已被确定')
            stage_text.set_color('#2E7D32')
            marker.set_alpha(1.0)

        line.set_ydata(curr)
        # Refresh fill
        for coll in ax.collections[:]:
            coll.remove()
        fill = ax.fill_between(x, 0, curr, color='#1E88E5', alpha=0.22)
        return line, stage_text, marker

    anim = FuncAnimation(fig, update, frames=total_frames, interval=90, blit=False)

    fig.text(0.5, -0.03,
             '测量前：|ψ|² 是一片"可能性的云"。测量瞬间：云坍缩成一个确定的点。'
             '这不是云"原本就在那里只是我们看不清"——是看这个动作本身改写了云。',
             ha='center', fontsize=10.5, color='#444', style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD',
                       edgecolor='#42A5F5', linewidth=1.0))

    plt.tight_layout()
    out = os.path.join(OUT, 'wave_collapse.gif')
    anim.save(out, writer=PillowWriter(fps=10), dpi=110)
    plt.close(fig)
    print(f"  ✓ {out}  ({os.path.getsize(out)//1024} KB)")


def make_qbism_update():
    """两栏对照：医学检测的贝叶斯更新 vs 量子测量的坍缩。同一条公式，不同实在。"""
    print("→ qbism_update.png ...")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.8))
    fig.patch.set_facecolor('white')

    # ======== LEFT: Bayesian medical update (prior 0.1% -> posterior 9%) ========
    cats = ['先验\nP(患病)', '似然\nP(+|病)', '后验\nP(病|+)']
    vals_medical = [0.1, 99.0, 9.0]  # %
    colors_m = ['#90CAF9', '#FFB74D', '#E53935']
    bars_m = axL.bar(cats, vals_medical, color=colors_m, edgecolor='#333',
                     linewidth=0.8, width=0.6)
    axL.set_ylim(0, 105)
    axL.set_ylabel('概率 (%)', fontsize=11)
    axL.set_title('贝叶斯更新：医学检测',
                  fontsize=14, fontweight='bold', pad=12, color='#1565C0')
    axL.spines['top'].set_visible(False)
    axL.spines['right'].set_visible(False)
    axL.grid(axis='y', alpha=0.25)

    for bar, v in zip(bars_m, vals_medical):
        axL.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                 f'{v}%', ha='center', fontsize=11, fontweight='bold',
                 color='#222')

    axL.annotate('', xy=(2, 20), xytext=(0, 3),
                 arrowprops=dict(arrowstyle='->', color='#555', lw=1.5,
                                 connectionstyle='arc3,rad=-0.25'))
    axL.text(1.0, 85, '证据到了\n信念更新',
             fontsize=11, color='#555', ha='center', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFF8E1',
                       edgecolor='#e0b030'))

    # ======== RIGHT: Quantum measurement (|psi|^2 -> delta) ========
    x = np.linspace(-4, 4, 400)
    psi = np.exp(-(x + 1)**2 / 2.0) + 0.8 * np.exp(-(x - 1.2)**2 / 1.5)
    psi /= np.trapz(psi, x)

    collapsed_x = 1.2
    delta = np.exp(-(x - collapsed_x)**2 / (2 * 0.12**2))
    delta /= np.trapz(delta, x)

    axR.plot(x, psi, color='#1E88E5', linewidth=2.4,
             label='测量前：|ψ|²（先验）', alpha=0.9)
    axR.fill_between(x, 0, psi, color='#1E88E5', alpha=0.20)
    axR.plot(x, delta, color='#E53935', linewidth=2.4,
             label='测量后：δ（后验）', alpha=0.9)
    axR.fill_between(x, 0, delta, color='#E53935', alpha=0.20)
    axR.axvline(collapsed_x, color='#E53935', linestyle=':', linewidth=1.3,
                alpha=0.7)
    axR.set_xlim(-4, 4)
    axR.set_ylim(0, delta.max() * 1.08)
    axR.set_xlabel('粒子位置 x', fontsize=11)
    axR.set_ylabel('概率密度', fontsize=11)
    axR.set_title('量子测量：波函数坍缩',
                  fontsize=14, fontweight='bold', pad=12, color='#C62828')
    axR.spines['top'].set_visible(False)
    axR.spines['right'].set_visible(False)
    axR.legend(loc='upper left', fontsize=10, frameon=True)

    axR.annotate('测量事件',
                 xy=(collapsed_x, delta.max() * 0.6),
                 xytext=(-2.8, delta.max() * 0.85),
                 fontsize=11, color='#555', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#555', lw=1.4),
                 bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFF8E1',
                           edgecolor='#e0b030'))

    fig.text(0.5, -0.02,
             '左边：证据（阳性）进来，后验压缩到 9%。右边：测量发生，后验压缩到一个点。\n'
             '数学是同一个——先验 × 证据 = 后验。量子只是把这句话推到了极限：先验就是世界的全部。',
             ha='center', fontsize=11.5, color='#333',
             bbox=dict(boxstyle='round,pad=0.55', facecolor='#F3E5F5',
                       edgecolor='#9C27B0', linewidth=1.1))

    plt.tight_layout()
    out = os.path.join(OUT, 'qbism_update.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {out}  ({os.path.getsize(out)//1024} KB)")


if __name__ == '__main__':
    make_double_slit_gif()
    make_wave_collapse_gif()
    make_qbism_update()
    print("\n✓ All figures done.")
