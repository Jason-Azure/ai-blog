#!/usr/bin/env python3
"""Figures for 看见物理（六）：相变.
Outputs:
  - emergence.png        涌现能力曲线
  - double_descent.gif   Double Descent 动画
  - grokking.gif         Grokking 动画
  - micro_macro.png      微观平滑 vs 宏观跳变 对比图
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.expanduser("~/ai-blog/content/posts/see-physics-6-phase-transition")
os.makedirs(OUT, exist_ok=True)


def make_emergence():
    print("→ emergence.png ...")
    x = np.linspace(8, 12, 200)

    def sigmoid(x, x0, k=4):
        return 1.0 / (1.0 + np.exp(-k * (x - x0)))

    y_add = sigmoid(x, 10.2) * 0.95
    y_cot = sigmoid(x, 10.8) * 0.80
    y_math = sigmoid(x, 11.3) * 0.60

    rng = np.random.default_rng(42)
    y_add += rng.normal(0, 0.005, size=x.size)
    y_cot += rng.normal(0, 0.005, size=x.size)
    y_math += rng.normal(0, 0.005, size=x.size)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor('white')

    ax.plot(x, y_add * 100, color='#2196F3', linewidth=2.5, label='三位数加法')
    ax.plot(x, y_cot * 100, color='#4CAF50', linewidth=2.5, label='思维链推理')
    ax.plot(x, y_math * 100, color='#9C27B0', linewidth=2.5, label='大学数学')

    for x0, color in [(10.2, '#2196F3'), (10.8, '#4CAF50'), (11.3, '#9C27B0')]:
        ax.axvline(x0, color=color, linestyle=':', alpha=0.5, linewidth=1.2)

    ax.annotate('平，平，平 ...',
                xy=(9.2, 8), xytext=(9.2, 25),
                fontsize=12, color='#555', ha='center',
                arrowprops=dict(arrowstyle='->', color='#888', lw=1.2))
    ax.annotate('然后突然起飞',
                xy=(11.0, 60), xytext=(11.4, 85),
                fontsize=12, color='#c0392b', ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.4))

    ax.set_xlabel('模型参数量 (log10)', fontsize=12)
    ax.set_ylabel('任务准确率 (%)', fontsize=12)
    ax.set_title('涌现能力：像相变一样锐利',
                 fontsize=15, fontweight='bold', pad=14)
    ax.set_xticks([8, 9, 10, 11, 12])
    ax.set_xticklabels(['100M', '1B', '10B', '100B', '1T'])
    ax.set_ylim(-3, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', fontsize=11, frameon=True)

    fig.text(0.5, -0.02,
             '每条曲线都像居里温度附近的磁化曲线：低于临界点接近零，过了临界点集体起飞。',
             ha='center', fontsize=11, color='#444', style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff8e1',
                       edgecolor='#e0b030', linewidth=1.0))

    plt.tight_layout()
    out = os.path.join(OUT, 'emergence.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {out}  ({os.path.getsize(out)//1024} KB)")


def make_double_descent_gif():
    """Double Descent 动画：经典 U 型 → 峰值 → 第二次下降"""
    print("→ double_descent.gif ...")

    # Full double descent curve
    x_full = np.linspace(0, 10, 500)
    # Model complexity (params / data ratio)
    # Piece 1: classic U-curve descent (0 -> 3)
    # Piece 2: rise to interpolation peak (3 -> 5)
    # Piece 3: second descent (5 -> 10)
    def dd_curve(x):
        # smooth double descent shape
        base = 0.8 * np.exp(-0.8 * x)  # initial decay
        peak = 1.8 * np.exp(-2.0 * (x - 4.5)**2)  # interpolation peak
        second = 0.15 / (1 + 0.06 * (x - 4.5)**2)  # slow second descent floor
        return base + peak + second + 0.05

    y_full = dd_curve(x_full)
    # add slight noise
    rng = np.random.default_rng(7)
    y_noisy = y_full + rng.normal(0, 0.008, size=x_full.size)
    y_noisy = np.clip(y_noisy, 0.03, None)

    # train error: monotonically decreasing to near zero
    y_train = 0.9 * np.exp(-0.6 * x_full) + 0.01
    y_train = np.clip(y_train + rng.normal(0, 0.005, x_full.size), 0.005, None)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor('white')

    line_test, = ax.plot([], [], color='#E53935', linewidth=2.8, label='测试误差')
    line_train, = ax.plot([], [], color='#1E88E5', linewidth=2.0, alpha=0.7,
                          linestyle='--', label='训练误差')
    vline = ax.axvline(x=4.5, color='#FF9800', linestyle=':', linewidth=1.5, alpha=0)
    stop_line = ax.axvline(x=3.0, color='#888', linestyle='--', linewidth=1.2, alpha=0)

    # Static decorations
    ax.set_xlim(-0.2, 10.5)
    ax.set_ylim(-0.05, 1.6)
    ax.set_xlabel('模型复杂度（参数量 / 数据量）', fontsize=12)
    ax.set_ylabel('误差', fontsize=12)
    ax.set_title('Double Descent：误差下降了两次',
                 fontsize=15, fontweight='bold', pad=14)
    ax.set_xticks([0, 2, 4.5, 7, 10])
    ax.set_xticklabels(['极小', '小', '临界点\n(参数≈数据)', '大', '极大'])
    ax.set_yticks([])
    ax.grid(alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize=11, frameon=True)

    # Annotation objects (hidden initially)
    ann_classical = ax.annotate('', xy=(0, 0), xytext=(0, 0), fontsize=0, alpha=0)
    ann_peak = ax.annotate('', xy=(0, 0), xytext=(0, 0), fontsize=0, alpha=0)
    ann_second = ax.annotate('', xy=(0, 0), xytext=(0, 0), fontsize=0, alpha=0)
    ann_stop = ax.annotate('', xy=(0, 0), xytext=(0, 0), fontsize=0, alpha=0)

    # Zones fill (hidden initially)
    zone_classical = ax.axvspan(0, 3, alpha=0, color='#E3F2FD')
    zone_danger = ax.axvspan(3, 6, alpha=0, color='#FFF3E0')
    zone_modern = ax.axvspan(6, 10.5, alpha=0, color='#E8F5E9')

    N = len(x_full)
    # Animation: 80 frames total
    # frames 0-29: draw curve up to the classical minimum (~x=3)
    # frames 30-39: pause, show "everyone stopped here"
    # frames 40-59: continue through peak
    # frames 60-79: continue to second descent, add labels
    total_frames = 80

    def init():
        line_test.set_data([], [])
        line_train.set_data([], [])
        return line_test, line_train

    def update(frame):
        if frame <= 29:
            # Phase 1: draw up to classical minimum
            idx = int((frame / 29) * N * 0.30)
            line_test.set_data(x_full[:idx], y_noisy[:idx])
            line_train.set_data(x_full[:idx], y_train[:idx])
        elif frame <= 39:
            # Phase 2: pause at classical minimum, show annotation
            idx = int(N * 0.30)
            line_test.set_data(x_full[:idx], y_noisy[:idx])
            line_train.set_data(x_full[:idx], y_train[:idx])
            if frame == 30:
                stop_line.set_alpha(0.7)
                ax.annotate('传统观点：\n到这里就该停了',
                            xy=(3.0, 0.22), xytext=(1.0, 0.75),
                            fontsize=12, color='#555', ha='center',
                            fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color='#888', lw=1.5),
                            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff',
                                      edgecolor='#ccc'))
                zone_classical.set_alpha(0.15)
        elif frame <= 59:
            # Phase 3: continue through the peak
            progress = (frame - 40) / 19  # 0 to 1
            idx = int(N * (0.30 + progress * 0.35))
            line_test.set_data(x_full[:idx], y_noisy[:idx])
            line_train.set_data(x_full[:idx], y_train[:idx])
            if frame == 50:
                vline.set_alpha(0.8)
                zone_danger.set_alpha(0.12)
                ax.annotate('插值峰值\n（过拟合最严重）',
                            xy=(4.5, dd_curve(4.5) + 0.05),
                            xytext=(6.5, 1.4),
                            fontsize=11, color='#E65100', ha='center',
                            fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color='#FF9800', lw=1.5),
                            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8E1',
                                      edgecolor='#FFB74D'))
        else:
            # Phase 4: second descent
            progress = (frame - 60) / 19  # 0 to 1
            idx = int(N * (0.65 + progress * 0.35))
            idx = min(idx, N)
            line_test.set_data(x_full[:idx], y_noisy[:idx])
            line_train.set_data(x_full[:idx], y_train[:idx])
            if frame == 70:
                zone_modern.set_alpha(0.12)
                ax.annotate('第二次下降！\n现代大模型在这里',
                            xy=(8.5, dd_curve(8.5)),
                            xytext=(8.5, 0.8),
                            fontsize=12, color='#2E7D32', ha='center',
                            fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.8),
                            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9',
                                      edgecolor='#81C784'))

        return line_test, line_train

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=total_frames, interval=100, blit=False)

    out = os.path.join(OUT, 'double_descent.gif')
    anim.save(out, writer=PillowWriter(fps=10), dpi=120)
    plt.close(fig)
    print(f"  ✓ {out}  ({os.path.getsize(out)//1024} KB)")


def make_grokking_gif():
    """Grokking 动画：训练秒满，测试平躺万步后突然起飞"""
    print("→ grokking.gif ...")

    # x axis: training steps (log scale feel, but linear for animation)
    steps = np.arange(0, 15001)

    # Train accuracy: jumps to 100% within first ~100 steps
    train_acc = np.clip(1.0 / (1 + np.exp(-0.1 * (steps - 50))), 0, 1.0) * 100

    # Test accuracy: flat near 1% until ~10000, then sudden jump
    def grok_curve(s):
        base = 1.0  # random guess = 1/97 ≈ 1%
        jump = 99.0 / (1 + np.exp(-0.005 * (s - 10500)))
        return base + jump

    test_acc = grok_curve(steps)
    rng = np.random.default_rng(42)
    test_acc += rng.normal(0, 0.3, size=steps.size)
    test_acc = np.clip(test_acc, 0, 100)
    train_acc += rng.normal(0, 0.15, size=steps.size)
    train_acc = np.clip(train_acc, 0, 100)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor('white')

    line_train, = ax.plot([], [], color='#1E88E5', linewidth=2.5, label='训练准确率')
    line_test, = ax.plot([], [], color='#E53935', linewidth=2.8, label='测试准确率')

    ax.set_xlim(-200, 15500)
    ax.set_ylim(-5, 110)
    ax.set_xlabel('训练步数', fontsize=12)
    ax.set_ylabel('准确率 (%)', fontsize=12)
    ax.set_title('Grokking：模型突然"开窍"',
                 fontsize=15, fontweight='bold', pad=14)
    ax.set_xticks([0, 2500, 5000, 7500, 10000, 12500, 15000])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='center right', fontsize=11, frameon=True)

    # Phase markers
    total_frames = 80
    annotated = set()

    def init():
        line_train.set_data([], [])
        line_test.set_data([], [])
        return line_train, line_test

    def update(frame):
        # Map frame to step index (slow down around the grokking moment)
        if frame <= 10:
            # First 100 steps (fast train phase)
            idx = int(frame * 100 / 10)
        elif frame <= 30:
            # Steps 100 -> 9000 (long boring plateau)
            progress = (frame - 10) / 20
            idx = int(100 + progress * 8900)
        elif frame <= 55:
            # Steps 9000 -> 12000 (the grokking zone, slow it down)
            progress = (frame - 30) / 25
            idx = int(9000 + progress * 3000)
        else:
            # Steps 12000 -> 15000 (aftermath)
            progress = (frame - 55) / 24
            idx = int(12000 + progress * 3000)
            idx = min(idx, len(steps) - 1)

        # Subsample for smooth drawing
        indices = np.linspace(0, idx, min(idx + 1, 600), dtype=int)
        line_train.set_data(steps[indices], train_acc[indices])
        line_test.set_data(steps[indices], test_acc[indices])

        # Annotations at key moments
        if frame == 8 and 'overfit' not in annotated:
            annotated.add('overfit')
            ax.annotate('100 步：训练集已完全记住',
                        xy=(100, 100), xytext=(3000, 95),
                        fontsize=11, color='#1565C0', ha='center',
                        arrowprops=dict(arrowstyle='->', color='#1E88E5', lw=1.3),
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD',
                                  edgecolor='#90CAF9'))

        if frame == 20 and 'plateau' not in annotated:
            annotated.add('plateau')
            ax.annotate('测试准确率 ≈ 1%\n（随机猜 = 1/97）\n模型只是在"背答案"',
                        xy=(5000, 1), xytext=(5000, 45),
                        fontsize=11, color='#C62828', ha='center',
                        arrowprops=dict(arrowstyle='->', color='#E53935', lw=1.3),
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFEBEE',
                                  edgecolor='#EF9A9A'))
            # Add shaded "memorization zone"
            ax.axvspan(100, 9500, alpha=0.06, color='#E53935')
            ax.text(4800, 108, '← 记忆阶段 →', fontsize=10, color='#C62828',
                    ha='center', style='italic')

        if frame == 45 and 'grok' not in annotated:
            annotated.add('grok')
            ax.annotate('突然"开窍"！\n从背答案 → 理解规则',
                        xy=(10500, 50), xytext=(13000, 45),
                        fontsize=12, color='#2E7D32', ha='center',
                        fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1.8),
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9',
                                  edgecolor='#81C784'))
            ax.axvspan(9500, 11500, alpha=0.08, color='#FF9800')
            ax.text(10500, 108, '相变！', fontsize=11, color='#E65100',
                    ha='center', fontweight='bold')

        return line_train, line_test

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=total_frames, interval=120, blit=False)

    out = os.path.join(OUT, 'grokking.gif')
    anim.save(out, writer=PillowWriter(fps=8), dpi=120)
    plt.close(fig)
    print(f"  ✓ {out}  ({os.path.getsize(out)//1024} KB)")


def make_micro_macro():
    """微观平滑 vs 宏观跳变：同一个模型，两种视角"""
    print("→ micro_macro.png ...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor('white')

    # Shared x axis: model scale
    x = np.linspace(8, 12, 300)

    # Left panel: continuous metric (neg log-prob, perplexity-like)
    # Smooth, monotonic improvement
    y_smooth = 4.5 - 0.35 * (x - 8) - 0.01 * (x - 8)**2
    rng = np.random.default_rng(42)
    y_smooth += rng.normal(0, 0.02, x.size)

    ax1.plot(x, y_smooth, color='#1E88E5', linewidth=2.5)
    ax1.fill_between(x, y_smooth, y_smooth.max() + 0.3, alpha=0.06, color='#1E88E5')
    ax1.set_xlabel('模型参数量 (log10)', fontsize=11)
    ax1.set_ylabel('负对数概率（越低越好）', fontsize=11)
    ax1.set_title('显微镜视角：连续指标',
                  fontsize=14, fontweight='bold', pad=12, color='#1565C0')
    ax1.set_xticks([8, 9, 10, 11, 12])
    ax1.set_xticklabels(['100M', '1B', '10B', '100B', '1T'])
    ax1.grid(alpha=0.2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Add annotation
    ax1.annotate('平滑改善\n看不到任何"跳变"',
                xy=(10, y_smooth[150]), xytext=(10, 3.2),
                fontsize=11, color='#1565C0', ha='center',
                arrowprops=dict(arrowstyle='->', color='#1E88E5', lw=1.3),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD',
                          edgecolor='#90CAF9'))

    ax1.text(10.0, 4.7, '"Schaeffer 说：涌现是幻觉"',
             fontsize=10, color='#666', ha='center', style='italic')

    # Right panel: discrete metric (0/1 accuracy)
    def sigmoid(x, x0, k=5):
        return 1.0 / (1.0 + np.exp(-k * (x - x0)))

    y_discrete = sigmoid(x, 10.5) * 92 + rng.normal(0, 1.0, x.size)
    y_discrete = np.clip(y_discrete, 0, 100)

    ax2.plot(x, y_discrete, color='#E53935', linewidth=2.5)
    ax2.fill_between(x, 0, y_discrete, alpha=0.06, color='#E53935')
    ax2.set_xlabel('模型参数量 (log10)', fontsize=11)
    ax2.set_ylabel('精确匹配准确率 (%)', fontsize=11)
    ax2.set_title('望远镜视角：离散指标',
                  fontsize=14, fontweight='bold', pad=12, color='#C62828')
    ax2.set_xticks([8, 9, 10, 11, 12])
    ax2.set_xticklabels(['100M', '1B', '10B', '100B', '1T'])
    ax2.set_ylim(-5, 105)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.grid(alpha=0.2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Mark the "jump"
    ax2.axvline(10.5, color='#FF9800', linestyle=':', linewidth=1.5, alpha=0.6)
    ax2.annotate('锐利跳变！',
                xy=(10.5, 50), xytext=(11.5, 35),
                fontsize=12, color='#C62828', ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#E53935', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE',
                          edgecolor='#EF9A9A'))

    ax2.text(10.0, 102, '"Wei 说：这就是涌现"',
             fontsize=10, color='#666', ha='center', style='italic')

    # Big middle arrow and conclusion
    fig.text(0.5, -0.06,
             '同一个模型，同一组数据，不同的测量方式 → 看到完全不同的图景。\n'
             '物理学的回答：两者都对。微观平滑，宏观跳变。这不是幻觉，是涌现。',
             ha='center', fontsize=12, color='#333',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFF8E1',
                       edgecolor='#e0b030', linewidth=1.2))

    plt.tight_layout()
    out = os.path.join(OUT, 'micro_macro.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ {out}  ({os.path.getsize(out)//1024} KB)")


if __name__ == '__main__':
    make_emergence()
    make_double_descent_gif()
    make_grokking_gif()
    make_micro_macro()
    print("\n✓ All figures done.")
