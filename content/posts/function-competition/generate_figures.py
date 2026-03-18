#!/usr/bin/env python3
"""
Generate all figures for the "函数的竞赛" blog post.
7 figures total: 3 PNG + 4 GIF

Usage:
    source ~/ai-lab-venv/bin/activate
    cd ~/ai-blog/content/posts/function-competition/
    python generate_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.gridspec import GridSpec
import imageio
import io
import os
from scipy.interpolate import CubicSpline
from math import factorial

# ============================================================
# Global settings
# ============================================================
plt.rcParams['font.family'] = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    size = os.path.getsize(path)
    print(f"  ✓ {name} ({size/1024:.0f} KB)")

def save_gif(frames, name, fps=3):
    path = os.path.join(OUTPUT_DIR, name)
    imageio.mimsave(path, frames, fps=fps, loop=0)
    size = os.path.getsize(path)
    print(f"  ✓ {name} ({size/1024:.0f} KB)")

# ============================================================
# Figure 1: Function Black Box (static PNG)
# ============================================================
def generate_function_blackbox():
    print("Generating function-blackbox.png...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    fig.set_facecolor('#ffffff')
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.5, 3.5)
    ax.axis('off')

    # Title
    ax.text(5, 3.2, 'Everything is a Function', fontsize=18, fontweight='bold',
            ha='center', va='center', color='#333333')

    # Three rows of examples
    examples = [
        {'input': 'Image (x, y)', 'output': 'RGB color', 'label': 'Computer Vision', 'color': '#FF9800', 'y': 2.0},
        {'input': 'Context words', 'output': 'Next word', 'label': 'Language Model', 'color': '#2196F3', 'y': 0.8},
        {'input': 'T, humidity, ...', 'output': 'Weather', 'label': 'Forecasting', 'color': '#4CAF50', 'y': -0.4},
    ]

    for ex in examples:
        y = ex['y']
        c = ex['color']

        # Input box
        inp_box = FancyBboxPatch((0.5, y - 0.35), 2.8, 0.7, boxstyle="round,pad=0.1",
                                  facecolor=c, alpha=0.15, edgecolor=c, linewidth=1.5)
        ax.add_patch(inp_box)
        ax.text(1.9, y, ex['input'], fontsize=10, ha='center', va='center', color='#333')

        # Arrow 1
        ax.annotate('', xy=(4.0, y), xytext=(3.5, y),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))

        # Function box (black box)
        func_box = FancyBboxPatch((4.0, y - 0.4), 2.5, 0.8, boxstyle="round,pad=0.1",
                                   facecolor='#333333', edgecolor='#333333', linewidth=1.5)
        ax.add_patch(func_box)
        ax.text(5.25, y + 0.05, 'f( )', fontsize=14, ha='center', va='center',
                color='white', fontweight='bold', family='monospace')
        ax.text(5.25, y - 0.25, ex['label'], fontsize=7, ha='center', va='center',
                color='#aaaaaa')

        # Arrow 2
        ax.annotate('', xy=(7.2, y), xytext=(6.7, y),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))

        # Output box
        out_box = FancyBboxPatch((7.2, y - 0.35), 2.8, 0.7, boxstyle="round,pad=0.1",
                                  facecolor=c, alpha=0.15, edgecolor=c, linewidth=1.5)
        ax.add_patch(out_box)
        ax.text(8.6, y, ex['output'], fontsize=10, ha='center', va='center', color='#333')

    # Bottom question
    ax.text(5, -1.2, 'How do we find f ?  What methods has humanity tried?',
            fontsize=12, ha='center', va='center', color='#666', style='italic')

    save_fig(fig, 'function-blackbox.png')


# ============================================================
# Figure 2: Taylor Approximation (animated GIF)
# ============================================================
def generate_taylor_gif():
    print("Generating taylor-approximation.gif...")
    x = np.linspace(-2 * np.pi, 2 * np.pi, 500)
    y_true = np.sin(x)

    frames = []
    orders = [1, 3, 5, 7, 9, 11, 13, 15]

    for n in orders:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.set_facecolor('#ffffff')

        # True function
        ax.plot(x, y_true, 'k-', linewidth=2.5, label='sin(x)', alpha=0.7)

        # Taylor approximation
        y_taylor = np.zeros_like(x)
        for k in range(n + 1):
            if k % 2 == 1:  # sin has only odd terms
                sign = (-1) ** ((k - 1) // 2)
                y_taylor += sign * x**k / factorial(k)

        # Clip for display
        y_display = np.clip(y_taylor, -3, 3)
        ax.plot(x, y_display, '-', linewidth=2, color='#2196F3',
                label=f'Taylor (order {n})')

        # Highlight convergence region
        ax.axvspan(-np.pi, np.pi, alpha=0.05, color='#4CAF50')
        ax.text(0, 2.6, 'convergence zone', fontsize=9, ha='center',
                color='#4CAF50', alpha=0.7)

        ax.set_xlim(-2 * np.pi, 2 * np.pi)
        ax.set_ylim(-3, 3)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'Taylor Series Approximation of sin(x)  —  Order {n}',
                     fontsize=14, fontweight='bold', color='#333')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)

        # Render to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#ffffff')
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)

    # Duplicate last frame for pause
    frames.extend([frames[-1]] * 4)

    save_gif(frames, 'taylor-approximation.gif', fps=2)


# ============================================================
# Figure 3: Fourier + Gibbs Phenomenon (animated GIF)
# ============================================================
def generate_fourier_gif():
    print("Generating fourier-gibbs.gif...")
    x = np.linspace(-np.pi, 3 * np.pi, 1000)

    # Square wave
    y_square = np.sign(np.sin(x))

    frames = []
    n_terms_list = [1, 2, 3, 5, 7, 10, 15, 25, 50]

    for n_terms in n_terms_list:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.set_facecolor('#ffffff')

        # True square wave
        ax.plot(x, y_square, 'k--', linewidth=1.5, alpha=0.4, label='Square wave')

        # Fourier approximation (only odd harmonics for square wave)
        y_fourier = np.zeros_like(x)
        for k in range(1, n_terms + 1):
            n_harm = 2 * k - 1  # odd harmonics: 1, 3, 5, ...
            y_fourier += (4 / np.pi) * np.sin(n_harm * x) / n_harm

        ax.plot(x, y_fourier, '-', linewidth=2, color='#4CAF50',
                label=f'Fourier ({n_terms} terms)')

        # Mark Gibbs phenomenon
        if n_terms >= 5:
            overshoot = np.max(y_fourier)
            ax.annotate(f'Gibbs! ({overshoot:.2f})', xy=(0.02, overshoot),
                       xytext=(1.5, 1.4), fontsize=9, color='#E91E63',
                       arrowprops=dict(arrowstyle='->', color='#E91E63', lw=1.2))

        ax.set_xlim(-np.pi, 3 * np.pi)
        ax.set_ylim(-1.8, 1.8)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'Fourier Series Approximation  —  {n_terms} Terms',
                     fontsize=14, fontweight='bold', color='#333')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#ffffff')
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)

    frames.extend([frames[-1]] * 4)
    save_gif(frames, 'fourier-gibbs.gif', fps=2)


# ============================================================
# Figure 4: Runge Phenomenon vs Spline (animated GIF)
# ============================================================
def generate_runge_gif():
    print("Generating runge-vs-spline.gif...")
    # Runge function
    def runge(x):
        return 1 / (1 + 25 * x**2)

    x_fine = np.linspace(-1, 1, 500)
    y_true = runge(x_fine)

    frames = []
    n_points_list = [3, 5, 7, 9, 11, 13, 15, 17, 21]

    for n in n_points_list:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        fig.set_facecolor('#ffffff')
        fig.suptitle(f'Runge Phenomenon  —  {n} Points', fontsize=14,
                     fontweight='bold', color='#333', y=0.98)

        # Equidistant points
        x_pts = np.linspace(-1, 1, n)
        y_pts = runge(x_pts)

        for idx, (ax, title) in enumerate(zip(axes, ['Polynomial Fit', 'Cubic Spline'])):
            # True function
            ax.plot(x_fine, y_true, 'k-', linewidth=2, alpha=0.5, label='f(x) = 1/(1+25x²)')

            if idx == 0:
                # Polynomial interpolation
                coeffs = np.polyfit(x_pts, y_pts, n - 1)
                y_interp = np.polyval(coeffs, x_fine)
                y_display = np.clip(y_interp, -2, 3)
                ax.plot(x_fine, y_display, '-', linewidth=2, color='#E91E63',
                        label=f'Polynomial (deg {n-1})')
            else:
                # Cubic spline
                cs = CubicSpline(x_pts, y_pts)
                y_spline = cs(x_fine)
                ax.plot(x_fine, y_spline, '-', linewidth=2, color='#4CAF50',
                        label='Cubic Spline')

            ax.plot(x_pts, y_pts, 'ko', markersize=5, zorder=5)
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.5, 2.5)
            ax.set_title(title, fontsize=12, color='#555')
            ax.legend(fontsize=9, loc='upper left')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#ffffff')
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)

    frames.extend([frames[-1]] * 4)
    save_gif(frames, 'runge-vs-spline.gif', fps=2)


# ============================================================
# Figure 5: Kernel SVM (static PNG)
# ============================================================
def generate_kernel_svm():
    print("Generating kernel-svm.png...")
    fig = plt.figure(figsize=(12, 4.5))
    fig.set_facecolor('#ffffff')

    # Panel 1: 2D data (not linearly separable)
    ax1 = fig.add_subplot(131)
    np.random.seed(42)
    n_inner = 30
    n_outer = 40

    # Inner circle (class 0)
    r_inner = np.random.uniform(0, 0.7, n_inner)
    theta_inner = np.random.uniform(0, 2*np.pi, n_inner)
    x_inner = r_inner * np.cos(theta_inner)
    y_inner = r_inner * np.sin(theta_inner)

    # Outer ring (class 1)
    r_outer = np.random.uniform(1.2, 2.0, n_outer)
    theta_outer = np.random.uniform(0, 2*np.pi, n_outer)
    x_outer = r_outer * np.cos(theta_outer)
    y_outer = r_outer * np.sin(theta_outer)

    ax1.scatter(x_inner, y_inner, c='#2196F3', s=30, zorder=3, label='Class A')
    ax1.scatter(x_outer, y_outer, c='#E91E63', s=30, zorder=3, label='Class B')
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_title('2D: Not Separable', fontsize=12, fontweight='bold', color='#333')
    ax1.set_xlabel('x₁')
    ax1.set_ylabel('x₂')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.2)
    ax1.set_aspect('equal')

    # Panel 2: Arrow showing kernel mapping
    ax2 = fig.add_subplot(132)
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.text(5, 7, 'Kernel Trick', fontsize=16, fontweight='bold', ha='center',
             va='center', color='#9C27B0')
    ax2.text(5, 5.5, 'φ: (x₁, x₂) → (x₁, x₂, x₁²+x₂²)', fontsize=10,
             ha='center', va='center', color='#666', family='monospace')
    ax2.annotate('', xy=(8, 4), xytext=(2, 4),
                arrowprops=dict(arrowstyle='->', color='#9C27B0', lw=3))
    ax2.text(5, 3, 'Map to higher\ndimension', fontsize=10, ha='center',
             va='center', color='#888')

    # Panel 3: 3D data (separable with hyperplane)
    ax3 = fig.add_subplot(133, projection='3d')
    z_inner = x_inner**2 + y_inner**2
    z_outer = x_outer**2 + y_outer**2

    ax3.scatter(x_inner, y_inner, z_inner, c='#2196F3', s=20, label='Class A')
    ax3.scatter(x_outer, y_outer, z_outer, c='#E91E63', s=20, label='Class B')

    # Separating plane
    xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 20), np.linspace(-2.5, 2.5, 20))
    zz = np.ones_like(xx) * 1.0  # z = 1.0 separating plane
    ax3.plot_surface(xx, yy, zz, alpha=0.15, color='#4CAF50')

    ax3.set_title('3D: Linearly Separable!', fontsize=12, fontweight='bold', color='#333')
    ax3.set_xlabel('x₁', fontsize=9)
    ax3.set_ylabel('x₂', fontsize=9)
    ax3.set_zlabel('x₁²+x₂²', fontsize=9)
    ax3.legend(fontsize=8, loc='upper left')
    ax3.view_init(elev=25, azim=45)

    plt.tight_layout()
    save_fig(fig, 'kernel-svm.png')


# ============================================================
# Figure 6: Neural Network Learning (animated GIF)
# ============================================================
def generate_nn_learning_gif():
    print("Generating nn-learning.gif...")

    # Target function: a complex multi-modal function
    def target_fn(x):
        return np.sin(2 * x) + 0.5 * np.sin(5 * x) + 0.3 * np.cos(8 * x)

    x = np.linspace(-np.pi, np.pi, 300)
    y_target = target_fn(x)

    frames = []

    # Simulate progressive learning: interpolate from flat to target
    steps = [0, 50, 100, 200, 500, 1000, 2000, 5000]
    max_step = max(steps)

    for step in steps:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.set_facecolor('#ffffff')

        # Target
        ax.plot(x, y_target, 'k--', linewidth=1.5, alpha=0.5, label='Target f(x)')

        if step == 0:
            # Initial random-ish prediction
            np.random.seed(42)
            y_pred = np.zeros_like(x) + 0.1 * np.random.randn(len(x))
        else:
            # Simulate learning: blend of flat and target with increasing fidelity
            progress = min(1.0, step / 2000)

            # Low-frequency first (spectral bias)
            y_pred = np.zeros_like(x)
            # Gradually add frequency components
            freqs = [(2, 1.0), (5, 0.5), (8, 0.3)]
            for freq, amp in freqs:
                freq_progress = min(1.0, progress * (3.0 / (freq / 2)))
                if freq == 2 or freq == 5:
                    y_pred += freq_progress * amp * np.sin(freq * x)
                else:
                    y_pred += freq_progress * amp * np.cos(freq * x)

            # Add small noise at early stages
            noise_level = 0.1 * max(0, 1 - progress * 2)
            np.random.seed(step)
            y_pred += noise_level * np.random.randn(len(x))

        y_pred_clipped = np.clip(y_pred, -3, 3)
        ax.plot(x, y_pred_clipped, '-', linewidth=2.5, color='#E91E63',
                label=f'Neural Network')
        ax.fill_between(x, y_target, y_pred_clipped, alpha=0.1, color='#E91E63')

        # Error metric
        mse = np.mean((y_target - y_pred)**2)
        ax.text(0.02, 0.95, f'Step {step}  |  MSE = {mse:.4f}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        if step >= 200 and step < 2000:
            ax.text(0.02, 0.82, 'Low frequencies learned first!',
                    transform=ax.transAxes, fontsize=9, color='#2196F3',
                    style='italic')
        elif step >= 2000:
            ax.text(0.02, 0.82, 'High frequencies gradually refined',
                    transform=ax.transAxes, fontsize=9, color='#4CAF50',
                    style='italic')

        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title('Neural Network Learning Process  —  Spectral Bias',
                     fontsize=14, fontweight='bold', color='#333')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#ffffff')
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig)

    frames.extend([frames[-1]] * 4)
    save_gif(frames, 'nn-learning.gif', fps=2)


# ============================================================
# Figure 7: Radar Chart Comparison (static PNG)
# ============================================================
def generate_method_radar():
    print("Generating method-radar.png...")

    categories = ['Global\nAccuracy', 'High-Dim\nScalability', 'Auto\nLearning',
                  'Generation\nAbility', 'Compute\nEfficiency', 'Interpretability']
    N = len(categories)

    methods = {
        'Taylor':       [2, 1, 1, 1, 5, 5],
        'Fourier':      [4, 2, 1, 1, 4, 4],
        'Polynomial':   [3, 1, 1, 1, 4, 4],
        'Spline':       [4, 2, 1, 1, 4, 3],
        'Kernel/SVM':   [4, 3, 3, 1, 2, 4],
        'Neural Net':   [5, 5, 5, 5, 2, 1],
    }

    colors = ['#2196F3', '#4CAF50', '#9C27B0', '#607D8B', '#FF9800', '#E91E63']

    # Angles for radar chart
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.set_facecolor('#ffffff')

    for idx, (method, values) in enumerate(methods.items()):
        vals = values + values[:1]
        ax.plot(angles, vals, 'o-', linewidth=2, label=method, color=colors[idx],
                markersize=5, alpha=0.8)
        ax.fill(angles, vals, alpha=0.06, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=8, color='#999')
    ax.set_title('Function Approximation Methods  —  Ultimate Comparison',
                 fontsize=14, fontweight='bold', color='#333', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
    ax.grid(True, alpha=0.3)

    save_fig(fig, 'method-radar.png')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("=" * 50)
    print("Generating figures for 函数的竞赛")
    print("=" * 50)

    generate_function_blackbox()
    generate_taylor_gif()
    generate_fourier_gif()
    generate_runge_gif()
    generate_kernel_svm()
    generate_nn_learning_gif()
    generate_method_radar()

    print("\n" + "=" * 50)
    print("All figures generated successfully!")
    print("=" * 50)
