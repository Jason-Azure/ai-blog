"""
Generate animated GIFs for 《看见数学》第八篇《圆与波》
1. three-changes.gif   - Three types of change (linear, exponential, periodic)
2. circle-to-sine.gif  - Circle rotation unfolds into sine wave
3. fourier-superposition.gif - Sine waves superposition approaching square wave
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# ============================================================
# GIF 1: Three Types of Change (linear, exponential, periodic)
# ============================================================
def create_three_changes_gif(output_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#ffffff')
    ax.set_xlim(0, 10)
    ax.set_ylim(-2, 8)
    ax.set_title('Three Types of Change', fontsize=15, fontweight='bold', pad=12)
    ax.axhline(y=0, color='#e0e0e0', linewidth=0.5)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    for spine in ax.spines.values():
        spine.set_color('#ccc')
    ax.tick_params(labelsize=9)

    # Use 500 points for ultra-smooth curves
    t_full = np.linspace(0, 10, 500)
    linear_full = 0.5 * t_full
    exponential_full = np.clip(0.3 * np.exp(0.4 * t_full) - 0.3, -2, 8)
    periodic_full = 2 * np.sin(t_full * 1.5) + 3

    colors = ['#1565C0', '#C62828', '#2E7D32']
    labels_list = ['Linear: y = 0.5t', 'Exponential: y = 0.3e^(0.4t)', 'Periodic: y = 2sin(1.5t)+3']

    # Draw faint full curves first (the "ghost" target)
    ax.plot(t_full, linear_full, color=colors[0], linewidth=1.5, alpha=0.15, zorder=1)
    ax.plot(t_full, exponential_full, color=colors[1], linewidth=1.5, alpha=0.15, zorder=1)
    ax.plot(t_full, periodic_full, color=colors[2], linewidth=1.5, alpha=0.15, zorder=1)

    # Animated thick lines
    line1, = ax.plot([], [], linewidth=3, color=colors[0], label=labels_list[0], zorder=3)
    line2, = ax.plot([], [], linewidth=3, color=colors[1], label=labels_list[1], zorder=3)
    line3, = ax.plot([], [], linewidth=3, color=colors[2], label=labels_list[2], zorder=3)

    # Animated dots traveling along each curve
    dot1, = ax.plot([], [], 'o', color=colors[0], markersize=9, zorder=5)
    dot2, = ax.plot([], [], 'o', color=colors[1], markersize=9, zorder=5)
    dot3, = ax.plot([], [], 'o', color=colors[2], markersize=9, zorder=5)

    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    n_frames = 120

    def animate(frame):
        n = max(2, int((frame + 1) / n_frames * len(t_full)))
        t = t_full[:n]

        line1.set_data(t, linear_full[:n])
        line2.set_data(t, exponential_full[:n])
        line3.set_data(t, periodic_full[:n])

        dot1.set_data([t[-1]], [linear_full[n-1]])
        dot2.set_data([t[-1]], [exponential_full[n-1]])
        dot3.set_data([t[-1]], [periodic_full[n-1]])

        return line1, line2, line3, dot1, dot2, dot3

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=67, blit=True)
    plt.tight_layout()
    anim.save(output_path, writer=PillowWriter(fps=15), dpi=80)
    plt.close()
    print(f"Created: {output_path}")


# ============================================================
# GIF 2: Circle rotation unfolds into sine wave
# ============================================================
def create_circle_sine_gif(output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5),
                                    gridspec_kw={'width_ratios': [1, 2.2]})
    fig.patch.set_facecolor('#ffffff')

    # Left: Circle
    ax1.set_xlim(-1.8, 1.8)
    ax1.set_ylim(-1.8, 1.8)
    ax1.set_aspect('equal')
    ax1.set_title('Rotation on Circle', fontsize=14, fontweight='bold', pad=10)
    ax1.axhline(y=0, color='#e0e0e0', linewidth=0.5)
    ax1.axvline(x=0, color='#e0e0e0', linewidth=0.5)

    # Draw circle prominently
    theta_circle = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(np.cos(theta_circle), np.sin(theta_circle),
             color='#90A4AE', linewidth=2.5, zorder=1)

    # Right: Sine wave
    ax2.set_xlim(0, 4 * np.pi)
    ax2.set_ylim(-1.8, 1.8)
    ax2.set_title('Unfold to Sine Wave: y = sin(t)', fontsize=14, fontweight='bold', pad=10)
    ax2.axhline(y=0, color='#e0e0e0', linewidth=0.5)
    ax2.set_xlabel('Angle t', fontsize=12)
    ax2.set_ylabel('y = sin(t)', fontsize=12)

    # Faint full sine wave target
    t_full = np.linspace(0, 4 * np.pi, 600)
    ax2.plot(t_full, np.sin(t_full), color='#BBDEFB', linewidth=2, zorder=0)

    # Animated elements on circle
    point, = ax1.plot([], [], 'o', color='#C62828', markersize=12, zorder=6)
    radius_line, = ax1.plot([], [], '-', color='#C62828', linewidth=2.5, zorder=4)
    h_line, = ax1.plot([], [], '--', color='#1565C0', linewidth=1.5, alpha=0.7, zorder=3)
    y_marker, = ax1.plot([], [], 's', color='#1565C0', markersize=7, zorder=5)

    # Animated elements on sine plot
    sine_line, = ax2.plot([], [], '-', color='#1565C0', linewidth=3, zorder=3)
    current_point, = ax2.plot([], [], 'o', color='#C62828', markersize=10, zorder=5)
    connect_line, = ax2.plot([], [], '--', color='#C62828', linewidth=1.5, alpha=0.5, zorder=2)

    for ax in [ax1, ax2]:
        ax.tick_params(labelsize=9)
        for spine in ax.spines.values():
            spine.set_color('#ccc')

    n_frames = 200
    angles = np.linspace(0, 4 * np.pi, n_frames)

    def animate(i):
        theta = angles[i]
        x = np.cos(theta)
        y = np.sin(theta)

        # Point on circle
        point.set_data([x], [y])
        radius_line.set_data([0, x], [0, y])

        # y-projection line from point to axis
        h_line.set_data([x, 1.6], [y, y])
        y_marker.set_data([0], [y])

        # Sine wave trail (use dense interpolation for smoothness)
        t_trail = np.linspace(0, theta, max(2, i * 4))
        sine_line.set_data(t_trail, np.sin(t_trail))
        current_point.set_data([theta], [y])

        # Connection line between circle and sine wave
        connect_line.set_data([1.6, theta], [y, y])

        return point, radius_line, h_line, y_marker, sine_line, current_point, connect_line

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=83, blit=True)
    plt.tight_layout()
    anim.save(output_path, writer=PillowWriter(fps=12), dpi=80)
    plt.close()
    print(f"Created: {output_path}")


# ============================================================
# GIF 3: Fourier Superposition (sine waves approaching square wave)
# ============================================================
def create_fourier_gif(output_path):
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    fig.patch.set_facecolor('#ffffff')
    fig.suptitle('Fourier: Sine Waves Build a Square Wave',
                 fontsize=16, fontweight='bold', y=0.98)

    t = np.linspace(0, 4 * np.pi, 600)

    # Color palette with strong contrast
    colors = ['#1565C0', '#2E7D32', '#E65100', '#C62828']
    labels = ['sin(x)', '+ sin(3x)/3', '+ sin(5x)/5', 'Sum (5 terms)']

    waves = [
        np.sin(t),
        np.sin(t) + np.sin(3 * t) / 3,
        np.sin(t) + np.sin(3 * t) / 3 + np.sin(5 * t) / 5,
        np.sin(t) + np.sin(3 * t) / 3 + np.sin(5 * t) / 5
        + np.sin(7 * t) / 7 + np.sin(9 * t) / 9,
    ]

    # Build a proper square wave target for the last panel
    square_wave = np.sign(np.sin(t)) * (4 / np.pi)  # amplitude-matched

    lines = []
    for idx, ax in enumerate(axes):
        ax.set_ylim(-2.0, 2.0)
        ax.axhline(y=0, color='#e0e0e0', linewidth=0.5)

        # Draw faint target curve for each panel
        ax.plot(t, waves[idx], color=colors[idx], linewidth=1.5, alpha=0.12, zorder=0)

        # For the last panel, also draw the faint square wave target
        if idx == 3:
            ax.plot(t, square_wave, color='#888888', linewidth=1.5,
                    linestyle='--', alpha=0.35, zorder=0, label='Target: square wave')

        line, = ax.plot([], [], linewidth=3, color=colors[idx], zorder=3)
        lines.append(line)

        ax.set_ylabel(labels[idx], fontsize=11, color=colors[idx], fontweight='bold')
        for spine in ax.spines.values():
            spine.set_color('#ddd')
        ax.tick_params(labelsize=8)

    # Add legend to last panel
    axes[3].legend(loc='upper right', fontsize=9, framealpha=0.8)
    axes[3].set_xlabel('x', fontsize=12)

    n_frames = 120

    # Animated dots at the leading edge
    dots = []
    for idx, ax in enumerate(axes):
        dot, = ax.plot([], [], 'o', color=colors[idx], markersize=7, zorder=5)
        dots.append(dot)

    def animate(frame):
        progress = (frame + 1) / n_frames
        n_pts = max(2, int(progress * len(t)))

        for i, (line, wave, dot) in enumerate(zip(lines, waves, dots)):
            line.set_data(t[:n_pts], wave[:n_pts])
            dot.set_data([t[n_pts - 1]], [wave[n_pts - 1]])

        return lines + dots

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=67, blit=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    anim.save(output_path, writer=PillowWriter(fps=15), dpi=80)
    plt.close()
    print(f"Created: {output_path}")


# ============================================================
# Run all
# ============================================================
if __name__ == '__main__':
    output_dir = '/home/azureuser/ai-blog/content/posts/see-math-8-waves/'

    print("Generating GIFs...")
    create_three_changes_gif(f'{output_dir}three-changes.gif')
    create_circle_sine_gif(f'{output_dir}circle-to-sine.gif')
    create_fourier_gif(f'{output_dir}fourier-superposition.gif')
    print("All 3 GIFs generated!")
