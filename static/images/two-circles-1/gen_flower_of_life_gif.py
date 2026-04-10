#!/usr/bin/env python3
"""
Flower of Life construction animation GIF.
800x800, ~12 seconds at 8 FPS (96 frames), loop forever.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.font_manager import FontProperties
from PIL import Image
import io
import math

# --- Config ---
OUTPUT = "/home/azureuser/ai-blog/static/images/two-circles-1/flower_construction_steps.gif"
FIG_SIZE = (8, 8)
DPI = 100
FPS = 8
TOTAL_FRAMES = 96

# Colors
BG_COLOR = '#FFFFFF'
NAVY = '#1a237e'
GOLDEN = '#FFB300'
LIGHT_BLUE = '#E3F2FD'
FILL_ALPHA = 0.08

# Font
FONT = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', size=22)
FONT_SMALL = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', size=16)

# --- Circle positions ---
r = 1.0

# Circle 0: center
# Circles 1-6: Ring 1 (Seed of Life inner ring) at distance r
# Circles 7-12: Ring 2a at distance 2r
# Circles 13-18: Ring 2b at distance r*sqrt(3)

def ring1_positions():
    """6 circles at distance r from origin, 60° apart starting at 0°"""
    positions = []
    for i in range(6):
        angle = math.radians(i * 60)
        positions.append((r * math.cos(angle), r * math.sin(angle)))
    return positions

def ring2a_positions():
    """6 circles at distance 2r, angles 0°,60°,...,300°"""
    positions = []
    for i in range(6):
        angle = math.radians(i * 60)
        positions.append((2*r * math.cos(angle), 2*r * math.sin(angle)))
    return positions

def ring2b_positions():
    """6 circles at distance r*sqrt(3), angles 30°,90°,...,330°"""
    positions = []
    for i in range(6):
        angle = math.radians(30 + i * 60)
        d = r * math.sqrt(3)
        positions.append((d * math.cos(angle), d * math.sin(angle)))
    return positions

# All 19 circle centers in order
all_centers = [(0, 0)] + ring1_positions() + ring2a_positions() + ring2b_positions()

# --- Animation schedule ---
# Phase 1: Title (frames 0-5)
# Phase 2: Center circle (frames 6-15)  - circle 0
# Phase 3: Circle 2 + vesica piscis glow (frames 16-25) - circle 1
# Phase 4: Circles 3-7 one by one (frames 26-55) ~6 frames each - circles 2-6
# Phase 5: Circles 8-19 faster (frames 56-85) ~2.5 frames each - circles 7-18
# Phase 6: Final hold (frames 86-95)

def get_circle_schedule():
    """Returns list of (circle_index, appear_frame) tuples."""
    schedule = []
    # Circle 0 (center): appears at frame 6
    schedule.append((0, 6))
    # Circle 1 (first ring, right): appears at frame 16
    schedule.append((1, 16))
    # Circles 2-6 (rest of seed): frames 26-55, ~6 frames apart
    for i, idx in enumerate(range(2, 7)):
        schedule.append((idx, 26 + i * 6))
    # Circles 7-18 (ring 2): frames 56-85, ~2.5 frames apart
    for i, idx in enumerate(range(7, 19)):
        frame = 56 + int(i * 2.5)
        schedule.append((idx, frame))
    return schedule

circle_schedule = get_circle_schedule()

# Label events
SEED_LABEL_FRAME = 53   # After circle 6 settles (appeared at 50, +3 frames)
FLOWER_LABEL_FRAME = 83 # After circle 18 settles

def get_circle_state(circle_idx, frame):
    """
    Returns state of a circle at given frame:
    None = not yet visible
    'golden' = just appeared (flash)
    'navy' = settled
    """
    for idx, appear_frame in circle_schedule:
        if idx == circle_idx:
            if frame < appear_frame:
                return None
            elif frame < appear_frame + 2:
                return 'golden'
            else:
                return 'navy'
    return None

def show_vesica_glow(frame):
    """Show vesica piscis highlight between frames 18-24"""
    return 18 <= frame <= 24

def render_frame(frame):
    """Render a single frame and return as PIL Image."""
    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_aspect('equal')
    ax.axis('off')

    # --- Phase 1: Title only ---
    if frame <= 5:
        # Fade in title
        alpha = min(1.0, (frame + 1) / 4)
        ax.text(0, 0.3, '生命之花的诞生', fontproperties=FONT,
                ha='center', va='center', color=NAVY, alpha=alpha, fontsize=28)
        ax.text(0, -0.3, 'Birth of the Flower of Life', fontproperties=FONT_SMALL,
                ha='center', va='center', color='#5C6BC0', alpha=alpha, fontsize=16)

    else:
        # --- Draw circles ---
        # Vesica Piscis glow (between circles 0 and 1)
        if show_vesica_glow(frame):
            glow_alpha = 0.15
            if frame <= 20:
                glow_alpha = 0.05 + 0.1 * (frame - 18) / 2
            elif frame >= 23:
                glow_alpha = 0.15 - 0.1 * (frame - 23) / 2
            # Draw the overlapping region approximation with a subtle glow
            theta = np.linspace(0, 2*np.pi, 100)
            # Highlight both circles with a glow
            for cx, cy in [all_centers[0], all_centers[1]]:
                glow_circle = Circle((cx, cy), r, facecolor='#FFD54F',
                                    edgecolor='none', alpha=glow_alpha, zorder=1)
                ax.add_patch(glow_circle)

        for circle_idx in range(19):
            state = get_circle_state(circle_idx, frame)
            if state is None:
                continue

            cx, cy = all_centers[circle_idx]

            if state == 'golden':
                # Golden flash - slightly thicker, golden color
                edge_color = GOLDEN
                lw = 2.0
                fill_alpha = 0.06
                fill_color = '#FFF8E1'
                zorder = 10
            else:
                # Settled - navy, with subtle light blue fill
                edge_color = NAVY
                lw = 1.2
                fill_alpha = FILL_ALPHA
                fill_color = LIGHT_BLUE
                zorder = 5

            # Fill circle
            fill_patch = Circle((cx, cy), r, facecolor=fill_color,
                               edgecolor='none', alpha=fill_alpha, zorder=zorder-1)
            ax.add_patch(fill_patch)

            # Edge circle
            edge_patch = Circle((cx, cy), r, facecolor='none',
                               edgecolor=edge_color, linewidth=lw, zorder=zorder)
            ax.add_patch(edge_patch)

        # --- Labels ---
        if SEED_LABEL_FRAME <= frame < FLOWER_LABEL_FRAME:
            # Seed of Life label
            label_alpha = min(1.0, (frame - SEED_LABEL_FRAME + 1) / 3)
            ax.text(0, -2.6, '生命种子  Seed of Life', fontproperties=FONT,
                    ha='center', va='center', color=NAVY, alpha=label_alpha,
                    fontsize=20,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor=NAVY, alpha=label_alpha * 0.8, linewidth=1))

        if frame >= FLOWER_LABEL_FRAME:
            # Flower of Life label
            label_alpha = min(1.0, (frame - FLOWER_LABEL_FRAME + 1) / 3)
            ax.text(0, -2.6, '生命之花  Flower of Life', fontproperties=FONT,
                    ha='center', va='center', color=NAVY, alpha=label_alpha,
                    fontsize=20,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor=GOLDEN, alpha=label_alpha * 0.8, linewidth=1.5))

    # Convert to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, facecolor=BG_COLOR)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGBA')
    return img


def main():
    print(f"Rendering {TOTAL_FRAMES} frames...")
    raw_frames = []
    for i in range(TOTAL_FRAMES):
        if i % 10 == 0:
            print(f"  Frame {i}/{TOTAL_FRAMES}...")
        img = render_frame(i)
        # Convert to RGB for GIF (no transparency in GIF)
        img_rgb = Image.new('RGB', img.size, (255, 255, 255))
        img_rgb.paste(img, mask=img.split()[3])
        raw_frames.append(img_rgb)

    # All frames are 800x800 (no bbox_inches='tight'), so no padding needed
    print(f"  Frame size: {raw_frames[0].width}x{raw_frames[0].height}")

    # Deduplicate consecutive identical frames and accumulate durations
    base_duration = int(1000 / FPS)  # 125ms per frame
    unique_frames = [raw_frames[0]]
    durations = [base_duration]

    for i in range(1, len(raw_frames)):
        if list(raw_frames[i].getdata()) == list(raw_frames[i-1].getdata()):
            # Same as previous frame - extend its duration
            durations[-1] += base_duration
        else:
            unique_frames.append(raw_frames[i])
            durations.append(base_duration)

    total_duration = sum(durations) / 1000
    print(f"  Unique frames: {len(unique_frames)} (from {TOTAL_FRAMES} total)")
    print(f"  Total duration: {total_duration:.1f}s")

    # Save GIF with per-frame durations
    print(f"  Saving GIF...")
    unique_frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=unique_frames[1:],
        duration=durations,
        loop=0,
        optimize=False
    )

    import os
    size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
    print(f"  Done! {OUTPUT}")
    print(f"  File size: {size_mb:.2f} MB")
    if size_mb > 3:
        print("  WARNING: File is larger than 3MB!")
    else:
        print("  OK: File is under 3MB.")


if __name__ == '__main__':
    main()
