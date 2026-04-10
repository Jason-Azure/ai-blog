#!/usr/bin/env python3
"""
flower_construction_steps.png
4-panel horizontal strip showing the construction progression:
  1 circle → 2 (Vesica Piscis) → 7 (Seed of Life) → 19 (Flower of Life)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'
plt.rcParams['mathtext.fontset'] = 'cm'

DARK_BLUE  = '#1a237e'
BLUE_LIGHT = '#E8EAF6'
GOLD       = '#FFB300'
WHITE      = '#FFFFFF'
GRAY_LIGHT = '#F5F5F5'

def draw_circle(ax, cx, cy, r, fill=False):
    c = plt.Circle((cx, cy), r, fill=fill,
                    facecolor=BLUE_LIGHT if fill else 'none',
                    edgecolor=DARK_BLUE, lw=1.4, alpha=0.85)
    ax.add_patch(c)

def seed_of_life_centres(r):
    """Return 7 centres: 1 central + 6 surrounding."""
    centres = [(0, 0)]
    for k in range(6):
        angle = k * np.pi / 3
        centres.append((r * np.cos(angle), r * np.sin(angle)))
    return centres

def flower_of_life_centres(r):
    """Return 19 centres: seed + 12 outer ring."""
    centres = seed_of_life_centres(r)
    # second ring: 12 circles
    for k in range(6):
        angle = k * np.pi / 3
        # at distance 2r along each axis
        centres.append((2*r * np.cos(angle), 2*r * np.sin(angle)))
        # between axes
        angle2 = (k + 0.5) * np.pi / 3
        centres.append((np.sqrt(3)*r * np.cos(angle2),
                         np.sqrt(3)*r * np.sin(angle2)))
    return centres

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
fig.patch.set_facecolor(WHITE)

r = 1.0
panels = [
    ('一', [(0, 0)]),
    ('二', [(- r/2, 0), (r/2, 0)]),
    ('七 · 生命种子', seed_of_life_centres(r)),
    ('十九 · 生命之花', flower_of_life_centres(r)),
]

for ax, (title, centres) in zip(axes, panels):
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(WHITE)

    for cx, cy in centres:
        draw_circle(ax, cx, cy, r, fill=True)

    # auto-range
    if len(centres) <= 2:
        lim = 2.0
    elif len(centres) <= 7:
        lim = 2.8
    else:
        lim = 3.8
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.text(0, -lim + 0.15, title, ha='center', va='bottom',
            fontsize=14, color=DARK_BLUE, weight='bold')

# arrows between panels
for i in range(3):
    x_right = axes[i].get_position().x1
    x_left  = axes[i+1].get_position().x0
    x_mid   = (x_right + x_left) / 2
    y_mid   = 0.50
    fig.text(x_mid, y_mid, '→', fontsize=24, ha='center', va='center',
             color=GOLD, weight='bold')

plt.subplots_adjust(wspace=0.05)
plt.savefig('/home/azureuser/ai-blog/static/images/two-circles-1/flower_construction_steps.png',
            dpi=200, bbox_inches='tight', facecolor=WHITE)
plt.close()
print('✓ flower_construction_steps.png')
