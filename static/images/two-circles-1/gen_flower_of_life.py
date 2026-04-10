#!/usr/bin/env python3
"""
flower_of_life_full.png
Complete Flower of Life pattern (19 circles inside a bounding circle)
with subtle golden shading in the petal/lens areas.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.collections import PatchCollection

plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'
plt.rcParams['mathtext.fontset'] = 'cm'

DARK_BLUE    = '#1a237e'
GOLD         = '#FFB300'
GOLD_PALE    = '#FFF8E1'
WHITE        = '#FFFFFF'

def flower_of_life_centres(r):
    """19 centres: 1 + 6 + 12."""
    centres = [(0.0, 0.0)]
    # first ring – 6
    for k in range(6):
        a = k * np.pi / 3
        centres.append((r * np.cos(a), r * np.sin(a)))
    # second ring – 12 (alternating 2r and √3·r)
    for k in range(6):
        a = k * np.pi / 3
        centres.append((2*r * np.cos(a), 2*r * np.sin(a)))
        a2 = (k + 0.5) * np.pi / 3
        centres.append((np.sqrt(3)*r * np.cos(a2),
                         np.sqrt(3)*r * np.sin(a2)))
    return centres

r = 1.0
centres = flower_of_life_centres(r)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(WHITE)

# ── Subtle petal shading ──
# For each pair of circles that overlap, fill the lens area
theta_res = 120
for i in range(len(centres)):
    for j in range(i+1, len(centres)):
        x1, y1 = centres[i]
        x2, y2 = centres[j]
        d = np.hypot(x2 - x1, y2 - y1)
        if d > 2*r - 1e-9:
            continue  # no overlap
        if d < 1e-9:
            continue
        # Intersection points of two circles of radius r
        # midpoint
        mx, my = (x1+x2)/2, (y1+y2)/2
        # half-distance
        a = d / 2
        h = np.sqrt(r**2 - a**2)
        # direction perpendicular to line between centres
        dx, dy = (x2-x1)/d, (y2-y1)/d
        px, py = -dy, dx
        # intersection points
        ix1, iy1 = mx + h*px, my + h*py
        ix2, iy2 = mx - h*px, my - h*py

        # Build lens from two arcs
        # Arc on circle i from intersection1 to intersection2
        ang1_i = np.arctan2(iy1 - y1, ix1 - x1)
        ang2_i = np.arctan2(iy2 - y1, ix2 - x1)
        # ensure we take the short arc
        if ang2_i < ang1_i:
            ang2_i += 2*np.pi
        if ang2_i - ang1_i > np.pi:
            ang1_i, ang2_i = ang2_i, ang1_i + 2*np.pi

        t1 = np.linspace(ang1_i, ang2_i, theta_res)
        arc1_x = x1 + r * np.cos(t1)
        arc1_y = y1 + r * np.sin(t1)

        ang1_j = np.arctan2(iy2 - y2, ix2 - x2)
        ang2_j = np.arctan2(iy1 - y2, ix1 - x2)
        if ang2_j < ang1_j:
            ang2_j += 2*np.pi
        if ang2_j - ang1_j > np.pi:
            ang1_j, ang2_j = ang2_j, ang1_j + 2*np.pi

        t2 = np.linspace(ang1_j, ang2_j, theta_res)
        arc2_x = x2 + r * np.cos(t2)
        arc2_y = y2 + r * np.sin(t2)

        lx = np.concatenate([arc1_x, arc2_x])
        ly = np.concatenate([arc1_y, arc2_y])
        ax.fill(lx, ly, color=GOLD, alpha=0.10, zorder=1, edgecolor='none')

# ── Draw circles ──
for cx, cy in centres:
    c = plt.Circle((cx, cy), r, fill=False,
                    edgecolor=DARK_BLUE, lw=1.3, zorder=3)
    ax.add_patch(c)

# ── Bounding circle ──
bound_r = 2*r + r * 0.15  # slightly larger than outermost centres + r
bound = plt.Circle((0, 0), 2*r + r, fill=False,
                    edgecolor=DARK_BLUE, lw=2.5, zorder=4,
                    linestyle='-')
ax.add_patch(bound)

lim = 3.5
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)

# Title below
ax.text(0, -lim + 0.1, '生命之花  Flower of Life',
        ha='center', va='bottom', fontsize=15,
        color=DARK_BLUE, weight='bold')

plt.savefig('/home/azureuser/ai-blog/static/images/two-circles-1/flower_of_life_full.png',
            dpi=200, bbox_inches='tight', facecolor=WHITE)
plt.close()
print('✓ flower_of_life_full.png')
