#!/usr/bin/env python3
"""
sqrt_chain.png
Visual chain: √3 → √2 → √5 → φ = (1+√5)/2
Each shown as a small geometric construction with arrows.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'
plt.rcParams['mathtext.fontset'] = 'cm'

DARK_BLUE  = '#1a237e'
GOLD       = '#FFB300'
RED        = '#D32F2F'
GREEN      = '#2E7D32'
PURPLE     = '#7B1FA2'
WHITE      = '#FFFFFF'
GRAY       = '#757575'

fig, axes = plt.subplots(1, 4, figsize=(10, 3.2))
fig.patch.set_facecolor(WHITE)

# ───────── Panel 1: √3 from equilateral triangle ─────────
ax = axes[0]
ax.set_aspect('equal'); ax.axis('off')
# equilateral triangle side = 2, height = √3
tri_x = [-1, 1, 0, -1]
tri_y = [0, 0, np.sqrt(3), 0]
ax.fill(tri_x, tri_y, color=DARK_BLUE, alpha=0.08)
ax.plot(tri_x, tri_y, '-', color=DARK_BLUE, lw=2)
# height dashed
ax.plot([0, 0], [0, np.sqrt(3)], '--', color=RED, lw=1.8)
ax.text(0.12, np.sqrt(3)/2, r'$\sqrt{3}$', fontsize=13, color=RED, weight='bold')
# base label
ax.text(0, -0.20, '2', ha='center', fontsize=11, color=DARK_BLUE)
# side label
ax.text(-0.7, np.sqrt(3)/2+0.1, '2', fontsize=11, color=DARK_BLUE, rotation=60, ha='center')
# title
ax.text(0, -0.65, r'$\sqrt{3}$' + '\nVesica Piscis',
        ha='center', fontsize=10, color=DARK_BLUE, weight='bold')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-0.9, 2.3)

# ───────── Panel 2: √2 from unit square diagonal ─────────
ax = axes[1]
ax.set_aspect('equal'); ax.axis('off')
sq = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]])
ax.fill(sq[:,0], sq[:,1], color=GREEN, alpha=0.08)
ax.plot(sq[:,0], sq[:,1], '-', color=GREEN, lw=2)
# diagonal
ax.plot([0,1], [0,1], '-', color=RED, lw=2.2)
ax.text(0.55, 0.38, r'$\sqrt{2}$', fontsize=13, color=RED, weight='bold', rotation=45)
ax.text(0.5, -0.18, '1', ha='center', fontsize=11, color=GREEN)
ax.text(1.15, 0.5, '1', ha='left', fontsize=11, color=GREEN)
ax.text(0.5, -0.58, r'$\sqrt{2}$' + '\n正方形对角线',
        ha='center', fontsize=10, color=GREEN, weight='bold')
ax.set_xlim(-0.5, 1.8)
ax.set_ylim(-0.85, 1.65)

# ───────── Panel 3: √5 from 1×2 rectangle diagonal ─────────
ax = axes[2]
ax.set_aspect('equal'); ax.axis('off')
rect = np.array([[0,0],[2,0],[2,1],[0,1],[0,0]])
ax.fill(rect[:,0], rect[:,1], color=PURPLE, alpha=0.08)
ax.plot(rect[:,0], rect[:,1], '-', color=PURPLE, lw=2)
# diagonal
ax.plot([0,2], [0,1], '-', color=RED, lw=2.2)
ax.text(1.05, 0.35, r'$\sqrt{5}$', fontsize=13, color=RED, weight='bold',
        rotation=np.degrees(np.arctan(0.5)))
ax.text(1, -0.18, '2', ha='center', fontsize=11, color=PURPLE)
ax.text(2.15, 0.5, '1', ha='left', fontsize=11, color=PURPLE)
ax.text(1, -0.58, r'$\sqrt{5}$' + '\n1:2 矩形对角线',
        ha='center', fontsize=10, color=PURPLE, weight='bold')
ax.set_xlim(-0.4, 2.8)
ax.set_ylim(-0.85, 1.65)

# ───────── Panel 4: φ = (1+√5)/2 ─────────
ax = axes[3]
ax.set_aspect('equal'); ax.axis('off')
# Golden ratio from a line segment
# Show a line split into a+b where a/b = φ
phi = (1 + np.sqrt(5)) / 2
total = phi + 1   # a=phi, b=1
# normalize to fit
s = 2.2 / total
a = phi * s
b = 1 * s
y0 = 0.6
ax.plot([0, a+b], [y0, y0], '-', color=GOLD, lw=3)
ax.plot([0, 0], [y0-0.08, y0+0.08], '-', color=GOLD, lw=3)
ax.plot([a, a], [y0-0.08, y0+0.08], '-', color=RED, lw=2.5)
ax.plot([a+b, a+b], [y0-0.08, y0+0.08], '-', color=GOLD, lw=3)
# labels
ax.text(a/2, y0+0.15, '$a$', ha='center', fontsize=12, color=DARK_BLUE, weight='bold')
ax.text(a+b/2, y0+0.15, '$b$', ha='center', fontsize=12, color=DARK_BLUE, weight='bold')
# formula
ax.text((a+b)/2, y0-0.45,
        r'$\varphi = \frac{a}{b} = \frac{1+\sqrt{5}}{2}$',
        ha='center', fontsize=13, color=GOLD, weight='bold')
ax.text((a+b)/2, y0-0.95, r'$\approx 1.618$',
        ha='center', fontsize=11, color=GRAY)
ax.text((a+b)/2, -0.75, r'$\varphi$' + ' 黄金比例',
        ha='center', fontsize=10, color='#E65100', weight='bold')
ax.set_xlim(-0.4, 2.8)
ax.set_ylim(-1.0, 1.4)

# ── Arrows between panels ──
for i in range(3):
    x_right = axes[i].get_position().x1
    x_left  = axes[i+1].get_position().x0
    x_mid   = (x_right + x_left) / 2
    fig.text(x_mid, 0.52, '→', fontsize=22, ha='center', va='center',
             color=GOLD, weight='bold')

plt.subplots_adjust(wspace=0.15)
plt.savefig('/home/azureuser/ai-blog/static/images/two-circles-1/sqrt_chain.png',
            dpi=200, bbox_inches='tight', facecolor=WHITE)
plt.close()
print('✓ sqrt_chain.png')
