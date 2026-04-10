#!/usr/bin/env python3
"""
vesica_piscis_sqrt3.png
Two equal circles overlapping → Vesica Piscis highlighted,
with right-triangle derivation showing height = r√3.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc
import numpy as np

# ── font ──
plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'
plt.rcParams['mathtext.fontset'] = 'cm'

DARK_BLUE  = '#1a237e'
GOLD       = '#FFB300'
GOLD_LIGHT = '#FFF8E1'
GRAY       = '#9E9E9E'
RED        = '#D32F2F'
WHITE      = '#FFFFFF'

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(WHITE)

r = 2.0
d = r          # distance between centres = r  (standard Vesica Piscis)
cx1, cx2 = -d/2, d/2
cy = 0.0

# ── Vesica Piscis fill (lens intersection) ──
theta = np.linspace(0, 2*np.pi, 500)
# intersection points
# For two circles of radius r with centres at (±d/2,0),
# the intersection x = 0; y = ±sqrt(r² - (d/2)²) = ±sqrt(r² - r²/4) = ±r√3/2
h = np.sqrt(r**2 - (d/2)**2)   # = r*sqrt(3)/2

# Build the lens shape
t_top = np.arcsin(h / r)
# Left arc: from circle centred at cx1
angles_left = np.linspace(-t_top, t_top, 200)
lens_x_left = cx1 + r * np.cos(angles_left)
lens_y_left = cy  + r * np.sin(angles_left)
# Right arc: from circle centred at cx2
angles_right = np.linspace(np.pi - t_top, np.pi + t_top, 200)
lens_x_right = cx2 + r * np.cos(angles_right)
lens_y_right = cy  + r * np.sin(angles_right)

lens_x = np.concatenate([lens_x_left, lens_x_right])
lens_y = np.concatenate([lens_y_left, lens_y_right])
ax.fill(lens_x, lens_y, color=GOLD, alpha=0.30, zorder=2)
ax.fill(lens_x, lens_y, color='none', edgecolor=GOLD, lw=2.5, zorder=3)

# ── Draw the two circles ──
for cx in [cx1, cx2]:
    circle = plt.Circle((cx, cy), r, fill=False,
                         edgecolor=DARK_BLUE, lw=2.2, zorder=4)
    ax.add_patch(circle)
    ax.plot(cx, cy, 'o', color=DARK_BLUE, ms=4, zorder=5)

# ── Label centres ──
ax.text(cx1, cy - 0.25, '$O_1$', ha='center', va='top',
        fontsize=13, color=DARK_BLUE, weight='bold')
ax.text(cx2, cy - 0.25, '$O_2$', ha='center', va='top',
        fontsize=13, color=DARK_BLUE, weight='bold')

# ── Mark intersection points ──
ax.plot(0, h,  'o', color=RED, ms=6, zorder=6)
ax.plot(0, -h, 'o', color=RED, ms=6, zorder=6)
ax.text(0.15, h + 0.15, '$A$', fontsize=13, color=RED, weight='bold')
ax.text(0.15, -h - 0.30, '$B$', fontsize=13, color=RED, weight='bold')

# ── Dashed line AB (height of Vesica Piscis) ──
ax.plot([0, 0], [-h, h], '--', color=RED, lw=1.5, zorder=3)

# ── Right triangle O1-M-A  (M is midpoint of O1O2 at origin) ──
# O1 = (cx1, 0),  M = (0, 0),  A = (0, h)
triangle_x = [cx1, 0, 0, cx1]
triangle_y = [0,   0, h, 0  ]
ax.plot(triangle_x, triangle_y, '-', color=RED, lw=1.8, zorder=5)
# Fill triangle lightly
ax.fill(triangle_x, triangle_y, color=RED, alpha=0.07, zorder=2)

# Right-angle mark at M
sq = 0.18
ax.plot([sq, sq, 0], [0, sq, sq], '-', color=RED, lw=1.2, zorder=5)

# ── Label triangle sides ──
# O1-M side (bottom) = r/2
ax.annotate('', xy=(0, -0.15), xytext=(cx1, -0.15),
            arrowprops=dict(arrowstyle='<->', color=GRAY, lw=1.2))
ax.text(cx1/2, -0.45, r'$\frac{r}{2}$', ha='center', va='top',
        fontsize=14, color=GRAY)

# M-A side (vertical) = r√3/2
ax.text(0.25, h/2, r'$\frac{r\sqrt{3}}{2}$', ha='left', va='center',
        fontsize=14, color=RED, weight='bold')

# O1-A hypotenuse = r
ax.text(cx1/2 - 0.30, h/2 + 0.20, '$r$', ha='center', va='center',
        fontsize=15, color=DARK_BLUE, weight='bold',
        rotation=np.degrees(np.arctan2(h, -cx1)))

# ── O1-O2 distance label ──
ax.annotate('', xy=(cx2, 0.12), xytext=(cx1, 0.12),
            arrowprops=dict(arrowstyle='<->', color=DARK_BLUE, lw=1.2))
ax.text(0, 0.30, '$r$（圆心距）', ha='center', va='bottom',
        fontsize=12, color=DARK_BLUE)

# ── Height label (full AB = r√3) ──
ax.annotate('', xy=(-0.30, h), xytext=(-0.30, -h),
            arrowprops=dict(arrowstyle='<->', color=GOLD, lw=1.8))
ax.text(-0.55, 0, r'$r\sqrt{3}$', ha='right', va='center',
        fontsize=16, color='#E65100', weight='bold', rotation=90)

# ── Title ──
ax.text(0, -r - 0.9, 'Vesica Piscis（鱼形囊）与 $\\sqrt{3}$',
        ha='center', va='top', fontsize=16, color=DARK_BLUE, weight='bold')

# ── Derivation note ──
note = (r'$O_1A = r,\;\; O_1M = \frac{r}{2}$' '\n'
        r'$MA = \sqrt{r^2 - \frac{r^2}{4}} = \frac{r\sqrt{3}}{2}$' '\n'
        r'$AB = r\sqrt{3}$')
ax.text(cx2 + 1.1, -1.3, note, fontsize=11, color='#424242',
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.4', fc='#F5F5F5', ec=GRAY, lw=0.8))

ax.set_xlim(-3.5, 4.8)
ax.set_ylim(-3.3, 2.8)

plt.savefig('/home/azureuser/ai-blog/static/images/two-circles-1/vesica_piscis_sqrt3.png',
            dpi=200, bbox_inches='tight', facecolor=WHITE)
plt.close()
print('✓ vesica_piscis_sqrt3.png')
