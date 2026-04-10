#!/usr/bin/env python3
"""
Animated GIF: Vesica Piscis and √3 derivation
Slow, meditative step-by-step geometric construction.
Output: vesica_piscis_sqrt3.gif (800x600, ~10s loop)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# ── Font & style ──
plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'
plt.rcParams['mathtext.fontset'] = 'cm'

# ── Colors ──
NAVY      = '#1a237e'
GOLD      = '#FFB300'
CORAL     = '#E53935'
WHITE     = '#FFFFFF'
GRAY      = '#9E9E9E'
LGRAY     = '#BDBDBD'
ORANGE_DK = '#E65100'

# ── Geometry ──
r  = 2.0
d  = r                          # centre-to-centre distance = r
cx1, cx2 = -d / 2, d / 2       # circle centres
cy = 0.0
h  = r * np.sqrt(3) / 2        # half-height of vesica = r√3/2

# Pre-compute vesica lens polygon
t_top = np.arcsin(h / r)
_al = np.linspace(-t_top, t_top, 300)
_ar = np.linspace(np.pi - t_top, np.pi + t_top, 300)
lens_x = np.concatenate([cx1 + r * np.cos(_al), cx2 + r * np.cos(_ar)])
lens_y = np.concatenate([cy  + r * np.sin(_al), cy  + r * np.sin(_ar)])

# Pre-compute full circle arcs (500 points each, very smooth)
theta_full = np.linspace(0, 2 * np.pi, 500)
c1_x = cx1 + r * np.cos(theta_full)
c1_y = cy  + r * np.sin(theta_full)
c2_x = cx2 + r * np.cos(theta_full)
c2_y = cy  + r * np.sin(theta_full)

# ── Figure ──
fig, ax = plt.subplots(figsize=(800 / 150, 600 / 150), dpi=150)
fig.patch.set_facecolor(WHITE)
fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)

# ── Animation timeline (frame index milestones) ──
# Each frame = 100 ms.  Designed for ~10 s total.
T = {
    'blank_end':        5,    # 0.0–0.5 s   empty canvas
    'title_end':        10,   # 0.5–1.0 s   title fades in
    'c1_draw_end':      22,   # 1.0–2.2 s   circle 1 draws
    'c1_hold_end':      32,   # 2.2–3.2 s   hold + label
    'c2_draw_end':      44,   # 3.2–4.4 s   circle 2 draws
    'c2_hold_end':      54,   # 4.4–5.4 s   hold + label
    'vesica_fade_end':  64,   # 5.4–6.4 s   vesica fills
    'vesica_hold_end':  76,   # 6.4–7.6 s   hold + label
    'tri_draw_end':     84,   # 7.6–8.4 s   triangle draws
    'tri_hold_end':     90,   # 8.4–9.0 s   hold
    'annot_fade_end':   97,   # 9.0–9.7 s   annotations fade in
    'final_end':        115,  # 9.7–11.5 s  final hold
}
TOTAL_FRAMES = T['final_end']

# ── Helper functions ──

def ease(t):
    """Smooth ease-in-out (Hermite interpolation)."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def progress(frame, start, end):
    """Return 0..1 progress of frame within [start, end)."""
    if frame < start:
        return 0.0
    if frame >= end:
        return 1.0
    return (frame - start) / (end - start)

def draw_circle_arc(ax, cx_arr, cy_arr, frac, **kwargs):
    """Draw a partial circle (fraction 0..1 of full arc)."""
    n = max(int(len(cx_arr) * frac), 2)
    ax.plot(cx_arr[:n], cy_arr[:n], **kwargs)

# ── Triangle path (O1 → M → A → back to O1) ──
tri_pts = np.array([[cx1, 0], [0, 0], [0, h], [cx1, 0]])
tri_segs = np.sqrt(np.sum(np.diff(tri_pts, axis=0)**2, axis=1))
tri_cumlen = np.concatenate([[0], np.cumsum(tri_segs)])
tri_total  = tri_cumlen[-1]


def triangle_path(frac):
    """Return (xs, ys) for the triangle drawn up to fraction frac."""
    target = frac * tri_total
    xs, ys = [tri_pts[0, 0]], [tri_pts[0, 1]]
    for i in range(len(tri_segs)):
        if target <= 0:
            break
        if target >= tri_segs[i]:
            xs.append(tri_pts[i + 1, 0])
            ys.append(tri_pts[i + 1, 1])
            target -= tri_segs[i]
        else:
            f = target / tri_segs[i]
            xs.append(tri_pts[i, 0] + f * (tri_pts[i + 1, 0] - tri_pts[i, 0]))
            ys.append(tri_pts[i, 1] + f * (tri_pts[i + 1, 1] - tri_pts[i, 1]))
            break
    return xs, ys


# ── Main animation function ──

def animate(frame):
    ax.clear()
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-2.9, 2.9)

    # ────────────────────────────────────────────────
    # 0. Title  (fades in after blank)
    # ────────────────────────────────────────────────
    if frame >= T['blank_end']:
        ta = ease(progress(frame, T['blank_end'], T['title_end']))
        ax.text(0, 2.55, '两个圆相遇的地方',
                ha='center', va='center', fontsize=15,
                color=NAVY, alpha=ta, fontweight='bold')

    # ────────────────────────────────────────────────
    # 1. First circle
    # ────────────────────────────────────────────────
    c1_started = frame >= T['title_end']
    if c1_started:
        frac = ease(progress(frame, T['title_end'], T['c1_draw_end']))
        # Glow (thicker, semi-transparent behind)
        draw_circle_arc(ax, c1_x, c1_y, frac,
                        color=NAVY, lw=6, alpha=0.08, solid_capstyle='round')
        # Main stroke
        draw_circle_arc(ax, c1_x, c1_y, frac,
                        color=NAVY, lw=2.5, alpha=1.0, solid_capstyle='round')
        # Centre dot (appears early)
        if frac > 0.15:
            ax.plot(cx1, cy, 'o', color=NAVY, ms=4, zorder=5)
        # Label
        if frame >= T['c1_draw_end']:
            la = ease(progress(frame, T['c1_draw_end'], T['c1_draw_end'] + 5))
            ax.text(cx1, -r - 0.35, '第一个圆', ha='center', va='top',
                    fontsize=9, color=LGRAY, alpha=la)

    # ────────────────────────────────────────────────
    # 2. Second circle
    # ────────────────────────────────────────────────
    c2_started = frame >= T['c1_hold_end']
    if c2_started:
        frac = ease(progress(frame, T['c1_hold_end'], T['c2_draw_end']))
        draw_circle_arc(ax, c2_x, c2_y, frac,
                        color=NAVY, lw=6, alpha=0.08, solid_capstyle='round')
        draw_circle_arc(ax, c2_x, c2_y, frac,
                        color=NAVY, lw=2.5, alpha=1.0, solid_capstyle='round')
        if frac > 0.15:
            ax.plot(cx2, cy, 'o', color=NAVY, ms=4, zorder=5)
        if frame >= T['c2_draw_end']:
            la = ease(progress(frame, T['c2_draw_end'], T['c2_draw_end'] + 5))
            ax.text(cx2, -r - 0.35, '第二个圆', ha='center', va='top',
                    fontsize=9, color=LGRAY, alpha=la)

    # ────────────────────────────────────────────────
    # 3. Vesica Piscis fill
    # ────────────────────────────────────────────────
    vesica_started = frame >= T['c2_hold_end']
    if vesica_started:
        vp = ease(progress(frame, T['c2_hold_end'], T['vesica_fade_end']))
        fill_alpha = vp * 0.30
        edge_alpha = vp * 0.65

        # Golden fill
        ax.fill(lens_x, lens_y, color=GOLD, alpha=fill_alpha, zorder=2)
        # Golden edge glow
        ax.fill(lens_x, lens_y, color='none', edgecolor=GOLD,
                lw=2.5, alpha=edge_alpha, zorder=3)

        # Labels (appear during hold)
        if frame >= T['vesica_fade_end']:
            la = ease(progress(frame, T['vesica_fade_end'],
                               T['vesica_fade_end'] + 6))
            ax.text(0, -h - 0.45, 'Vesica Piscis', ha='center', va='top',
                    fontsize=12, color=ORANGE_DK, alpha=la, style='italic',
                    fontweight='bold')
            ax.text(0, -h - 0.85, '鱼形囊', ha='center', va='top',
                    fontsize=10, color=ORANGE_DK, alpha=la * 0.85)

    # ────────────────────────────────────────────────
    # 4. Triangle  (O1 → M → A → O1)
    # ────────────────────────────────────────────────
    tri_started = frame >= T['vesica_hold_end']
    if tri_started:
        tp = ease(progress(frame, T['vesica_hold_end'], T['tri_draw_end']))
        tx, ty = triangle_path(tp)

        # Draw triangle path
        ax.plot(tx, ty, '-', color=CORAL, lw=2.2, zorder=5,
                solid_capstyle='round', solid_joinstyle='round')

        # When complete, add fill + right-angle mark + dots
        if tp >= 1.0:
            fa = ease(progress(frame, T['tri_draw_end'], T['tri_draw_end'] + 4))
            ax.fill([cx1, 0, 0], [0, 0, h],
                    color=CORAL, alpha=0.07 * fa, zorder=2)
            # Right-angle square at M=(0,0)
            sq = 0.15
            ax.plot([sq, sq, 0], [0, sq, sq], '-',
                    color=CORAL, lw=1.2, alpha=fa, zorder=5)
            # Vertex dots
            for px, py in [(cx1, 0), (0, 0), (0, h)]:
                ax.plot(px, py, 'o', color=CORAL, ms=4.5,
                        alpha=fa, zorder=6)
            # Vertex labels
            ax.text(cx1 - 0.15, -0.2, '$O_1$', ha='right', va='top',
                    fontsize=10, color=NAVY, alpha=fa, fontweight='bold')
            ax.text(0.12, -0.22, '$M$', ha='left', va='top',
                    fontsize=10, color=CORAL, alpha=fa, fontweight='bold')
            ax.text(0.15, h + 0.1, '$A$', ha='left', va='bottom',
                    fontsize=10, color=CORAL, alpha=fa, fontweight='bold')

    # ────────────────────────────────────────────────
    # 5. √3 annotations
    # ────────────────────────────────────────────────
    annot_started = frame >= T['tri_hold_end']
    if annot_started:
        aa = ease(progress(frame, T['tri_hold_end'], T['annot_fade_end']))

        # Bottom side label: r/2
        ax.text(cx1 / 2, -0.32, r'$\frac{r}{2}$',
                ha='center', va='top', fontsize=12,
                color=GRAY, alpha=aa)

        # Vertical side label: r√3/2
        ax.text(0.3, h / 2, r'$\frac{r\sqrt{3}}{2}$',
                ha='left', va='center', fontsize=12,
                color=CORAL, alpha=aa, fontweight='bold')

        # Hypotenuse label: r
        mid_x = cx1 / 2
        mid_y = h / 2
        ang = np.degrees(np.arctan2(h, -cx1))
        ax.text(mid_x - 0.28, mid_y + 0.18, '$r$',
                ha='center', va='center', fontsize=13,
                color=NAVY, alpha=aa, fontweight='bold',
                rotation=ang)

        # Dimension arrows for full height AB = r√3
        arr_alpha = aa * 0.7
        ax.annotate('', xy=(-0.35, h), xytext=(-0.35, -h),
                    arrowprops=dict(arrowstyle='<->', color=GOLD,
                                   lw=1.6, mutation_scale=12),
                    alpha=arr_alpha)
        ax.text(-0.6, 0, r'$r\sqrt{3}$', ha='right', va='center',
                fontsize=14, color=ORANGE_DK, alpha=aa,
                fontweight='bold', rotation=90)

        # Main formula at bottom
        ax.text(0, -2.45, r'$h = r\sqrt{3}$',
                ha='center', va='top', fontsize=16,
                color=CORAL, alpha=aa, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.35', fc=WHITE,
                          ec=CORAL, lw=1.5, alpha=aa * 0.5))

        # Second intersection point B (bottom)
        ax.plot(0, -h, 'o', color=CORAL, ms=4.5, alpha=aa, zorder=6)
        ax.text(0.15, -h - 0.15, '$B$', ha='left', va='top',
                fontsize=10, color=CORAL, alpha=aa, fontweight='bold')

        # Dashed line A-B
        ax.plot([0, 0], [-h, h], '--', color=CORAL, lw=1.2,
                alpha=aa * 0.5, zorder=3)

    return []


# ── Build & save ──
print(f'Rendering {TOTAL_FRAMES} frames ({TOTAL_FRAMES * 0.1:.1f} s) ...')
ani = animation.FuncAnimation(fig, animate, frames=TOTAL_FRAMES,
                              interval=100, blit=False, repeat=True)
out_path = '/home/azureuser/ai-blog/static/images/two-circles-1/vesica_piscis_sqrt3.gif'
ani.save(out_path, writer='pillow', fps=10, dpi=150,
         savefig_kwargs={'facecolor': WHITE, 'pad_inches': 0.05})
plt.close()

import os
size_kb = os.path.getsize(out_path) / 1024
print(f'Done  {out_path}')
print(f'Size: {size_kb:.0f} KB  ({size_kb/1024:.1f} MB)')
print(f'Frames: {TOTAL_FRAMES}, Duration: {TOTAL_FRAMES * 0.1:.1f} s')
