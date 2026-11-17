"""Energy model diagram: radio state machine with literature-LTE and measured parameters, plus device terms.
python3 fig_energy_model.py <out dir>"""
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT = sys.argv[1]
INK, INK2 = "#0b0b0b", "#52514e"
C = {"idle": "#e4e3de", "pro": "#fbe3b0", "act": "#cfe0f6", "tail": "#fad3c2", "dev": "#d6efe4"}
E = {"idle": "#9b9a93", "pro": "#eda100", "act": "#2a78d6", "tail": "#eb6834", "dev": "#1baf7a"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
W, H = 3.5, 2.95
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100 * H / W * 1.0); ax.axis("off")
Y = 100 * H / W

def rbox(x, y, w, h, key):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5", fc=C[key], ec=E[key], lw=0.9))

def arrow(p, q, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=INK2,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))

ax.text(1.5, Y - 1.5, "(a) Radio state machine", fontsize=7.2, fontweight="bold", color=INK, va="top")
states = [("idle", "IDLE", r"$P_{\mathrm{idle}}$"), ("pro", "PROMO.", r"$P_{\mathrm{pro}},\,T_{\mathrm{pro}}$"),
          ("act", "TRANSFER", r"$\beta+\alpha r$"), ("tail", "TAIL", r"$P_{\mathrm{tail}},\,T_{\mathrm{tail}}$")]
xs = [1.5, 26.5, 51.5, 76.5]; bw, by, bh = 22, Y - 31, 14
for (k, name, sym), x in zip(states, xs):
    rbox(x, by, bw, bh, k)
    ax.text(x + bw / 2, by + bh - 2.6, name, ha="center", va="top", fontsize=7, fontweight="bold", color=INK)
    ax.text(x + bw / 2, by + 3.6, sym, ha="center", va="center", fontsize=7, color=INK)
for i in range(3):
    arrow((xs[i] + bw + 0.4, by + bh / 2), (xs[i + 1] - 0.4, by + bh / 2))
arrow((xs[3] + 6, by + bh + 0.4), (xs[2] + bw - 4, by + bh + 0.4), rad=0.55)
ax.text((xs[2] + xs[3] + bw) / 2 + 1, by + bh + 6.6, r"request within $T_{\mathrm{tail}}$", ha="center", fontsize=6, color=INK2, style="italic")
arrow((xs[3] + bw / 2, by - 0.4), (xs[0] + bw / 2, by - 0.4), rad=-0.12)
ax.text(50, by - 7.4, r"no request within $T_{\mathrm{tail}}$", ha="center", fontsize=6, color=INK2, style="italic")

# parameter table under the boxes
ty = by - 14.5
rows = [("Lit. LTE", ["11 mW", "1.21 W, 0.26 s", "1.29 W + 0.052 W/Mbps", "1.06 W, 11.6 s"]),
        ("Measured", ["0", "none", "0.65 W", "0.25 W, 1 s"])]
for j, (lab, vals) in enumerate(rows):
    yy = ty - 5.2 * j
    for v, x in zip(vals, xs):
        ax.text(x + bw / 2, yy, v, ha="center", va="center", fontsize=5.9 if len(v) > 14 else 6.3, color=INK,
                fontweight="bold" if j == 1 else "normal")
ax.text(50, ty - 10.6, "regular: literature LTE model    bold: measured phone", ha="center", fontsize=6, color=INK2, va="center")

# device terms
dy = ty - 15
ax.text(1.5, dy, "(b) Device playback", fontsize=7.2, fontweight="bold", color=INK, va="top")
items = [(r"$p_{\mathrm{base}}\,W$", "screen, SoC", "1.10 W"), (r"$c\,B$", "per decoded MB", "0.044 J/MB"),
         (r"$p_{\mathrm{dec}}\,\phi\,W$", r"AV1 extra", "0–1.43 W"), (r"$p_{\mathrm{stall}}\,S$", "stalls", "1.10 W")]
for i, (sym, desc, val) in enumerate(items):
    x = xs[i]
    rbox(x, 1.5, bw, dy - 8, "dev")
    top = dy - 6.5
    ax.text(x + bw / 2, top - 1.5, sym, ha="center", va="top", fontsize=7, color=INK)
    ax.text(x + bw / 2, top - 8.5, desc, ha="center", va="top", fontsize=6, color=INK2)
    ax.text(x + bw / 2, 5.0, val, ha="center", va="center", fontsize=6.3, color=INK, fontweight="bold")
    if i < 3:
        ax.text(x + bw + 1.5, (dy - 8) / 2 + 1.5, "+", ha="center", va="center", fontsize=8, color=INK2)
fig.savefig(os.path.join(OUT, "fig_energy_model.png"), dpi=300)
