"""Where the headroom comes from: energy composition of tuned energy-unaware players and the component
split of H_aware (exact; same fronts and interpolation as the headroom metric).
python3 fig_energy_split.py <comp.csv> <out dir>"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
CSV, OUT = sys.argv[1], sys.argv[2]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"
COL = {"Eradio": "#eb6834", "Ebase": "#b9b8b1", "Ec": "#2a78d6", "Estall": "#eda100"}
LAB = {"Eradio": "radio", "Ebase": "base playback", "Ec": "per-MB decoding", "Estall": "stalls"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False})
d = pd.read_csv(CSV)
ANCH = ["legacy_lte", "modern_lte", "modern_5g", "meas"]
ALAB = {"legacy_lte": "Prior-work", "modern_lte": "Modern + lit. LTE", "modern_5g": "Modern + lit. 5G", "meas": "Measured"}
rows = [(f, m) for f in ("short", "long") for m in ANCH]
y = np.array([i + (1.25 if f == "long" else 0) for i, (f, m) in enumerate(rows)])[::-1]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 2.55), sharey=True, layout="constrained",
                             gridspec_kw={"width_ratios": [1.25, 1]})
for (f, m), yy in zip(rows, y):
    r = d[(d.form == f) & (d.model == m)].iloc[0]
    left = 0
    for k in ("Eradio", "Ebase", "Ec", "Estall"):
        v = r["U_" + k]; a1.barh(yy, v, left=left, color=COL[k], height=0.72, lw=0.3, edgecolor="white"); left += v
    left = 0
    for k in ("Eradio", "Ec", "Estall"):
        v = 100 * r[k + "_share"]
        if v > 0:
            a2.barh(yy, v, left=left, color=COL[k], height=0.72, lw=0.3, edgecolor="white"); left += v
    a2.text(left + 0.8, yy, f"{100 * r.H_total:.1f}", va="center", fontsize=7, color=INK)
a1.set_yticks(y); a1.set_yticklabels([ALAB[m] for f, m in rows], fontsize=7)
for ax in (a1, a2):
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
a1.set_xlabel("Energy-unaware\nenergy [J/min]", fontsize=7); a2.set_xlabel("Headroom $H_A$\nby component [%]", fontsize=7)
a2.set_xlim(0, 38)
for f, yy in (("Short-video feed", y[0] + 0.75), ("Long-form video", y[4] + 0.75)):
    a1.text(-0.02, yy + 0.05, f, transform=a1.get_yaxis_transform(), ha="right", va="bottom", fontsize=7, fontweight="bold", color=INK)
a1.set_ylim(y.min() - 0.6, y.max() + 1.3)
h = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in COL]
fig.legend(h, [LAB[k] for k in COL], loc="outside lower center", ncol=4, fontsize=7, handlelength=1.0, columnspacing=0.9)
fig.savefig(os.path.join(OUT, "fig_energy_split.png"), dpi=300)
