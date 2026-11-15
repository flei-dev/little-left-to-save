"""Radio tail after a 200-KB transfer on two phones (LTE, LTE, 5G) vs the literature models.  PNG 300 dpi.
python3 make_fig_phone.py [out dir]"""
import sys, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "phone"))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "figures", "out")
os.makedirs(OUT, exist_ok=True)
PD = os.path.join(ROOT, "phone", "data")
from analyze_phone import load, transfers, marks
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "legend.frameon": False})
RUNS = [(os.path.join(PD, "huawei_lte"), "Huawei MAO-AL00, LTE", S1), (os.path.join(PD, "redmi_lte"), "Redmi 2311DRK48C, LTE", S2),
        (os.path.join(PD, "redmi_5g"), "Redmi 2311DRK48C, 5G (NR)", S3)]
rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(3.5, 2.7), layout="constrained")
grid = np.arange(0, 30, 1.0)
rows = []
for path, lab, col in RUNS:
    s, e, m = load(path)
    b1 = marks(e, "P1_baseline"); w1 = s[(s.t >= b1["start"] + 10) & (s.t < b1["end"])]
    base = float(w1.P.median())
    C, E = [], []
    for _, r in transfers(e, "P2_tail").iterrows():
        w = s[(s.t >= r.t_end) & (s.t < r.t_end + 40)]
        c = np.interp(np.arange(0, 40, 1.0) + 0.5, w.t - r.t_end, w.P - base)
        C.append(c); E.append(c[1:20].sum() - 19 * np.median(c[25:]))
    C, E = np.array(C), np.array(E)
    bs = [rng.choice(E, len(E)).mean() for _ in range(4000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    rows.append((lab, E.mean(), lo, hi))
    med = np.median(C, 0)[:30]
    q1, q3 = np.percentile(C, [25, 75], axis=0)[:, :30]
    ax.fill_between(grid, q1, q3, color=col, alpha=0.15, lw=0)
    ax.plot(grid, med, color=col, lw=1.4, label=lab.replace("Redmi 2311DRK48C", "Redmi").replace("Huawei MAO-AL00", "Huawei").replace(", 5G (NR)", ", 5G"))
ax.step([0, 11.576, 30], [1.06, 0, 0], where="post", color=INK, lw=1.0, ls="--", label="lit. LTE model")
ax.step([0, 20, 30], [2.12, 0, 0], where="post", color=INK2, lw=1.0, ls=":", label="5G anchor")
ax.axhline(0, color=INK2, lw=0.5)
ax.set_xlim(0, 30); ax.set_ylim(-0.4, 2.3)
ax.set_xlabel("Time after the end of a 200-KB transfer [s]")
ax.set_ylabel("Excess power over idle [W]")
ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(loc="upper right", fontsize=7, frameon=True, framealpha=0.95, edgecolor="white", borderpad=0.3, handlelength=1.6)
fig.savefig(os.path.join(OUT, "fig_phone_tail.png"), dpi=300)
json.dump(rows, open(os.path.join(OUT, "tail_energy_table.json"), "w"))
for r in rows: print(r)
