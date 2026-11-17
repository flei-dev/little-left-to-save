"""Fig. 1 (problem): radio power over one minute of a feed under per-video fetching and batched fetching,
with the literature LTE model and the measured phone.  Powers and energies from svs.model.  PNG 300 dpi.
python3 fig_problem.py <repo root> <out dir>"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(ROOT, "code"))
from svs.model import RADIOS, radio_energy

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"
C_ACT, C_TAIL, C_PRO = "#2a78d6", "#eb6834", "#eda100"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
                     "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False})
T_END, THR = 60.0, 10.0                      # s, Mbps (median LTE test throughput 10.4 Mbps)
MB = 2.5                                      # MB per video
def schedule(batch):
    """(t_req, n_videos) of each transfer: one video per 10-s swipe, or three videos every 30 s."""
    return [(t, 1) for t in range(0, 60, 10)] if batch == 1 else [(0, 3), (30, 3)]

def trace(radio, sched):
    t = np.linspace(0, T_END, 6001); p = np.zeros_like(t); kind = np.zeros_like(t)  # 1 pro, 2 act, 3 tail
    transfers = []; last_end = None
    for tr, n in sched:
        promo = radio.t_pro > 0 and (last_end is None or tr - last_end > radio.t_tail)
        ds = tr + (radio.t_pro if promo else 0.0); de = ds + n * MB * 8 / THR
        transfers.append((tr, ds, de, n * MB * 1e6, promo))
        m = (t >= tr) & (t < ds); p[m] = radio.p_pro; kind[m] = 1
        m = (t >= ds) & (t < de); p[m] = radio.beta + radio.alpha * THR; kind[m] = 2
        last_end = de
    for i, (tr, ds, de, nb, pr) in enumerate(transfers):
        nxt = transfers[i + 1][0] if i + 1 < len(transfers) else T_END
        m = (t >= de) & (t < min(de + radio.t_tail, nxt)); p[m] = radio.p_tail; kind[m] = 3
    e = radio_energy(transfers, T_END, radio, full_tail=False)
    return t, p, kind, e

fig, axs = plt.subplots(2, 2, figsize=(3.5, 2.55), sharex=True, sharey=True, layout="constrained")
cols = (("lte", "Literature LTE model"), ("lte_meas", "Measured phone"))
rows = ((1, "One video per swipe"), (3, "Three videos per burst"))
E = {}
for j, (rk, rlab) in enumerate(cols):
    for i, (b, blab) in enumerate(rows):
        ax = axs[i, j]; t, p, kind, e = trace(RADIOS[rk], schedule(b)); E[(rk, b)] = e
        for k, c in ((1, C_PRO), (2, C_ACT), (3, C_TAIL)):
            ax.fill_between(t, 0, np.where(kind == k, p, 0), color=c, lw=0, step="post")
        ax.plot(t, p, color=INK, lw=0.5, drawstyle="steps-post")
        ax.text(0.97, 0.93, f"{e:.0f} J", transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                fontweight="bold", color=INK)
        ax.set_ylim(0, 2.3); ax.set_xlim(0, 60); ax.set_xticks([0, 20, 40, 60]); ax.set_yticks([0, 1, 2])
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        if i == 0: ax.set_title(rlab, fontsize=7.5, color=INK)
        if j == 0: ax.set_ylabel(blab.replace(" per ", "\nper "), fontsize=7)
for j, (rk, _) in enumerate(cols):
    s = 1 - E[(rk, 3)] / E[(rk, 1)]
    axs[1, j].text(0.03, 0.93, f"batching: −{100 * s:.0f}%", transform=axs[1, j].transAxes,
                   ha="left", va="top", fontsize=7, color=INK2)
for ax in axs[1]: ax.set_xlabel("Time [s]", fontsize=7)
fig.supylabel("Radio power [W]", fontsize=7)
h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_ACT, C_TAIL, C_PRO)]
fig.legend(h, ["transfer", "tail", "promotion"], loc="outside lower center", ncol=3, fontsize=7, handlelength=1.2)
fig.savefig(os.path.join(OUT, "fig_problem.png"), dpi=300)
print({k: round(v, 1) for k, v in E.items()})
