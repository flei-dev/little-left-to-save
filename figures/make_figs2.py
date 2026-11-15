"""Paper-C figures for E3 (DRL vs rules) and E4 (AV1 vs H.264).  PNG 300 dpi.
python3 make_figs2.py <cpu2 results> <cpu1 results> <out dir>"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from cstudy.analyze import pareto  # noqa: E402
from svs.grids import AWARE, UNAWARE  # noqa: E402

R2, R1, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
                     "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False})
REG = [("legacy", "Galaxy S4 + lit. LTE"), ("meas", "Modern + measured LTE"),
       ("5g", "Modern + lit. 5G (5G traces)"), ("low", "Modern + measured LTE\n(low-throughput traces)")]
FORM = {"short": "Feed", "long": "Long-form"}

# ---------------------------------------------------------------- E3: DRL vs tuned rules
fig, axs = plt.subplots(2, 4, figsize=(7.16, 3.9), layout="constrained")
for i, form in enumerate(("short", "long")):
    for j, (reg, lab) in enumerate(REG):
        ax = axs[i, j]
        d = pd.read_csv(os.path.join(R2, f"E3_{form}_{reg}", "summary", "cfg_means.csv"))
        rules = d[d.family.isin(UNAWARE[form] + AWARE[form])]
        drl = d[d.family == "DRL"]
        ax.scatter(rules.qoe_min, rules.E_min, s=3, color="#c9c8c1", lw=0, zorder=1)
        f = pareto(rules[["qoe_min", "E_min"]].to_numpy())
        ax.plot(f[:, 0], f[:, 1], "-", lw=1.4, color=S1, zorder=3, label="tuned rules (front)")
        ax.scatter(drl.qoe_min, drl.E_min, s=12, color=S2, zorder=4, ec="white", lw=0.4,
                   label="DQN (4 λ × 3 seeds)")
        allq = np.r_[f[:, 0], drl.qoe_min]
        ax.set_xlim(allq.min() - 4, allq.max() + 3)
        e = np.r_[f[:, 1], drl.E_min]
        ax.set_ylim(e.min() * 0.95, e.max() * 1.05)
        ax.xaxis.grid(True, color=GRID, lw=0.5); ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.set_title(lab, fontsize=7)
        if j == 0:
            ax.set_ylabel(f"{FORM[form]}\nenergy [J/min]")
        if i == 1:
            ax.set_xlabel("QoE per minute")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=2, fontsize=7)
fig.savefig(os.path.join(OUT, "fig_drl.png"), dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- E4: codec
rows = []
for form in ("short", "long"):
    for ts in ("old", "lte", "5g"):
        p = os.path.join(R1, f"E4_{form}_{ts}", "summary", "codec.csv")
        t = pd.read_csv(p)
        t["form"], t["ts"] = form, ts
        rows.append(t)
t = pd.concat(rows)
DEC = [("av1hw_prop_lte_meas", "hardware\n(0 W)"), ("av1dav1d_prop_lte_meas", "dav1d\n(+0.115 W)"),
       ("av1sysdec_prop_lte_meas", "system dec.\n(+0.69 W)"), ("av1coconut_prop_lte_meas", "COCONUT\nSW (+1.43 W)")]
TS = [("old", "low-throughput (3.2 Mbps)", S1), ("lte", "4G (10 Mbps)", S2), ("5g", "5G (68 Mbps)", S3)]
fig, axs = plt.subplots(2, 2, figsize=(7.16, 3.5), layout="constrained", sharex=True)
for i, form in enumerate(("short", "long")):
    for j, (key, ylab) in enumerate((("av1_vs_h264", "Always-AV1 vs always-H.264"),
                                    ("greedy_vs_h264", "Quality-greedy codec switching vs H.264"))):
        ax = axs[i, j]
        x = np.arange(len(DEC)); w = 0.26
        for k, (ts, lab, col) in enumerate(TS):
            v = [t[(t.form == form) & (t.ts == ts) & (t.model == m)][key].iloc[0] * 100 for m, _ in DEC]
            ax.bar(x + (k - 1) * w, v, w * 0.9, color=col, label=lab, zorder=2)
            lo = [t[(t.form == form) & (t.ts == ts) & (t.model == m)][key + "_lo"].iloc[0] * 100 for m, _ in DEC] \
                if key + "_lo" in t else None
            hi = [t[(t.form == form) & (t.ts == ts) & (t.model == m)][key + "_hi"].iloc[0] * 100 for m, _ in DEC] \
                if key + "_hi" in t else None
            if lo is not None:
                ax.errorbar(x + (k - 1) * w, v, [np.array(v) - lo, np.array(hi) - v], fmt="none", ecolor=INK,
                            elinewidth=0.6, capsize=1.2, zorder=3)
        ax.axhline(0, color=INK2, lw=0.7)
        ax.set_xticks(x, [lab for _, lab in DEC], fontsize=7)
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_ylabel(f"{'Feed' if form == 'short' else 'Long-form'}\nsaving [%]", fontsize=7)
        if i == 0:
            ax.set_title(ylab, fontsize=7.5)
    axs[1, 0].set_xlabel("AV1 decoder of the phone (extra power at 1080p)")
    axs[1, 1].set_xlabel("AV1 decoder of the phone (extra power at 1080p)")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=3, fontsize=7, title="network traces", title_fontsize=7)
fig.savefig(os.path.join(OUT, "fig_codec.png"), dpi=300)
plt.close(fig)

# data vs energy (AV1 vs H.264), measured radio
fig, ax = plt.subplots(figsize=(3.5, 2.9), layout="constrained")
mk = {"short": "o", "long": "s"}
for m, lab in DEC:
    pass
cols = {"av1hw_prop_lte_meas": S1, "av1dav1d_prop_lte_meas": S3, "av1sysdec_prop_lte_meas": S4,
        "av1coconut_prop_lte_meas": S2}
for m, lab in DEC:
    for form in ("short", "long"):
        y = t[(t.model == m) & (t.form == form)]
        ax.scatter(y.data_av1_vs_h264 * 100, y.av1_vs_h264 * 100, marker=mk[form], s=22, color=cols[m],
                   ec="white", lw=0.4, label=lab.replace("\n", " ") if form == "short" else None, zorder=3)
ax.axhline(0, color=INK2, lw=0.7)
ax.set_xlabel("Data saving of AV1 over H.264 at matched QoE [%]")
ax.set_ylabel("Energy saving [%]")
ax.set_title("AV1 saves data, not necessarily energy\n(circles: feed, squares: long-form)", fontsize=7)
ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.legend(fontsize=7, loc="outside lower center", ncol=2, title="AV1 decoder", title_fontsize=7)
fig.savefig(os.path.join(OUT, "fig_codec_data_energy.png"), dpi=300)
plt.close(fig)
print("written")
