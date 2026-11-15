"""Paper-C figures from the CPU2 summaries (E1, E2, E5).  PNG, 300 dpi.
python3 make_figs.py <results dir> <out dir>"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

R, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
UNAWARE = "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
                     "axes.spines.top": False, "axes.spines.right": False})
ANCH = ["legacy_lte", "modern_lte", "modern_5g", "meas"]
ALAB = {"legacy_lte": "Galaxy S4\n+ lit. LTE\n(prior work)", "modern_lte": "Modern\n+ lit. LTE",
        "modern_5g": "Modern\n+ lit. 5G", "meas": "Modern\n+ measured\n(this work)"}
TLAB_A = {"legacy_lte": "Galaxy S4 + lit. LTE (prior work)", "modern_lte": "Modern phone + lit. LTE",
          "modern_5g": "Modern phone + lit. 5G", "meas": "Modern phone + measured LTE"}
FORM = {"short": "Short-video feed", "long": "Long-form video"}


def hr(stage):
    return pd.read_csv(os.path.join(R, stage, "summary", "headroom.csv")).set_index("model")


def grid_y(ax):
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- Fig 1: headline bars
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.5), sharey=True, constrained_layout=True)
for ax, form in zip(axs, ("short", "long")):
    t = hr(f"E1E2_{form}").loc[ANCH]
    x = np.arange(len(ANCH)); w = 0.36
    for k, (col, key, lab) in enumerate(((S1, "H_aware", "Best energy-aware policy"),
                                         (S2, "H_oracle", "Clairvoyant reference"))):
        v = t[key].to_numpy() * 100
        err = np.vstack([v - t[key + "_lo"] * 100, t[key + "_hi"] * 100 - v])
        ax.bar(x + (k - 0.5) * w, v, w * 0.92, color=col, label=lab, zorder=2)
        ax.errorbar(x + (k - 0.5) * w, v, err, fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.8, zorder=3)
        for xi, vi, hi in zip(x, v, t[key + "_hi"] * 100):
            ax.text(xi + (k - 0.5) * w, hi + 0.8, f"{vi:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(x, [ALAB[a] for a in ANCH], fontsize=7)
    ax.set_title(FORM[form], fontsize=8.5, color=INK)
    grid_y(ax)
axs[0].set_ylabel("Energy saving at matched QoE\nover tuned energy-unaware players [%]")
axs[1].legend(loc="upper right", fontsize=7)
fig.savefig(os.path.join(OUT, "fig_headline.png"), dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Fig 2: headroom maps
C_GRID = (0.0444, 0.1, 0.2, 0.4, 0.65); TAILS = (0.0, 1.0, 5.0, 11.576, 20.0); BETAS = (0.65, 1.29, 2.58)
TLAB = ["none", "1 s, 0.25 W\n(measured)", "5 s, 1.06 W", "11.6 s, 1.06 W\n(LTE)", "20 s, 2.12 W\n(5G NSA)"]
blues = LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def gridmat(t, key, b):
    M = np.full((len(TAILS), len(C_GRID)), np.nan)
    for m, r in t.iterrows():
        if not m.startswith("c"):
            continue
        c, T, bb = (float(s[1:]) for s in m.split("_"))
        if abs(bb - b) > 1e-6:
            continue
        M[int(np.argmin([abs(T - x) for x in TAILS])), int(np.argmin([abs(c - x) for x in C_GRID]))] = r[key]
    return M * 100


for key, fname, title in (("H_aware", "fig_map_aware.png", "best energy-aware policy"),
                          ("H_oracle", "fig_map_oracle.png", "clairvoyant reference")):
    tabs = {f: hr(f"E1E2_{f}") for f in ("short", "long")}
    vmax = max(np.nanmax(tabs[f][key].to_numpy()) for f in tabs) * 100
    fig, axs = plt.subplots(2, 3, figsize=(7.16, 3.6), sharex=True, sharey=True, constrained_layout=True)
    for i, form in enumerate(("short", "long")):
        for j, b in enumerate(BETAS):
            ax = axs[i, j]
            M = gridmat(tabs[form], key, b)
            im = ax.imshow(M, origin="lower", cmap=blues, vmin=0, vmax=vmax, aspect="auto")
            for y in range(M.shape[0]):
                for xx in range(M.shape[1]):
                    ax.text(xx, y, f"{M[y, xx]:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if M[y, xx] > 0.55 * vmax else INK)
            if b == 0.65:
                ax.add_patch(plt.Rectangle((-0.5, 0.5), 1, 1, fill=False, ec=S2, lw=1.4))
            ax.set_xticks(range(len(C_GRID)), [f"{c:g}" for c in C_GRID], fontsize=7)
            ax.set_yticks(range(len(TAILS)), TLAB, fontsize=7)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(length=0)
            if i == 0:
                ax.set_title(f"active radio power {b:g} W" + (" (measured)" if b == 0.65 else ""), fontsize=7.5)
            if i == 1:
                ax.set_xlabel("decode energy [J/MB]")
            if j == 0:
                ax.set_ylabel(f"{FORM[form]}\nradio tail after a transfer")
    cb = fig.colorbar(im, ax=axs, shrink=0.75, pad=0.01)
    cb.set_label(f"Energy saving at matched QoE, {title} [%]", fontsize=7)
    cb.outline.set_visible(False)
    fig.savefig(os.path.join(OUT, fname), dpi=300)
    plt.close(fig)

# ---------------------------------------------------------------- Fig 3: fronts
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cstudy.analyze import pareto  # noqa: E402
from svs.grids import AWARE, ORACLE, UNAWARE as UNA  # noqa: E402
fig, axs = plt.subplots(2, 2, figsize=(7.16, 4.4), constrained_layout=True)
for i, form in enumerate(("short", "long")):
    d = pd.read_csv(os.path.join(R, f"E1E2_{form}", "summary", "cfg_means.csv"))
    for j, m in enumerate(("legacy_lte", "meas")):
        ax = axs[i, j]
        x = d[d.model == m]
        u = x.family.isin(UNA[form]); a = u | x.family.isin(AWARE[form]); o = a | x.family.isin(ORACLE[form])
        ax.scatter(x.qoe_min, x.E_min, s=3, color="#c9c8c1", lw=0, zorder=1, label="all configurations")
        for mask, col, lab, z in ((o, S2, "+ clairvoyant", 2), (a, S1, "+ energy-aware", 3),
                                  (u, UNAWARE, "energy-unaware", 4)):
            f = pareto(x.loc[mask, ["qoe_min", "E_min"]].to_numpy())
            ax.plot(f[:, 0], f[:, 1], "-o", ms=2.6, lw=1.4, color=col, label=lab, zorder=z,
                    mec="white", mew=0.4)
        fr = [pareto(x.loc[mm, ["qoe_min", "E_min"]].to_numpy()) for mm in (u, a, o)]
        uf = fr[0]
        qlo = uf[:, 0].min() - 15
        pts = np.vstack([f[f[:, 0] >= qlo] for f in fr])
        ax.set_xlim(qlo, pts[:, 0].max() + 2)
        ax.set_ylim(pts[:, 1].min() * 0.96, pts[:, 1].max() * 1.04)
        ax.set_title(f"{FORM[form]}: {TLAB_A[m]}", fontsize=7.5)
        ax.set_xlabel("QoE per minute (SQoE-III, phone VMAF)")
        ax.set_ylabel("Device energy [J/min]")
        grid_y(ax); ax.xaxis.grid(True, color=GRID, lw=0.6)
axs[0, 0].legend(fontsize=7, loc="upper left")
fig.savefig(os.path.join(OUT, "fig_fronts.png"), dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Fig 4: what the clairvoyant knows (feed)
t = hr("E2_decomp_short").loc[ANCH]
fig, ax = plt.subplots(figsize=(3.5, 2.8), layout="constrained")
keys = (("H_aware", S1, "energy-aware policies"), ("H_thronly", S2, "+ knows future throughput"),
        ("H_watchonly", S3, "+ knows watch times"), ("H_oracle", S4, "+ knows both"))
x = np.arange(len(ANCH)); w = 0.2
for k, (key, col, lab) in enumerate(keys):
    ax.bar(x + (k - 1.5) * w, t[key] * 100, w * 0.9, color=col, label=lab, zorder=2)
ax.set_xticks(x, [ALAB[a] for a in ANCH], fontsize=7)
ax.set_ylabel("Energy saving at matched QoE [%]")
ax.set_title("Short-video feed: where the remaining headroom is", fontsize=7.5)
grid_y(ax)
fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=7)
ax.set_ylim(0, 50)
fig.savefig(os.path.join(OUT, "fig_decomp.png"), dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Fig 5: robustness
VAR = [("E1E2_{f}", "main (4-s chunks, phone VMAF, 4G)"), ("E5_{f}_2s", "2-s chunks"),
       ("E5_{f}_vmafstd", "standard VMAF"), ("E5_{f}_5g", "5G traces"),
       ("E5_{f}_old", "low-throughput traces (3G/FCC)"), ("E5_{f}_sampled", "watch times from retention curves")]
fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.8), layout="constrained")
for ax, form in zip(axs, ("short", "long")):
    y = np.arange(len(VAR))[::-1]
    for k, (m, col, lab) in enumerate((("legacy_lte", S2, "Galaxy S4 + lit. LTE (prior work)"),
                                       ("meas", S1, "Modern phone + measured LTE"))):
        for (st, _), yi in zip(VAR, y):
            st = st.format(f=form)
            if not os.path.exists(os.path.join(R, st)):
                continue
            r = hr(st).loc[m]
            v, lo, hi = r.H_aware * 100, r.H_aware_lo * 100, r.H_aware_hi * 100
            ax.errorbar([v], [yi + (0.14 if k == 0 else -0.14)], xerr=[[v - lo], [hi - v]], fmt="o", ms=4,
                        color=col, elinewidth=1, label=lab if st.startswith("E1E2") else None,
                        mec="white", mew=0.5)
    ax.set_yticks(y, [lab for _, lab in VAR] if form == "short" else [""] * len(VAR))
    ax.set_ylim(-0.6, len(VAR) - 0.4)
    ax.set_title(FORM[form], fontsize=8.5)
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlim(0, 36)
    if form == "long":
        ax.text(1, y[-1], "n/a (one long video per session)", va="center", fontsize=7, color=INK2)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=2, fontsize=7)
fig.supxlabel("Energy saving of energy-aware policies at matched QoE [%] (95% CI)", fontsize=7.5)
fig.savefig(os.path.join(OUT, "fig_robustness.png"), dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Fig S1: per genre (long form)
g = pd.read_csv(os.path.join(R, "E1E2_long", "summary", "genre.csv"))
gen = sorted(g.variant.unique())
fig, ax = plt.subplots(figsize=(3.5, 2.9), layout="constrained")
y = np.arange(len(gen))[::-1]
for k, (m, col, lab) in enumerate((("legacy_lte", S2, "Galaxy S4 + lit. LTE (prior work)"), ("meas", S1, "modern + measured"))):
    s = g[g.model == m].set_index("variant").loc[gen]
    v = s.H_aware * 100
    ax.errorbar(v, y + (0.14 if k == 0 else -0.14), xerr=[v - s.H_aware_lo * 100, s.H_aware_hi * 100 - v],
                fmt="o", ms=3.5, color=col, elinewidth=0.9, label=lab, mec="white", mew=0.5)
ax.set_yticks(y, gen)
ax.set_xlabel("Energy saving at matched QoE [%]")
ax.set_title("Long-form video, by content genre", fontsize=7.5)
ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.legend(fontsize=7, loc="outside lower center", ncol=2)
fig.savefig(os.path.join(OUT, "figS_genre.png"), dpi=300)
plt.close(fig)
print("figures written to", OUT)
