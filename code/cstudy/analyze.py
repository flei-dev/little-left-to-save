"""Pre-registered analysis of the paper-C study (PREREG_C.md).

Headroom of energy-aware adaptation at matched QoE (one energy model, one codec mode):
  U = energy-unaware configurations, A = U + energy-aware ones, O = A + clairvoyant references.
  H_aware  = mean over the points (q_b, E_b) of the Pareto front of U of  1 - E_A(q_b) / E_b
  H_oracle = the same with O instead of A.
E_X(q) = energy of the (linearly interpolated) Pareto front of X at QoE q; between two front points the
interpolation is achievable by time-sharing the two configurations across sessions.  Session bootstrap
(pairs resampled with replacement, 500 replicates) gives 95 % percentile intervals.

  python3 -m cstudy.analyze headroom results/E1E2_short [--boot 500]
  python3 -m cstudy.analyze drl      results/E3_short_meas
  python3 -m cstudy.analyze codec    results/E4_short_lte
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from svs.grids import AWARE, ORACLE, UNAWARE

Q, E, MB = "qoe_min", "E_min", "MB_min"


# ---------------------------------------------------------------------------------------------- loading
def load(stage):
    """Returns (cfg table with means, dict block -> (X, names)), X [n_cfg, n_pairs, n_metrics]."""
    rows, blocks = [], {}
    for f in sorted(glob.glob(os.path.join(stage, "raw", "*.npz"))):
        z = np.load(f, allow_pickle=False)
        X, names, mets = z["X"], [str(x) for x in z["names"]], [str(x) for x in z["metrics"]]
        fam, mode, model = str(z["family"]), str(z["mode"]), str(z["model"])
        orc = z["oracle"]
        mean = np.nanmean(X, axis=1)
        for i, n in enumerate(names):
            r = dict(model=model, mode=mode, family=fam, name=n, oracle=str(orc[i]))
            r.update({m: float(mean[i, k]) for k, m in enumerate(mets)})
            r["n_nan"] = int(np.isnan(X[i, :, 0]).sum())
            rows.append(r)
        blocks[(model, mode, fam)] = (X, names, mets)
    return pd.DataFrame(rows), blocks


def pairs(stage):
    return pd.read_csv(os.path.join(stage, "pairs.csv"))


def form_of(stage):
    return json.load(open(os.path.join(stage, "args.json")))["form"]


# ---------------------------------------------------------------------------------------------- fronts
def pareto(pts):
    """Non-dominated (quality up, energy down) points sorted by increasing quality."""
    if len(pts) == 0:
        return np.zeros((0, 2))
    pts = pts[np.argsort(-pts[:, 0], kind="stable")]
    out, best = [], np.inf
    for q, e in pts:
        if e < best - 1e-12:
            out.append((q, e))
            best = e
    return np.array(out[::-1])


def front_energy(front, q):
    if len(front) == 0 or q > front[-1, 0] + 1e-9:
        return np.nan
    if q <= front[0, 0]:
        return float(front[0, 1])
    return float(np.interp(q, front[:, 0], front[:, 1]))


def headroom(base_pts, method_pts, unreachable=0.0):
    """Mean over the base front of 1 - E_method(q)/E_base; method points that cannot reach q count
    `unreachable` (0 when the method set contains the base set, as here)."""
    bf, mf = pareto(base_pts), pareto(method_pts)
    s = []
    for q, e in bf:
        em = front_energy(mf, q)
        s.append(unreachable if np.isnan(em) else 1 - em / e)
    return (float(np.nanmean(s)) if s else np.nan), bf, mf


def at_top(base_pts, method_pts):
    """Saving at the highest QoE the base set reaches."""
    bf, mf = pareto(base_pts), pareto(method_pts)
    q, e = bf[-1]
    em = front_energy(mf, q)
    return 1 - em / e


# ---------------------------------------------------------------------------------------------- headroom
def _sets(df, form):
    u = df.family.isin(UNAWARE[form]).to_numpy()
    a = u | df.family.isin(AWARE[form]).to_numpy()
    o = a | df.family.isin(ORACLE[form]).to_numpy()
    return u, a, o


def _stack(blocks, df_model, metric):
    """[n_cfg, n_pairs] of one metric in the row order of df_model."""
    out = []
    cache = {}
    for _, r in df_model.iterrows():
        key = (r.model, r["mode"], r.family)
        if key not in cache:
            X, names, mets = blocks[key]
            cache[key] = (X, {n: i for i, n in enumerate(names)}, mets)
        X, idx, mets = cache[key]
        out.append(X[idx[r["name"]], :, mets.index(metric)])
    return np.array(out, np.float64)


def headroom_table(stage, boot=500, pair_mask=None, label="", models=None, loaded=None):
    df, blocks = loaded or load(stage)
    form = form_of(stage)
    out = []
    rng = np.random.default_rng(0)
    for (model, mode), d in df.groupby(["model", "mode"], sort=False):
        if models is not None and model not in models:
            continue
        d = d.reset_index(drop=True)
        u, a, o = _sets(d, form)
        if not u.any():
            continue
        fam_pre = {p: d.family.str.startswith(p).to_numpy() for p in ("Ow-", "Ot-")}
        lam0 = d.name.str.contains(r"lam=0[,)]", regex=True).to_numpy()
        if form == "short":
            rung_only = (d.family == "DashletE").to_numpy()
            sched_only = d.family.isin(["BurstE", "SPARE"]).to_numpy() & lam0
        else:
            rung_only = (d.family == "MPC-E").to_numpy()
            sched_only = (d.family == "BurstMPC").to_numpy() & lam0
        qx, ex, mx = (_stack(blocks, d, m) for m in (Q, E, MB))
        if pair_mask is not None:
            qx, ex, mx = qx[:, pair_mask], ex[:, pair_mask], mx[:, pair_mask]
        npair = qx.shape[1]

        def stats(w):
            qm, em, mm = (x @ w / w.sum() for x in (qx, ex, mx))
            P = np.c_[qm, em]
            D = np.c_[qm, mm]
            r = {}
            r["H_aware"] = headroom(P[u], P[a])[0]
            r["H_oracle"] = headroom(P[u], P[o])[0] if o.sum() > a.sum() else np.nan
            r["Htop_aware"] = at_top(P[u], P[a])
            r["Htop_oracle"] = at_top(P[u], P[o]) if o.sum() > a.sum() else np.nan
            r["Hdata_aware"] = headroom(D[u], D[a])[0]
            r["Hdata_oracle"] = headroom(D[u], D[o])[0] if o.sum() > a.sum() else np.nan
            r["H_rungonly"] = headroom(P[u], P[u | rung_only])[0]
            r["H_schedonly"] = headroom(P[u], P[u | sched_only])[0]
            for pre, key in (("Ow-", "H_watchonly"), ("Ot-", "H_thronly")):
                mk = fam_pre[pre]
                if mk.any():
                    r[key] = headroom(P[u], P[a | mk])[0]
            return r

        point = stats(np.ones(npair))
        row = dict(model=model, mode=mode, variant=label, n_pairs=npair, **point)
        bf = pareto(np.c_[qx.mean(1), ex.mean(1)][u])
        row.update(q_top_unaware=bf[-1, 0], E_top_unaware=bf[-1, 1], n_front_unaware=len(bf))
        if boot:
            bs = []
            for _ in range(boot):
                w = np.bincount(rng.integers(npair, size=npair), minlength=npair).astype(float)
                bs.append(stats(w))
            bs = pd.DataFrame(bs)
            for k in point:
                lo, hi = np.nanpercentile(bs[k], [2.5, 97.5]) if bs[k].notna().any() else (np.nan, np.nan)
                row[k + "_lo"], row[k + "_hi"] = lo, hi
        out.append(row)
    return pd.DataFrame(out), df


def parse_grid(model):
    """'c0.2_T5_b1.29' -> (0.2, 5.0, 1.29)."""
    try:
        c, t, b = model.split("_")
        return float(c[1:]), float(t[1:]), float(b[1:])
    except ValueError:
        return None


# ---------------------------------------------------------------------------------------------- DRL (E3)
def drl_table(stage, boot=500):
    df, blocks = load(stage)
    form = form_of(stage)
    out = []
    rng = np.random.default_rng(1)
    for (model, mode), d in df.groupby(["model", "mode"], sort=False):
        d = d.reset_index(drop=True)
        rules = d.family.isin(UNAWARE[form] + AWARE[form]).to_numpy()
        drl = (d.family == "DRL").to_numpy()
        if not drl.any():
            continue
        qx, ex = _stack(blocks, d, Q), _stack(blocks, d, E)
        npair = qx.shape[1]

        def stats(w):
            P = np.c_[qx @ w / w.sum(), ex @ w / w.sum()]
            rf, df_ = pareto(P[rules]), pareto(P[drl])
            s_drl, cov = [], 0
            for q, e in rf:                     # DRL vs the rule envelope, where DRL reaches q
                em = front_energy(df_, q)
                if not np.isnan(em):
                    s_drl.append(1 - em / e)
                    cov += 1
            return {"S_drl_vs_rules": float(np.mean(s_drl)) if s_drl else np.nan,
                    "coverage": cov / len(rf),
                    "G_drl": headroom(P[rules], P[rules | drl])[0],
                    "qtop_rules": rf[-1, 0], "qtop_drl": df_[-1, 0]}

        point = stats(np.ones(npair))
        row = dict(model=model, n_pairs=npair, n_drl=int(drl.sum()), **point)
        bs = pd.DataFrame([stats(np.bincount(rng.integers(npair, size=npair), minlength=npair).astype(float))
                           for _ in range(boot)])
        for k in point:
            row[k + "_lo"], row[k + "_hi"] = np.nanpercentile(bs[k], [2.5, 97.5])
        out.append(row)
    return pd.DataFrame(out), df


# ---------------------------------------------------------------------------------------------- codec (E4)
def codec_table(stage, boot=300):
    df, blocks = load(stage)
    form = form_of(stage)
    fams = set(UNAWARE[form] + AWARE[form])
    out = []
    rng = np.random.default_rng(2)
    for model, d in df.groupby("model", sort=False):
        d = d[d.family.isin(fams)].reset_index(drop=True)
        m = {k: (d["mode"] == k).to_numpy() for k in ("h264", "av1", "both")}
        un = d.family.isin(UNAWARE[form]).to_numpy()
        qx, ex, mx = (_stack(blocks, d, k) for k in (Q, E, MB))
        av = _stack(blocks, d, "av1_frac")
        npair = qx.shape[1]
        cap_mode = "av1" if model.startswith("av1hw") else "h264"

        def stats(w):
            P = np.c_[qx @ w / w.sum(), ex @ w / w.sum()]
            D = np.c_[qx @ w / w.sum(), mx @ w / w.sum()]
            fixed = m["h264"] | m["av1"]
            r = {}
            r["av1_vs_h264"] = headroom(P[m["h264"]], P[m["av1"]], unreachable=np.nan)[0]
            r["both_vs_bestfixed"] = headroom(P[fixed], P[fixed | m["both"]])[0]
            r["both_vs_caprule"] = headroom(P[m[cap_mode]], P[m[cap_mode] | m["both"]])[0]
            r["greedy_vs_h264"] = headroom(P[m["h264"] & un], P[m["both"] & un], unreachable=np.nan)[0]
            r["data_both_vs_h264"] = headroom(D[m["h264"]], D[m["h264"] | m["both"]])[0]
            r["data_av1_vs_h264"] = headroom(D[m["h264"]], D[m["av1"]], unreachable=np.nan)[0]
            return r

        point = stats(np.ones(npair))
        row = dict(model=model, n_pairs=npair, **point)
        row["av1_frac_both_aware"] = float(np.nanmean(av[m["both"] & ~un].mean(1)))
        row["av1_frac_both_unaware"] = float(np.nanmean(av[m["both"] & un].mean(1)))
        if boot:
            bs = pd.DataFrame([stats(np.bincount(rng.integers(npair, size=npair), minlength=npair).astype(float))
                               for _ in range(boot)])
            for k in point:
                row[k + "_lo"], row[k + "_hi"] = np.nanpercentile(bs[k], [2.5, 97.5])
        out.append(row)
    return pd.DataFrame(out), df


# ---------------------------------------------------------------------------------------------- figures
def fig_map(table, form, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .models import BETAS, C_GRID, TAILS
    t = table.copy()
    g = t.model.map(parse_grid)
    t = t[g.notna()].copy()
    t[["c", "T", "b"]] = pd.DataFrame(g[g.notna()].tolist(), index=t.index)
    tails = [T for T, _ in TAILS]
    fig, axs = plt.subplots(2, len(BETAS), figsize=(7.2, 4.6), sharex=True, sharey=True, constrained_layout=True)
    vmax = max(0.05, float(np.nanmax(t[["H_aware", "H_oracle"]].to_numpy())))
    for i, key in enumerate(("H_aware", "H_oracle")):
        for j, b in enumerate(BETAS):
            ax = axs[i, j]
            M = np.full((len(tails), len(C_GRID)), np.nan)
            for _, r in t[np.isclose(t.b, b)].iterrows():
                M[tails.index(min(tails, key=lambda x: abs(x - r["T"]))), list(C_GRID).index(
                    min(C_GRID, key=lambda x: abs(x - r.c)))] = r[key]
            im = ax.imshow(M * 100, origin="lower", cmap="Blues", vmin=0, vmax=vmax * 100, aspect="auto")
            for y in range(M.shape[0]):
                for x in range(M.shape[1]):
                    if not np.isnan(M[y, x]):
                        ax.text(x, y, f"{M[y, x] * 100:.0f}", ha="center", va="center", fontsize=6.5,
                                color="white" if M[y, x] > 0.6 * vmax else "#1f1f1e")
            ax.set_xticks(range(len(C_GRID)), [f"{c:g}" for c in C_GRID], fontsize=6.5)
            ax.set_yticks(range(len(tails)), [f"{T:g}" for T in tails], fontsize=6.5)
            if i == 0:
                ax.set_title(f"active radio {b:g} W", fontsize=8)
            if j == 0:
                ax.set_ylabel(("energy-aware" if i == 0 else "clairvoyant") + "\nradio tail [s]", fontsize=7.5)
            if i == 1:
                ax.set_xlabel("decode energy [J/MB]", fontsize=7.5)
    cb = fig.colorbar(im, ax=axs, shrink=0.8)
    cb.set_label("energy saving at matched QoE [%]", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    fig.suptitle(f"{'Short-video feed' if form == 'short' else 'Long-form video'}: headroom over tuned "
                 f"energy-unaware players", fontsize=8.5)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_fronts(df, form, models, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = [m for m in models if m in set(df.model)]
    if not models:
        return
    fig, axs = plt.subplots(1, len(models), figsize=(1.9 * len(models) + 0.6, 2.4), constrained_layout=True)
    axs = np.atleast_1d(axs)
    for ax, mname in zip(axs, models):
        d = df[(df.model == mname) & (df["mode"] == "")]
        u, a, o = _sets(d.reset_index(drop=True), form)
        P = d[[Q, E]].to_numpy()
        for mask, col, lab in ((o, "#9b9a93", "+ clairvoyant"), (a, "#2a78d6", "+ energy-aware"),
                               (u, "#1f1f1e", "energy-unaware")):
            f = pareto(P[mask])
            ax.plot(f[:, 0], f[:, 1], "-o", ms=2, lw=1, color=col, label=lab)
        ax.set_title(mname, fontsize=8)
        ax.set_xlabel("QoE per minute", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
    axs[0].set_ylabel("device energy [J/min]", fontsize=7.5)
    axs[-1].legend(fontsize=6.5, frameon=False, loc="best")
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["headroom", "drl", "codec", "genre"])
    ap.add_argument("stage")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or os.path.join(a.stage, "summary")
    os.makedirs(out, exist_ok=True)
    figs = []
    if a.kind == "headroom":
        t, df = headroom_table(a.stage, a.boot)
        form = form_of(a.stage)
        if t.model.map(parse_grid).notna().sum() >= 10:
            figs.append(lambda: fig_map(t, form, os.path.join(out, f"map_{form}.png")))
        figs.append(lambda: fig_fronts(df, form, ["meas", "modern_lte", "modern_5g", "legacy_lte"],
                                       os.path.join(out, f"fronts_{form}.png")))
    elif a.kind == "drl":
        t, df = drl_table(a.stage, a.boot)
    elif a.kind == "codec":
        t, df = codec_table(a.stage, min(a.boot, 300))
    else:                                     # per-genre headroom of the anchor models (long form)
        pr = pairs(a.stage)
        ts = []
        ld = load(a.stage)
        for g in sorted(pr.genre.unique()):
            mask = (pr.sort_values("pid").genre == g).to_numpy()
            t1, df = headroom_table(a.stage, a.boot, mask, label=g, loaded=ld,
                                    models=["meas", "modern_lte", "modern_5g", "legacy_lte"])
            ts.append(t1)
        t = pd.concat(ts)
    df.to_csv(os.path.join(out, "cfg_means.csv"), index=False)
    t.to_csv(os.path.join(out, f"{a.kind}.csv"), index=False)
    for f in figs:                            # figures are optional (matplotlib may be missing)
        try:
            f()
        except Exception as e:                # noqa: BLE001
            print(f"figure skipped: {type(e).__name__}: {e}")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        cols = [c for c in t.columns if not c.endswith(("_lo", "_hi"))]
        print(t[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
