#!/usr/bin/env python3
"""Analyse one svs_phone.py measurement (samples.csv, events.csv, meta.json).

  python3 analyze_phone.py <folder or zip> --out <dir>

Outputs <dir>/phone_summary.json and <dir>/phone_measurement.png:
  baseline power, tail time / tail power after a small transfer, active power vs throughput,
  energy of trickle vs batch downloading, and of the replayed SPARE / BurstE schedules,
  compared with the radio model (literature LTE parameters and the parameters fitted here).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from svs.model import RADIOS, Radio, radio_energy  # noqa: E402


def load(path):
    p = Path(path)
    if p.suffix == ".zip":
        tmp = Path(tempfile.mkdtemp())
        zipfile.ZipFile(p).extractall(tmp)
        p = tmp
    s = pd.read_csv(p / "samples.csv")
    e = pd.read_csv(p / "events.csv")
    m = json.load(open(p / "meta.json"))
    s = s.dropna(subset=["current"])
    cur = pd.to_numeric(s.current, errors="coerce")
    vol = pd.to_numeric(s.voltage, errors="coerce")
    # Termux:API / BatteryManager: microamperes, negative while discharging (some phones report mA)
    scale_i = 1e-6 if cur.abs().median() > 5000 else 1e-3
    # sign differs between vendors (Huawei: negative, Xiaomi: positive while discharging); runs are unplugged
    i_a = cur.abs() * scale_i
    if vol.notna().any():
        v = vol.where(vol > 0)
        v_v = v / (1e6 if v.median() > 1e5 else 1e3)
        v_v = v_v.interpolate().bfill().ffill()
    else:
        v_v = pd.Series(3.85, index=s.index)
    s = pd.DataFrame({"t": s.t.astype(float), "I": i_a, "V": v_v})
    s["P"] = s.I * s.V
    return s, e, m


def update_interval(s):
    ch = s.P.diff().fillna(1).ne(0)
    tt = s.t[ch].diff().dropna()
    return float(tt.median()) if len(tt) else float("nan")


def energy(s, t0, t1, base):
    w = s[(s.t >= t0) & (s.t < t1)]
    if len(w) < 2:
        return float("nan")
    tt = np.append(w.t.to_numpy(), t1)
    dt = np.diff(tt)
    return float(np.sum((w.P.to_numpy() - base) * dt))


def marks(e, phase):
    d = e[(e.phase == phase) & e.t_start.isna()]
    return dict(zip(d.label, d.t_req.astype(float)))


def transfers(e, phase, label_prefix=""):
    d = e[(e.phase == phase) & e.t_start.notna()]
    if label_prefix:
        d = d[d.label.astype(str).str.startswith(label_prefix)]
    return d.astype({"t_req": float, "t_start": float, "t_end": float, "bytes": float})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    s, e, m = load(a.path)
    out = {"device": f"{m.get('brand')} {m.get('model')} Android {m.get('android')}", "net": m.get("net"),
           "minutes": m.get("minutes"), "n_samples": len(s), "update_interval_s": update_interval(s)}

    # baseline: P1 (skip first 10 s) and P6
    b1, b6 = marks(e, "P1_baseline"), marks(e, "P6_baseline")
    w1 = s[(s.t >= b1.get("start", 0) + 10) & (s.t < b1.get("end", 0))]
    base = float(w1.P.median()) if len(w1) else float(s.P.quantile(0.1))
    out["baseline_W"] = base
    if b6:
        w6 = s[(s.t >= b6["start"] + 5) & (s.t < b6.get("end", b6["start"] + 60))]
        out["baseline_end_W"] = float(w6.P.median()) if len(w6) else None

    # tail: excess power after each 200 KB transfer, aligned at its end
    tr = transfers(e, "P2_tail")
    grid = np.arange(0, 40, 1.0)
    curves = []
    for _, r in tr.iterrows():
        w = s[(s.t >= r.t_end) & (s.t < r.t_end + 40)]
        if len(w) > 5:
            curves.append(np.interp(grid + 0.5, w.t - r.t_end, w.P - base))
    if curves:
        c = np.median(np.array(curves), axis=0)
        late = float(np.median(c[30:]))                        # residual excess at 30-40 s (should be ~0)
        thr = late + 0.3 * (c[:3].mean() - late)
        below = np.where(c < thr)[0]
        t_tail = float(grid[below[0]]) if len(below) else float("nan")
        out["tail_curve_W"] = [round(x, 4) for x in c.tolist()]
        out["tail_time_s"] = t_tail
        out["tail_power_W"] = float(np.mean(c[1:max(int(t_tail), 2)]) - late) if np.isfinite(t_tail) else None
        out["tail_residual_W"] = late

    # active power vs throughput (20 MB transfers)
    tb = transfers(e, "P3_bulk")
    act = []
    for _, r in tb.iterrows():
        w = s[(s.t >= r.t_start + 0.5) & (s.t < r.t_end)]
        if len(w) > 2 and r.t_end > r.t_start:
            act.append((r.bytes * 8e-6 / (r.t_end - r.t_start), float(w.P.mean() - base)))
    out["active_points_Mbps_W"] = act

    # fitted LTE radio (beta from the active points with the literature alpha; tail from P2)
    lit = RADIOS["lte"]
    if act and out.get("tail_power_W") and np.isfinite(out.get("tail_time_s", np.nan)):
        beta = float(np.median([p - lit.alpha * thr for thr, p in act]))
        fit = Radio("lte_measured", 0.0, lit.t_pro, out["tail_power_W"], beta, lit.alpha, out["tail_time_s"],
                    out["tail_power_W"])
        out["fitted_radio"] = fit.__dict__
    else:
        fit = None

    def window_stats(phase, start_label, end_label, prefix):
        mk = marks(e, phase)
        t0, t1 = mk.get(start_label), mk.get(end_label)
        if t0 is None or t1 is None:
            return None
        trs = transfers(e, phase, prefix)
        trs = trs[(trs.t_req >= t0) & (trs.t_req < t1)]
        tl = [(r.t_req - t0, r.t_start - t0, r.t_end - t0, r.bytes, False) for _, r in trs.iterrows()]
        res = {"measured_excess_J": energy(s, t0, t1, base), "MB": float(trs.bytes.sum() / 1e6),
               "n": len(trs), "seconds": t1 - t0}
        for name, rad in (("lit", lit), ("fit", fit)):
            if rad is None or not tl:
                continue
            first_promo = [(x[0], x[1], x[2], x[3], i == 0 or (x[0] - tl[i - 1][2]) > rad.t_tail)
                           for i, x in enumerate(tl)]
            res[f"model_{name}_J"] = radio_energy(first_promo, t1 - t0, rad) - rad.p_idle * (t1 - t0)
        return res

    pat = {}
    for rep in (0, 1):
        for kind in ("trickle", "batch"):
            pat[f"{kind}{rep}"] = window_stats("P4_pattern", f"{kind}_rep{rep}_start", f"{kind}_rep{rep}_end", kind)
    out["pattern"] = pat
    rep = {}
    for key in sorted({str(l).rsplit("_", 1)[0] for l in e[e.phase == "P5_replay"].label if str(l).endswith("_start")}):
        rep[key] = window_stats("P5_replay", f"{key}_start", f"{key}_end", key)
    out["replay"] = rep

    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(a.out) / "phone_summary.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "tail_curve_W"}, indent=1, default=float))
    plot(out, Path(a.out) / "phone_measurement.png")


def plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    if "tail_curve_W" in out:
        c = np.array(out["tail_curve_W"])
        ax[0].plot(np.arange(len(c)) + 0.5, c, color="#eb6834")
        if np.isfinite(out.get("tail_time_s", np.nan)):
            ax[0].axvline(out["tail_time_s"], color="#333", ls="--", lw=1)
        ax[0].set_xlabel("Time after a 200 KB transfer (s)")
        ax[0].set_ylabel("Power above idle (W)")
        ax[0].set_title("(a) Radio tail, median of 8 trials", fontsize=10)
    labels, meas, mod = [], [], []
    for k, v in out.get("pattern", {}).items():
        if v:
            labels.append(k); meas.append(v["measured_excess_J"]); mod.append(v.get("model_fit_J", np.nan))
    x = np.arange(len(labels))
    ax[1].bar(x - 0.2, meas, 0.4, color="#2a78d6", label="measured")
    ax[1].bar(x + 0.2, mod, 0.4, color="#c9c4b8", label="model (fitted)")
    ax[1].set_xticks(x, labels, fontsize=8)
    ax[1].set_ylabel("Energy above idle (J)")
    ax[1].set_title("(b) Same 18 MB: trickle vs batch", fontsize=10)
    ax[1].legend(fontsize=7, frameon=False)
    labels, meas, mod = [], [], []
    for k, v in out.get("replay", {}).items():
        if v:
            tr, pol = k.split("|")
            labels.append(f"{pol}\n{tr[:10]}"); meas.append(v["measured_excess_J"]); mod.append(v.get("model_fit_J", np.nan))
    x = np.arange(len(labels))
    ax[2].bar(x - 0.2, meas, 0.4, color="#1baf7a", label="measured")
    ax[2].bar(x + 0.2, mod, 0.4, color="#c9c4b8", label="model (fitted)")
    ax[2].set_xticks(x, labels, fontsize=7)
    ax[2].set_title("(c) Replayed download schedules (340 s)", fontsize=10)
    ax[2].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)


if __name__ == "__main__":
    main()
