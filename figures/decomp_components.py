"""Exact split of headroom H_aware into energy components (radio, base playback, per-MB decoding, stall).
Uses the same Pareto fronts and linear interpolation as cstudy.analyze.headroom (point estimate)."""
import sys, os, numpy as np, pandas as pd
ROOT = sys.argv[1]
sys.path.insert(0, os.path.join(ROOT, "code"))
from cstudy.analyze import UNAWARE, AWARE, ORACLE
COMP = ["Eradio_min", "Ebase_min", "Ec_min", "Edec_min", "Estall_min"]

def front_idx(q, e):
    order = np.argsort(-q, kind="stable"); out = []; best = np.inf
    for i in order:
        if e[i] < best - 1e-12:
            out.append(i); best = e[i]
    return np.array(out[::-1])

def interp(fi, q, X, qq):
    fq = q[fi]
    if qq > fq[-1] + 1e-9: return None
    if qq <= fq[0]: return X[fi[0]]
    j = np.searchsorted(fq, qq); w = (qq - fq[j-1]) / (fq[j] - fq[j-1])
    return (1-w) * X[fi[j-1]] + w * X[fi[j]]

rows = []
for form, stage in (("short", "E1E2_short"), ("long", "E1E2_long")):
    d = pd.read_csv(os.path.join(ROOT, "results", stage, "summary", "cfg_means.csv"))
    d = d[d["mode"].isna() | (d["mode"] == "")]
    for m in ["legacy_lte", "modern_lte", "modern_5g", "meas"]:
        x = d[d.model == m].reset_index(drop=True)
        u = x.family.isin(UNAWARE[form]).to_numpy(); a = u | x.family.isin(AWARE[form]).to_numpy()
        q = x.qoe_min.to_numpy(); e = x.E_min.to_numpy()
        X = x[COMP + ["E_min", "MB_min", "n_promo"]].to_numpy()
        iu = np.where(u)[0]; ia = np.where(a)[0]
        fu = iu[front_idx(q[iu], e[iu])]; fa = ia[front_idx(q[ia], e[ia])]
        parts = []
        for k in fu:
            xa = interp(fa, q, X, q[k]); xu = X[k]
            parts.append(np.r_[(xu[:5] - xa[:5]) / xu[5], (xu[5]-xa[5])/xu[5], xu[5], xu[6], xa[6], xu[7], xa[7], xu[:5]])
        P = np.array(parts).mean(0)
        r = dict(form=form, model=m, H_total=P[5])
        r.update({c.replace("_min", "") + "_share": P[i] for i, c in enumerate(COMP)})
        r.update(E_U=P[6], MB_U=P[7], MB_A=P[8], promo_U=P[9], promo_A=P[10])
        r.update({"U_" + c.replace("_min", ""): P[11+i] for i, c in enumerate(COMP)})
        rows.append(r)
out = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(out.round(4).to_string())
out.to_csv(sys.argv[2], index=False)
