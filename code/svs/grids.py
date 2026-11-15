"""Policy grids of the paper-C study.  Each entry: (family, factory, clairvoyant?, buffer cap).

Energy-unaware families (no energy term, no radio awareness): short Fixed / Dashlet / PDAS; long Rate / BBA /
BOLA / MPC.  Energy-aware: short DashletE / BurstE / SPARE; long MPC-E / BurstMPC.  Clairvoyant references
(run on an OracleSession): short O-SPARE / O-Burst; long O-MPC / O-BurstMPC.
"""

from __future__ import annotations

import math

from .longform import LBBA, LBOLA, LMPC, LRate, OBurst
from .policies import BurstE, Dashlet, Fixed, Pdas, Spare

LAMS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4)
CAPS = (30.0, 60.0, 120.0)
BURST_LH = ((10.0, 60.0), (30.0, 60.0), (20.0, 120.0), (60.0, 120.0))
INF = math.inf

UNAWARE = {"short": ("Fixed", "Dashlet", "PDAS"), "long": ("Rate", "BBA", "BOLA", "MPC")}
AWARE = {"short": ("DashletE", "BurstE", "SPARE"), "long": ("MPC-E", "BurstMPC")}
ORACLE = {"short": ("O-SPARE", "O-Burst"), "long": ("O-MPC", "O-BurstMPC")}


def short_base(lams=LAMS):
    g = []
    for P in (1, 2, 3, 4):
        for B in (4, 8, 16, 30):
            g.append(("Fixed", lambda P=P, B=B: Fixed(P, B), False, INF))
    for th in (0.2, 0.4, 0.6, 0.8):
        for bm in (12, 24, 48):
            g.append(("Dashlet", lambda th=th, bm=bm: Dashlet(th, bm), False, INF))
    for q in (0.5, 0.7, 0.9):
        for bm in (8, 16, 32):
            g.append(("PDAS", lambda q=q, bm=bm: Pdas(q, bm), False, INF))
    for th in (0.4, 0.6):
        for lam in lams:
            g.append(("DashletE", lambda th=th, lam=lam: Dashlet(th, 24, lam), False, INF))
    for L in (4, 8):
        for H in (12, 24, 48):
            for lam in lams:
                g.append(("BurstE", lambda L=L, H=H, lam=lam: BurstE(L, H, lam), False, INF))
    for lam in lams:
        g.append(("SPARE", lambda lam=lam: Spare(lam), False, INF))
    return g


def short_oracle(lams=LAMS, kind="full"):
    """kind: 'full' (knows watch times and throughput), 'watch' or 'thr' (E2 decomposition)."""
    pre = {"full": "O", "watch": "Ow", "thr": "Ot"}[kind]
    g = []
    for nb in (1, 2, 4, 8):
        for lam in lams:
            g.append((f"{pre}-SPARE", lambda lam=lam, nb=nb: Spare(lam, nbar=nb, label=f"{pre}-SPARE"), kind, INF))
    for L in (4, 8):
        for H in (12, 24, 48):
            for lam in lams:
                g.append((f"{pre}-Burst", lambda L=L, H=H, lam=lam: OBurst(L, H, lam, label=f"{pre}-Burst"), kind, INF))
    return g


def long_base(lams=LAMS, caps=CAPS, burst=BURST_LH):
    g = []
    for cap in caps:
        for s in (0.6, 0.8, 0.9, 1.0):
            g.append(("Rate", lambda s=s, cap=cap: LRate(s, cap), False, cap))
        for r, c in ((5, 10), (10, 20), (5, 30)):
            g.append(("BBA", lambda r=r, c=c, cap=cap: LBBA(r, c, cap), False, cap))
        for gp in (1, 2, 5, 10):
            g.append(("BOLA", lambda gp=gp, cap=cap: LBOLA(gp, cap), False, cap))
        g.append(("MPC", lambda cap=cap: LMPC(0.0, cap), False, cap))
        for lam in lams:
            if lam > 0:
                g.append(("MPC-E", lambda lam=lam, cap=cap: LMPC(lam, cap), False, cap))
    for L, H in burst:
        for lam in lams:
            g.append(("BurstMPC", lambda lam=lam, L=L, H=H: LMPC(lam, H, burst=(L, H)), False, H))
    return g


def long_oracle(lams=LAMS, caps=CAPS, burst=BURST_LH):
    g = []
    for cap in caps:
        for lam in lams:
            g.append(("O-MPC", lambda lam=lam, cap=cap: LMPC(lam, cap, label="O-MPC"), "full", cap))
    for L, H in burst:
        for lam in lams:
            g.append(("O-BurstMPC", lambda lam=lam, L=L, H=H: LMPC(lam, H, burst=(L, H), label="O-BurstMPC"),
                      "full", H))
    return g


def configs(form: str, sets=("base", "oracle")):
    """sets: 'base' (unaware + aware), 'oracle' (clairvoyant), 'decomp' (short: watch-only and throughput-only
    clairvoyant variants)."""
    out = []
    if "base" in sets:
        out += short_base() if form == "short" else long_base()
    if "oracle" in sets:
        out += short_oracle() if form == "short" else long_oracle()
    if "decomp" in sets and form == "short":
        out += short_oracle(kind="watch") + short_oracle(kind="thr")
    return out
