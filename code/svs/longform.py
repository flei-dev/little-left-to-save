"""Long-form (VoD) streaming on the same simulator, energy model and QoE as the short-video feed.

A long-form session is one video of the content library played from start to end (300-s window, user watches
it all), with the standard player buffer cap.  The rule families are the classic ABR algorithms:
  Rate     : throughput rule (highest bitrate <= safety x harmonic-mean throughput)
  BBA      : buffer-based (reservoir / cushion), Huang et al. SIGCOMM'14
  BOLA     : BOLA-BASIC, Spiteri et al. INFOCOM'16 (utility ln(size / smallest size))
  MPC      : RobustMPC, Yin et al. SIGCOMM'15 (5-chunk horizon, SQoE-III objective)
  MPC-E    : MPC whose objective subtracts lam x predicted energy of the plan (GreenABR / MPC-E idea)
  BurstMPC : on/off downloading (EnDASH / ON-OFF style): idle until the buffer falls to L, then download
             back-to-back up to H, rungs from MPC-E (lam may be 0)
Every family runs with buffer caps of 30 / 60 / 120 s (the cap is a standard player knob).

  python3 -m svs.longform --library content_library_x264.csv --chunk 4 --out long_4s.pkl
"""

from __future__ import annotations

import argparse
import itertools
import math
import pickle

import numpy as np
import pandas as pd

from .model import BITRATES, PX, RTT, QOE_UNIT_S, W_Q, W_RB, W_SM
from .policies import Policy, best_at_top

GENRES = ("animation", "film", "documentary", "sports", "talk", "gaming", "music", "travel", "screen")
WIN_S = 300.0          # session length [s]
MAX_WIN = 2            # windows per video


def video_folds(genre_of: dict, k: int = 5):
    """Same genre-stratified folds as tmm_abr.library.video_folds."""
    folds = [[] for _ in range(k)]
    i = 0
    for g in GENRES:
        for v in sorted(n for n, x in genre_of.items() if x == g):
            folds[i % k].append(v)
            i += 1
    return folds


def load_lib(path, kbps=BITRATES):
    from .content import load_library
    return load_library(path)


def build(library_csv: str, chunk_s: float):
    from svs.content import Clip, load_library
    lib = load_library(library_csv)
    folds = video_folds({n: v[0] for n, v in lib.items()})
    test_videos = set(folds[0] + folds[1])                  # held out from DRL training (E3)
    L = int(round(WIN_S / chunk_s))
    clips, sessions, split = {}, [], {"dev": [], "test": [], "main": []}
    cid = 0
    for name in sorted(lib):
        genre, sz, q, qs = lib[name]
        n = sz.shape[0]
        starts = list(range(0, n - L + 1, L))[:MAX_WIN] or [0]
        for s0 in starts:
            m = min(L, n - s0)
            dur = m * chunk_s
            S = np.ones(int(round(dur)) + 1)
            cumS = np.arange(len(S), dtype=float)
            clips[cid] = Clip(cid, genre, name, sz[s0:s0 + m].copy(), q[s0:s0 + m].copy(), qs[s0:s0 + m].copy(),
                              S, cumS, chunk_s)
            i = len(sessions)
            sessions.append((cid, [(cid, dur)]))
            split["main"].append(i)
            split["test" if name in test_videos else "dev"].append(i)
            cid += 1
    return clips, sessions, split


# ---------------------------------------------------------------------------------------------- policies
def _levels(S, j):
    """Distinct bitrate levels of the allowed rungs (lists of rungs per level, increasing bitrate)."""
    rs, kb = S.rungs(j), S.kb[j]
    out = []
    for r in rs:
        if out and kb[out[-1][0]] == kb[r]:
            out[-1].append(r)
        else:
            out.append([r])
    return out


class _Long(Policy):
    """Common part: plays out when everything is downloaded; optional on/off (burst) schedule."""

    def __init__(self, cap: float, burst=None):
        self.cap, self.burst = cap, burst

    def reset(self, S):
        self.on = True

    def act(self, S):
        k = S.next_k(S.cur)
        if k is None:
            return ("sleep", S.buf_ahead(S.cur) + 0.01)
        b = S.buf_ahead(S.cur)
        if self.burst is not None:
            L, H = self.burst
            if not self.on:
                if b <= L + 1e-9:
                    self.on = True
                else:
                    return ("sleep", b - L + 1e-3)
            if b + S.chunk_s > H + 1e-9:
                self.on = False
                return ("sleep", max(b - L, 0.0) + 1e-3)
        elif b + S.chunk_s > self.cap + 1e-9:          # buffer full: wait until the next chunk fits
            return ("sleep", b + S.chunk_s - self.cap + 1e-6)
        return ("dl", S.cur, k, self.rung(S, k, b))


class LRate(_Long):
    def __init__(self, safety: float, cap: float):
        super().__init__(cap)
        self.safety = safety
        self.name = f"Rate(s={safety:g},B={cap:g})"

    def rung(self, S, k, b):
        rs = S.rungs(S.cur)
        if not S.thr:
            return rs[0]
        hm = S.thr_hm()
        ok = [r for r in rs if S.kb[S.cur][r] / 1000.0 <= self.safety * hm]
        return best_at_top(S, S.cur, k, ok) if ok else rs[0]


class LBBA(_Long):
    def __init__(self, res: float, cush: float, cap: float):
        super().__init__(cap)
        self.res, self.cush = res, cush
        self.name = f"BBA(r={res:g},c={cush:g},B={cap:g})"

    def rung(self, S, k, b):
        lv = _levels(S, S.cur)
        if b < self.res:
            i = 0
        elif b >= self.res + self.cush:
            i = len(lv) - 1
        else:
            i = int((b - self.res) / self.cush * (len(lv) - 1))
        c = lv[i]
        return c[0] if len(c) == 1 else max(c, key=lambda r: S.C[S.cur].q[k, r])


class LBOLA(_Long):
    def __init__(self, gp: float, cap: float):
        super().__init__(cap)
        self.gp = gp
        self.name = f"BOLA(gp={gp:g},B={cap:g})"

    def rung(self, S, k, b):
        rs = np.array(S.rungs(S.cur))
        s = S.C[S.cur].sizes[k, rs]
        v = np.log(s / s.min())
        qmax = self.cap / S.chunk_s
        V = (qmax - 1.0) / (v.max() + self.gp)
        q = b / S.chunk_s
        return int(rs[int(np.argmax((V * (v + self.gp) - q) / s))])


_PLANS = {}


def _plans(S, j, h):
    """Rung plans over the horizon; with two codecs, plans keep one codec over the horizon."""
    rs = tuple(S.rungs(j))
    cod = S.C[j].codec
    key = (rs, None if cod is None else tuple(int(cod[r]) for r in rs), h)
    if key not in _PLANS:
        if cod is None:
            groups = [list(rs)]
        else:
            groups = [[r for r in rs if int(cod[r]) == c] for c in sorted({int(cod[r]) for r in rs})]
        _PLANS[key] = np.vstack([np.array(list(itertools.product(g, repeat=h)), int) for g in groups])
    return _PLANS[key]


_PRE = {}


def clear_cache():
    _PRE.clear()


def _pre(S, j, k, h):
    """Per (clip, chunk, horizon) plan tables; reused by every MPC configuration run on the same session."""
    c = S.C[j]
    P = _plans(S, j, h)
    key = (id(c), k, h, id(P))
    v = _PRE.get(key)
    if v is None:
        if len(_PRE) > 400:
            _PRE.clear()
        rows = k + np.arange(h)
        sizes = c.sizes[rows[None, :], P]
        q = c.q[rows[None, :], P]
        intra = np.abs(np.diff(q, axis=1)).sum(1) if h > 1 else np.zeros(len(P))
        v = (P, sizes, q[:, 0].copy(), W_Q * q.sum(1) * S.chunk_s / QOE_UNIT_S - W_SM * intra,
             sizes.sum(1) / 1e6, S.pdec[j][P].sum(1) * S.chunk_s)
        _PRE[key] = v
    return v


class LMPC(_Long):
    """RobustMPC with an optional energy term; on an OracleSession the plan is simulated on the true trace."""

    def __init__(self, lam: float, cap: float, horizon: int = 5, burst=None, label: str | None = None):
        super().__init__(cap, burst)
        self.lam, self.h = lam, horizon
        fam = label or ("BurstMPC" if burst else ("MPC" if lam == 0 else "MPC-E"))
        knob = f"L={burst[0]:g},H={burst[1]:g}" if burst else f"B={cap:g}"
        self.name = f"{fam}(lam={lam:g},{knob})"

    def rung(self, S, k, b):
        j, c = S.cur, S.C[S.cur]
        if not S.thr and not S.knows_thr:
            return S.rungs(j)[0]
        h = min(self.h, c.n - k)
        P, sizes, q0, base, mb, edec = _pre(S, j, k, h)
        cs = S.chunk_s
        cap = self.burst[1] if self.burst else self.cap
        n = len(P)
        buf = np.full(n, b)
        rebuf = np.zeros(n)
        tdl = np.zeros(n)
        if S.knows_thr:
            t = np.full(n, S.t)
        else:
            thr = S.thr_robust()
        for i in range(h):
            if i > 0 and cap < math.inf:          # the player waits while the buffer is full
                wait = np.maximum(buf + cs - cap, 0.0)
                buf -= wait
                if S.knows_thr:
                    t += wait
            if S.knows_thr:
                de = S.trace.finish_vec(t + RTT, sizes[:, i])
                dl = de - t
                t = de
            else:
                dl = sizes[:, i] * (8e-6 / thr) + RTT
            rebuf += np.maximum(dl - buf, 0.0)
            buf = np.maximum(buf - dl, 0.0) + cs
            tdl += dl
        obj = base - W_RB * rebuf
        if k > 0:
            obj -= W_SM * np.abs(q0 - c.q[k - 1, S.dl[j][k - 1]])
        if self.lam > 0:
            rad = S.radio_belief
            e = rad.beta * tdl + (rad.alpha * 8.0 + S.device.c) * mb + edec + S.device.p_stall * rebuf
            obj = obj - self.lam * e
        return int(P[int(np.argmax(obj)), 0])


class OBurst(Policy):
    """Clairvoyant on/off preloader for the feed (run on an OracleSession): when the current video is about
    to run dry or the next video has no first chunk, wake the radio and fetch, in deadline order, every chunk
    that will be watched within the next H s of playback; then sleep.  Rungs: quality / energy rule."""

    def __init__(self, L: float, H: float, lam: float, label: str = "O-Burst"):
        self.L, self.H, self.lam = L, H, lam
        self.name = f"{label}(L={L:g},H={H:g},lam={lam:g})"

    def reset(self, S):
        self.on = False

    def _todo(self, S):
        out = []
        for j in S.window():
            k = S.next_k(j)
            if k is not None and S.p_watch(j, k) > 0.5:
                t = S.time_to_need(j, k)
                if t < self.H:
                    out.append((t, j, k))
        return out

    def act(self, S):
        from .policies import rung_energy
        if not self.on:
            k = S.next_k(S.cur)
            low = k is not None and S.p_watch(S.cur, k) > 0.5 and S.buf_ahead(S.cur) < self.L
            nxt = S.cur + 1 < S.nv and not S.dl[S.cur + 1]
            self.on = low or nxt
        if self.on:
            todo = self._todo(S)
            if todo:
                _, j, k = min(todo)
                return ("dl", j, k, rung_energy(S, j, k, self.lam, pweight=True))
            self.on = False
        return ("sleep", 0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", required=True)
    ap.add_argument("--chunk", type=float, default=4.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from svs import longform as mod
    clips, sessions, split = mod.build(a.library, a.chunk)
    with open(a.out, "wb") as f:
        pickle.dump({"clips": clips, "sessions": sessions, "splits": split, "form": "long"}, f)
    print(f"{len(clips)} long-form windows from {len({c.src for c in clips.values()})} videos; "
          f"dev {len(split['dev'])}, test {len(split['test'])}")


if __name__ == "__main__":
    main()
