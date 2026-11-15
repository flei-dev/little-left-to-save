"""Trace-driven short-video session simulator (swipe feed, sequential chunk downloads, one link).

The player sees the current video and the next WINDOW-1 videos of the feed: their manifests (per-chunk size and
VMAF of every rung) and their population retention curves, but not how long this user will watch.
At every decision the policy either downloads the next chunk of one video in the window at some rung,
or sleeps.  Playback and the user's swipes advance while chunks download.
"""

from __future__ import annotations

import math

import numpy as np

from .model import BITRATES, PX, QOE_UNIT_S, RTT, WINDOW, W_Q, W_RB, W_RBN, W_SM, W_SMJ, Device, Radio, radio_energy

EPS = 1e-9


class Trace:
    def __init__(self, times, mbps, offset: float = 0.0):
        t = np.asarray(times, float)
        b = np.maximum(np.asarray(mbps, float), 0.01)
        dt = np.diff(t)
        dt = np.append(dt, dt.mean() if len(dt) else 1.0)
        self.edges = np.concatenate([[0.0], np.cumsum(dt)])
        self.cum = np.concatenate([[0.0], np.cumsum(b * dt)])          # Mbit
        self.period, self.total = float(self.edges[-1]), float(self.cum[-1])
        self.offset = offset % self.period

    def _bits(self, t):
        k, r = divmod(t + self.offset, self.period)
        return k * self.total + np.interp(r, self.edges, self.cum)

    def finish(self, t0: float, nbytes: float) -> float:
        target = self._bits(t0) + nbytes * 8e-6
        k, r = divmod(target, self.total)
        return k * self.period + float(np.interp(r, self.cum, self.edges)) - self.offset

    def finish_vec(self, t0, nbytes):
        """Vectorized finish(): t0 and nbytes are arrays of the same shape."""
        k0, r0 = np.divmod(np.asarray(t0, float) + self.offset, self.period)
        target = k0 * self.total + np.interp(r0, self.edges, self.cum) + np.asarray(nbytes, float) * 8e-6
        k, r = np.divmod(target, self.total)
        return k * self.period + np.interp(r, self.cum, self.edges) - self.offset

    def mean_mbps(self, t0: float, h: float) -> float:
        return float(self._bits(t0 + h) - self._bits(t0)) / h


_DEF_KB = np.array(BITRATES, float)
_DEF_CODEC = np.zeros(len(BITRATES), int)
_DEF_PX = np.array(PX, float)


class Session:
    knows_watch = False          # clairvoyant about the user's watch times (OracleSession)
    knows_thr = False            # clairvoyant about the throughput of the next seconds

    @property
    def is_oracle(self):
        return self.knows_watch or self.knows_thr

    def __init__(self, clips, items, trace: Trace, radio: Radio, device: Device, radio_belief: Radio | None = None,
                 buf_cap: float = math.inf, codecs=None):
        self.C = [clips[v] for v, _ in items]
        self.W = [min(w, c.dur) for (_, w), c in zip(items, self.C)]
        self.nv = len(items)
        self.chunk_s = float(getattr(self.C[0], "chunk_s", 4.0))
        self.buf_cap = buf_cap                         # long-form players stop downloading at this buffer level
        # per-video rung description: nominal kbps, codec, extra decoding power; allowed rungs sorted by kbps
        self.kb, self.pdec, self.order = [], [], []
        for c in self.C:
            kb = c.kbps if getattr(c, "kbps", None) is not None else _DEF_KB
            cod = c.codec if getattr(c, "codec", None) is not None else _DEF_CODEC
            px = c.px if getattr(c, "px", None) is not None else _DEF_PX
            scale = px if getattr(device, "dec_scale", "prop") == "prop" else np.ones(len(kb))
            pd = np.asarray(getattr(device, "p_dec", (0.0, 0.0)), float)[cod] * scale
            ok = [r for r in range(len(kb)) if codecs is None or int(cod[r]) in codecs]
            self.kb.append(kb)
            self.pdec.append(pd)
            self.order.append(sorted(ok, key=lambda r: (kb[r], r)))
        self.trace, self.radio, self.device = trace, radio, device
        self.radio_belief = radio_belief or radio      # what the policies assume (may differ from the truth)
        self.t, self.cur, self.pos = 0.0, 0, 0.0
        self.dl = [[] for _ in range(self.nv)]
        self.watched = [np.zeros(c.n) for c in self.C]
        self.stall, self.stall_n, self.stalling = 0.0, 0, False
        self.transfers, self.last_end = [], None
        self.thr, self.err, self._pred = [], [], None
        self.bytes_down = 0.0
        self.aborted = 0.0
        self.done = False

    # ------------------------------------------------------------------ playback
    def advance(self, t_to: float, abort_video: int | None = None):
        """Play until t_to.  If abort_video is given, return early as soon as the user swipes past it."""
        while self.t < t_to - EPS and not self.done:
            v = self.cur
            if self.pos >= self.W[v] - EPS:              # swipe (or finished) -> next video
                self.cur += 1
                self.pos = 0.0
                self.stalling = False
                if self.cur >= self.nv:
                    self.done = True
                if abort_video is not None and self.cur > abort_video:
                    return
                continue
            k = int(self.pos / self.chunk_s + EPS)
            if k >= len(self.dl[v]):                      # needed chunk missing -> stall until t_to
                if not self.stalling:
                    self.stall_n += 1
                    self.stalling = True
                self.stall += t_to - self.t
                self.t = t_to
                return
            self.stalling = False
            end = min((k + 1) * self.chunk_s, self.W[v])
            dt = min(end - self.pos, t_to - self.t)
            self.watched[v][k] += dt
            self.pos += dt
            self.t += dt

    def fetch(self, j: int, r: int):
        k = len(self.dl[j])
        nb = float(self.C[j].sizes[k, r])
        promo = self.last_end is None or (self.t - self.last_end) > self.radio.t_tail
        t_req = self.t
        ds = t_req + (self.radio.t_pro if promo else 0.0) + RTT
        de = self.trace.finish(ds, nb)
        self._pred = self.thr_hm()
        self.advance(de, abort_video=j)
        if self.cur > j and self.t < de - EPS:           # the user swiped past this video: abort the download
            t_ab = max(self.t, ds)
            part = max(self.trace._bits(t_ab) - self.trace._bits(ds), 0.0) / 8e-6 if t_ab > ds else 0.0
            self.transfers.append((t_req, min(ds, t_ab), t_ab, part, promo))
            self.last_end = t_ab
            self.bytes_down += part
            self.aborted += part
            self.t = max(self.t, t_ab)
            return
        self.t = max(self.t, de)
        self.dl[j].append(r)
        thr = nb * 8e-6 / max(de - ds, 1e-6)
        if self.thr:
            self.err.append(abs(self._pred - thr) / thr)
        self.thr.append(thr)
        self.transfers.append((t_req, ds, de, nb, promo))
        self.last_end = de
        self.bytes_down += nb

    def run(self, policy, t_max: float = 4 * 3600.0):
        policy.reset(self)
        while not self.done and self.t < t_max:
            a = policy.act(self)
            if a[0] == "dl":                            # ("dl", video, chunk, rung)
                assert a[2] == len(self.dl[a[1]]), "chunks of a video are fetched in order"
                if a[1] == self.cur and self.buf_cap < math.inf:
                    wait = self.buf_ahead(a[1]) + self.chunk_s - self.buf_cap
                    if wait > EPS:                      # buffer full: wait until the chunk fits
                        self.advance(self.t + wait)
                        continue
                self.fetch(a[1], a[3])
            else:
                self.advance(self.t + max(a[1], 0.01))
        return self.metrics()

    # ------------------------------------------------------------------ helpers for policies
    def window(self):
        return range(self.cur, min(self.cur + WINDOW, self.nv))

    def rungs(self, j):
        """Allowed rungs of video j, by increasing nominal bitrate."""
        return self.order[j]

    def next_k(self, j):
        k = len(self.dl[j])
        return k if k < self.C[j].n else None

    def buf_ahead(self, j):
        b = len(self.dl[j]) * self.chunk_s
        return max(b - self.pos, 0.0) if j == self.cur else b

    def p_watch(self, j, k):
        c = self.C[j]
        if j == self.cur:
            if k * self.chunk_s <= self.pos + EPS:
                return 1.0
            s0 = c.surv(self.pos)
            return c.surv(k * self.chunk_s) / s0 if s0 > 0 else 1.0
        return c.surv(k * self.chunk_s)

    def exp_watch(self, j, k):
        """Expected seconds of chunk k of video j that will be watched."""
        c = self.C[j]
        a, b = k * self.chunk_s, (k + 1) * self.chunk_s
        if j == self.cur:
            s0 = c.surv(self.pos)
            a = max(a, self.pos)
            return max(c.integ(b) - c.integ(a), 0.0) / s0 if s0 > 0 else b - a
        return c.integ(b) - c.integ(a)

    def exp_remaining(self, j):
        c = self.C[j]
        if j == self.cur:
            s0 = c.surv(self.pos)
            return (c.integ(c.dur) - c.integ(self.pos)) / s0 if s0 > 0 else 0.0
        return c.integ(c.dur)

    def time_to_need(self, j, k):
        if j == self.cur:
            return max(k * self.chunk_s - self.pos, 0.0)
        return sum(self.exp_remaining(i) for i in range(self.cur, j)) + k * self.chunk_s

    def thr_hm(self):
        if not self.thr:
            return 1.0
        x = self.thr[-5:]
        return len(x) / sum(1.0 / v for v in x)

    def thr_robust(self):
        e = max(self.err[-5:]) if self.err else 0.0
        return self.thr_hm() / (1.0 + e)

    def since_last(self):
        return math.inf if self.last_end is None else self.t - self.last_end

    # ------------------------------------------------------------------ results
    def snapshot(self):
        """(QoE so far, device energy so far [J]) at the current time; used for per-step RL rewards."""
        qoe, watch, mb_play, e_dec = 0.0, 0.0, 0.0, 0.0
        for v, c in enumerate(self.C[:self.cur + 1]):
            prev = None
            for k, r in enumerate(self.dl[v]):
                s = self.watched[v][k]
                if s <= EPS:
                    continue
                q = c.q[k, r]
                watch += s
                qoe += W_Q * q * s / QOE_UNIT_S
                mb_play += c.sizes[k, r] / 1e6 * s / self.chunk_s
                e_dec += self.pdec[v][r] * s
                if prev is not None:
                    d = abs(q - prev)
                    qoe -= W_SM * d + W_SMJ * math.floor(d / 20.0)
                prev = q
        qoe -= W_RB * self.stall + W_RBN * self.stall_n
        e = (radio_energy(self.transfers, self.t, self.radio, full_tail=False) + e_dec
             + self.device.p_base * watch + self.device.c * mb_play + self.device.p_stall * self.stall)
        return qoe, e

    def metrics(self):
        dev, watch = self.device, 0.0
        qoe = qsum = qstd = mb_play = waste = e_dec = 0.0
        n_av1 = 0.0
        for v, c in enumerate(self.C):
            prev = None
            for k, r in enumerate(self.dl[v]):
                s = self.watched[v][k]
                if s <= EPS:
                    waste += c.sizes[k, r]
                    continue
                q = c.q[k, r]
                watch += s
                qsum += q * s
                qstd += c.q_std[k, r] * s
                qoe += W_Q * q * s / QOE_UNIT_S
                mb_play += c.sizes[k, r] / 1e6 * s / self.chunk_s
                e_dec += self.pdec[v][r] * s
                if c.codec is not None and c.codec[r] == 1:
                    n_av1 += s
                if prev is not None:
                    d = abs(q - prev)
                    qoe -= W_SM * d + W_SMJ * math.floor(d / 20.0)
                prev = q
        qoe -= W_RB * self.stall + W_RBN * self.stall_n
        e_radio = radio_energy(self.transfers, self.t, self.radio)
        e_play = dev.p_base * watch + dev.c * mb_play + e_dec
        e_stall = dev.p_stall * self.stall
        waste += self.aborted
        m = max(watch, EPS) / 60.0
        return {
            "watch_s": watch, "qoe_min": qoe / m, "vmaf": qsum / max(watch, EPS), "vmaf_std": qstd / max(watch, EPS),
            "rebuf_s_min": self.stall / m, "rebuf_n_min": self.stall_n / m,
            "E_min": (e_radio + e_play + e_stall) / m, "Eradio_min": e_radio / m, "Eplay_min": e_play / m,
            "Estall_min": e_stall / m, "MB_min": self.bytes_down / 1e6 / m, "wasteMB_min": waste / 1e6 / m,
            "n_dl": len(self.transfers), "n_promo": sum(1 for x in self.transfers if x[4]), "t_end": self.t,
            "Ebase_min": dev.p_base * watch / m, "Ec_min": dev.c * mb_play / m, "Edec_min": e_dec / m,
            "av1_frac": n_av1 / max(watch, EPS),
        }


class OracleSession(Session):
    """Clairvoyant player: knows how long the user will watch every video in the window and the throughput of
    the next seconds of the trace.  Policies run unchanged on it; their P(watched), expected watch time,
    time-until-needed and throughput estimates become exact.  A reference, not a proven optimum."""

    knows_watch = True
    knows_thr = True
    HORIZON = 2.0           # s of future throughput the estimate averages over

    def p_watch(self, j, k):
        if not self.knows_watch:
            return Session.p_watch(self, j, k)
        if j == self.cur and k * self.chunk_s <= self.pos + EPS:
            return 1.0
        return 1.0 if k * self.chunk_s < self.W[j] - EPS else 0.0

    def exp_watch(self, j, k):
        if not self.knows_watch:
            return Session.exp_watch(self, j, k)
        a, b = k * self.chunk_s, (k + 1) * self.chunk_s
        if j == self.cur:
            a = max(a, self.pos)
        return max(min(b, self.W[j]) - a, 0.0)

    def exp_remaining(self, j):
        if not self.knows_watch:
            return Session.exp_remaining(self, j)
        return max(self.W[j] - (self.pos if j == self.cur else 0.0), 0.0)

    def thr_hm(self):
        if not self.knows_thr:
            return Session.thr_hm(self)
        return max(self.trace.mean_mbps(self.t + RTT, self.HORIZON), 0.01)

    def thr_robust(self):
        if not self.knows_thr:
            return Session.thr_robust(self)
        return self.thr_hm()


class WatchOracleSession(OracleSession):
    """Knows the watch times only (throughput estimated as usual)."""
    knows_thr = False


class ThrOracleSession(OracleSession):
    """Knows the near-future throughput only (watch behaviour from the retention curves as usual)."""
    knows_watch = False


SESSIONS = {False: Session, True: OracleSession, "full": OracleSession, "watch": WatchOracleSession,
            "thr": ThrOracleSession}
