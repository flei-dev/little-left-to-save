"""Download policies for the swipe feed.

Baselines (each with a knob grid that traces its QoE-energy front):
  Fixed     : TikTok / MMGC-demo style - keep >= 4 s of the current video, preload the first chunk of the next
              P videos, then fill the current video up to B s (B acts as the buffer cap of PSWA / BE-ABR).
  Dashlet   : Dashlet-style (NSDI'23) - next chunks of all videos in the window ranked by
              P(watched) / (expected time until needed); download if P >= theta and total buffer < Bmax.
  DashletE  : Dashlet + an energy-aware rung (GreenABR / MPC-E style objective quality - lambda * energy).
  BurstE    : radio-aware on/off downloading (EnDASH / burst style): idle until the current buffer < L, then
              download back-to-back until the current buffer >= H and every upcoming video has a first chunk;
              energy-aware rung.
Proposed:
  SPARE     : rung = argmax  E[watched fraction] * quality - lambda * energy   (saturation- and swipe-aware);
              schedule = must-fetch (current video about to run dry, first chunk of the next video), plus
              piggy-backing while the radio is still up: prefetch a chunk now iff
                  cost_now <= P(watched) * (E_active + wake_cost / nbar)
              where cost_now = E_active + tail energy re-spent; a chunk is never prefetched from radio idle.
  Ablations : SPARE-noradio (wake cost ignored), SPARE-noprob (P(watched) := 1 everywhere),
              SPARE-nosat (throughput-rule rung instead of the quality/energy rung).
"""

from __future__ import annotations

from .model import RTT, W_Q, W_RB, W_RBN, W_SM

SLEEP = 0.25


def dl_time(S, j, k, r, thr=None):
    return S.C[j].sizes[k, r] * 8e-6 / (thr or S.thr_robust()) + RTT


def feasible(S, j, k):
    """Allowed rungs (by increasing bitrate) whose bitrate is sustainable and, for chunks needed soon, that
    arrive in time.  With two codecs, a rung that would arrive late is skipped instead of ending the scan
    (the other codec's rung of the same bitrate may be smaller)."""
    hm = S.thr_robust()
    ttn = S.time_to_need(j, k)
    rs, kb = S.rungs(j), S.kb[j]
    multi = S.C[j].codec is not None
    out = [rs[0]]
    t0 = None
    for r in rs[1:]:
        if kb[r] / 1000.0 > hm:
            break
        if ttn < 2 * S.chunk_s:
            if t0 is None:
                t0 = dl_time(S, j, k, rs[0])
            if dl_time(S, j, k, r) > max(ttn - 0.5, t0):
                if multi:
                    continue
                break
        out.append(r)
    return out


def best_at_top(S, j, k, ok):
    """Among candidate rungs, the highest bitrate level; between codecs of that level, the better quality."""
    kb, q = S.kb[j], S.C[j].q[k]
    top = kb[ok[-1]]
    cands = [r for r in ok if kb[r] == top]
    return max(cands, key=lambda r: (q[r], -r)) if len(cands) > 1 else cands[0]


def rung_thr(S, j, k, safety: float = 0.9):
    hm = S.thr_hm()
    fe = feasible(S, j, k)
    ok = [r for r in fe if S.kb[j][r] / 1000.0 <= safety * hm]
    return best_at_top(S, j, k, ok) if ok else fe[0]


def rung_energy(S, j, k, lam: float, pweight: bool):
    c = S.C[j]
    if pweight:
        f = S.exp_watch(j, k) / S.chunk_s
        g = S.p_watch(j, k)
    else:
        f = g = 1.0
    qprev = c.q[k - 1, S.dl[j][k - 1]] if k > 0 else None
    thr = S.thr_hm()
    best, best_u = 0, -1e18
    for r in feasible(S, j, k):
        q = c.q[k, r]
        u = f * W_Q * q
        if qprev is not None:
            u -= g * W_SM * abs(q - qprev)
        if lam > 0:
            mb = c.sizes[k, r] / 1e6
            u -= lam * (S.radio_belief.active_energy(c.sizes[k, r], thr) + S.device.c * mb * f
                        + S.pdec[j][r] * S.chunk_s * f)
        if u > best_u:
            best, best_u = r, u
    return best


class Policy:
    name = "policy"

    def reset(self, S):
        pass


class Fixed(Policy):
    def __init__(self, P: int, B: float):
        self.P, self.B = P, B
        self.name = f"Fixed(P={P},B={B:g})"

    def act(self, S):
        cur = S.cur
        k = S.next_k(cur)
        if k is not None and S.buf_ahead(cur) < S.chunk_s:
            return ("dl", cur, k, rung_thr(S, cur, k))
        for j in list(S.window())[1:1 + self.P]:
            if not S.dl[j]:
                return ("dl", j, 0, rung_thr(S, j, 0))
        if k is not None and S.buf_ahead(cur) < self.B:
            return ("dl", cur, k, rung_thr(S, cur, k))
        return ("sleep", SLEEP)


class Dashlet(Policy):
    def __init__(self, theta: float, bmax: float, lam: float | None = None):
        self.theta, self.bmax, self.lam = theta, bmax, lam
        e = "" if lam is None else f",lam={lam:g}"
        self.name = f"{'DashletE' if lam is not None else 'Dashlet'}(th={theta:g},B={bmax:g}{e})"

    def _rung(self, S, j, k):
        return rung_thr(S, j, k) if self.lam is None else rung_energy(S, j, k, self.lam, pweight=False)

    def act(self, S):
        win = list(S.window())
        total = sum(S.buf_ahead(i) for i in win)
        best = None
        for j in win:
            k = S.next_k(j)
            if k is None:
                continue
            p = S.p_watch(j, k)
            urgent = (j == S.cur and S.buf_ahead(j) < S.chunk_s) or (j == S.cur + 1 and k == 0)
            if urgent or (p >= self.theta and total < self.bmax):
                score = (1e6 if urgent else 0.0) + p / (S.time_to_need(j, k) + 1.0)
                if best is None or score > best[0]:
                    best = (score, j, k)
        if best is None:
            return ("sleep", SLEEP)
        _, j, k = best
        return ("dl", j, k, self._rung(S, j, k))


class Pdas(Policy):
    """PDAS-style (ACM MM'22 short-video grand challenge winner, rule adaptation from its description):
    a per-video maximum buffer from the retention curve (the q-quantile of the remaining watch time, capped at
    bmax), videos served in playback order, and a retention-weighted quality objective for the rung."""

    def __init__(self, q: float, bmax: float):
        self.q, self.bmax = q, bmax
        self.name = f"PDAS(q={q:g},B={bmax:g})"

    def _target(self, S, j):
        c = S.C[j]
        pos = S.pos if j == S.cur else 0.0
        s0 = c.surv(pos)
        lvl = (1.0 - self.q) * s0
        t = c.dur
        for g in range(int(pos) + 1, int(c.dur) + 1):           # first second where survival drops below lvl
            if c.S[g] < lvl:
                t = float(g)
                break
        return min(self.bmax, max(t - pos, S.chunk_s))

    def act(self, S):
        cur = S.cur
        k = S.next_k(cur)
        if k is not None and S.buf_ahead(cur) < S.chunk_s:
            return ("dl", cur, k, rung_energy(S, cur, k, 0.0, pweight=True))
        win = list(S.window())
        if len(win) > 1 and not S.dl[win[1]]:
            return ("dl", win[1], 0, rung_energy(S, win[1], 0, 0.0, pweight=True))
        for j in win:
            k = S.next_k(j)
            if k is not None and S.buf_ahead(j) < self._target(S, j):
                return ("dl", j, k, rung_energy(S, j, k, 0.0, pweight=True))
        return ("sleep", SLEEP)


class BurstE(Policy):
    def __init__(self, L: float, H: float, lam: float):
        self.L, self.H, self.lam = L, H, lam
        self.name = f"BurstE(L={L:g},H={H:g},lam={lam:g})"

    def reset(self, S):
        self.on = False

    def _todo(self, S):
        win = list(S.window())
        out = []
        k = S.next_k(S.cur)
        if k is not None and S.buf_ahead(S.cur) < self.H:
            out.append((S.cur, k))
        for j in win[1:]:
            if not S.dl[j]:
                out.append((j, 0))
        return out

    def act(self, S):
        if not self.on:
            k = S.next_k(S.cur)
            low = k is not None and S.buf_ahead(S.cur) < self.L
            nxt = S.cur + 1 < S.nv and not S.dl[S.cur + 1]
            if low or nxt:
                self.on = True
        if self.on:
            todo = self._todo(S)
            if todo:
                j, k = max(todo, key=lambda x: S.p_watch(*x) / (S.time_to_need(*x) + 1.0))
                return ("dl", j, k, rung_energy(S, j, k, self.lam, pweight=False))
            self.on = False
        return ("sleep", SLEEP)


class Spare(Policy):
    def __init__(self, lam: float, nbar: float = 2.0, radio_aware: bool = True, prob: bool = True,
                 sat: bool = True, label: str = "SPARE", eager0: bool = False):
        self.lam, self.nbar, self.radio_aware, self.prob, self.sat = lam, nbar, radio_aware, prob, sat
        self.eager0 = eager0      # True: first chunks of all upcoming videos are must-fetch (wakes the radio)
        self.name = f"{label}(lam={lam:g},n={nbar:g})"

    def _p(self, S, j, k):
        return S.p_watch(j, k) if self.prob else 1.0

    def _rung(self, S, j, k):
        if not self.sat:
            return rung_thr(S, j, k)
        return rung_energy(S, j, k, self.lam, pweight=self.prob)

    def act(self, S):
        cur, rad = S.cur, S.radio_belief
        win = list(S.window())
        # 1) must-fetch, in deadline order: the current video about to run dry, and the first chunk of every
        #    upcoming video in the window (always watched, P = 1; the user may swipe at any moment)
        k = S.next_k(cur)
        if k is not None and S.knows_watch and S.p_watch(cur, k) < 0.5:
            k = None                                     # clairvoyant: the user leaves before this chunk
        r = self._rung(S, cur, k) if k is not None else None
        if k is not None and S.buf_ahead(cur) < dl_time(S, cur, k, r) + 1.0:
            return ("dl", cur, k, r)
        if len(win) > 1 and not S.dl[win[1]]:
            return ("dl", win[1], 0, self._rung(S, win[1], 0))
        if k is not None and S.buf_ahead(cur) < S.chunk_s + dl_time(S, cur, k, r) + 1.0:
            return ("dl", cur, k, r)
        if self.eager0 or not self.radio_aware:
            for j in win[2:]:
                if not S.dl[j]:
                    return ("dl", j, 0, self._rung(S, j, 0))
        # 2) piggy-back deeper chunks while the radio is up
        since = S.since_last()
        if since > rad.t_tail:
            return ("sleep", SLEEP)
        thr = S.thr_hm()
        wake = rad.wake_cost() / self.nbar if self.radio_aware else 0.0
        tail_respent = (rad.p_tail - rad.p_idle) * since if self.radio_aware else 0.0
        best = None
        for j in S.window():
            k = S.next_k(j)
            if k is None:
                continue
            r = self._rung(S, j, k)
            e_act = rad.active_energy(S.C[j].sizes[k, r], thr)
            p = self._p(S, j, k)
            if e_act + tail_respent <= p * (e_act + wake) + 1e-12:
                score = p / (S.time_to_need(j, k) + 1.0)
                if best is None or score > best[0]:
                    best = (score, j, k, r)
        if best is None:
            return ("sleep", SLEEP)
        return ("dl", best[1], best[2], best[3])


class Spare2(Spare):
    """SPARE v2: the prefetch decision also counts the QoE value of prefetching (stall risk avoided), and may
    wake the radio for it.  Prefetch chunk (j, k) now iff
        lam * (cost_now - P * (E_active + wake_cost / nbar)) <= V_stall(j, k)
    cost_now = E_active + tail energy re-spent (radio up) or + a full wake-up (radio idle);
    V_stall  = expected QoE loss if the chunk is fetched just in time instead (SQoE-III rebuffering terms).
    """

    def __init__(self, lam: float, nbar: float = 2.0, label: str = "SPARE2", **kw):
        super().__init__(lam, nbar, label=label, **kw)

    def _v_stall(self, S, j, k, r, rad):
        if j == S.cur:
            return 0.0                                   # protected by the must-fetch rule
        t = dl_time(S, j, k, r) + rad.t_pro
        if k == 0:                                       # stalls if the user leaves video j-1 within t
            q = 1.0 - S.C[j - 1].surv(t) if j - 1 > S.cur else 1.0 - S.C[j - 1].surv(S.pos + t) / max(S.C[j - 1].surv(S.pos), 1e-9)
            return q * (W_RB * t / 2.0 + W_RBN)
        slack = k * S.chunk_s                              # fetched when video j starts, needed k*4 s later
        if t <= slack:
            return 0.0
        return self._p(S, j, k) * (W_RB * (t - slack) + W_RBN)

    def act(self, S):
        cur, rad = S.cur, S.radio_belief
        win = list(S.window())
        k = S.next_k(cur)
        r = self._rung(S, cur, k) if k is not None else None
        if k is not None and S.buf_ahead(cur) < dl_time(S, cur, k, r) + 1.0:
            return ("dl", cur, k, r)
        if len(win) > 1 and not S.dl[win[1]]:
            return ("dl", win[1], 0, self._rung(S, win[1], 0))
        if k is not None and S.buf_ahead(cur) < S.chunk_s + dl_time(S, cur, k, r) + 1.0:
            return ("dl", cur, k, r)
        since = S.since_last()
        awake = since <= rad.t_tail
        thr = S.thr_hm()
        wake = rad.wake_cost() / self.nbar
        extra = (rad.p_tail - rad.p_idle) * since if awake else rad.wake_cost()
        best = None
        for j in win:
            k = S.next_k(j)
            if k is None:
                continue
            r = self._rung(S, j, k)
            e_act = rad.active_energy(S.C[j].sizes[k, r], thr)
            p = self._p(S, j, k)
            if p <= 0:
                continue
            if self.lam * (e_act + extra - p * (e_act + wake)) <= self._v_stall(S, j, k, r, rad) + 1e-12:
                score = p / (S.time_to_need(j, k) + 1.0)
                if best is None or score > best[0]:
                    best = (score, j, k, r)
        if best is None:
            return ("sleep", SLEEP)
        return ("dl", best[1], best[2], best[3])


LAMS_SPARE = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6)


def grid():
    """All (family, policy factory) pairs of the main experiment."""
    g = []
    for P in (1, 2, 3, 4):
        for B in (4, 8, 16, 30):
            g.append(("Fixed", lambda P=P, B=B: Fixed(P, B)))
    for th in (0.2, 0.4, 0.6, 0.8):
        for bm in (12, 24, 48):
            g.append(("Dashlet", lambda th=th, bm=bm: Dashlet(th, bm)))
    for th in (0.4, 0.6):
        for lam in LAMS_SPARE:
            g.append(("DashletE", lambda th=th, lam=lam: Dashlet(th, 24, lam)))
    for q in (0.5, 0.7, 0.9):
        for bm in (8, 16, 32):
            g.append(("PDAS", lambda q=q, bm=bm: Pdas(q, bm)))
    for L in (4, 8):
        for H in (12, 24, 48):
            for lam in LAMS_SPARE:
                g.append(("BurstE", lambda L=L, H=H, lam=lam: BurstE(L, H, lam)))
    for lam in LAMS_SPARE:
        g.append(("SPARE", lambda lam=lam: Spare(lam)))
        g.append(("SPARE-noradio", lambda lam=lam: Spare(lam, radio_aware=False, label="SPARE-noradio")))
        g.append(("SPARE-noprob", lambda lam=lam: Spare(lam, prob=False, label="SPARE-noprob")))
        g.append(("SPARE-eager0", lambda lam=lam: Spare(lam, eager0=True, label="SPARE-eager0")))
    for nb in (1, 2, 4, 8):
        g.append(("SPARE-nosat", lambda nb=nb: Spare(0.0, nbar=nb, sat=False, label="SPARE-nosat")))
    return g
