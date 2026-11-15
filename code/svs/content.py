"""Short-video catalogue and viewing sessions.

Content: every short video is a clip of the 46-video x264 library (per-chunk sizes and VMAF for 6 rungs).
Viewing behaviour: KuaiRand-Pure (Kuaishou), immersive feed (tab 1), videos of 8-120 s.
  * the player's knowledge of a video = its population retention curve, estimated from period 1 (04-08..04-21);
  * the sessions (who watched what, for how long, in which order) come from period 2 (04-22..05-08),
    so the retention curves never see the evaluated views.
A KuaiRand video of duration D gets a clip of n = max(2, round(D/4)) chunks; watch times are rescaled by 4n/D.

  python3 -m svs.content --library content_library_x264.csv --kuairand <KuaiRand-Pure/data> --out svs_data.pkl
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .model import BITRATES, CHUNK_S

MIN_VIEWS = 50
SESSION_LEN = 20
DUR_RANGE = (8_000, 120_000)          # ms


@dataclass
class Clip:
    vid: int
    genre: str
    src: str
    sizes: np.ndarray      # [n, rungs] bytes
    q: np.ndarray          # [n, rungs] phone VMAF (used by QoE)
    q_std: np.ndarray      # [n, rungs] default VMAF (reported)
    S: np.ndarray          # survival P(watch >= t) at t = 0, 1, ..., dur  (S[0] = 1, S[dur] = P(complete))
    cumS: np.ndarray       # integral of S from 0 to t, same grid
    chunk_s: float = CHUNK_S
    kbps: np.ndarray | None = None    # nominal bitrate of every rung (None: the 6-rung x264 ladder)
    codec: np.ndarray | None = None   # codec of every rung, 0 = H.264, 1 = AV1 (None: all H.264)
    px: np.ndarray | None = None      # picture size of every rung relative to 1080p (None: the x264 ladder)

    @property
    def n(self) -> int:
        return self.sizes.shape[0]

    @property
    def dur(self) -> float:
        return self.n * self.chunk_s

    def surv(self, t: float) -> float:
        if t <= 0:
            return 1.0
        if t >= self.dur:
            return float(self.S[-1])
        i = int(t)
        f = t - i
        return float(self.S[i] * (1 - f) + self.S[i + 1] * f)

    def integ(self, t: float) -> float:
        """Integral of S from 0 to t (expected watched seconds up to t)."""
        t = min(max(t, 0.0), self.dur)
        i = int(t)
        if i >= len(self.S) - 1:
            return float(self.cumS[-1])
        f = t - i
        return float(self.cumS[i] + self.S[i] * f + 0.5 * (self.S[i + 1] - self.S[i]) * f * f)


def load_library(path):
    df = pd.read_csv(path)
    lib = {}
    for name, d in df.groupby("video"):
        d = d[(d.size_bytes > 0) & d.vmaf.notna() & d.vmaf_phone.notna()]
        p = d.pivot_table(index="chunk", columns="kbps", values=["size_bytes", "vmaf_phone", "vmaf"])
        p = p.dropna()
        p = p.loc[:p.index[(p.index.to_series().diff().fillna(1) != 1).argmax() - 1]
                  if (p.index.to_series().diff().fillna(1) != 1).any() else p.index[-1]]   # contiguous prefix
        lib[name] = (d.genre.iloc[0], p["size_bytes"][list(BITRATES)].to_numpy(float),
                     p["vmaf_phone"][list(BITRATES)].to_numpy(float), p["vmaf"][list(BITRATES)].to_numpy(float))
    return lib


def survival(pt_scaled: np.ndarray, dur: float) -> np.ndarray:
    grid = np.arange(0, int(dur) + 1, dtype=float)
    x = np.minimum(np.sort(pt_scaled), dur)
    S = 1.0 - np.searchsorted(x, grid, side="left") / len(x)
    S[0] = 1.0
    return np.minimum.accumulate(S)


def build(library_csv, kuairand_dir, seed: int = 0, chunk_s: float = CHUNK_S):
    lib = load_library(library_csv)
    names = sorted(lib)
    kd = Path(kuairand_dir)
    cols = ["user_id", "video_id", "time_ms", "play_time_ms", "duration_ms", "tab"]
    p1 = pd.read_csv(kd / "log_standard_4_08_to_4_21_pure.csv", usecols=cols)
    p2 = pd.read_csv(kd / "log_standard_4_22_to_5_08_pure.csv", usecols=cols)
    keep = lambda d: d[(d.tab == 1) & d.duration_ms.between(*DUR_RANGE)]
    p1, p2 = keep(p1), keep(p2)
    vc = p1.video_id.value_counts()
    good = set(vc[vc >= MIN_VIEWS].index)
    dur_ms = p1[p1.video_id.isin(good)].groupby("video_id").duration_ms.median()

    clips = {}
    for vid, g in p1[p1.video_id.isin(good)].groupby("video_id"):
        D = float(dur_ms[vid]) / 1000.0
        n = max(2, int(round(D / chunk_s)))
        scale = n * chunk_s / D
        pt = g.play_time_ms.to_numpy(float) / 1000.0 * scale
        pt[g.play_time_ms.to_numpy(float) >= 0.98 * g.duration_ms.to_numpy(float)] = n * chunk_s   # completed
        S = survival(pt, n * chunk_s)
        cumS = np.concatenate([[0.0], np.cumsum(0.5 * (S[1:] + S[:-1]))])
        rng = np.random.default_rng(int(vid) * 7919 + seed)
        cand = [m for m in names if lib[m][1].shape[0] >= n]
        src = cand[int(rng.integers(len(cand)))]
        genre, sz, q, qs = lib[src]
        off = int(rng.integers(sz.shape[0] - n + 1))
        clips[int(vid)] = Clip(int(vid), genre, src, sz[off:off + n].copy(), q[off:off + n].copy(),
                               qs[off:off + n].copy(), S, cumS, chunk_s)

    s2 = p2[p2.video_id.isin(good)].sort_values(["user_id", "time_ms"])
    sessions = []
    for uid, g in s2.groupby("user_id"):
        vids = g.video_id.to_numpy()
        pt = g.play_time_ms.to_numpy(float) / 1000.0
        dms = g.duration_ms.to_numpy(float)
        for s in range(0, len(g) - SESSION_LEN + 1, SESSION_LEN):
            items = []
            for v, p, dm in zip(vids[s:s + SESSION_LEN], pt[s:s + SESSION_LEN], dms[s:s + SESSION_LEN]):
                c = clips[int(v)]
                w = c.dur if p * 1000.0 >= 0.98 * dm else min(p * c.dur / (dm / 1000.0), c.dur)
                items.append((int(v), float(max(w, 0.05))))
            sessions.append((int(uid), items))
    return clips, sessions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default="content_library_x264.csv")
    ap.add_argument("--kuairand", required=True)
    ap.add_argument("--out", default="svs_data.pkl")
    ap.add_argument("--chunk", type=float, default=CHUNK_S, help="segment length of the library [s]")
    a = ap.parse_args()
    from svs import content as mod          # pickle the classes under their importable module name
    clips, sessions = mod.build(a.library, a.kuairand, chunk_s=a.chunk)
    with open(a.out, "wb") as f:
        pickle.dump({"clips": clips, "sessions": sessions}, f)
    w = np.array([w for _, it in sessions for _, w in it])
    d = np.array([clips[v].dur for _, it in sessions for v, _ in it])
    print(f"{len(clips)} clips, {len(sessions)} sessions of {SESSION_LEN} views; "
          f"clip duration median {np.median(d):.0f} s; watch time median {np.median(w):.1f} s; "
          f"watched fraction mean {np.mean(w / d):.2f}; completed {np.mean(w >= d - 1e-6):.2f}")


if __name__ == "__main__":
    main()
