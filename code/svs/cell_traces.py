"""Real cellular throughput traces (Raca et al.: Irish 4G dataset, MMSys'18; Irish 5G dataset, MMSys'20).

Only file-download sessions are used (saturating downloads, so DL_bitrate approximates the available throughput;
the 5G dataset's video-app sessions are rate-limited by the player; the 4G dataset contains downloads only).  Each file is cut into 300-s windows; a window
is kept when >= 80 % of its 1-s samples are in one network mode (LTE or 5G).  Train/test split by source file
(every 5th file, offset 0 and 2 -> test), so windows of one drive never appear on both sides.

  python3 -m svs.cell_traces --g5 <5G-production-dataset> [--g4 <LTE dataset folder>] --out traces_cell
  -> traces_cell/{lte,5g}/{train,test}/<file>_<start>
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

WIN = 300
MODE = {"LTE": "lte", "5G": "5g"}


def per_second(path):
    """1-s series: duplicate timestamps averaged, missing seconds filled with the previous value."""
    d = pd.read_csv(path)
    if "DL_bitrate" not in d or "NetworkMode" not in d:
        return None
    d["t"] = pd.to_datetime(d.Timestamp, format="%Y.%m.%d_%H.%M.%S", errors="coerce")
    d = d.dropna(subset=["t"])
    d["bw"] = pd.to_numeric(d.DL_bitrate, errors="coerce").fillna(0) / 1000.0            # kbps -> Mbps
    g = d.groupby("t").agg(bw=("bw", "mean"), mode=("NetworkMode", lambda x: x.astype(str).mode().iloc[0]))
    g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="1s")).ffill()
    return g


def windows(path):
    g = per_second(path)
    if g is None or len(g) < WIN:
        return []
    bw = g.bw.to_numpy()
    mode = g["mode"].to_numpy()
    out = []
    for s in range(0, len(g) - WIN + 1, WIN):
        m = pd.Series(mode[s:s + WIN]).value_counts()
        if m.index[0] in MODE and m.iloc[0] >= 0.8 * WIN and bw[s:s + WIN].mean() > 0.2:
            out.append((MODE[m.index[0]], s, bw[s:s + WIN]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g5", required=True, help="5G-production-dataset folder")
    ap.add_argument("--g4", default="", help="4G LTE dataset folder")
    ap.add_argument("--out", default="traces_cell")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.g5, "Download", "*", "*.csv")))
    if a.g4:
        files += sorted(glob.glob(os.path.join(a.g4, "**", "*.csv"), recursive=True))   # all are download sessions
    stats = {}
    for i, f in enumerate(files):
        part = "test" if i % 5 in (0, 2) else "train"
        tag = ("g4" if a.g4 and f.startswith(a.g4) else "g5") + "_" + Path(f).parent.name + "_" + Path(f).stem
        for net, s, bw in windows(f):
            d = Path(a.out) / net / part
            d.mkdir(parents=True, exist_ok=True)
            np.savetxt(d / f"{tag}_{s}", np.column_stack([np.arange(WIN, dtype=float), bw]), fmt="%.3f")
            stats.setdefault((net, part), []).append(bw.mean())
    for k in sorted(stats):
        v = np.array(stats[k])
        print(k, len(v), "windows, mean Mbps p10/p50/p90:", np.percentile(v, [10, 50, 90]).round(1))


if __name__ == "__main__":
    main()
