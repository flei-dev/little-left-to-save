"""Environment check and run-time estimate:  python3 -m cstudy.check [cpu1|cpu2]"""

from __future__ import annotations

import os
import pickle
import shutil
import sys
import time
from pathlib import Path

ROOT = Path.cwd()


def bench():
    """Seconds per simulated session on this machine: (feed, long-form MPC)."""
    import numpy as np
    from svs.grids import configs
    from svs.sim import SESSIONS, Trace
    from cstudy.models import registry
    dev, rad = registry()["meas"]
    tr = sorted((ROOT / "traces" / "lte" / "test").iterdir())[:3]
    d = pickle.load(open(ROOT / "data" / "svs_data.pkl", "rb"))
    g = [c for c in configs("short") if c[0] in ("SPARE", "Dashlet", "O-Burst")][::4]
    t0, n = time.time(), 0
    for f in tr:
        a = np.loadtxt(f, ndmin=2)
        for fam, mk, orc, cap in g:
            SESSIONS[orc](d["clips"], d["sessions"][1][1], Trace(a[:, 0], a[:, 1], 0.0), rad, dev, buf_cap=cap).run(mk())
            n += 1
    s_short = (time.time() - t0) / n
    d = pickle.load(open(ROOT / "data" / "long_4s.pkl", "rb"))
    g = [c for c in configs("long") if c[0] in ("MPC-E", "O-MPC")][::6]
    t0, n = time.time(), 0
    for f in tr[:2]:
        a = np.loadtxt(f, ndmin=2)
        for fam, mk, orc, cap in g:
            SESSIONS[orc](d["clips"], d["sessions"][0][1], Trace(a[:, 0], a[:, 1], 0.0), rad, dev, buf_cap=cap).run(mk())
            n += 1
    return s_short, (time.time() - t0) / n


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "cpu2"
    ok = True
    print(f"python {sys.version.split()[0]}, {os.cpu_count()} CPU threads, "
          f"free disk {shutil.disk_usage(ROOT).free / 1e9:.0f} GB")
    for m in ("numpy", "pandas", "matplotlib"):
        try:
            mod = __import__(m)
            print(f"  {m} {mod.__version__}")
        except ImportError:
            print(f"  MISSING: {m}  (pip install {m})")
            ok = ok and m == "matplotlib"
    need = ["data/svs_data.pkl", "data/svs_data_2s.pkl", "data/long_4s.pkl", "data/long_2s.pkl",
            "data/content_library_x264_2s.csv", "traces/lte/test", "traces/5g/test", "traces/old/test"]
    for p in need:
        if not (ROOT / p).exists():
            print(f"  MISSING: {p}")
            ok = False
    if which == "cpu1":
        for k in ("VIDEOS", "FFMPEG", "FFPROBE"):
            v = os.path.expanduser(os.environ.get(k, ""))
            print(f"  {k} = {v}  {'ok' if v and os.path.exists(v) else 'NOT FOUND'}")
            ok = ok and bool(v) and os.path.exists(v)
        if ok:
            import subprocess
            enc = subprocess.run([os.path.expanduser(os.environ["FFMPEG"]), "-hide_banner", "-encoders"],
                                 capture_output=True, text=True).stdout
            flt = subprocess.run([os.path.expanduser(os.environ["FFMPEG"]), "-hide_banner", "-filters"],
                                 capture_output=True, text=True).stdout
            print(f"  libsvtav1 {'ok' if 'libsvtav1' in enc else 'MISSING'}, libvmaf {'ok' if 'libvmaf' in flt else 'MISSING'}")
            ok = ok and "libsvtav1" in enc and "libvmaf" in flt
    if not ok:
        print("CHECK FAILED")
        sys.exit(1)
    s, l = bench()
    w = max(1, (os.cpu_count() or 2) - 1)
    # simulated sessions per stage group (see cstudy.main) x measured cost / workers
    if which == "cpu2":
        short_sims = (79 * 246 * 378                 # E1/E2
                      + 4 * 510 * 378                # E2 decomposition
                      + 4 * 246 * 378 * (2 + 1 + 1 + 1 + 1)   # E5: 2-s (x2 cost), std VMAF, 5G, old, sampled
                      + 4 * 136 * 375)               # E3 rule evaluation
        long_sims = 0.8 * (79 * 187 * 252 + 4 * 187 * 252 * (2 + 1 + 1 + 1) + 4 * 110 * 252)
        sim_h = (short_sims * s + long_sims * l) / w / 3600
        train_h = -(-96 // w) * 5000 * 0.45 / 3600
        print(f"benchmark: {s * 1000:.0f} ms per feed session, {l * 1000:.0f} ms per long-form MPC session")
        print(f"estimated run time with {w} workers: simulations {sim_h:.1f} h + DRL training {train_h:.1f} h "
              f"= about {sim_h + train_h:.0f} h")
    else:
        e4 = 3 * 11 * 3 * (136 * 375 * s * 2 + 110 * 252 * l * 0.8 * 2 * 1.5) / w / 3600
        print(f"benchmark: {s * 1000:.0f} ms / {l * 1000:.0f} ms; E4 after encoding: about {e4:.1f} h with {w} workers")
        print("AV1 encoding of 46 videos: roughly 1-1.5 days (VMAF dominates)")
    print("CHECK OK")


if __name__ == "__main__":
    main()
