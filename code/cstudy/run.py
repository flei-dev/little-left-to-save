"""Stage runner of the paper-C study.

Simulates policy configurations under energy models on (trace, session) pairs and stores per-session metrics:
  <out>/pairs.csv                          one row per pair (trace file, session id, genre, ...)
  <out>/raw/<model>[__<mode>]__<family>.npz  X [n_cfg, n_pairs, n_metrics] float32 + config names
A job = one (model, codec mode, family) block; finished blocks are skipped, so a stage can be restarted.

  python3 -m cstudy.run --data svs_data.pkl --form short --split main --traces traces_cell/lte/test \
      --spt 3 --seed0 9000 --models grid --sets base,oracle --out results/E1E2_short
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from svs.grids import configs
from svs.longform import clear_cache
from svs.sim import SESSIONS, Trace

from .models import model_set, registry

METRICS = ["qoe_min", "vmaf", "vmaf_std", "rebuf_s_min", "rebuf_n_min", "E_min", "Eradio_min", "Eplay_min",
           "Estall_min", "Ebase_min", "Ec_min", "Edec_min", "MB_min", "wasteMB_min", "n_promo", "watch_s",
           "av1_frac"]
CODECS = {"": None, "h264": (0,), "av1": (1,), "both": (0, 1)}
G = {}


def trace_files(d, n=0):
    fs = sorted(p for p in Path(d).iterdir() if p.is_file() and not p.name.startswith("."))
    return fs[:n] if n else fs


def sample_watch(c, rng):
    """Watch time drawn from the clip's population retention curve (E5 behaviour variant)."""
    u = rng.uniform()
    S = c.S
    if u < S[-1]:
        return c.dur
    i = int(np.argmax(S <= u))
    if i == 0:
        return 0.05
    return max(i - 1 + (S[i - 1] - u) / max(S[i - 1] - S[i], 1e-9), 0.05)


def load_data(path, quality="phone"):
    d = pickle.load(open(path, "rb"))
    if quality == "std":                     # E5: QoE from the standard VMAF model instead of the phone model
        clips = {}
        for k, c in d["clips"].items():
            c2 = copy.copy(c)
            c2.q, c2.q_std = c.q_std, c.q
            clips[k] = c2
        d["clips"] = clips
    return d


def build_pairs(d, split, traces, spt, seed0, n_traces=0, watch="real"):
    if "splits" in d:
        pool = [d["sessions"][i] for i in d["splits"][split]]
    else:                                    # feed: even sessions = development, odd = evaluation
        pool = d["sessions"][0::2] if split == "dev" else d["sessions"][1::2]
    pairs = []
    for i, f in enumerate(trace_files(traces, n_traces)):
        a = np.loadtxt(f, ndmin=2)
        for s in range(spt):
            rng = np.random.default_rng(seed0 * (i + 1) + s)
            uid, items = pool[(i * spt + s) % len(pool)]
            off = float(rng.uniform(0, a[-1, 0]))
            if watch == "sampled":
                items = [(v, sample_watch(d["clips"][v], rng)) for v, _ in items]
            pairs.append(dict(trace=f.name, pid=len(pairs), uid=uid, items=items, a=a, off=off,
                              genre=d["clips"][items[0][0]].genre))
    return pairs


def _init(args):
    d = load_data(args["data"], args["quality"])
    G["clips"] = d["clips"]
    G["pairs"] = build_pairs(d, args["split"], args["traces"], args["spt"], args["seed0"], args["n_traces"],
                             args["watch"])
    G["cfgs"] = configs(args["form"], tuple(args["sets"]))
    G["reg"] = registry()
    G["args"] = args


def _job(job):
    model, mode, fam, out = job
    dev, rad = G["reg"][model]
    codecs = CODECS[mode]
    if fam == "DRL":
        from svs.drl import load_policy
        runs = G["args"]["drl_runs"]
        items = [(r, lambda r=r: load_policy(r, json.loads((Path(r) / "config.json").read_text())["lam"]))
                 for r in runs]
        cfgs = [("DRL", mk, False, None) for _, mk in items]
    else:
        cfgs = [c for c in G["cfgs"] if c[0] == fam]
    X = np.full((len(cfgs), len(G["pairs"]), len(METRICS)), np.nan, np.float32)
    t0 = time.time()
    names = [None] * len(cfgs)
    for p in G["pairs"]:                       # session outer, configurations inner (MPC tables are reused)
        clear_cache()
        for i, (f, make, orc, cap) in enumerate(cfgs):
            pol = make()
            names[i] = pol.name if fam != "DRL" else f"{pol.name}[{Path(runs[i]).name}]"
            bc = cap if cap is not None else getattr(pol, "cap", math.inf)
            cls = SESSIONS[orc]
            S = cls(G["clips"], p["items"], Trace(p["a"][:, 0], p["a"][:, 1], p["off"]), rad, dev,
                    buf_cap=bc, codecs=codecs)
            m = S.run(pol)
            X[i, p["pid"]] = [m.get(k, np.nan) for k in METRICS]
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, X=X, names=np.array(names), family=fam, mode=mode, model=model,
                        seconds=time.time() - t0,
                        oracle=np.array([str(c[2]) for c in cfgs]), metrics=np.array(METRICS))
    os.replace(tmp, out)
    return job, time.time() - t0


def run_stage(a):
    out = Path(a["out"])
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({k: v for k, v in a.items()}, indent=1, default=str))
    d = load_data(a["data"], a["quality"])
    pairs = build_pairs(d, a["split"], a["traces"], a["spt"], a["seed0"], a["n_traces"], a["watch"])
    with open(out / "pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pid", "trace", "uid", "genre", "offset", "n_items", "watch_s"])
        for p in pairs:
            w.writerow([p["pid"], p["trace"], p["uid"], p["genre"], round(p["off"], 3), len(p["items"]),
                        round(sum(x for _, x in p["items"]), 2)])
    fams = []
    for c in configs(a["form"], tuple(a["sets"])):
        if c[0] not in fams and (not a["families"] or c[0] in a["families"]):
            fams.append(c[0])
    if a.get("drl_runs"):
        fams.append("DRL")
    jobs = []
    for m in model_set(a["models"]):
        for mode in a["modes"]:
            for fam in fams:
                tag = f"{m}__{mode}__{fam}" if mode else f"{m}__{fam}"
                path = str(out / "raw" / f"{tag}.npz")
                if not os.path.exists(path):
                    jobs.append((m, mode, fam, path))
    total = len(jobs)
    prog = out / "progress.json"
    if not jobs:
        prog.write_text(json.dumps({"done": 0, "total": 0, "status": "finished"}))
        print(f"{out.name}: nothing to do", flush=True)
        return
    # longest blocks first (better packing)
    weight = {f: sum(1 for c in configs(a["form"], tuple(a["sets"])) if c[0] == f) for f in fams}
    jobs.sort(key=lambda j: -weight.get(j[2], 4) * (3 if j[2].startswith("O") or "MPC" in j[2] else 1))
    t0 = time.time()
    done = 0
    workers = min(a["workers"], total)
    with Pool(workers, initializer=_init, initargs=(a,)) as pool:
        for job, dt in pool.imap_unordered(_job, jobs):
            done += 1
            el = time.time() - t0
            eta = el / done * (total - done)
            prog.write_text(json.dumps({"stage": out.name, "done": done, "total": total,
                                        "elapsed_min": round(el / 60, 1), "eta_min": round(eta / 60, 1),
                                        "status": "running" if done < total else "finished"}))
            if done % max(1, total // 50) == 0 or done == total:
                print(f"{out.name}: {done}/{total} blocks  {el / 60:.1f} min  ETA {eta / 60:.0f} min", flush=True)


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--form", choices=["short", "long"], required=True)
    ap.add_argument("--split", default="main")
    ap.add_argument("--traces", required=True)
    ap.add_argument("--spt", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--n_traces", type=int, default=0)
    ap.add_argument("--models", default="anchors")
    ap.add_argument("--sets", default="base,oracle")
    ap.add_argument("--families", default="")
    ap.add_argument("--modes", default="", help="codec modes (E4): h264,av1,both")
    ap.add_argument("--quality", default="phone", choices=["phone", "std"])
    ap.add_argument("--watch", default="real", choices=["real", "sampled"])
    ap.add_argument("--drl_runs", default="", help="comma-separated DRL run folders to evaluate")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", required=True)
    a = vars(ap.parse_args(argv))
    a["sets"] = a["sets"].split(",")
    a["families"] = [x for x in a["families"].split(",") if x]
    a["modes"] = a["modes"].split(",") if a["modes"] else [""]
    a["drl_runs"] = [x for x in a["drl_runs"].split(",") if x]
    return a


if __name__ == "__main__":
    run_stage(parse())
