"""One-command driver of the paper-C experiments (see PREREG_C.md).

  python3 -m cstudy.main cpu2      # E1, E2, E5, E3 (simulation + DRL); resumable
  python3 -m cstudy.main cpu1      # AV1 encode of the 46 videos (2-s chunks), then E4; resumable
  python3 -m cstudy.main status    # progress of every stage
  python3 -m cstudy.main zip       # pack the summaries finished so far

Run from the package root (the folder with run.sh).  Every stage writes results/<stage>/ and a DONE marker;
after each group of stages a zip of the summaries is written to the package root (results_<server>_<n>.zip).
"""

from __future__ import annotations

import glob
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path

ROOT = Path.cwd()
RES = ROOT / "results"
LOG = ROOT / "logs"
SMOKE = os.environ.get("CSTUDY_SMOKE") == "1"          # tiny run for testing the pipeline
WORKERS = int(os.environ.get("WORKERS", 0)) or max(1, (os.cpu_count() or 2) - 1)

TR = {"lte": ("traces/lte/train", "traces/lte/test"), "5g": ("traces/5g/train", "traces/5g/test"),
      "old": ("traces/old/train", "traces/old/test")}
SPT_SHORT = {"lte": 3, "5g": 8, "old": 4}             # sessions per trace -> ~370 evaluation sessions
SPT_LONG = {"lte": 2, "5g": 5, "old": 3}              # -> ~250 evaluation sessions
REGIMES = {                                           # E3: (radio, device, energy model, trace set)
    "legacy": ("lte", "legacy", "legacy_lte", "lte"),
    "meas": ("lte_meas", "modern", "meas", "lte"),
    "5g": ("5g", "modern", "modern_5g", "5g"),
    "low": ("lte_meas", "modern", "meas", "old"),
}
DRL_LAMS = (0.02, 0.1, 0.4, 1.6)
DRL_SEEDS = (0, 1, 2)
DRL_EPISODES = 5000
FORM_DATA = {"short": "data/svs_data.pkl", "long": "data/long_4s.pkl"}
FORM_SPLIT = {"short": "main", "long": "main"}


def sim(name, data, form, traces, models, sets="base,oracle", split="main", spt=None, seed0=9000, extra=(),
        analyses=("headroom",)):
    tset = Path(traces).parts[1]
    spt = spt or (SPT_SHORT if form == "short" else SPT_LONG)[tset]
    args = ["--data", data, "--form", form, "--split", split, "--traces", traces, "--spt", str(spt),
            "--seed0", str(seed0), "--models", models, "--sets", sets, "--out", str(RES / name),
            "--workers", str(WORKERS)] + list(extra)
    if SMOKE:
        args += ["--n_traces", "2"]
    return dict(name=name, kind="sim", args=args, analyses=analyses)


def cpu2_stages():
    st = []
    # E1 + E2: energy-model grid, unaware / aware / clairvoyant, main evaluation sessions, real 4G traces
    st.append(sim("E1E2_short", "data/svs_data.pkl", "short", "traces/lte/test", "grid", seed0=9000))
    st.append(sim("E1E2_long", "data/long_4s.pkl", "long", "traces/lte/test", "grid", seed0=9100,
                  analyses=("headroom", "genre")))
    st.append(dict(name="cpu2_ckpt1", kind="zip", tag="ckpt1"))
    # E2 decomposition (feed): what the clairvoyant gains from knowing watch times vs throughput
    st.append(sim("E2_decomp_short", "data/svs_data.pkl", "short", "traces/lte/test", "anchors",
                  sets="base,oracle,decomp", seed0=9000))
    # E5 robustness, anchor energy models
    for form, d2 in (("short", "data/svs_data_2s.pkl"), ("long", "data/long_2s.pkl")):
        s0 = 9000 if form == "short" else 9100
        st.append(sim(f"E5_{form}_2s", d2, form, "traces/lte/test", "anchors", seed0=s0))
        st.append(sim(f"E5_{form}_vmafstd", FORM_DATA[form], form, "traces/lte/test", "anchors", seed0=s0,
                      extra=("--quality", "std")))
        st.append(sim(f"E5_{form}_5g", FORM_DATA[form], form, "traces/5g/test", "anchors", seed0=s0))
        st.append(sim(f"E5_{form}_old", FORM_DATA[form], form, "traces/old/test", "anchors", seed0=s0))
    st.append(sim("E5_short_sampled", "data/svs_data.pkl", "short", "traces/lte/test", "anchors", seed0=9000,
                  extra=("--watch", "sampled")))
    st.append(dict(name="cpu2_ckpt2", kind="zip", tag="ckpt2"))
    # E3: DRL vs tuned rules in four regimes, both forms
    st.append(dict(name="E3_train", kind="drl_train"))
    for form in ("short", "long"):
        for reg, (_, _, model, tset) in REGIMES.items():
            split = "main" if form == "short" else "test"        # long form: videos held out from training
            st.append(dict(sim(f"E3_{form}_{reg}", FORM_DATA[form], form, TR[tset][1], model, sets="base",
                               split=split, seed0=9200), kind="drl_eval", form=form, regime=reg,
                           analyses=("drl",)))
    st.append(dict(name="cpu2_ckpt3", kind="zip", tag="ckpt3"))
    return st


def cpu1_stages():
    st = [dict(name="E4_encode", kind="encode"), dict(name="E4_data", kind="codec_data")]
    for tset in ("lte", "old", "5g"):
        for form, data in (("short", "work/data_codec/svs_data_codec.pkl"), ("long", "work/data_codec/long_codec.pkl")):
            st.append(sim(f"E4_{form}_{tset}", data, form, TR[tset][1], "codec", sets="base",
                          seed0=9300, extra=("--modes", "h264,av1,both"), analyses=("codec",)))
        if tset == "lte":
            st.append(dict(name="cpu1_ckpt1", kind="zip", tag="ckpt1"))
    st.append(dict(name="cpu1_ckpt2", kind="zip", tag="ckpt2"))
    return st


# ---------------------------------------------------------------------------------------------- stage kinds
def py(mod, *args, log=None):
    cmd = [sys.executable, "-m", mod] + [str(a) for a in args]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "code"))
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise SystemExit(f"FAILED ({mod}); see {log}")


def drl_jobs():
    from svs.drl import DQNConfig
    jobs = []
    for form in ("short", "long"):
        for reg, (radio, device, _, tset) in REGIMES.items():
            for lam in (DRL_LAMS[2:3] if SMOKE else DRL_LAMS):
                for seed in (DRL_SEEDS[:1] if SMOKE else DRL_SEEDS):
                    cfg = DQNConfig(radio=radio, device=device, lam=lam, seed=seed, regime=reg, form=form,
                                    buf_cap=math.inf if form == "short" else 60.0,
                                    episodes=60 if SMOKE else DRL_EPISODES,
                                    warmup_episodes=10 if SMOKE else 30)
                    jobs.append((cfg, FORM_DATA[form], TR[tset][0]))
    return jobs


def _train(job):
    from svs.drl import train
    cfg, data, traces = job
    t0 = time.time()
    with open(LOG / "E3_train" / f"{cfg.run_id}.log", "a") as fh:
        so, se = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = fh
        try:
            train(cfg, data, traces, str(RES / "E3_train" / "runs"), log_every=100)
        finally:
            sys.stdout, sys.stderr = so, se
    return cfg.run_id, time.time() - t0


def run_drl_train(stage):
    (LOG / "E3_train").mkdir(parents=True, exist_ok=True)
    jobs = [j for j in drl_jobs() if not (RES / "E3_train" / "runs" / j[0].run_id / "done.json").exists()]
    total = len(drl_jobs())
    prog = RES / "E3_train" / "progress.json"
    prog.parent.mkdir(parents=True, exist_ok=True)
    jobs.sort(key=lambda j: j[0].form != "short")
    t0, n0 = time.time(), total - len(jobs)
    done = n0
    if jobs:
        with Pool(min(WORKERS, len(jobs)), maxtasksperchild=1) as pool:
            for rid, dt in pool.imap_unordered(_train, jobs):
                done += 1
                el = time.time() - t0
                prog.write_text(json.dumps({"done": done, "total": total, "elapsed_min": round(el / 60, 1),
                                            "eta_min": round(el / (done - n0) * (total - done) / 60, 1)}))
                print(f"E3_train: {done}/{total} runs  {el / 60:.0f} min", flush=True)


def drl_runs(form, regime):
    return sorted(str(p.parent) for p in (RES / "E3_train" / "runs").glob(f"{form}_{regime}_lam*/done.json"))


def run_encode(stage):
    v, ff, fp = os.environ.get("VIDEOS", ""), os.environ.get("FFMPEG", ""), os.environ.get("FFPROBE", "")
    for k, p in (("VIDEOS", v), ("FFMPEG", ff), ("FFPROBE", fp)):
        if not p or not os.path.exists(os.path.expanduser(p)):
            raise SystemExit(f"{k} not found: '{p}'  (edit the top of run.sh)")
    enc = subprocess.run([os.path.expanduser(ff), "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if "libsvtav1" not in enc:
        raise SystemExit("this ffmpeg has no libsvtav1 encoder")
    parts = int(os.environ.get("ENC_PARTS", 3))
    threads = max(1, (os.cpu_count() or 4) // parts)
    out = ROOT / "work" / "encoded_av1"
    out.mkdir(parents=True, exist_ok=True)
    (LOG / "E4_encode").mkdir(parents=True, exist_ok=True)
    procs = []
    for i in range(parts):
        cmd = [sys.executable, str(ROOT / "code" / "run_all.py"), "--videos", os.path.expanduser(v),
               "--out", str(out), "--ffmpeg", os.path.expanduser(ff), "--ffprobe", os.path.expanduser(fp),
               "--codec", "av1", "--chunk", "2", "--threads", str(threads), "--part", str(i), "--nparts", str(parts),
               "--subsample", os.environ.get("SUBSAMPLE", "1")]
        if SMOKE:
            cmd += ["--only", os.environ.get("SMOKE_VIDEOS", "")]
        procs.append(subprocess.Popen(cmd, stdout=open(LOG / "E4_encode" / f"part{i}.log", "a"),
                                      stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    n = len(glob.glob(str(out / "*" / "av1" / "chunks.csv")))
    if n < (1 if SMOKE else 46):
        raise SystemExit(f"only {n} of 46 videos encoded; see logs/E4_encode/*.log, then run again")


def run_stage(st):
    name = st["name"]
    done = RES / name / "DONE"
    if done.exists():
        return
    (RES / name).mkdir(parents=True, exist_ok=True)
    LOG.mkdir(exist_ok=True)
    log = LOG / f"{name}.log"
    print(f"[{time.strftime('%m-%d %H:%M')}] stage {name}", flush=True)
    t0 = time.time()
    if st["kind"] in ("sim", "drl_eval"):
        args = list(st["args"])
        if st["kind"] == "drl_eval":
            runs = drl_runs(st["form"], st["regime"])
            if not runs:
                raise SystemExit(f"no trained DRL runs for {st['form']}/{st['regime']}")
            args += ["--drl_runs", ",".join(runs)]
        py("cstudy.run", *args, log=log)
        for an in st["analyses"]:            # an analysis failure must not stop the simulations
            try:
                py("cstudy.analyze", an, RES / name, "--boot", "20" if SMOKE else "500", log=log)
            except SystemExit as e:
                (RES / name / f"ANALYSIS_FAILED_{an}").write_text(str(e))
                print(f"WARNING: analysis {an} of {name} failed (simulation results kept): {e}", flush=True)
    elif st["kind"] == "drl_train":
        run_drl_train(st)
    elif st["kind"] == "encode":
        run_encode(st)
    elif st["kind"] == "codec_data":
        py("cstudy.codec_data", "--h264", "data/content_library_x264_2s.csv", "--av1_dir", "work/encoded_av1",
           "--short", "data/svs_data_2s.pkl", "--out_dir", "work/data_codec", log=log)
        for f in ("av1_vs_h264_by_rung.csv", "content_library_av1_2s.csv"):
            src = ROOT / "work" / "data_codec" / f
            if src.exists():
                (RES / name / f).write_bytes(src.read_bytes())
    elif st["kind"] == "zip":
        make_zip(st["tag"])
    done.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + f"  {(time.time() - t0) / 60:.1f} min\n")


def make_zip(tag):
    server = os.environ.get("SERVER", "x")
    z = ROOT / f"results_{server}_{tag}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(RES.rglob("*")):
            rel = p.relative_to(ROOT)
            if p.is_dir() or "raw" in p.parts or p.suffix in (".npz",):
                continue
            if "E3_train" in p.parts and p.name == "weights.npz":
                continue
            zf.write(p, rel)
        for p in sorted(LOG.rglob("*.log")):
            zf.write(p, p.relative_to(ROOT))
        for f in ("PREREG_C.md", "MANIFEST.txt"):
            if (ROOT / f).exists():
                zf.write(ROOT / f, f)
    print(f"wrote {z.name} ({z.stat().st_size / 1e6:.1f} MB)", flush=True)


# ---------------------------------------------------------------------------------------------- status
def status():
    for server, stages in (("cpu2", cpu2_stages()), ("cpu1", cpu1_stages())):
        rows = []
        for st in stages:
            d = RES / st["name"]
            if not any((RES / s["name"]).exists() for s in stages):
                continue
            if st["kind"] == "zip":
                if (d / "DONE").exists():
                    rows.append(f"  {'-> results_' + server + '_' + st['tag'] + '.zip':28s} written")
                continue
            if (d / "DONE").exists():
                rows.append(f"  {st['name']:20s} done   {(d / 'DONE').read_text().strip()}")
            elif (d / "progress.json").exists():
                p = json.loads((d / "progress.json").read_text())
                rows.append(f"  {st['name']:20s} {p.get('done')}/{p.get('total')}  "
                            f"elapsed {p.get('elapsed_min', 0):.0f} min, ETA {p.get('eta_min', 0):.0f} min")
            elif st["kind"] == "encode" and (ROOT / "work" / "encoded_av1").exists():
                n = len(glob.glob(str(ROOT / "work" / "encoded_av1" / "*" / "av1" / "chunks.csv")))
                rows.append(f"  {st['name']:20s} {n}/46 videos encoded")
            else:
                rows.append(f"  {st['name']:20s} waiting")
        if rows:
            print(f"{server}:")
            print("\n".join(rows))
    if not RES.exists():
        print("nothing started yet")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "status":
        return status()
    if cmd == "zip":
        return make_zip("now")
    if cmd not in ("cpu1", "cpu2"):
        print(__doc__)
        return
    os.environ["SERVER"] = cmd
    for st in (cpu2_stages() if cmd == "cpu2" else cpu1_stages()):
        run_stage(st)
    print(f"[{time.strftime('%m-%d %H:%M')}] ALL DONE ({cmd})", flush=True)


if __name__ == "__main__":
    main()
