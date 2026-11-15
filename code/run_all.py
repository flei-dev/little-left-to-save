#!/usr/bin/env python3
"""Run encode_vmaf.py over every video in a folder tree (skips videos already finished).

Split the work over several servers with --part/--nparts, e.g. server A: --part 0 --nparts 2,
server B: --part 1 --nparts 2.  Videos are assigned round-robin in name order, so the split is stable.

  python3 run_all.py --videos videos --out encoded --ffmpeg ./ffmpeg/ffmpeg --ffprobe ./ffmpeg/ffprobe \
          --part 0 --nparts 2 --threads 32 [--chunk 2]
"""
import argparse
import os
import subprocess
import sys
import time

EXT = (".mp4", ".mkv", ".mov", ".webm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="videos")
    ap.add_argument("--out", default="encoded")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--codec", default="x264")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--subsample", type=int, default=1)
    ap.add_argument("--chunk", type=float, default=4.0, help="segment length in seconds")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--only", default="", help="comma-separated names to run (default: all)")
    a = ap.parse_args()

    files = []
    for root, _, names in os.walk(a.videos):
        for n in names:
            if n.lower().endswith(EXT) and ".full." not in n:
                files.append(os.path.join(root, n))
    files.sort(key=lambda p: os.path.basename(p).lower())
    if a.only:
        keep = set(a.only.split(","))
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] in keep]
    mine = [f for i, f in enumerate(files) if i % a.nparts == a.part]
    print(f"{len(files)} videos found, {len(mine)} in part {a.part}/{a.nparts}", flush=True)

    here = os.path.dirname(os.path.abspath(__file__))
    for k, f in enumerate(mine, 1):
        name = os.path.splitext(os.path.basename(f))[0]
        done = os.path.join(a.out, name, a.codec, "chunks.csv")
        if os.path.exists(done):
            print(f"[{k}/{len(mine)}] skip {name} (done)", flush=True)
            continue
        print(f"[{k}/{len(mine)}] {name}  {time.strftime('%H:%M:%S')}", flush=True)
        r = subprocess.run([sys.executable, os.path.join(here, "encode_vmaf.py"), f"--src={f}", f"--name={name}",
                            "--out", a.out, "--ffmpeg", a.ffmpeg, "--ffprobe", a.ffprobe, "--codec", a.codec,
                            "--threads", str(a.threads), "--subsample", str(a.subsample), "--chunk", str(a.chunk)])
        if r.returncode != 0:
            print(f"  FAILED {name}", flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
