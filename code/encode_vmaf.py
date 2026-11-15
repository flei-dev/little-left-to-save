#!/usr/bin/env python3
"""Encode one source video into a bitrate ladder and compute per-chunk size and VMAF.

For every rung: encode the whole video (keyframe every CHUNK seconds), take per-chunk byte sizes
from the packet list, compute per-frame VMAF (default model + phone model) against the source,
and average over each chunk.  Output: <out>/<name>/<codec>/chunks.csv  (+ timing.csv).

Only the Python standard library and an ffmpeg/ffprobe build with libvmaf are needed.

Example (Windows):
  python encode_vmaf.py --src videos\\youtube\\lqiN98z6Dak.mp4 --name lqiN98z6Dak --out encoded ^
         --ffmpeg tools\\ffmpeg.exe --ffprobe tools\\ffprobe.exe
Example (Linux server):
  python3 encode_vmaf.py --src videos/youtube/lqiN98z6Dak.mp4 --name lqiN98z6Dak --out encoded \
         --ffmpeg ./ffmpeg --ffprobe ./ffprobe --threads 16
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time

# (kbps, box width, box height) -- same bitrates as the simulator; Pensieve-style resolutions
LADDER = [(300, 426, 240), (750, 640, 360), (1200, 854, 480),
          (1850, 1280, 720), (2850, 1920, 1080), (4300, 1920, 1080)]

VMAF_MODELS = "version=vmaf_v0.6.1\\:name=vmaf|version=vmaf_v0.6.1\\:name=vmaf_phone\\:enable_transform=true"


def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        sys.stderr.write("COMMAND FAILED:\n" + " ".join(cmd) + "\n" + p.stderr[-3000:] + "\n")
        sys.exit(1)
    return p.stdout


def probe(ffprobe, path):
    out = run([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
               "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_packets:format=duration",
               "-of", "json", path])
    j = json.loads(out)
    s = j["streams"][0]
    num, den = (s.get("avg_frame_rate") or s["r_frame_rate"]).split("/")
    fps = float(num) / float(den) if float(den) else float(s["r_frame_rate"].split("/")[0])
    return dict(w=int(s["width"]), h=int(s["height"]), fps=fps,
                frames=int(s.get("nb_read_packets", 0)), dur=float(j["format"]["duration"]))


def fit(src_w, src_h, box_w, box_h):
    """Largest even size with the source aspect ratio inside the box, never above the source size."""
    s = min(box_w / src_w, box_h / src_h, 1.0)
    w = max(2, int(round(src_w * s / 2)) * 2)
    h = max(2, int(round(src_h * s / 2)) * 2)
    return w, h


def encoder_args(codec, kbps, fps, chunk):
    gop = max(1, int(round(fps * chunk)))
    kf = ["-force_key_frames", f"expr:gte(t,n_forced*{chunk})"]
    rate = ["-b:v", f"{kbps}k", "-maxrate", f"{int(kbps * 1.5)}k", "-bufsize", f"{kbps * 2}k"]
    if codec == "x264":
        return ["-c:v", "libx264", "-preset", "medium", "-sc_threshold", "0", "-g", str(gop)] + kf + rate
    if codec == "x265":
        return ["-c:v", "libx265", "-preset", "medium", "-x265-params",
                f"keyint={gop}:min-keyint={gop}:scenecut=0:log-level=error"] + kf + rate
    if codec == "vp9":
        return ["-c:v", "libvpx-vp9", "-deadline", "good", "-cpu-used", "2", "-row-mt", "1",
                "-g", str(gop), "-keyint_min", str(gop)] + kf + ["-b:v", f"{kbps}k", "-maxrate", f"{int(kbps * 1.5)}k"]
    if codec == "av1":
        return ["-c:v", "libsvtav1", "-preset", "8", "-svtav1-params", f"keyint={gop}:scd=0",
                "-b:v", f"{kbps}k"]
    raise ValueError(codec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="encoded")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--codec", default="x264", choices=["x264", "x265", "vp9", "av1"])
    ap.add_argument("--chunk", type=float, default=4.0)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--subsample", type=int, default=1, help="VMAF on every n-th frame (1 = all)")
    ap.add_argument("--seconds", type=float, default=0, help="only use the first N seconds (0 = all)")
    ap.add_argument("--keep", action="store_true", help="keep the encoded video files")
    a = ap.parse_args()

    src = os.path.abspath(a.src)
    # make tool paths absolute: the VMAF step runs inside the output folder
    for k in ("ffmpeg", "ffprobe"):
        p = getattr(a, k)
        if os.path.exists(p):
            setattr(a, k, os.path.abspath(p))
    info = probe(a.ffprobe, src)
    odir = os.path.abspath(os.path.join(a.out, a.name, a.codec))
    os.makedirs(odir, exist_ok=True)
    dur = info["dur"] if a.seconds <= 0 else min(info["dur"], a.seconds)
    n_chunks = int(math.floor(dur / a.chunk + 1e-6))
    limit = ["-t", f"{n_chunks * a.chunk:.3f}"]
    print(f"{a.name}: {info['w']}x{info['h']} {info['fps']:.3f} fps {info['dur']:.1f} s -> {n_chunks} chunks of {a.chunk:g} s, codec {a.codec}")

    rows, timing = [], []
    for kbps, bw, bh in LADDER:
        w, h = fit(info["w"], info["h"], bw, bh)
        ext = "webm" if a.codec == "vp9" else "mp4"
        enc = os.path.join(odir, f"{kbps}k.{ext}")
        t0 = time.time()
        run([a.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", src] + limit +
            ["-map", "0:v:0", "-an", "-fps_mode", "passthrough",
             "-vf", f"scale={w}:{h}:flags=bicubic,format=yuv420p", "-threads", str(a.threads)] +
            encoder_args(a.codec, kbps, info["fps"], a.chunk) + [enc])
        t_enc = time.time() - t0

        # per-chunk sizes from packet timestamps
        sizes = [0] * n_chunks
        pk = run([a.ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time,size",
                  "-of", "csv=p=0", enc])
        for line in pk.splitlines():
            parts = line.strip().split(",")
            if len(parts) < 2 or parts[0] in ("", "N/A"):
                continue
            c = int(float(parts[0]) / a.chunk + 1e-6)
            if 0 <= c < n_chunks:
                sizes[c] += int(parts[1])

        # per-frame VMAF, distorted upscaled to the source size (bicubic)
        t0 = time.time()
        log = f"vmaf_{kbps}k.json"
        # pair frames by index, not by timestamp: the encode and the source use different time bases,
        # and rounding made libvmaf compare frame n with frame n-1 for NTSC-type frame rates (bug found 2026-11-02)
        rate = f"{info['fps']:.9f}"
        # settb first so both streams get bit-identical timestamps (second bug, found 2026-11-03: with different
        # time bases the rounding of N/rate/TB still shifted one stream by a frame when rate is not exact)
        lavfi = (f"[0:v]scale={info['w']}:{info['h']}:flags=bicubic,format=yuv420p,settb=AVTB,setpts=N/({rate})/TB[d];"
                 f"[1:v]format=yuv420p,settb=AVTB,setpts=N/({rate})/TB[r];"
                 f"[d][r]libvmaf=model='{VMAF_MODELS}':feature='name=psnr':n_threads={a.threads}:n_subsample={a.subsample}"
                 f":log_fmt=json:log_path={log}")
        run([a.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", enc] + limit + ["-i", src] + limit +
            ["-lavfi", lavfi, "-f", "null", "-"], cwd=odir)
        t_vmaf = time.time() - t0

        with open(os.path.join(odir, log), encoding="utf-8") as f:
            frames = json.load(f)["frames"]
        acc = [[[], [], []] for _ in range(n_chunks)]
        for fr in frames:
            c = int(fr["frameNum"] / info["fps"] / a.chunk + 1e-6)
            if 0 <= c < n_chunks:
                m = fr["metrics"]
                acc[c][0].append(m["vmaf"])
                acc[c][1].append(m["vmaf_phone"])
                acc[c][2].append(m.get("psnr_y", float("nan")))
        for c in range(n_chunks):
            v, p, ps = acc[c]
            rows.append(dict(video=a.name, codec=a.codec, chunk=c, kbps=kbps, width=w, height=h,
                             size_bytes=sizes[c],
                             vmaf=round(sum(v) / len(v), 3) if v else "",
                             vmaf_phone=round(sum(p) / len(p), 3) if p else "",
                             vmaf_min=round(min(v), 3) if v else "",
                             psnr_y=round(sum(ps) / len(ps), 2) if ps else "",
                             psnr_y_min=round(min(ps), 2) if ps else "", n_frames=len(v)))
        mean_v = sum(r["vmaf"] for r in rows[-n_chunks:] if r["vmaf"] != "") / max(1, n_chunks)
        real_kbps = sum(sizes) * 8 / (n_chunks * a.chunk) / 1000
        timing.append(dict(kbps=kbps, width=w, height=h, real_kbps=round(real_kbps), vmaf_mean=round(mean_v, 2),
                           encode_s=round(t_enc, 1), vmaf_s=round(t_vmaf, 1), video_s=n_chunks * a.chunk))
        print(f"  {kbps:5d}k {w}x{h}: real {real_kbps:6.0f} kbps, VMAF {mean_v:6.2f}, "
              f"encode {t_enc:6.1f} s, vmaf {t_vmaf:6.1f} s")
        if not a.keep:
            os.remove(enc)

    for fn, data in (("chunks.csv", rows), ("timing.csv", timing)):
        with open(os.path.join(odir, fn), "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(data[0]))
            wr.writeheader()
            wr.writerows(data)
    tot = sum(t["encode_s"] + t["vmaf_s"] for t in timing)
    print(f"done: {os.path.join(odir, 'chunks.csv')}  ({tot / 60:.1f} min for {n_chunks * a.chunk:.0f} s of video, "
          f"{tot / (n_chunks * a.chunk):.2f} s per video-second)")


if __name__ == "__main__":
    main()
