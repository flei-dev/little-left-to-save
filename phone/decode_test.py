"""Phone power during real-time decoding (Termux).  Clips sports_{h264,av1}.mp4 and talk_{h264,av1}.mp4 in ~/svs/codec.
python decode_test.py [check]"""
import os, sys, time, subprocess, csv, shutil, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svs_phone import Sampler, battery_api, sh
D = os.path.expanduser("~/svs/codec")
def has(dec):
    return dec in subprocess.run(["ffmpeg", "-hide_banner", "-decoders"], capture_output=True, text=True).stdout
def ok(dec, f):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-c:v", dec, "-i", f, "-t", "2", "-f", "null", "-"], capture_output=True, text=True)
    return r.returncode == 0
tests = []
for g in ("sports", "talk"):
    for dec, f in (("h264", "h264"), ("h264_mediacodec", "h264"), ("libdav1d", "av1"), ("av1_mediacodec", "av1")):
        p = os.path.join(D, f"{g}_{f}.mp4")
        if has(dec) and ok(dec, p):
            tests.append((g, dec, p))
print("testable:", [(g, d) for g, d, _ in tests], flush=True)
if len(sys.argv) > 1 and sys.argv[1] == "check":
    sys.exit()
out = os.path.expanduser(f"~/storage/downloads/svs_dec_{time.strftime('%m%d_%H%M')}")
os.makedirs(out, exist_ok=True)
sh(["termux-wake-lock"])
smp = Sampler(os.path.join(out, "samples.csv")); smp.start()
ev = open(os.path.join(out, "events.csv"), "w", newline=""); w = csv.writer(ev); w.writerow(["label", "t"])
def mark(l):
    w.writerow([l, f"{time.time():.3f}"]); ev.flush(); print(time.strftime("%H:%M:%S"), l, flush=True)
try:
    print("Starting, about 20 min; do not touch the phone.", flush=True)
    time.sleep(5); mark("baseline_start"); time.sleep(60); mark("baseline_end")
    for rep, order in enumerate((tests, tests[::-1])):
        for g, dec, p in order:
            mark(f"{g}|{dec}|rep{rep}|start")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-re", "-c:v", dec, "-i", p, "-f", "null", "-"])
            mark(f"{g}|{dec}|rep{rep}|end"); time.sleep(10)
    mark("baseline2_start"); time.sleep(60); mark("baseline2_end")
finally:
    smp.close(); ev.close(); sh(["termux-wake-unlock"])
    json.dump({"battery_end": battery_api()}, open(os.path.join(out, "meta.json"), "w"))
    print("done:", shutil.make_archive(out, "zip", out))
