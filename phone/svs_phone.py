#!/usr/bin/env python3
"""Radio energy measurement on an Android phone (Termux + Termux:API, no root).

  python svs_phone.py check            # 30 s self-check: current reading, download server, network type
  python svs_phone.py run --net 4g     # full measurement (~50 min, ~400 MB mobile data)
  python svs_phone.py run --net 4g --quick   # tail + bulk only (P1, P2, P3, P6): ~12 min, ~70 MB

Phases (all with the screen on, WiFi off, phone unplugged):
  P1 baseline 120 s                              -> idle power
  P2 tail: 8 x (200 KB download, 40 s idle)      -> tail time and tail power after a transfer
  P3 bulk: 3 x (20 MB download, 40 s idle)       -> active power vs throughput
  P4 pattern: same 18 MB as trickle (0.5 MB / 5 s) vs batch (6 MB / 60 s), 2 repetitions, alternating order
  P5 replay: download schedules of two simulated short-video sessions (SPARE vs BurstE), 300 s each
  P6 baseline 60 s                               -> drift check
Output: a folder and a zip in the phone's Download folder (svs_measure_<net>_<time>).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
URLS = [
    "https://mirrors.tuna.tsinghua.edu.cn/kernel/v6.x/linux-6.6.tar.xz",
    "https://mirrors.ustc.edu.cn/kernel.org/linux/kernel/v6.x/linux-6.6.tar.xz",
    "https://mirrors.aliyun.com/linux-kernel/v6.x/linux-6.6.tar.xz",
    "https://mirrors.huaweicloud.com/kernel/v6.x/linux-6.6.tar.xz",
]
SYS_I = ["/sys/class/power_supply/battery/current_now", "/sys/class/power_supply/bms/current_now",
         "/sys/class/power_supply/Battery/current_now"]
SYS_V = ["/sys/class/power_supply/battery/voltage_now", "/sys/class/power_supply/bms/voltage_now",
         "/sys/class/power_supply/Battery/voltage_now"]


def sh(cmd, timeout=20):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:      # noqa: BLE001
        return f"ERR {e}"


def battery_api():
    out = sh(["termux-battery-status"], timeout=10)
    try:
        return json.loads(out)
    except Exception:           # noqa: BLE001
        return {}


def find_sysfs(paths):
    for p in paths:
        try:
            with open(p) as f:
                int(f.read().strip())
            return p
        except Exception:       # noqa: BLE001
            continue
    return None


class Sampler(threading.Thread):
    """Logs (t, current, voltage, source) as fast as the source allows (sysfs ~5 Hz, Termux:API ~1 Hz)."""

    def __init__(self, path):
        super().__init__(daemon=True)
        self.pi, self.pv = find_sysfs(SYS_I), find_sysfs(SYS_V)
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["t", "current", "voltage", "src"])
        self.stop_flag = False
        self.n = 0

    def read(self):
        if self.pi:
            with open(self.pi) as f:
                i = int(f.read().strip())
            v = ""
            if self.pv:
                with open(self.pv) as f:
                    v = int(f.read().strip())
            return i, v, "sysfs"
        b = battery_api()
        return b.get("current", ""), b.get("voltage", ""), "api"

    def run(self):
        while not self.stop_flag:
            t0 = time.time()
            try:
                i, v, s = self.read()
                self.w.writerow([f"{t0:.3f}", i, v, s])
                self.n += 1
                if self.n % 50 == 0:
                    self.f.flush()
            except Exception:   # noqa: BLE001
                pass
            time.sleep(max(0.0, 0.2 - (time.time() - t0)))

    def close(self):
        self.stop_flag = True
        time.sleep(1.5)
        self.f.close()


class Runner:
    def __init__(self, outdir, url, size):
        self.url, self.size = url, size
        self.ev = open(os.path.join(outdir, "events.csv"), "w", newline="")
        self.w = csv.writer(self.ev)
        self.w.writerow(["phase", "label", "t_req", "t_start", "t_end", "bytes_req", "bytes", "http"])
        self.phase = ""

    def mark(self, label):
        t = time.time()
        self.w.writerow([self.phase, label, f"{t:.3f}", "", "", "", "", ""])
        self.ev.flush()
        print(f"{time.strftime('%H:%M:%S')}  {self.phase}: {label}", flush=True)

    def fetch(self, nbytes, label="", t_req=None):
        t_req = t_req or time.time()
        off = random.randint(0, max(0, self.size - nbytes - 1))
        t0 = time.time()
        out = sh(["curl", "-s", "-o", "/dev/null", "-r", f"{off}-{off + nbytes - 1}", "--max-time", "90",
                  "-w", "%{size_download} %{http_code}", self.url], timeout=100)
        t1 = time.time()
        try:
            got, code = out.split()
        except ValueError:
            got, code = 0, out[:20]
        self.w.writerow([self.phase, label, f"{t_req:.3f}", f"{t0:.3f}", f"{t1:.3f}", nbytes, got, code])
        self.ev.flush()
        return t1 - t0, int(float(got or 0))

    def idle(self, seconds):
        time.sleep(max(0.0, seconds))


def pick_url():
    for u in URLS:
        h = sh(["curl", "-sI", "--max-time", "10", u])
        size = 0
        for line in h.splitlines():
            if line.lower().startswith("content-length"):
                size = int(line.split(":")[1].strip())
        if size < 50_000_000:
            continue
        r = sh(["curl", "-s", "-o", "/dev/null", "-r", "0-99999", "--max-time", "15", "-w", "%{http_code}", u])
        if r == "206":
            return u, size
    return None, 0


def meta():
    return {
        "model": sh(["getprop", "ro.product.model"]), "brand": sh(["getprop", "ro.product.brand"]),
        "android": sh(["getprop", "ro.build.version.release"]),
        "telephony": sh(["termux-telephony-deviceinfo"], timeout=15),
        "battery": battery_api(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def check():
    print("1) Reading battery current ...")
    pi = find_sysfs(SYS_I)
    print("   sysfs current file:", pi or "not readable, using Termux:API (about 1 sample/s)")
    vals = []
    t_end = time.time() + 10
    while time.time() < t_end:
        if pi:
            with open(pi) as f:
                vals.append(int(f.read().strip()))
            time.sleep(0.2)
        else:
            vals.append(battery_api().get("current"))
    print(f"   {len(vals)} values in 10 s, first: {vals[:6]}")
    b = battery_api()
    print("   battery:", {k: b.get(k) for k in ("percentage", "plugged", "status", "current", "voltage")})
    print("2) Download server ...")
    u, size = pick_url()
    print("   available:", u, f"{size / 1e6:.0f} MB" if u else "none reachable")
    if u:
        t0 = time.time()
        sh(["curl", "-s", "-o", "/dev/null", "-r", "0-4999999", "--max-time", "60", u], timeout=70)
        print(f"   5 MB in {time.time() - t0:.1f} s, about {40 / max(time.time() - t0, 0.01):.1f} Mbps")
    print("3) Network type:", sh(["termux-telephony-deviceinfo"], timeout=15)[:400])
    print("\nCheck finished.")


def run(net, quick=False):
    plugged = str(battery_api().get("plugged", "")).upper()
    if plugged and plugged != "UNPLUGGED":
        sys.exit("The phone is charging; unplug it and run again.")
    url, size = pick_url()
    if not url:
        sys.exit("No download server is reachable; see the output of the check command.")
    sched = json.load(open(os.path.join(HERE, "replay_schedules.json")))
    stamp = time.strftime("%m%d_%H%M")
    root = os.path.expanduser("~/storage/downloads")
    if not os.path.isdir(root):
        root = os.path.expanduser("~")
    outdir = os.path.join(root, f"svs_measure_{net}_{stamp}")
    os.makedirs(outdir, exist_ok=True)
    m = meta()
    m.update(net=net, url=url, size=size)
    json.dump(m, open(os.path.join(outdir, "meta.json"), "w"), ensure_ascii=False, indent=1)
    sh(["termux-wake-lock"])
    smp = Sampler(os.path.join(outdir, "samples.csv"))
    smp.start()
    R = Runner(outdir, url, size)
    random.seed(7)
    t_start = time.time()
    try:
        mins = 12 if quick else 50
        print(f"Starting in 20 s, about {mins} min. Keep the screen on in Termux and do not touch the phone.", flush=True); time.sleep(20)
        R.phase = "P1_baseline"; R.mark("start"); R.idle(120); R.mark("end")

        R.phase = "P2_tail"
        for k in range(8):
            R.mark(f"trial{k}"); R.fetch(200_000, f"tail{k}"); R.idle(40)
        R.mark("end")

        R.phase = "P3_bulk"
        for k in range(3):
            R.mark(f"bulk{k}"); R.fetch(20_000_000, f"bulk{k}"); R.idle(40)
        R.mark("end")

        if not quick:
            R.phase = "P4_pattern"
            for rep, order in enumerate((("trickle", "batch"), ("batch", "trickle"))):
                for kind in order:
                    R.mark(f"{kind}_rep{rep}_start")
                    t0 = time.time()
                    if kind == "trickle":
                        for j in range(36):
                            R.idle(t0 + 5 * j - time.time()); R.fetch(500_000, f"{kind}{rep}", t0 + 5 * j)
                    else:
                        for j in range(3):
                            R.idle(t0 + 60 * j - time.time()); R.fetch(6_000_000, f"{kind}{rep}", t0 + 60 * j)
                    R.idle(t0 + 180 - time.time())
                    R.mark(f"{kind}_rep{rep}_end_transfers")
                    R.idle(40)
                    R.mark(f"{kind}_rep{rep}_end")

            R.phase = "P5_replay"
            keys = sorted(sched)
            sess = sorted({k.split("|")[0] for k in keys})
            order = [(sess[0], "SPARE"), (sess[0], "BurstE"), (sess[1], "BurstE"), (sess[1], "SPARE")]
            for s, pol in order:
                key = f"{s}|{pol}"
                R.mark(f"{key}_start")
                t0 = time.time()
                for t_rel, nb in sched[key]:
                    R.idle(t0 + t_rel - time.time())          # sequential: wait for the scheduled time or the previous download
                    R.fetch(int(nb), key, t0 + t_rel)
                R.idle(t0 + 300 - time.time())
                R.mark(f"{key}_end_window")
                R.idle(40)
                R.mark(f"{key}_end")

        R.phase = "P6_baseline"; R.mark("start"); R.idle(60); R.mark("end")
    except KeyboardInterrupt:
        R.mark("interrupted")
    finally:
        smp.close()
        R.ev.close()
        sh(["termux-wake-unlock"])
        m["battery_end"] = battery_api()
        m["minutes"] = (time.time() - t_start) / 60
        json.dump(m, open(os.path.join(outdir, "meta.json"), "w"), ensure_ascii=False, indent=1)
        z = shutil.make_archive(outdir, "zip", outdir)
        print(f"\nDone in {m['minutes']:.0f} min. Result: {z} (in the phone's Download folder).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "run"])
    ap.add_argument("--net", default="4g")
    ap.add_argument("--quick", action="store_true", help="only idle, tail and bulk phases (~12 min, ~70 MB)")
    a = ap.parse_args()
    check() if a.cmd == "check" else run(a.net, a.quick)


if __name__ == "__main__":
    main()
