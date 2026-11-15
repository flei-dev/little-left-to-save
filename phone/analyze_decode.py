#!/usr/bin/env python3
"""Mean phone power while decoding each clip in real time (output of decode_test.py).

  python3 analyze_decode.py [folder]          (default: data/huawei_decode)

Power = |battery current| x voltage, averaged over each labelled 60-s interval (skipping the first 3 s for
decoder start-up and the last 1 s) and then over the two repetitions.
Prints the table used in Section III-C of the paper.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyze_phone import load  # noqa: E402

SKIP_START, SKIP_END = 3.0, 1.0   # s


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "data", "huawei_decode")
    s, e, _ = load(path)
    t = {r.label: r.t for r in e.itertuples()}
    rows = []
    for lab, t0 in t.items():
        if not lab.endswith("|start"):
            continue
        key = lab[: -len("|start")]
        t1 = t[key + "|end"]
        clip, dec, rep = key.split("|")
        w = s[(s.t >= t0 + SKIP_START) & (s.t <= t1 - SKIP_END)]
        rows.append(dict(clip=clip, decoder=dec, rep=rep, p_w=w.P.mean()))
    d = pd.DataFrame(rows)
    tab = d.groupby(["decoder", "clip"]).p_w.mean().unstack("clip").round(3)
    print(tab.to_string())
    base = tab.loc["h264_mediacodec"]
    for dec in ("libdav1d", "av1_mediacodec"):
        extra = tab.loc[dec] - base
        print(f"{dec}: extra power over hardware H.264 = {extra.round(3).to_dict()}, mean {extra.mean():.3f} W")


if __name__ == "__main__":
    main()
