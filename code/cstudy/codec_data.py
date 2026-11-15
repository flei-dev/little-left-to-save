"""E4 data: add the AV1 encodes (2-s chunks, same 6-rung ladder) to the 2-s H.264 library.

Every clip / long-form window gets 12 rungs ordered by bitrate, H.264 then AV1 at each bitrate
(kbps = 300, 300, 750, 750, ...; codec = 0, 1, 0, 1, ...).  Short-video clips keep the source segment of
svs_data_2s.pkl (found by matching the H.264 chunk sizes), so the H.264-only mode of E4 is exactly the
2-s feed of E5.

  python3 -m cstudy.codec_data --h264 content_library_x264_2s.csv --av1_dir encoded_av1 \
      --short svs_data_2s.pkl --out_dir data_codec
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd

from svs.model import BITRATES, PX


def merge_av1(av1_dir, h264_csv, out_csv):
    genre = pd.read_csv(h264_csv, usecols=["video", "genre"]).drop_duplicates().set_index("video").genre
    frames = []
    for f in sorted(glob.glob(os.path.join(av1_dir, "*", "av1", "chunks.csv"))):
        d = pd.read_csv(f)
        v = str(d.video.iloc[0])
        if v not in genre.index:
            continue
        d["genre"] = genre[v]
        frames.append(d)
    if not frames:
        raise SystemExit(f"no AV1 chunks.csv under {av1_dir}")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_csv, index=False)
    return df


def interleave(h, a):
    n = min(h.shape[0], a.shape[0])
    out = np.empty((n, 2 * h.shape[1]))
    out[:, 0::2], out[:, 1::2] = h[:n], a[:n]
    return out


KBPS = np.repeat(np.array(BITRATES, float), 2)
CODEC = np.tile([0, 1], len(BITRATES))
PXS = np.repeat(np.array(PX, float), 2)


def find_offset(lib_sz, clip_sz):
    n = clip_sz.shape[0]
    hits = np.flatnonzero(np.all(lib_sz[:, :] == clip_sz[0][None, :], axis=1))
    for o in hits:
        if o + n <= lib_sz.shape[0] and np.array_equal(lib_sz[o:o + n], clip_sz):
            return int(o)
    return None


def build(h264_csv, av1_csv, short_pkl, out_dir):
    from svs.content import Clip, load_library
    from svs import longform
    hl, al = load_library(h264_csv), load_library(av1_csv)
    os.makedirs(out_dir, exist_ok=True)
    # ---- short-video feed
    d = pickle.load(open(short_pkl, "rb"))
    clips, bad = {}, set()
    for vid, c in d["clips"].items():
        if c.src not in hl or c.src not in al:
            bad.add(vid)
            continue
        o = find_offset(hl[c.src][1], c.sizes)
        g, asz, aq, aqs = al[c.src]
        if o is None or o + c.n > asz.shape[0]:
            bad.add(vid)
            continue
        sl = slice(o, o + c.n)
        clips[vid] = Clip(c.vid, c.genre, c.src, interleave(c.sizes, asz[sl]), interleave(c.q, aq[sl]),
                          interleave(c.q_std, aqs[sl]), c.S, c.cumS, c.chunk_s, KBPS, CODEC, PXS)
    sessions, split = [], {"dev": [], "main": []}
    for i, (uid, items) in enumerate(d["sessions"]):
        if any(v in bad for v, _ in items):
            continue
        split["dev" if i % 2 == 0 else "main"].append(len(sessions))
        sessions.append((uid, items))
    pickle.dump({"clips": clips, "sessions": sessions, "splits": split, "form": "short"},
                open(os.path.join(out_dir, "svs_data_codec.pkl"), "wb"))
    print(f"feed: {len(clips)} clips with AV1, {len(bad)} without; sessions kept {len(sessions)} of "
          f"{len(d['sessions'])} (main {len(split['main'])})")
    # ---- long form: same windows as long_2s.pkl
    lclips, lsess, lsplit = longform.build(h264_csv, 2.0)
    keep_c, ls2, sp2 = {}, [], {"dev": [], "test": [], "main": []}
    for i, (cid, items) in enumerate(lsess):
        c = lclips[cid]
        if c.src not in al:
            continue
        o = find_offset(hl[c.src][1], c.sizes)
        g, asz, aq, aqs = al[c.src]
        if o is None or o + c.n > asz.shape[0]:
            continue
        sl = slice(o, o + c.n)
        keep_c[cid] = Clip(cid, c.genre, c.src, interleave(c.sizes, asz[sl]), interleave(c.q, aq[sl]),
                           interleave(c.q_std, aqs[sl]), c.S, c.cumS, c.chunk_s, KBPS, CODEC, PXS)
        for k in ("dev", "test", "main"):
            if i in lsplit[k]:
                sp2[k].append(len(ls2))
        ls2.append((cid, items))
    pickle.dump({"clips": keep_c, "sessions": ls2, "splits": sp2, "form": "long"},
                open(os.path.join(out_dir, "long_codec.pkl"), "wb"))
    print(f"long form: {len(ls2)} of {len(lsess)} windows with AV1")
    # ---- rate-quality summary (AV1 vs H.264 per rung)
    h = pd.read_csv(h264_csv)
    a = pd.read_csv(av1_csv)
    m = h.merge(a, on=["video", "chunk", "kbps"], suffixes=("_h264", "_av1"))
    s = m.groupby("kbps").agg(size_h264=("size_bytes_h264", "mean"), size_av1=("size_bytes_av1", "mean"),
                              vmaf_h264=("vmaf_h264", "mean"), vmaf_av1=("vmaf_av1", "mean"),
                              vmafp_h264=("vmaf_phone_h264", "mean"), vmafp_av1=("vmaf_phone_av1", "mean"))
    s["size_ratio"] = s.size_av1 / s.size_h264
    s.round(3).to_csv(os.path.join(out_dir, "av1_vs_h264_by_rung.csv"))
    print(s.round(3).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h264", required=True)
    ap.add_argument("--av1_dir", default="")
    ap.add_argument("--av1_csv", default="")
    ap.add_argument("--short", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    av1_csv = a.av1_csv or os.path.join(a.out_dir, "content_library_av1_2s.csv")
    if a.av1_dir:
        merge_av1(a.av1_dir, a.h264, av1_csv)
    build(a.h264, av1_csv, a.short, a.out_dir)


if __name__ == "__main__":
    main()
