"""Energy models of the study: the E1 grid, the anchors, and the codec (E4) device variants."""

from __future__ import annotations

import os

from svs.model import DEVICES, RADIOS, Device, make_radio

P_BASE = 0.286 * 3.85                                      # W, playback base power of the modern phone (COCONUT)
C_GRID = (0.0444, 0.1, 0.2, 0.4, 0.65)               # J/MB decoded (0.0444 COCONUT HW decode ... 0.648 GreenABR S4)
TAILS = ((0.0, 0.0), (1.0, 0.25), (5.0, 1.06), (11.576, 1.06), (20.0, 2.12))   # (s, W): none, measured, ..., 5G
BETAS = (0.65, 1.29, 2.58)                           # W, active radio power: measured, LTE (Huang'12), 2x LTE

ANCHORS = {
    "meas": ("modern", "lte_meas"),        # our phone: modern device + measured LTE radio (= grid c0.0444_T1_b0.65)
    "modern_lte": ("modern", "lte"),       # modern device + literature LTE radio
    "modern_5g": ("modern", "5g"),         # modern device + literature-derived 5G NSA radio (20 s tail)
    "legacy_lte": ("legacy", "lte"),       # Galaxy S4 + literature LTE radio (the setting of most prior work)
}

# E4: extra decoding power of AV1 over hardware H.264 at 1080p [W]
AV1_DEC = {
    "hw": 0.0,         # AV1 hardware decoder
    "dav1d": 0.115,    # our phone, dav1d software decoder (sports +0.17 W, talk +0.06 W)
    "sysdec": 0.69,    # our phone, Android system AV1 decoder (sports +0.89 W, talk +0.49 W)
    "coconut": 1.43,   # COCONUT: AV1 software decoding +130 % of 1.10 W
}


def grid_name(c, T, b):
    return f"c{c:g}_T{T:g}_b{b:g}"


def registry():
    reg = {}
    for c in C_GRID:
        for T, P in TAILS:
            for b in BETAS:
                n = grid_name(c, T, b)
                reg[n] = (Device(n, P_BASE, c, P_BASE), make_radio(n, T, P, b))
    for n, (d, r) in ANCHORS.items():
        reg[n] = (DEVICES[d], RADIOS[r])
    for k, w in AV1_DEC.items():
        for scale in ("prop", "flat"):
            for rn in ("lte_meas", "lte"):
                base = DEVICES["modern"]
                n = f"av1{k}_{scale}_{rn}"
                reg[n] = (Device(n, base.p_base, base.c, base.p_stall, (0.0, w), scale), RADIOS[rn])
    return reg


def model_set(spec: str):
    """'grid' = 75 grid points + 4 anchors; 'anchors'; 'codec' (E4); or a comma-separated list of names."""
    reg = registry()
    if spec == "grid":
        g = [n for n in reg if n.startswith("c")]
        if os.environ.get("CSTUDY_SMOKE") == "1":
            g = g[::25]
        return g + list(ANCHORS)
    if spec == "anchors":
        return list(ANCHORS)
    if spec == "codec":
        c = [n for n in reg if n.startswith("av1") and (n.endswith("lte_meas") or "_prop_" in n)
             and not n.startswith("av1hw_flat")]
        return c[::4] if os.environ.get("CSTUDY_SMOKE") == "1" else c
    return spec.split(",")
