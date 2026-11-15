"""Constants, QoE weights and the device / radio energy models of the short-video simulator.

All powers in W, energies in J, times in s, sizes in bytes, throughput in Mbps.
"""

from __future__ import annotations

from dataclasses import dataclass

CHUNK_S = 4.0                                         # default segment length [s] (content may override)
QOE_UNIT_S = 4.0                                      # SQoE-III quality term is per 4 s of playback
BITRATES = (300, 750, 1200, 1850, 2850, 4300)        # kbps (same ladder as the content library)
N_RUNGS = len(BITRATES)
# picture size of each rung relative to 1080p (ladder of encode_vmaf.py: 240p, 360p, 480p, 720p, 1080p, 1080p)
PX = tuple(w * h / (1920 * 1080) for w, h in ((426, 240), (640, 360), (854, 480), (1280, 720), (1920, 1080), (1920, 1080)))
RTT = 0.08                                            # request latency per chunk [s]
WINDOW = 5                                            # current video + 4 upcoming videos visible to the player

# SQoE-III linear QoE (Duanmu et al.), quality in VMAF per 4-s chunk
W_Q, W_RB, W_RBN, W_SM, W_SMJ = 0.07713539, 1.24971639, 2.87757412, 0.04938335, 1.436473


@dataclass(frozen=True)
class Device:
    """Playback energy while a video plays: p_base * seconds + c * MB of the decoded (watched) part
    + p_dec[codec] * (picture size relative to 1080p, or 1 if dec_scale == "flat") * seconds."""
    name: str
    p_base: float      # W, playback with screen on, independent of bitrate
    c: float           # J per MB decoded
    p_stall: float     # W while the player waits (screen on, rebuffering / startup)
    p_dec: tuple = (0.0, 0.0)   # extra decoding power [W] at 1080p per codec (0 = H.264, 1 = AV1)
    dec_scale: str = "prop"     # "prop": extra power proportional to the picture size; "flat": independent of it


# legacy: GreenABR Galaxy S4 fit (fit_energy_model.py, model M3): 1870.19 mJ per 4-s chunk + 648.35 mJ/MB;
#         stall power 1016 mW as in the ESM evaluation script.
# modern: COCONUT Android phone, ExoPlayer, x264 hardware decoding, 24 fps: 0.286 A at the lowest rung and
#         +0.024 A from 145 kbps to 16.8 Mbps; 3.85 V nominal battery voltage assumed (COCONUT gives current only).
#         -> 1.10 W base; 0.024 A * 3.85 V / 16.655 Mbps = 5.55 mW/Mbps = 44.4 mJ/MB.
DEVICES = {
    "legacy": Device("legacy", 1870.19 / 4 / 1000, 0.64835, 1.016),
    "modern": Device("modern", 0.286 * 3.85, 0.0444, 0.286 * 3.85),
}


@dataclass(frozen=True)
class Radio:
    """RRC-style radio power state machine: IDLE -> promotion -> ACTIVE (beta + alpha*thr) -> TAIL -> IDLE."""
    name: str
    p_idle: float
    t_pro: float
    p_pro: float
    beta: float        # W, active base power
    alpha: float       # W per Mbps of downlink throughput
    t_tail: float
    p_tail: float

    def wake_cost(self) -> float:
        """Energy of one promotion plus one full tail [J]."""
        return self.p_pro * self.t_pro + (self.p_tail - self.p_idle) * self.t_tail

    def active_energy(self, nbytes: float, thr_mbps: float) -> float:
        t = nbytes * 8e-6 / max(thr_mbps, 1e-3)
        return (self.beta + self.alpha * thr_mbps) * t


# lte : Huang et al., MobiSys 2012 (promotion 260 ms / 1210.7 mW, beta 1288.04 mW, alpha_d 51.97 mW/Mbps,
#       tail 11.576 s / 1060 mW, idle 11.36 mW).
# 5g  : NSA, derived from Xu et al., SIGCOMM 2020: tail ~20 s (vs ~10 s for 4G) and a 5G module drawing 2-3x the
#       4G power; we take the lower end (2x) for promotion, active base and tail power, the LTE alpha.
# wifi: Huang et al. 2012 WiFi model (beta 132.86 mW, alpha_d 137.01 mW/Mbps), tail power 120 mW;
#       0.24 s power-save timeout (assumption, typical PSM value), no promotion.
RADIOS = {
    "wifi": Radio("wifi", 0.0, 0.0, 0.0, 0.13286, 0.13701, 0.24, 0.12),
    "lte": Radio("lte", 0.01136, 0.26, 1.2107, 1.28804, 0.05197, 11.576, 1.06),
    "5g": Radio("5g", 0.01136, 0.26, 2 * 1.2107, 2 * 1.28804, 0.05197, 20.0, 2 * 1.06),
}
# measured on the user's phone (HUAWEI MAO-AL00, Android 12, China Mobile LTE, 2026-11-09; phone/analyze_phone.py):
# after a 200 KB transfer the excess power returns to idle within ~1-2 s (no long tail); active excess ~0.65 W
# at ~47 Mbps (3 bulk downloads; slope not identifiable).  First-second bump ~0.25 W taken as a 1-s tail.
RADIOS["lte_meas"] = Radio("lte_meas", 0.0, 0.0, 0.0, 0.65, 0.0, 1.0, 0.25)
# sensitivity variants (secondary analysis only)
RADIOS.update({
    "lte_tail5": Radio("lte_tail5", 0.01136, 0.26, 1.2107, 1.28804, 0.05197, 5.0, 1.06),
    "lte_tailpow50": Radio("lte_tailpow50", 0.01136, 0.26, 1.2107, 1.28804, 0.05197, 11.576, 0.53),
    "5g_tail10": Radio("5g_tail10", 0.01136, 0.26, 2 * 1.2107, 2 * 1.28804, 0.05197, 10.0, 2 * 1.06),
    "5g_x3": Radio("5g_x3", 0.01136, 0.26, 3 * 1.2107, 3 * 1.28804, 0.05197, 20.0, 3 * 1.06),
})


def make_radio(name: str, t_tail: float, p_tail: float, beta: float, alpha: float = 0.0) -> Radio:
    """Radio of the energy-model grid.  Tails of <= 1 s model a phone without an RRC-style tail (as measured):
    no promotion delay, zero idle power.  Longer tails get the LTE promotion (260 ms) at a power that scales
    with the tail power (1.2107 W at 1.06 W) and the LTE idle power."""
    if t_tail <= 1.0:
        return Radio(name, 0.0, 0.0, 0.0, beta, alpha, t_tail, p_tail)
    return Radio(name, 0.01136, 0.26, 1.2107 * p_tail / 1.06, beta, alpha, t_tail, p_tail)


def radio_energy(transfers, t_end: float, radio: Radio, full_tail: bool = True) -> float:
    """Total radio energy of a session.

    transfers: time-ordered list of (t_req, t_data_start, t_data_end, nbytes, promoted).
    full_tail: the tail after the last transfer is counted in full (the radio stays up after the session);
    False counts it only up to t_end (energy spent so far, used for per-step rewards).
    """
    e, last = 0.0, None
    for t_req, t_ds, t_de, nb, promo in transfers:
        if last is None:
            e += radio.p_idle * t_req
        else:
            gap = t_req - last
            if gap > radio.t_tail:
                e += radio.p_tail * radio.t_tail + radio.p_idle * (gap - radio.t_tail)
            else:
                e += radio.p_tail * gap
        pro = radio.t_pro if promo else 0.0
        e += radio.p_pro * pro
        e += radio.beta * (t_ds - t_req - pro)                    # request latency (RTT): connected
        dur = max(t_de - t_ds, 1e-9)
        e += (radio.beta + radio.alpha * nb * 8e-6 / dur) * dur
        last = t_de
    if last is None:
        return radio.p_idle * t_end
    if full_tail:
        e += radio.p_tail * radio.t_tail
    else:
        e += radio.p_tail * min(radio.t_tail, max(t_end - last, 0.0))
    if t_end > last + radio.t_tail:
        e += radio.p_idle * (t_end - last - radio.t_tail)
    return e
