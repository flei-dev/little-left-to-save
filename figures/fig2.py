"""Study-design diagram (Fig. 2).  python3 fig2.py [out dir]"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
INK, INK2 = "#0b0b0b", "#52514e"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(7.16, 3.1))
ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

def box(x, y, w, h, title, body, ec=INK2, fc="#f7f7f5", tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", ec=ec, fc=fc, lw=1.0))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=6.3, weight="bold", color=tc)
    ax.text(x + w / 2, y + h - 6.4, body, ha="center", va="top", fontsize=5.3, color=INK2, linespacing=1.4)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=INK2))

# column 1: measurement
box(1, 21, 19, 18, "Phone measurement",
    "2 phones, LTE and 5G:\nradio tail, active power\n\n4 decoding paths:\nH.264 HW, H.264 SW,\nAV1 dav1d, AV1 system")
box(1, 1, 19, 17, "Inputs",
    "46 videos, 9 genres\nH.264 + AV1 ladders\nphone VMAF\nKuaiRand swipes\n4G / 5G / 3G traces")
# column 2: energy models
box(25, 1, 20, 38, "Energy models",
    "anchors:\nGalaxy S4 + lit. LTE\n(prior work)\nmodern + lit. LTE\nmodern + lit. 5G\nmodern + measured\n\n75-point grid:\ntail energy × 5\nactive power × 3\ndecode J/MB × 5")
# column 3: policy sets
box(50, 27, 22, 12, "U: energy-unaware", "tuned baselines\nfeed: Fixed, Dashlet, PDAS\nlong: Rate, BBA, BOLA, MPC", ec=INK2)
box(50, 14, 22, 12, "A: + energy-aware", "feed: DashletE, BurstE, SPARE\nlong: MPC-E, BurstMPC", ec=S1, fc="#eef4fc", tc=S1)
box(50, 1, 22, 12, "O: + clairvoyant", "knows watch times and\nnear-future throughput", ec=S2, fc="#fdf1ec", tc=S2)
# column 4: metric
box(77, 14, 22, 25, "Matched-QoE headroom",
    "on the Pareto front of U:\n\nH_A = mean of\n1 − E_A(q) / E_U(q)\n\nH_O: same with O\n\nsession bootstrap, 95% CI;\ncriteria pre-registered")
box(77, 1, 22, 11, "Also compared", "DQN vs tuned rules\nH.264 vs AV1 selection", ec=S3, fc="#edf8f3", tc=S3)
arrow(20.5, 30, 24.5, 30); arrow(20.5, 9.5, 24.5, 9.5)
for y in (33, 20, 7):
    arrow(45.5, 20, 49.5, y)
for y in (33, 20, 7):
    arrow(72.5, y, 76.5, 26)
import os, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig_study_design.png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
