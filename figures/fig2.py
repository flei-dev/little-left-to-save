"""Study-design figure (Fig. 3), double column.  python3 fig2.py [out dir]"""
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
INK, INK2, LINE = "#0b0b0b", "#52514e", "#77766f"
LANE = {1: ("#eb6834", "#fdf1ec"), 2: ("#c98a00", "#fdf6e3"), 3: ("#2a78d6", "#eef4fc"), 4: ("#14936a", "#edf8f3")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "mathtext.default": "regular"})
W, H = 7.16, 3.78
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1])
YT = 200 * H / W
ax.set_xlim(0, 200); ax.set_ylim(0, YT); ax.axis("off")
LH, TH, PAD = 3.25, 6.0, 2.2   # line height, title block, bottom pad (axis units)

def height(title, body):
    n = 0 if body is None else body.count("\n") + 1
    if body is None:
        return title.count("\n") * LH + LH + 2 * PAD
    return TH + n * LH + PAD

def lane(x, w, k, title):
    ec, fc = LANE[k]
    ax.add_patch(FancyBboxPatch((x, 1.5), w, YT - 3.5, boxstyle="round,pad=0,rounding_size=2", fc=fc, ec="none", zorder=0))
    ax.add_patch(FancyBboxPatch((x, YT - 10), w, 8, boxstyle="round,pad=0,rounding_size=2", fc=ec, ec="none", zorder=1))
    ax.text(x + w / 2, YT - 6, title, ha="center", va="center", fontsize=7.5, color="white", fontweight="bold", zorder=2)

def box(x, ytop, w, title, body=None, k=None, strong=False, fc="white", h=None):
    h = h or height(title, body)
    y = ytop - h
    ec = LANE[k][0] if k else LINE
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.3", fc=fc, ec=ec,
                                lw=1.2 if strong else 0.7, zorder=3))
    if body is None:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=7, color=INK, zorder=4, linespacing=1.2)
    else:
        ax.text(x + w / 2, y + h - 1.5, title, ha="center", va="top", fontsize=7, color=INK, fontweight="bold", zorder=4)
        ax.text(x + w / 2, y + h - TH, body, ha="center", va="top", fontsize=7, color=INK2, zorder=4, linespacing=1.2)
    return (x, y, w, h)

def arr(p, q, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=LINE, zorder=2.5,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0.6, shrinkB=0.6))
R = lambda b: (b[0] + b[2], b[1] + b[3] / 2)
L = lambda b: (b[0], b[1] + b[3] / 2)
T = lambda b: (b[0] + b[2] / 2, b[1] + b[3])
B = lambda b: (b[0] + b[2] / 2, b[1])
G = 4.5          # vertical gap
top = YT - 14

# lanes  (x, w)
X1, W1, X2, W2, X3, W3, X4, W4 = 1, 41, 45, 39, 87, 67, 157, 42
lane(X1, W1, 1, "Measure (Sec. III)"); lane(X2, W2, 2, "Model (Sec. III)")
lane(X3, W3, 3, "Simulate (Sec. IV)"); lane(X4, W4, 4, "Analyze (Sec. V)")

# lane 1
bw = W1 - 6; x = X1 + 3
p1 = box(x, top, bw / 2 - 1, "Huawei\nLTE", k=1)
p2 = box(x + bw / 2 + 1, top, bw / 2 - 1, "Redmi\nLTE, 5G", k=1)
lg = box(x, p1[1] - G, bw, "Termux logger", "battery current\n≈1 sample/s", k=1)
m1 = box(x, lg[1] - G, bw, "Radio tests", "200-KB transfers\nbulk transfers\n→ tail energy,\n   transfer power", k=1, strong=True)
m2 = box(x, m1[1] - G, bw, "Decoding tests", "1080p, 3 decoders\n→ extra decoding\n   power", k=1, strong=True)
arr(B(p1), (p1[0] + p1[2] / 2, lg[1] + lg[3])); arr(B(p2), (p2[0] + p2[2] / 2, lg[1] + lg[3]))
arr(B(lg), T(m1)); arr(B(m1), T(m2))

# lane 2
bw = W2 - 6; x = X2 + 3
rm = box(x, top, bw, "Radio model", "$P_{pro}, T_{pro}, \\beta, \\alpha,$\n$P_{tail}, T_{tail}$\nidle → promotion\n→ transfer → tail", k=2)
dm = box(x, rm[1] - G, bw, "Device model", "$p_{base}W + cB$\n$+\\, p_{dec}\\phi W + p_{stall}S$", k=2)
an = box(x, dm[1] - G, bw, "4 anchors", "prior-work\nmodern + lit. LTE\nmodern + lit. 5G\nmeasured", k=2, strong=True)
gr = box(x, an[1] - G, bw, "75-point grid", "5 tails × 3 $\\beta$\n× 5 values of $c$", k=2, strong=True)
arr(R(m1), L(rm)); arr(R(m2), L(dm)); arr(B(rm), T(dm)); arr(B(dm), T(an)); arr(B(an), T(gr))

# lane 3
x = X3 + 3; bw = W3 - 6; iw = (bw - 4) / 3
d1 = box(x, top, iw, "46 videos\nH.264, AV1", k=3)
d2 = box(x + iw + 2, top, iw, "KuaiRand\nswipes", k=3)
d3 = box(x + 2 * iw + 4, top, iw, "LTE, 5G,\nlow-thr.", k=3)
core = box(x, d1[1] - G, bw, "Unified session simulator", "feed: 20 views per session\nlong-form: 300-s windows", k=3, strong=True, fc="#dbe8f8")
for b in (d1, d2, d3):
    arr(B(b), (b[0] + b[2] / 2, core[1] + core[3]))
arr(R(an), (core[0] - 0.2, core[1] + core[3] / 2), rad=0.0)
pw = (bw - 3) / 2
pu = box(x, core[1] - G, pw, "U: unaware", "Fixed, Dashlet,\nPDAS, Rate, BBA,\nBOLA, MPC", k=3)
pa = box(x + pw + 3, core[1] - G, pw, "A: + aware", "DashletE, BurstE,\nSPARE, MPC-E,\nBurstMPC", k=3)
po = box(x, pu[1] - G, pw, "O: + clairvoyant", "knows watch times\nand throughput", k=3)
pl = box(x + pw + 3, pa[1] - G, pw, "L: learned", "96 DQNs\n4 regimes", k=3, h=po[3])
cx = box(x, po[1] - G, bw, "Codec modes", "H.264 · AV1 · both;  4 AV1 decoders", k=3)
arr((pu[0] + pu[2] / 2, core[1]), T(pu)); arr((pa[0] + pa[2] / 2, core[1]), T(pa))
arr(B(pu), T(po)); arr(B(pa), T(pl))

# lane 4
x = X4 + 3; bw = W4 - 6
f1 = box(x, top, bw, "Pareto fronts", "QoE vs. energy\nper policy set", k=4)
f2 = box(x, f1[1] - G, bw, "Headroom", "$H_X=\\overline{1-E_X(q)/E_U(q)}$\nX = A or O", k=4, strong=True, fc="#d9f1e6")
f3 = box(x, f2[1] - G, bw, "Uncertainty", "session bootstrap\npre-registered tests", k=4)
ow = (bw - 2) / 2
o1 = box(x, f3[1] - G, ow, "Map,\nanchors", k=4)
o2 = box(x + ow + 2, f3[1] - G, ow, "Energy\nsplit", k=4)
o3 = box(x, o1[1] - 2.5, ow, "DQN,\ncodecs", k=4)
o4 = box(x + ow + 2, o2[1] - 2.5, ow, "Robust-\nness", k=4)
arr(R(core), L(f1), rad=0.2); arr(R(pa), L(f2), rad=0.0); arr(R(pl), L(f3), rad=0.0)
arr(B(f1), T(f2)); arr(B(f2), T(f3)); arr(R(cx), L(o3), rad=0.0); arr((o1[0] + ow / 2, f3[1]), T(o1)); arr((o2[0] + ow / 2, f3[1]), T(o2))
fig.savefig(os.path.join(OUT, "fig_study_design.png"), dpi=300)

