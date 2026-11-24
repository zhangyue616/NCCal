import sys, os, csv
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import numpy as np

NET, INF, ROT, OUT = sys.argv[1:5]
NAME = "Figure_S_G2"
W, H = 504.0, 118.0
cv = Canvas(W, H)
ov, txt = cv.ov, cv.txt
DASH = (0, (3, 2))
MINUS = lambda v: ("%g" % v).replace("-", "−")


def rows_of(path):
    with io.open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def swarm(xpt, r, half):
    order = np.argsort(xpt, kind="mergesort")
    out, placed = np.zeros(len(xpt)), []
    step = 2 * r * 0.92
    cand = [0.0]
    k = 1
    while k * step <= half + 1e-9:
        cand += [k * step, -k * step]
        k += 1
    for i in order:
        near = [(px, py) for px, py in placed if abs(px - xpt[i]) < 2 * r]
        for dy in cand:
            if all((px - xpt[i]) ** 2 + (py - dy) ** 2 >= (2 * r * 0.92) ** 2 for px, py in near):
                out[i] = dy
                break
        placed.append((xpt[i], out[i]))
    return out


net = rows_of(NET)
un = dict((m, [r for r in net if r["model"] == m]) for m in ("8m", "35m"))
assert all(len(un[m]) == 199 for m in un)
inf = dict(((r["model"], r["endpoint"]), r) for r in rows_of(INF))
rot = rows_of(ROT)
tr = dict((r["model"], r) for r in rot if r["basis"] == "native")
rots = dict((m, [r for r in rot if r["model"] == m and r["basis"] != "native"]) for m in ("8m", "35m"))
assert tr["8m"]["axis_zero_based"] == "801" and tr["35m"]["axis_zero_based"] == "814" and all(len(rots[m]) == 9 for m in rots)
ENDS = (("primary_gain", "mae_gain"), ("secondary_gain", "brier_gain"))
TRG = dict(((m, e), float(inf[m, e]["trained_gain"])) for m in un for _, e in ENDS)
assert all(abs(TRG[m, e] - float(tr[m][k])) < 1e-8 for m in un for k, e in ENDS)
rank = dict(((m, e), 1 + sum(float(r[e]) >= TRG[m, e] for r in un[m])) for m in un for _, e in ENDS)
assert all(rank[m, e] == 1 + int(inf[m, e]["untrained_ge"]) for m in un for _, e in ENDS)

x0 = 60.0
x = x0
for mk, col, ms, lab in (("o", ORANGE, 5.0, "Trained"), ("o", BLUE, 3.0, "Untrained"), ("^", ROSE, 3.6, "Rotated")):
    ov.plot([x + 3.0], [9.3], marker=mk, ms=ms, mfc=col, mec="none", ls="none", zorder=30)
    x = cv.right_edge(txt(x + 8.5, 12.0, lab, 7.5)) + 8.0
GY, GL = 24.0, 12.0
GH, GYL = 4 * GL, (4.0, 0.0)
for (k, e), gx, gw, gl, ticks, title in (
        (ENDS[0], 60.0, 200.0, (-0.95, 1.35), [-0.5, 0, 0.5, 1], "Exposure MAE reduction (× 0.001)"),
        (ENDS[1], 296.0, 200.0, (-16.0, 32.0), [-10, 0, 10, 20, 30], "Paired Brier reduction (× 0.001)")):
    g_ = cv.axes(gx, GY, gw, GH, gl, GYL, xticks=ticks, left=False)
    g_.xaxis.set_major_formatter(FixedFormatter([MINUS(v) for v in ticks]))
    g_.axvline(0, color=MUTED, lw=0.6, ls=DASH, zorder=1)
    g_.axhline(2.0, color=LIGHT, lw=0.6, zorder=1)
    for r_, m in enumerate(("8m", "35m")):
        vals = [1000 * float(r[e]) for r in un[m]]
        xpt = (np.asarray(vals) - gl[0]) / (gl[1] - gl[0]) * gw
        assert xpt.min() > 0 and xpt.max() < gw
        g_.plot(vals, 2 * r_ + 0.55 + swarm(xpt, 1.0, 0.45 * GL) * 4.0 / GH, marker="o", ms=2.0, mfc=BLUE, mec="none", alpha=0.75, ls="none", zorder=3)
        g_.plot([1000 * float(r[k]) for r in rots[m]], [2 * r_ + 1.5] * 9, marker="^", ms=3.4, mfc=ROSE, mec="none", alpha=0.9, ls="none", zorder=4)
        g_.plot([1000 * TRG[m, e]], [2 * r_ + 0.55], marker="o", ms=5.6, mfc=ORANGE, mec="white", mew=0.6, ls="none", zorder=5)
    txt(gx + gw / 2.0, GY + GH + 21.0, title, 8, ha="center")
for r_, lab in enumerate(("801 (8M)", "814 (35M)")):
    txt(52.0, GY + (2 * r_ + 1.0) * GL + 2.7, lab, 8, ha="right")

cv.finish(OUT, NAME, "endpoint ranks of 200: 8m %d and %d, 35m %d and %d | trained gains x1000: %s"
          % (rank["8m", "mae_gain"], rank["8m", "brier_gain"], rank["35m", "mae_gain"], rank["35m", "brier_gain"],
             " ".join("%.3f" % (1000 * TRG[m, e]) for m in ("8m", "35m") for _, e in ENDS)))
