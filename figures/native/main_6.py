# -*- coding: utf-8 -*-
"""Draw Figure 6 from published protein annotations and paired-residue measurements.

The panels include the Q9I398 chain example and the E1 and ESM-2 reference comparisons.
Run from the repository root with python -B figures/render.py 6.
"""
import sys, os, io, json, csv, math
from pathlib import Path
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from style import *
from PIL import Image

ROOT, F6D, OUT = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)

table = lambda n: list(csv.DictReader(io.StringIO((ROOT / n).read_bytes().decode("utf-8-sig")), delimiter="\t"))
W, H = 504.0, 398.0
cv = Canvas(W, H)
ov, txt = cv.ov, cv.txt
DASH = (0, (3, 2))
MINUS = lambda v: ("%g" % v).replace("-", "−")
CLS = (("alpha_helix", "Helix", ROSE, "o"), ("beta_strand", "Strand", BLUE, "s"), ("disorder", "Disorder", SAGE, "^"), ("turn", "Turn", LILAC, "D"))
RY, UY = 11.0, 206.0


def swarm(xpt, r, half):
    order = np.argsort(xpt, kind="mergesort")
    out, placed = np.zeros(len(xpt)), []
    step = 2 * r * 0.92
    cand, k = [0.0], 1
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


def lane(a, w, xlim, ylim, h, yc, vals, col, ms, half, alpha=0.75, marker="o"):
    xpt = (np.asarray(vals, float) - xlim[0]) / (xlim[1] - xlim[0]) * w
    assert xpt.min() > 0 and xpt.max() < w, (xpt.min(), xpt.max(), w)
    dy = swarm(xpt, ms * 0.5, half) * abs(ylim[1] - ylim[0]) / h
    a.plot(vals, yc + dy, marker=marker, ms=ms, mfc=col, mec="none", alpha=alpha, ls="none", zorder=3)


def glyph_key(x, y, entries, size=6.5, gap=7.0):
    for mk, col, ms, lab, fill in entries:
        ov.plot([x + 3.0], [y - 2.4], marker=mk, ms=ms, mfc=col if fill else "white", mec=col, mew=0.7, ls="none", zorder=30)
        x = cv.right_edge(txt(x + 8.0, y, lab, size)) + gap
    return x - gap


# ---------------------------------------------------------------- stored values
lab = [r for r in table("results/artwork/use/6b_and_s_t1_public_labels.tsv") if r["source"] == "Hallee" and r["status"] == "EVALUABLE"]
T = dict((r["unit_id"], r) for r in lab if r["frame"] == "T")
N = dict((r["unit_id"], r) for r in lab if r["frame"] == "NEW")
assert len(T) == 98 and set(T) == set(N)
yes = lambda r, k: r[k] == "True"
n_same, n_strong = sum(yes(r, "same_AA_flag") for r in T.values()), sum(yes(r, "strong_increment_flag") for r in T.values())
n_same2, n_strong2 = sum(yes(r, "same_AA_flag") for r in N.values()), sum(yes(r, "strong_increment_flag") for r in N.values())
med_acc = float(np.median([float(r["pair_accuracy"]) for r in T.values()]))
med_auc = float(np.median([float(r["auc_mean"]) for r in T.values()]))
own = [r for r in table("results/artwork/use/6c_and_s_t2_own_units.tsv") if r["model"] == "e1_150m" and r["concept"] == "OWN_HELIX_VS_BETA" and r["frame"] == "NEW"]
ref = [r for r in table("results/artwork/use/6c_and_s_t2_references_new.tsv") if r["model"] == "e1_150m" and r["concept"] == "OWN_HELIX_VS_BETA" and r["frame"] == "NEW"]
e_un = np.array([float(r["pair_accuracy"]) for r in ref if r["reference"].startswith("random")])
e_rot = np.array([float(r["pair_accuracy"]) for r in ref if r["reference"].startswith("rot")])
assert len(own) == 4 and len(e_un) == 199 and len(e_rot) == 9, (len(own), len(e_un), len(e_rot))
net = table("results/artwork/use/6d_network_results.tsv")
seq_correct = int([r for r in table("results/artwork/use/6d_classification_summary.tsv") if r["model"] == "sequence"][0]["correct"])
correct, un = {}, {}
for m in ("8m", "35m", "650m"):
    tr = [r for r in net if r["model"] == m and r["network_type"] == "trained"]
    assert len(tr) == 1
    correct[m] = int(tr[0]["pair_correct"])
    un[m] = [int(r["pair_correct"]) for r in net if r["model"] == m and r["network_type"] != "trained"]
    assert len(un[m]) == 199
J6 = json.load(io.open(F6D, encoding="utf-8"))
SEL = J6["selected"]
RES, PAIR = SEL["residues"], SEL["pair"]
print("labels: same-AA %d, beyond sequence %d (confirmation); %d, %d (new) | median accuracy %.3f, median AUROC %.3f" % (n_same, n_strong, n_same2, n_strong2, med_acc, med_auc))
print("E1 channels:", [(r["block0"], r["unit0"], round(float(r["pair_accuracy"]) * int(r["groups"]))) for r in own], "| untrained %.3f-%.3f, rotations %.3f-%.3f" % (e_un.min(), e_un.max(), e_rot.min(), e_rot.max()))
print("ESM-2 pairs:", seq_correct, correct, dict((m, (min(v), max(v))) for m, v in un.items()))

# =============================================================== a  a same-amino-acid pair on one protein
cv.head(0.0, RY, "a", "one pair", 0)
im = np.asarray(Image.open(io.BytesIO((ROOT / "data/artwork/generated/prot_exposure_pair_gray.png").read_bytes())).convert("RGBA"))
G = json.loads((ROOT / "data/artwork/generated/protein_exposure_pair_manifest.json").read_bytes().decode("utf-8"))
G = G[0] if isinstance(G, list) else G
wpt = 94.0
hpt = wpt * im.shape[0] / im.shape[1]
x0, y0 = 38.0, RY + 4.0
ov.imshow(im, extent=(x0, x0 + wpt, y0 + hpt, y0), zorder=22, interpolation="lanczos", aspect="auto")
ov.set_xlim(0, W)
ov.set_ylim(H, 0)
CELLW, wy = 8.2, y0 + hpt + 6.0
try:
    for (k, lab_), wx in zip((("buried", "Buried"), ("exposed", "Exposed")), (18.0, 96.0)):
        seq, anc = G["sites"][k]["window_sequence"], G["anchor_xy_px"][k]
        for q, ch_ in enumerate(seq):
            mid = (q == 3)
            ov.add_patch(Rectangle((wx + q * CELLW, wy), CELLW - 0.6, 10.0, facecolor=ORANGE if mid else "white", edgecolor=ORANGE if mid else GREY, linewidth=0.5, zorder=23))
            ov.text(wx + q * CELLW + (CELLW - 0.6) / 2.0, wy + 7.4, ch_, fontsize=6.5, color="white" if mid else INK, fontweight="bold" if mid else "normal", ha="center", va="baseline", zorder=30)
        cv.texts.append((ov.texts[-1], seq))
        ov.plot([x0 + anc[0] * wpt / im.shape[1], wx + 3.5 * CELLW - 0.3], [y0 + anc[1] * hpt / im.shape[0], wy - 0.8], color=ORANGE, lw=0.6, zorder=23)
        txt(wx + 3.5 * CELLW - 0.3, wy + 20.0, lab_, 7.5, ha="center")
except Exception as err:
    raise RuntimeError("Required exposure-pair geometry or site manifest is incomplete") from err
print("panel a: image %d x %d px, drawn %.0f x %.0f pt, lowest text at %.0f" % (im.shape[1], im.shape[0], wpt, hpt, wy + 20.0))

# =============================================================== b  98 published labels
cv.head(166.0, RY, "b", "published labels", 0)
BX, BY, BS = 200.0, 46.0, 110.0
LIM = (0.44, 1.0)
tk = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
b = cv.axes(BX, BY, BS, BS, LIM, LIM, tk, tk)
assert all(LIM[0] < float(r["auc_mean"]) < LIM[1] and LIM[0] < float(r["pair_accuracy"]) < LIM[1] for r in T.values())
b.plot(LIM, LIM, color=MUTED, lw=0.6, ls=DASH, zorder=1)
b.axhline(0.5, color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=1)
for cat, name, col, mk in CLS:
    for fill in (False, True):
        q = [r for r in T.values() if r["annotation_category"] == cat and yes(r, "strong_increment_flag") == fill]
        b.scatter([float(r["auc_mean"]) for r in q], [float(r["pair_accuracy"]) for r in q], s=11, marker=mk,
                  facecolor=col if fill else "white", edgecolor=col, linewidths=0.7, alpha=0.9, zorder=3 if fill else 2)
b.set_xlabel("All-residue AUROC", fontsize=7.5, labelpad=2)
b.set_ylabel("Same-amino-acid pair accuracy", fontsize=7.5, labelpad=1)
m_ = cv.axes(BX + BS + 3.0, BY, 15.0, BS, (0, 38), LIM, None, None, left=False)
m_.spines["bottom"].set_visible(False)
edges = np.arange(0.45, 1.0001, 0.05)
bot = np.zeros(len(edges) - 1)
for cat, name, col, mk in CLS:
    h = np.histogram([float(r["pair_accuracy"]) for r in T.values() if r["annotation_category"] == cat], bins=edges)[0]
    m_.barh(edges[:-1] + 0.025, h, height=0.044, left=bot, color=col, linewidth=0, zorder=2)
    bot += h
assert bot.max() <= 38 and bot.sum() == 98, (bot.max(), bot.sum())               # no bar is cut, every label is counted
glyph_key(BX - 20.0, RY + 13.0, [(mk, col, 3.6, name, True) for cat, name, col, mk in CLS])
glyph_key(BX - 20.0, RY + 24.0, [("o", MUTED, 3.6, "filled: also improves the sequence predictor", True)])
txt(BX + 5.0, BY + 10.0, "%d of 98 above chance" % n_same, 6.5)
txt(BX + 5.0, BY + 19.0, "%d beyond sequence" % n_strong, 6.5, color=MUTED)

# =============================================================== c  the audit as a matrix
cv.head(338.0, RY, "c", "audit matrix", 0)
MX, MW, GAP = 348.0, 86.0, 2.6
cw = (MW - 3 * GAP) / 98.0
x = MX
cols = []
for cat, name, col, mk in CLS:
    ids = sorted((u for u in T if T[u]["annotation_category"] == cat), key=lambda u: -float(T[u]["pair_accuracy"]))
    txt(x + len(ids) * cw / 2.0, RY + (13.0 if name in ("Helix", "Disorder") else 21.0), name, 6.5, ha="center", color=col)
    for u in ids:
        cols.append((x, u, col))
        x += cw
    x += GAP
BYc, BHc = RY + 26.0, 24.0                                # the accuracy profile
ov.plot([MX, MX + MW], [BYc + BHc, BYc + BHc], color=GREY, lw=0.5, zorder=22)
for x, u, col in cols:
    a_ = float(T[u]["pair_accuracy"])
    hh = (a_ - 0.5) / 0.5 * BHc
    ov.add_patch(Rectangle((x, BYc + BHc - max(hh, 0.0)), cw, abs(hh), facecolor=col, edgecolor="none", zorder=23))
txt(MX + MW + 4.0, BYc + 4.0, "1.0", 6.5, color=MUTED)
txt(MX + MW + 4.0, BYc + BHc + 2.4, "0.5", 6.5, color=MUTED)
txt(MX + MW + 20.0, BYc + BHc / 2.0 + 2.4, "Pair accuracy", 6.5, color=MUTED)
ROWS = (("1,099 confirmation groups", T, (("same_AA_flag", "Separates pair", n_same), ("strong_increment_flag", "Adds to sequence", n_strong))),
        ("New proteins", N, (("same_AA_flag", "Separates pair", n_same2), ("strong_increment_flag", "Adds to sequence", n_strong2))))
y = BYc + BHc + 16.0
for head_, frame, tests in ROWS:
    txt(MX, y - 3.0, head_, 6.5, color=MUTED)
    for key, name, n in tests:
        for x, u, col in cols:
            ov.add_patch(Rectangle((x, y), cw, 9.5, facecolor=col if yes(frame[u], key) else LIGHT, edgecolor="none", zorder=23))
        txt(MX + MW + 4.0, y + 7.2, name, 6.5)
        txt(W - 1.0, y + 7.2, "%d" % n, 7.5, "bold", ha="right")
        y += 12.0
    y += 14.0
txt(MX, y - 6.0, "One column per label, ordered by", 6.5, color=MUTED)
txt(MX, y + 2.5, "pair accuracy within its class", 6.5, color=MUTED)

# =============================================================== d  one new chain
cv.head(0.0, UY, "d", "one chain", 0)
pos = np.array([r["position1"] for r in RES])
val = np.array([r["signed_measured_readout"] for r in RES], float)
ann = [r["frozen_test_annotation"] for r in RES]
aa = [r["amino_acid"] for r in RES]
AAP = PAIR["amino_acid"]
hp, sp = int(PAIR["helix_position1"]), int(PAIR["strand_position1"])
assert aa[hp - 1] == AAP and aa[sp - 1] == AAP and ann[hp - 1] == "helix" and ann[sp - 1] == "strand"
same = [i for i in range(len(RES)) if aa[i] == AAP]
print("chain %s: %d residues; mean readout helix %.2f strand %.2f other %.2f | %s residues: helix %s strand %s" % (
    SEL["accession"], len(RES), val[[a == "helix" for a in ann]].mean(), val[[a == "strand" for a in ann]].mean(), val[[a == "other" for a in ann]].mean(), AAP,
    [round(float(val[i]), 2) for i in same if ann[i] == "helix"], [round(float(val[i]), 2) for i in same if ann[i] == "strand"]))
DX, DY, DW, DH = 30.0, UY + 48.0, 102.0, 104.0
YL = (math.floor(val.min()) - 0.5, math.ceil(val.max()) + 0.5)
d = cv.axes(DX, DY, DW, DH, (0, len(RES) + 1), YL, [50, 100, 150, 200, 250], [-4, 0, 4])
d.yaxis.set_major_formatter(FixedFormatter([MINUS(v) for v in (-4, 0, 4)]))
d.axhline(0, color=GREY, lw=0.5, zorder=1)
d.plot(pos, val, color=GREY, lw=0.5, zorder=2)
ACOL = {"helix": ROSE, "strand": BLUE, "other": GREY}
for key in ("other", "strand", "helix"):
    q = [i for i in same if ann[i] == key]
    d.scatter(pos[q], val[q], s=9 if key != "other" else 5, color=ACOL[key], edgecolors="white", linewidths=0.3, zorder=4 if key != "other" else 3)
d.scatter([hp, sp], [val[hp - 1], val[sp - 1]], s=30, facecolor="none", edgecolor=INK, linewidths=0.8, zorder=5)
d.set_xlabel("Residue position, %s" % SEL["accession"], fontsize=7.5, labelpad=2)
d.set_ylabel("Channel 1900 readout", fontsize=7.5, labelpad=1)
xp = lambda p: DX + p / float(len(RES) + 1) * DW
yp = lambda v: DY + DH * (1 - (v - YL[0]) / (YL[1] - YL[0]))
txt(xp(hp) - 5.0, yp(val[hp - 1]) - 8.0, "%s%d" % (AAP, hp), 6.5, color=ROSE, ha="right")
txt(xp(sp) + 5.0, yp(val[sp - 1]) + 12.0, "%s%d" % (AAP, sp), 6.5, color=BLUE)
TY = DY - 11.0                                            # the annotation track
ov.add_patch(Rectangle((xp(0.5), TY + 2.0), xp(len(RES) + 0.5) - xp(0.5), 1.0, facecolor=GREY, edgecolor="none", zorder=22))
i = 0
while i < len(RES):
    j = i
    while j + 1 < len(RES) and ann[j + 1] == ann[i]:
        j += 1
    if ann[i] != "other":
        ov.add_patch(Rectangle((xp(i + 0.5), TY), xp(j + 1.5) - xp(i + 0.5), 5.0, facecolor=ACOL[ann[i]], edgecolor="none", zorder=23))
    i = j + 1
kx = DX
for col, name in ((ROSE, "Helix"), (BLUE, "Strand")):
    ov.add_patch(Rectangle((kx, TY - 11.4), 6.4, 5.0, facecolor=col, edgecolor="none", zorder=30))
    kx = cv.right_edge(txt(kx + 9.0, TY - 6.6, name, 6.5)) + 8.0
ov.plot([kx + 3.0], [TY - 9.0], marker="o", ms=3.4, mfc=MUTED, mec="none", ls="none", zorder=30)
txt(kx + 8.0, TY - 6.6, "arginine" if AAP == "R" else AAP, 6.5)
hs = [i for i in same if ann[i] == "helix"]
ss = [i for i in same if ann[i] == "strand"]
jit = lambda n: np.linspace(-0.24, 0.24, n) if n > 1 else np.zeros(1)
ZX, ZW = DX + DW + 24.0, 42.0
ZL = (-0.55, 1.1)
assert ZL[0] < min(val[hs + ss]) and max(val[hs + ss]) < ZL[1]
zz = cv.axes(ZX, DY, ZW, DH, (-0.55, 1.55), ZL, None, [-0.4, 0, 0.4, 0.8])
zz.yaxis.set_major_formatter(FixedFormatter([MINUS(v) for v in (-0.4, 0, 0.4, 0.8)]))
zz.axhline(0, color=GREY, lw=0.5, zorder=1)
hx, sx = 0 + jit(len(hs)), 1 + jit(len(ss))
zz.plot([hx[hs.index(hp - 1)], sx[ss.index(sp - 1)]], [val[hp - 1], val[sp - 1]], color=INK, lw=0.6, zorder=2)
zz.scatter(hx, val[hs], s=12, color=ROSE, edgecolors="white", linewidths=0.3, zorder=3)
zz.scatter(sx, val[ss], s=12, color=BLUE, edgecolors="white", linewidths=0.3, zorder=3)
zz.scatter([hx[hs.index(hp - 1)], sx[ss.index(sp - 1)]], [val[hp - 1], val[sp - 1]], s=30, facecolor="none", edgecolor=INK, linewidths=0.8, zorder=5)
zz.xaxis.set_major_locator(FixedLocator([0, 1]))
zz.xaxis.set_major_formatter(FixedFormatter(["Helix", "Strand"]))
zz.tick_params(axis="x", labelsize=6.5, length=0, pad=3)
txt(ZX + ZW / 2.0, DY - 5.0, "Arginines", 6.5, ha="center", color=MUTED)
print("arginines: helix %s strand %s" % (np.round(val[hs], 2).tolist(), np.round(val[ss], 2).tolist()))

# =============================================================== e  four E1 channels against their references
cv.head(204.0, UY, "e", "E1 channels", 0)
EX, EW, EY, ER = 268.0, 78.0, UY + 34.0, 23.6
EL, EYL = (0.38, 0.95), (5.0, 0.0)
e = cv.axes(EX, EY, EW, 5 * ER, EL, EYL, xticks=[0.4, 0.6, 0.8], left=False)
e.axvspan(e_un.min(), e_un.max(), ymin=0, ymax=1, color=tint(BLUE, 0.16), lw=0, zorder=0)
e.axvspan(e_rot.min(), e_rot.max(), ymin=0, ymax=1, color=tint(ROSE, 0.22), lw=0, zorder=0)
e.axvline(0.5, color=MUTED, lw=0.6, ls=DASH, zorder=1)
ept = lambda v: EX + (v - EL[0]) / (EL[1] - EL[0]) * EW
for k, r in enumerate(sorted(own, key=lambda r: -float(r["pair_accuracy"]))):
    a_, n_ = float(r["pair_accuracy"]), int(r["groups"])
    yc = k + 0.5
    e.plot([a_], [yc], marker="o", ms=5.6, mfc=ORANGE, mec="white", mew=0.6, ls="none", zorder=6)
    txt(ept(a_) - 6.0, EY + yc * ER + 2.7, "%d" % round(a_ * n_), 7.5, "bold", color=TAB, ha="right")
    txt(EX - 5.0, EY + yc * ER + 2.7, "Block %d, %s" % (int(r["block0"]) + 1, r["unit0"]), 7.5, ha="right")
lane(e, EW, EL, EYL, 5 * ER, 4.5, e_un, BLUE, 2.0, 0.42 * ER)
e.plot(e_rot, [4.5] * len(e_rot), marker="^", ms=3.4, mfc=ROSE, mec="white", mew=0.3, ls="none", zorder=5)
txt(EX - 5.0, EY + 4.5 * ER - 1.4, "199 untrained", 6.5, ha="right", color=BLUE)
txt(EX - 5.0, EY + 4.5 * ER + 6.6, "9 rotations", 6.5, ha="right", color=ROSE)
e.set_xlabel("Accuracy on 140 new pairs", fontsize=7.5, labelpad=2)
txt(EX + EW, EY - 4.0, "correct of 140", 6.5, ha="right", color=MUTED)

# =============================================================== f  ESM-2 exposure pairs
cv.head(356.0, UY, "f", "ESM-2 exposure", 0)
FX, FW, FY, FR = 424.0, 74.0, UY + 34.0, 29.5
FL, FYL = (472.0, 652.0), (4.0, 0.0)
f_ = cv.axes(FX, FY, FW, 4 * FR, FL, FYL, xticks=[500, 550, 600, 650], left=False)
f_.axvline(967 / 2.0, color=MUTED, lw=0.6, ls=DASH, zorder=1)
fpt = lambda v: FX + (v - FL[0]) / (FL[1] - FL[0]) * FW
txt(fpt(967 / 2.0), FY - 4.0, "50%", 6.5, color=MUTED, ha="center")
f_.plot([seq_correct], [0.5], marker="o", ms=5.6, mfc=REF, mec="white", mew=0.6, ls="none", zorder=5)
txt(fpt(seq_correct) + 6.0, FY + 0.5 * FR + 2.7, "%d" % seq_correct, 7.5)
txt(FX - 5.0, FY + 0.5 * FR + 2.7, "Sequence alone", 7.5, ha="right")
for r_, (m, lab_) in enumerate((("8m", "+ 801 (8M)"), ("35m", "+ 814 (35M)"), ("650m", "+ 4566 (650M)"))):
    yc = r_ + 1.5
    lane(f_, FW, FL, FYL, 4 * FR, yc, un[m], BLUE, 2.0, 0.42 * FR)
    f_.plot([correct[m]], [yc], marker="o", ms=5.6, mfc=ORANGE, mec="white", mew=0.6, ls="none", zorder=6)
    above = correct[m] > max(un[m])
    txt(fpt(correct[m]) - 6.0 if above else fpt(max(un[m])) + 5.0, FY + yc * FR + 2.7, "%d" % correct[m], 7.5, "bold" if above else "normal", color=TAB if above else INK, ha="right" if above else "left")
    txt(FX - 5.0, FY + yc * FR + 2.7, lab_, 7.5, ha="right")
f_.set_xlabel("Pairs identified, of 967", fontsize=7.5, labelpad=2)
cv.finish(OUT, "Figure_6", write_labels=True)
