import sys, os, csv
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
from PIL import Image
import numpy as np

(D3P, PROT, MANP, NETREF, BASIS, PAIRS, CLS, ACT, NET, PROT2, MAN2P, SCALE, OUT) = sys.argv[1:14]
D3, MAN, MAN2 = J(D3P), J(MANP), J(MAN2P)[0]
NAME = "Protein_panels"
W, H = 504.0, 404.0
cv = Canvas(W, H)
ov, txt = cv.ov, cv.txt
DASH = (0, (3, 2))
REFS = (("X", "gX", "Composition"), ("S", "gS", "Sequence window"), ("L", "gL", "Interaction"))
MINUS = lambda v: ("%g" % v).replace("-", "−")
RY = 11.0                                               # base line of the panel letters of the upper row
UY = 230.0                                              # base line of the panel letters of the lower row


def rows_of(path, delim):
    with io.open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=delim))


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


def lane(a, w, xlim, ylim, h, yc, vals, col, ms, half, alpha=0.75):
    xpt = (np.asarray(vals, float) - xlim[0]) / (xlim[1] - xlim[0]) * w
    assert xpt.min() > 0 and xpt.max() < w, (xpt.min(), xpt.max(), w)        # every network is inside the axes
    dy = swarm(xpt, ms * 0.5, half) * abs(ylim[1] - ylim[0]) / h
    a.plot(vals, yc + dy, marker="o", ms=ms, mfc=col, mec="none", alpha=alpha, ls="none", zorder=3)


def glyph_key(x, y, entries, size=7.5, gap=8.0):
    for mk, col, ms, lab in entries:
        ov.plot([x + 3.0], [y - 2.7], marker=mk, ms=ms, mfc=col, mec="none", ls="none", zorder=30)
        x = cv.right_edge(txt(x + 8.5, y, lab, size)) + gap
    return x - gap


def protein(path, man_sites, x0, y0, wpt, wy, order, masked):
    im = np.asarray(Image.open(path).convert("RGBA"))
    hpt = wpt * im.shape[0] / im.shape[1]
    ov.imshow(im, extent=(x0, x0 + wpt, y0 + hpt, y0), zorder=22, interpolation="lanczos", aspect="auto")
    ov.set_xlim(0, W)
    ov.set_ylim(H, 0)
    for (seq, anchor, lab), wx in zip(order, (10.0, 100.0)):
        for k, ch_ in enumerate(seq):
            centre = (k == 3)
            ov.add_patch(Rectangle((wx + k * CELLW, wy), CELLW - 0.6, 10.0, facecolor=ORANGE if centre else "white", edgecolor=ORANGE if centre else GREY, linewidth=0.5, zorder=23))
            if not (centre and masked):                 # a masked residue has no letter; a residue read in the natural sequence keeps it, in white
                ov.text(wx + k * CELLW + (CELLW - 0.6) / 2.0, wy + 7.4, ch_, fontsize=6.5, color="white" if centre else INK, fontweight="bold" if centre else "normal",
                        ha="center", va="baseline", zorder=30)
        cv.texts.append((ov.texts[-1], seq))
        ov.plot([x0 + anchor[0] * wpt / im.shape[1], wx + 3.5 * CELLW - 0.3], [y0 + anchor[1] * hpt / im.shape[0], wy - 0.8], color=ORANGE, lw=0.6, zorder=23)
        txt(wx + 3.5 * CELLW - 0.3, wy + 21.0, lab, 7.5, ha="center")
    return im, hpt


loss = dict((r["model"], float(r["paired_Brier_mean"])) for r in D3["absolute_losses"])
end = dict((e["endpoint"], e) for e in D3["endpoints"])
assert all(abs((loss[m] - loss[m + "j"]) - float(end[g]["mean"])) < 1e-9 for m, g, _ in REFS)
passed = sum(end[g]["Holm_reject"] == "true" for _, g, _ in REFS)
ref = rows_of(NETREF, ",")
assert len(ref) == 200 and ref[0]["network"] == "trained_gpu" and all(r["network"] != "trained_gpu" for r in ref[1:])
G_tr, A_tr = float(ref[0]["single_center_native"]), float(ref[0]["single_center_native_minus_rotated"])
G_un = np.array([float(r["single_center_native"]) for r in ref[1:]])
A_un = np.array([float(r["single_center_native_minus_rotated"]) for r in ref[1:]])
basis = [r for r in rows_of(BASIS, ",") if r["network"] == "trained_gpu"]
assert len(basis) == 10 and basis[0]["basis"] == "0" and basis[0]["selected_axis"] == "1246" and abs(float(basis[0]["single_center_G"]) - G_tr) < 1e-15
G_rot = np.array([float(r["single_center_G"]) for r in basis[1:]])
assert abs(G_rot.mean() - float(ref[0]["single_center_mean_rotated"])) < 1e-12 and abs((G_tr - G_rot.mean()) - A_tr) < 1e-12
rank_G, rank_A = 1 + int((G_un >= G_tr).sum()), 1 + int((A_un >= A_tr).sum())
rot_above = int((G_rot > G_tr).sum())
assert rank_G == 1 and rank_A == 1                      # what the one count of panel F says

cls = dict((r["model"], r) for r in rows_of(CLS, "\t"))
assert all(int(cls[m]["n_pairs"]) == 967 and int(cls[m]["tie"]) == 0 for m in ("sequence", "8m", "35m"))
correct = dict((m, int(cls[m]["correct"])) for m in ("sequence", "8m", "35m"))
net = rows_of(NET, "\t")
un = dict((m, [int(r["pair_correct"]) for r in net if r["model"] == m]) for m in ("8m", "35m"))
sc = rows_of(SCALE, "\t")                             # the scale extension: one row per network
for m in ("8m", "35m"):                                # its 8M and 35M rows repeat the archived networks
    assert sorted(int(r["pair_correct"]) for r in sc if r["model"] == m and r["network_type"] != "trained") == sorted(un[m])
    assert [int(r["pair_correct"]) for r in sc if r["model"] == m and r["network_type"] == "trained"] == [correct[m]]
t650 = [r for r in sc if r["model"] == "650m" and r["network_type"] == "trained"]
assert len(t650) == 1 and t650[0]["axis"] == "4566" and t650[0]["primary_block"] == "16"
correct["650m"] = int(t650[0]["pair_correct"])
un["650m"] = [int(r["pair_correct"]) for r in sc if r["model"] == "650m" and r["network_type"] != "trained"]
assert all(len(un[m]) == 199 for m in un)
rank_pairs = dict((m, 1 + sum(v >= correct[m] for v in un[m])) for m in un)
pairs = rows_of(PAIRS, "\t")
assert len(pairs) == 967 and all(sum(r[m + "_status"] == "correct" for r in pairs) == correct[m] for m in ("sequence", "8m", "35m"))
lookup={}
for r in rows_of(ACT,'\t'):
    lookup[r['accession'],int(r['buried_position'])]=float(r['buried_activation_814'])
    lookup[r['accession'],int(r['exposed_position'])]=float(r['exposed_activation_814'])
def act_at(acc,pos):return lookup[acc,pos]
a_bur = np.array([act_at(r["accession"], int(r["buried_position"])) for r in pairs])
a_exp = np.array([act_at(r["accession"], int(r["exposed_position"])) for r in pairs])
below, above = int((a_exp > a_bur).sum()), int((a_exp < a_bur).sum())
assert below + above == 967                             # no pair lies on the diagonal
S2 = MAN2["sites"]
assert all(abs(act_at(MAN2["accession"], S2[k]["position"]) - S2[k]["activation_neuron_814"]) < 1e-12 for k in S2)

CELLW = 8.2
site = [m for m in MAN if m["file"] == os.path.basename(PROT)][0]
K1 = 126.0 / Image.open(PROT).size[0]                   # points per pixel: the two proteins of the figure are drawn at one scale

cv.head(0.0, RY, "A", "Exposure pair", 168.0)
txt(163.0, RY, "ESM-2 35M, neuron 814", 7.5, color=MUTED, ha="right")
assert abs(MAN2["px_per_angstrom"] - site["px_per_angstrom"]) < 1e-9
w2 = MAN2["size_px"][0] * K1
h2 = MAN2["size_px"][1] * K1
QY = RY + 9.0
wins2 = [(S2[k]["window_sequence"], MAN2["anchor_xy_px"][k], lab) for k, lab in (("buried", "Buried"), ("exposed", "Exposed"))]
assert all(w_[0][3] == "A" for w_ in wins2) and MAN2["anchor_xy_px"]["buried"][0] < MAN2["anchor_xy_px"]["exposed"][0]      # the buried site is the left one
protein(PROT2, None, (168.0 - w2) / 2.0, QY, w2, QY + h2 + 6.0, wins2, masked=False)
LOW = QY + h2 + 6.0 + 21.0                              # base line of the two labels under the protein: the lowest text of the upper row
assert 19.0 < UY - LOW < 21.0                           

cv.head(176.0, RY, "B", "Neuron at paired residues", 160.0)
E1, DY, DS = 202.0, RY + 34.0, 132.0
ALIM = (-0.32, 2.2)
e_ = cv.axes(E1, DY, DS, DS, ALIM, ALIM, xticks=[0, 1, 2], yticks=[0, 1, 2])
e_.plot(ALIM, ALIM, color=MUTED, lw=0.6, ls=DASH, zorder=1)
assert ALIM[0] < min(a_exp.min(), a_bur.min()) and max(a_exp.max(), a_bur.max()) < ALIM[1]       # every pair is inside the axes
e_.plot(a_exp, a_bur, marker="o", ms=2.1, mfc=ORANGE, mec="none", alpha=0.40, ls="none", zorder=2, rasterized=True)
apt = lambda v: (v - ALIM[0]) / (ALIM[1] - ALIM[0]) * DS
txt(E1 + apt(2.1), DY + DS - apt(0.95), "%d" % below, 8, "bold", color=TAB, ha="right")      # pairs below the diagonal: higher at the exposed residue
txt(E1 + apt(-0.2), DY + DS - apt(1.85), "%d" % above, 7.5, color=MUTED)                      # pairs above it
txt(E1 + DS / 2.0, DY + DS + 21.0, "Activation, exposed residue", 8, ha="center")
txt(E1 - 24.0, DY + DS / 2.0, "Activation, buried residue", 8, rot=90, ha="center", va="center")

cv.head(344.0, RY, "C", "Pairs identified", 160.0)
FX, FW, FY, FR = 414.0, 80.0, DY + 6.0, 31.5            # the last tick label (650) ends inside the canvas
assert abs((FY + 4 * FR) - (DY + DS)) < 1e-9            # the axes of B and C end on one line
FL, FYL = (472.0, 652.0), (4.0, 0.0)
f_ = cv.axes(FX, FY, FW, 4 * FR, FL, FYL, xticks=[500, 550, 600, 650], left=False)
f_.axvline(967 / 2.0, color=MUTED, lw=0.6, ls=DASH, zorder=1)
fpt = lambda v: FX + (v - FL[0]) / (FL[1] - FL[0]) * FW
txt(fpt(967 / 2.0), FY - 3.0, "50%", 7, color=MUTED, ha="center")
f_.plot([correct["sequence"]], [0.5], marker="o", ms=5.6, mfc=REF, mec="white", mew=0.6, ls="none", zorder=5)
txt(fpt(correct["sequence"]) + 6.5, FY + 0.5 * FR + 2.7, "%d" % correct["sequence"], 7.5)
txt(FX - 6.0, FY + 0.5 * FR + 2.7, "Sequence alone", 8, ha="right")
for r_, (m, lab) in enumerate((("8m", "+ 801 (8M)"), ("35m", "+ 814 (35M)"), ("650m", "+ 4566 (650M)"))):
    yc = r_ + 1.5
    lane(f_, FW, FL, FYL, 4 * FR, yc, un[m], BLUE, 2.0, 0.42 * FR)
    f_.plot([correct[m]], [yc], marker="o", ms=5.6, mfc=ORANGE, mec="white", mew=0.6, ls="none", zorder=5)
    txt(FX - 6.0, FY + yc * FR + 2.7, lab, 8, ha="right")
    if correct[m] > max(un[m]):                         # the trained network lies beyond every untrained one: its number stands before it
        txt(fpt(correct[m]) - 6.5, FY + yc * FR + 2.7, "%d" % correct[m], 8, "bold", color=TAB, ha="right")
    else:                                               # inside the untrained range: its number stands after the range
        txt(fpt(max(un[m])) + 5.0, FY + yc * FR + 2.7, "%d" % correct[m], 7.5)
txt(FX + FW / 2.0, FY + 4 * FR + 21.0, "Pairs identified, of 967", 8, ha="center")

cv.head(0.0, UY, "D", "Paired sites", 168.0)
txt(163.0, UY, "ESM-2 8M, neuron 1246", 7.5, color=MUTED, ha="right")
ex = D3["sequence_example"]
assert ex["PDB"] == site["pdb_id"]
wins = []
for key, res_, lab in (("A164", ex["HELIX_site_1based"], "Helix"), ("A178", ex["STRAND_site_1based"], "Strand")):
    i = res_ - ex["fragment_start_1based"]
    seq = ex["sequence_fragment"][i - 3:i + 4]
    assert seq[3] == "A" and site["sites"][key]["residue"] == "ALA%d" % res_
    wins.append((seq, site["anchor_xy_px"][key], lab))
im, _ = protein(PROT, site, 20.0, UY + 9.0, 126.0, UY + 133.0, wins, masked=True)
assert abs(K1 - 126.0 / im.shape[1]) < 1e-12

cv.head(176.0, UY, "E", "Gain", 164.0)
BY, PH = UY + 31.0, 112.0
CX, CW = 254.0, 82.0
c = cv.axes(CX, BY, CW, PH, (-0.003, 0.031), (3.0, 0.0), xticks=[0, 0.01, 0.02, 0.03], left=False)
c.axvline(0, color=MUTED, lw=0.6, ls=DASH, zorder=1)
for r_, (_, key, lab) in enumerate(REFS):
    e = end[key]
    y = r_ + 0.5
    c.plot([float(e["CI98_75_lower"]), float(e["CI98_75_upper"])], [y, y], color=ORANGE, lw=0.8, solid_capstyle="butt", zorder=3)
    c.plot([float(e["CI95_lower"]), float(e["CI95_upper"])], [y, y], color=ORANGE, lw=2.4, solid_capstyle="butt", zorder=4)
    c.plot([float(e["mean"])], [y], marker="o", ms=5.6, mfc=ORANGE, mec="white", mew=0.6, ls="none", zorder=5)
    assert float(e["CI98_75_lower"]) > 0 and e["Holm_reject"] == "true"
    txt(CX - 6.0, BY + y * PH / 3.0 + 2.7, lab, 8, ha="right")
txt(CX + CW / 2.0, BY + PH + 21.0, "Paired Brier reduction", 8, ha="center")
txt(CX + CW, UY + 20.0, "%d of 3 pass" % passed, 8, "bold", color=TAB, ha="right")

cv.head(344.0, UY, "F", "Reference", 160.0)
glyph_key(352.0, UY + 20.0, (("o", ORANGE, 5.0, "Trained"), ("o", BLUE, 3.0, "Untrained"), ("^", ROSE, 3.6, "Rotated")))
PX, PW = 402.0, 96.0
XL, YL = (-0.0235, 0.0125), (5.0, 0.0)
p = cv.axes(PX, BY, PW, PH, XL, YL, xticks=[-0.02, -0.01, 0, 0.01], left=False)
p.xaxis.set_major_formatter(FixedFormatter([MINUS(v) for v in (-0.02, -0.01, 0, 0.01)]))
p.axvline(0, color=MUTED, lw=0.6, ls=DASH, zorder=1)
p.axhline(2.85, color=LIGHT, lw=0.6, zorder=1)
upt = PH / 5.0                                          # points per unit of the y axis
for yc, vals, tr_v in ((1.35, G_un, G_tr), (4.1, A_un, A_tr)):
    lane(p, PW, XL, YL, PH, yc, vals, BLUE, 2.0, 0.55 * upt)
    p.plot([tr_v], [yc - 0.85], marker="o", ms=5.6, mfc=ORANGE, mec="white", mew=0.6, ls="none", zorder=5)
p.plot(G_rot, [2.3] * 9, marker="^", ms=3.6, mfc=ROSE, mec="none", alpha=0.9, ls="none", zorder=4)
txt(PX + 1.0, BY + 0.5 * upt + 2.7, "Rank 1 of 200", 8, "bold", color=TAB)        # the trained network, in both lanes
txt(PX - 6.0, BY + 1.35 * upt + 2.7, "Gain", 8, ha="right")
txt(PX - 6.0, BY + 3.75 * upt, "Gain −\nrotated mean", 8, ha="right", va="center")
txt(PX + PW / 2.0, BY + PH + 21.0, "Single-site Brier reduction", 8, ha="center")
assert BY + PH + 21.0 + 10.0 <= H                       

cv.finish(OUT, NAME, "gains %s | gains that pass %d of 3 | 1246 gain %.5f, minus rotated mean %.5f, ranks %d and %d of 200, rotations above the trained axes %d of 9 | pairs %d, 814 higher at the exposed residue in %d, at the buried residue in %d | identified: sequence %d, 801 %d, 814 %d, 4566 %d | untrained ranges 8m %d-%d, 35m %d-%d, 650m %d-%d | ranks of the count 8m %d, 35m %d, 650m %d | example %s %s chain %s, buried A%d RSA %.2f, exposed A%d RSA %.2f"
          % (" ".join("%.4f" % float(end[g]["mean"]) for _, g, _ in REFS), passed, G_tr, A_tr, rank_G, rank_A, rot_above,
             len(pairs), below, above, correct["sequence"], correct["8m"], correct["35m"], correct["650m"], min(un["8m"]), max(un["8m"]), min(un["35m"]), max(un["35m"]), min(un["650m"]), max(un["650m"]),
             rank_pairs["8m"], rank_pairs["35m"], rank_pairs["650m"], MAN2["accession"], MAN2["pdb_id"], MAN2["chain"],
             S2["buried"]["position"], S2["buried"]["rsa"], S2["exposed"]["position"], S2["exposed"]["rsa"]))
