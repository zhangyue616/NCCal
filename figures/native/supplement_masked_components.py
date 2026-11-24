import sys, os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
from PIL import Image
import numpy as np

D3, S3, TP = J(sys.argv[1]), np.load(sys.argv[2], allow_pickle=False), np.load(sys.argv[3], allow_pickle=False)
PROT, MAN, OUT = sys.argv[4], J(sys.argv[5]), sys.argv[6]
NAME = "Masked_protein"
W, H = 504.0, 346.0
cv = Canvas(W, H)
ov, txt = cv.ov, cv.txt
DASH = (0, (3, 2))
REFS = (("X", "gX", "Composition"), ("S", "gS", "Sequence window"), ("L", "gL", "Interaction"))

loss = dict((r["model"], float(r["paired_Brier_mean"])) for r in D3["absolute_losses"])
end = dict((e["endpoint"], e) for e in D3["endpoints"])
pairs = dict((r["model"], r) for r in D3["C_plot_pairs"])
assert all(abs(pairs[m]["without_unit"] - loss[m]) < 1e-12 and abs(pairs[m]["with_unit"] - loss[m + "j"]) < 1e-12 for m, _, _ in REFS)
order = S3["endpoint_order"].tolist()
gc = S3["group_contributions"]
assert gc.shape == (1304, 4) and all(abs(gc[:, order.index(k)].mean() - float(end[k]["mean"])) < 1e-12 for k in order)      # the exported groups reproduce the stored means
assert all(abs((loss[m] - loss[m + "j"]) - float(end[g]["mean"])) < 1e-9 for m, g, _ in REFS)                              # and the stored losses
passed = sum(end[g]["Holm_reject"] == "true" for _, g, _ in REFS)

ids = TP["case_ids"].tolist()
acc, kind = [i.split(":")[0] for i in ids], [i.split(":")[1] for i in ids]
assert len(ids) == 2608 and acc[0::2] == acc[1::2] and all(set(kind[2 * k:2 * k + 2]) == {"HELIX", "STRAND"} for k in range(1304))
first_helix = np.array([k == "HELIX" for k in kind[0::2]])
assert np.array_equal(first_helix, TP["y_a"].astype(bool))                      # the stored label of the first site says the same
j0, j1 = TP["j1246"][0::2], TP["j1246"][1::2]
jh, js = np.where(first_helix, j0, j1), np.where(first_helix, j1, j0)
pc, gid = TP["pair_contributions"], S3["group_ids"].tolist()
assert TP["endpoint_order"].tolist() == order and np.array_equal(pc[:, 0], TP["X_loss"] - TP["Xj_loss"])
where = dict((g, i) for i, g in enumerate(gid))
assert np.array_equal(pc, gc[[where[g] for g in TP["groups"].tolist()]])        # the pairs are the groups of panel D, in another order
below, above = int((js > jh).sum()), int((js < jh).sum())
assert below + above == len(js)                                                 # no pair lies on the diagonal

cv.head(0.0, 11.0, "A", "Paired sites", 168.0)
if "2" in OPT:                                          # optional change 2: the encoder of the figure
    txt(163.0, 11.0, "ESM-2", 7.5, color=MUTED, ha="right")
site = [m for m in MAN if m["file"] == os.path.basename(PROT)][0]
ex = D3["sequence_example"]
assert ex["PDB"] == site["pdb_id"]
im = np.asarray(Image.open(PROT).convert("RGBA"))
pw = 126.0
ph = pw * im.shape[0] / im.shape[1]
px0, py0 = 20.0, 20.0
ov.imshow(im, extent=(px0, px0 + pw, py0 + ph, py0), zorder=22, interpolation="lanczos", aspect="auto")
ov.set_xlim(0, W)
ov.set_ylim(H, 0)
WY, CELLW = 144.0, 8.2
for (key, res, lab), wx in zip((("A164", ex["HELIX_site_1based"], "Helix"), ("A178", ex["STRAND_site_1based"], "Strand")), (10.0, 100.0)):
    i = res - ex["fragment_start_1based"]
    seq = ex["sequence_fragment"][i - 3:i + 4]
    assert seq[3] == "A" and site["sites"][key]["residue"] == "ALA%d" % res
    for k, ch_ in enumerate(seq):
        masked = (k == 3)
        ov.add_patch(Rectangle((wx + k * CELLW, WY), CELLW - 0.6, 10.0, facecolor=ORANGE if masked else "white", edgecolor=ORANGE if masked else GREY, linewidth=0.5, zorder=23))
        if not masked or "3" in OPT:                    # optional change 3: the masked alanine by its letter, in white
            ov.text(wx + k * CELLW + (CELLW - 0.6) / 2.0, WY + 7.4, ch_, fontsize=6.5, color="white" if masked else INK, fontweight="bold" if masked else "normal",
                    ha="center", va="baseline", zorder=30)
    cv.texts.append((ov.texts[-1], seq))
    ax_, ay_ = site["anchor_xy_px"][key]
    ov.plot([px0 + ax_ * pw / im.shape[1], wx + 3.5 * CELLW - 0.3], [py0 + ay_ * ph / im.shape[0], WY - 0.8], color=ORANGE, lw=0.6, zorder=23)
    txt(wx + 3.5 * CELLW - 0.3, WY + 21.0, lab, 7.5, ha="center")

cv.head(176.0, 11.0, "B", "Paired Brier loss", 164.0)
cv.key(178.0, 31.0, ((GREY, "Reference"), (ORANGE, "With coordinate")), size=7.5, gap=9.0)
BX, BY, BW, RH = 256.0, 42.0, 78.0, 28.0
b = cv.axes(BX, BY, BW, 3 * RH, (0, 0.275), (3.0, 0.0), xticks=[0, 0.1, 0.2], left=False)
b.axvline(0.25, color=MUTED, lw=0.6, ls=DASH, zorder=1)
b.axvline(0, color=INK, lw=0.7, zorder=4)
for r_, (m, _, lab) in enumerate(REFS):
    for off, key, col in ((-0.19, m, GREY), (0.19, m + "j", ORANGE)):
        b.barh(r_ + 0.5 + off, loss[key], height=0.34, color=col, linewidth=0, zorder=3)
        b.text(loss[key] - 0.006, r_ + 0.5 + off, "%.4f" % loss[key], fontsize=7, color=INK if col == GREY else "white", ha="right", va="center", zorder=6,
               fontweight="normal" if col == GREY else "bold")
    txt(BX - 6.0, BY + (r_ + 0.5) * RH + 2.7, lab, 8, ha="right")
txt(BX + BW / 2.0, BY + 3 * RH + 21.0, "Paired Brier loss", 8, ha="center")           

cv.head(348.0, 11.0, "C", "Gain", 156.0)
CX, CW = 356.0, 138.0
c = cv.axes(CX, BY, CW, 4 * RH, (-0.003, 0.031), (4.0, 0.0), xticks=[0, 0.01, 0.02, 0.03], left=False)
c.axvline(0, color=MUTED, lw=0.6, ls=DASH, zorder=1)
c.axhline(3.0, color=LIGHT, lw=0.6, zorder=1)
for r_, key in enumerate(("gX", "gS", "gL", "psi")):
    e = end[key]
    y, col = r_ + 0.5, (INK if key == "psi" else ORANGE)
    c.plot([float(e["CI98_75_lower"]), float(e["CI98_75_upper"])], [y, y], color=col, lw=0.8, solid_capstyle="butt", zorder=3)
    c.plot([float(e["CI95_lower"]), float(e["CI95_upper"])], [y, y], color=col, lw=2.4, solid_capstyle="butt", zorder=4)
    c.plot([float(e["mean"])], [y], marker="D" if key == "psi" else "o", ms=3.8 if key == "psi" else 5.6, mfc=col, mec="white", mew=0.6, ls="none", zorder=5)
    assert float(e["CI98_75_lower"]) > 0 and e["Holm_reject"] == "true"
txt(CX + (float(end["psi"]["CI98_75_upper"]) + 0.003 + 0.0016) / 0.034 * CW, BY + 3.5 * RH, "Composition −\nsequence window", 7.5, color=MUTED, va="center")   # beside its interval
txt(CX + CW / 2.0, BY + 4 * RH + 21.0, "Brier reduction", 8, ha="center")
txt(CX + CW, 31.0, "%d of 3 pass" % passed, 8, "bold", color=TAB, ha="right")

RY = 195.0
DY, DS = RY + 17.0, 104.0
MINUS = FixedFormatter(["−0.5", "0", "0.5"])


def scatter(x0, xs, ys, xlim, ylim, xticks, yticks, xfmt=None, yfmt=None, diag=False, zero=False):
    a = cv.axes(x0, DY, DS, DS, xlim, ylim, xticks=xticks, yticks=yticks)
    if xfmt:
        a.xaxis.set_major_formatter(xfmt)
    if yfmt:
        a.yaxis.set_major_formatter(yfmt)
    if diag:
        a.plot(xlim, xlim, color=MUTED, lw=0.6, ls=DASH, zorder=1)
    if zero:
        a.axhline(0, color=MUTED, lw=0.6, ls=DASH, zorder=1)
        a.axvline(0, color=MUTED, lw=0.6, ls=DASH, zorder=1)
    assert xlim[0] < xs.min() and xs.max() < xlim[1] and ylim[0] < ys.min() and ys.max() < ylim[1]      # every pair is inside the axes
    a.plot(xs, ys, marker="o", ms=2.1, mfc=ORANGE, mec="none", alpha=0.40, ls="none", zorder=2, rasterized=True)
    return a


cv.head(0.0, RY, "D", "Group by group", 168.0)
gx, gs = S3["x_composition_Brier_reduction"], S3["y_sequence_window_Brier_reduction"]
assert np.array_equal(gx, gc[:, order.index("gX")]) and np.array_equal(gs, gc[:, order.index("gS")])
LIM = (-0.9, 0.7)
DX = 48.0
d = scatter(DX, gx, gs, LIM, LIM, [-0.5, 0, 0.5], [-0.5, 0, 0.5], MINUS, MINUS, diag=True)
d.plot([float(end["gX"]["mean"])], [float(end["gS"]["mean"])], marker="o", ms=4.8, mfc="white", mec=INK, mew=0.9, ls="none", zorder=4)
txt(DX + DS / 2.0, DY + DS + 21.0, "Gain, composition", 8, ha="center")
txt(DX - 30.0, DY + DS / 2.0, "Gain, sequence window", 8, rot=90, ha="center", va="center")

cv.head(176.0, RY, "E", "Coordinate at paired sites", 328.0)
E1, E2 = 228.0, 392.0
ALIM = (-0.25, 1.25)
scatter(E1, js, jh, ALIM, ALIM, [0, 0.5, 1], [0, 0.5, 1], diag=True)
apt = lambda v: (v - ALIM[0]) / (ALIM[1] - ALIM[0]) * DS                     # activation to points along either axis
txt(E1 + apt(1.21), DY + DS - apt(0.60), "%d" % below, 8, "bold", color=TAB, ha="right")      # pairs below the diagonal: higher at the strand site
txt(E1 + apt(-0.19), DY + DS - apt(1.09), "%d" % above, 7.5, color=MUTED)                      # pairs above it
txt(E1 + DS / 2.0, DY + DS + 21.0, "Activation, strand site", 8, ha="center")          # the quantity by name: the title that said it is no longer drawn
txt(E1 - 26.0, DY + DS / 2.0, "Activation, helix site", 8, rot=90, ha="center", va="center")
scatter(E2, js - jh, pc[:, order.index("gX")], (-1.05, 1.15), LIM, [-1, 0, 1], [-0.5, 0, 0.5], FixedFormatter(["−1", "0", "1"]), MINUS, zero=True)
txt(E2 + DS / 2.0, DY + DS + 21.0, "Activation, strand − helix", 8, ha="center")
txt(E2 - 30.0, DY + DS / 2.0, "Gain, composition", 8, rot=90, ha="center", va="center")

cv.finish(OUT, NAME, "losses %s | gains %s | psi %.5f | gains that pass %d of 3 | groups %d | coordinate higher at the strand site in %d pairs, at the helix site in %d"
          % (" ".join("%s %.4f/%.4f" % (m, loss[m], loss[m + "j"]) for m, _, _ in REFS), " ".join("%.4f" % float(end[g]["mean"]) for _, g, _ in REFS),
             float(end["psi"]["mean"]), passed, len(gx), below, above))
