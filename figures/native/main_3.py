import sys, os
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
from matplotlib.patches import Circle
import numpy as np

D3, D4, F5, G2D = J(sys.argv[1]), J(sys.argv[2]), J(sys.argv[3]), J(sys.argv[4])
ACT, OUT = sys.argv[5], sys.argv[6]
NAME = "Figure_3"
W, H = 504.0, 368.0
cv = Canvas(W, H)
ov, txt = cv.ov, cv.txt
DASH = (0, (3, 2))
TWO = (("Aliphatic OH", "Hydroxyls", "293", "oh"), ("Carbonyl", "Carbonyls", "454", "co"))

cv.head(0.0, 11.0, "A", "Label-prediction gain", 150.0)
if "2" in OPT:
    txt(144.0, 40.0, "ChemBERTa-77M-MTR", 7.5, color=MUTED, ha="right")
rows = sorted(D3["concept_rows"], key=lambda r: -r["Delta_Brier"])
assert len(rows) == 12
AX0, AY0, AW, PH = 40.0, 34.0, 104.0, 96.0
a = cv.axes(AX0, AY0, AW, PH, (0, 12), (0, 4.0), yticks=[0, 2, 4])
for i, r in enumerate(rows):
    a.bar(i + 0.5, r["Delta_Brier"] * 1000.0, width=0.72, color=GROUPC[r["group"]], linewidth=0, zorder=3)
    ov.add_patch(Rectangle((AX0 + (i + 0.5) / 12.0 * AW - 2.2, AY0 + PH + 3.0), 4.4, 4.4, facecolor=GROUPC[r["group"]], edgecolor="none", zorder=30))
txt(AX0 - 26.0, AY0 + PH / 2.0, "ΔBrier (× 0.001)", 8, ha="center", va="center", rot=90)
ps = D3["chemical_transfer_record"]["panel_summary"]
mean_gain = float(ps["point"]) * 1000.0
assert abs(sum(r["Delta_Brier"] for r in rows) / 12.0 * 1000.0 - mean_gain) < 1e-9
a.axhline(mean_gain, color=MUTED, lw=0.7, ls=DASH, zorder=4)
txt(AX0 + AW, AY0 + PH * (1 - mean_gain / 4.0) - 3.0, "mean %.2f" % mean_gain, 7.5, color=MUTED, ha="right")

cv.head(160.0, 11.0, "B", "Two coordinates", 150.0)
arows = dict(((r["label"], r["condition"]), r["plot_values"]) for r in D4["activation_rows"])
BX0, BY0, BW, BH = 196.0, 34.0, 108.0, PH
b = cv.axes(BX0, BY0, BW, BH, (0, 2), (0, 1.0), yticks=[0, 0.5, 1])
for g, (lab, grp, _, _) in enumerate(TWO):
    assert abs(arows[(lab, "D_mean")][0]) < 1e-3
    for k, (cond, col) in enumerate((("M1", pale(GROUPC[grp])), ("M2", GROUPC[grp]))):
        est, lo, hi = arows[(lab, cond)]
        x = g + 0.5 + (k - 0.5) * 0.34
        b.bar(x, est, width=0.30, color=col, linewidth=0, zorder=3)
        ov.add_patch(Rectangle((BX0 + x / 2.0 * BW - 2.2, BY0 + BH + 3.0), 4.4, 4.4, facecolor=(PALE, DARK)[k], edgecolor="none", zorder=30))
        b.plot([x, x], [lo, hi], color=ERR, lw=0.8, zorder=5)
        for e_ in (lo, hi):
            b.plot([x - 0.045, x + 0.045], [e_, e_], color=ERR, lw=0.8, zorder=5)
        b.text(x, hi + 0.02, "%.2f" % est, fontsize=7.5, color=INK, ha="center", va="bottom", zorder=6, fontweight="bold" if cond == "M2" else "normal")
    txt(BX0 + (g + 0.5) / 2.0 * BW, BY0 + BH + 18.0, lab, 8, ha="center")
txt(BX0 - 22.0, BY0 + BH / 2.0, "R²", 8, ha="center", va="center", rot=90)

cv.head(320.0, 11.0, "C", "Nine per encoder", 184.0)
CX0, CW, CGAP = 348.0, 66.0, 8.0
RISE, MEANS = 0, {}
group_of = dict((r["concept_id"], r["group"]) for r in D3["concept_rows"])
for m, (model, mname) in enumerate((("chemberta_mtr", "ChemBERTa-77M-MTR"), ("molformer", "MoLFormer-XL"))):
    x0 = CX0 + m * (CW + CGAP)
    c = cv.axes(x0, BY0 + 10.0, CW, BH - 10.0, (-1.35, 1.45), (0, 1.0), yticks=[0, 0.5, 1], left=(m == 0))
    hue_of = dict((ro["coordinate"], GROUPC[group_of[ro["representative_concept_id"]]]) for ro in F5["row_order"][model])
    pts = {}
    for r in F5["coordinates"]:
        if r["model"] == model and r["role"] == "selected":
            pts.setdefault(r["coordinate"], {})[r["predictor"]] = r["point"]
    assert len(pts) == 9
    for coord, v in sorted(pts.items()):
        lo_, hi_ = v["Binary concept means"], v["Local graph"]
        RISE += 1 if hi_ > lo_ else 0
        c.plot([0, 1], [lo_, hi_], color=hue_of[coord], lw=1.1, zorder=2, solid_capstyle="round")
        c.plot([0, 1], [lo_, hi_], marker="o", ms=2.8, mfc=hue_of[coord], mec="none", ls="none", zorder=3)
    mb = sum(v["Binary concept means"] for v in pts.values()) / 9.0
    ml = sum(v["Local graph"] for v in pts.values()) / 9.0
    assert abs((ml - mb) - [q for q in F5["macro"] if q["model"] == model][0]["point"]) < 1e-9
    MEANS[model] = (mb, ml)
    c.plot([0, 1], [mb, ml], color=INK, lw=1.6, zorder=5, solid_capstyle="round")
    c.plot([0], [mb], marker="o", ms=5.6, mfc=PALE, mec=INK, mew=0.8, ls="none", zorder=6)
    c.plot([1], [ml], marker="o", ms=5.6, mfc=DARK, mec=INK, mew=0.8, ls="none", zorder=6)
    c.text(-0.16, mb, "%.2f" % mb, fontsize=7.5, color=INK, ha="right", va="center", zorder=7)
    c.text(1.16, ml, "%.2f" % ml, fontsize=7.5, color=INK, ha="left", va="center", zorder=7, fontweight="bold")
    for k, col in ((0, PALE), (1, DARK)):
        ov.add_patch(Rectangle((x0 + (k + 1.35) / 2.8 * CW - 3.2, BY0 + BH + 5.0), 6.4, 6.4, facecolor=col, edgecolor="none", zorder=30))
    txt(x0 + 1.85 / 2.8 * CW, BY0 + 2.0, mname, 7.5, ha="center", color=MUTED)
txt(CX0 - 21.5, BY0 + 10.0 + (BH - 10.0) / 2.0, "R²", 8, ha="center", va="center", rot=90)
txt(CX0 + CW + CGAP / 2.0, BY0 + BH + 25.0, "%d of 18 rise" % RISE, 8, "bold", color=TAB, ha="center")

LY = BY0 + BH + 42.0
lx = cv.key(6.0, LY, [(GROUPC[g], GROUPN.get(g, g)) for g in D3["group_order"]], size=8, gap=10.0)
cv.key(lx + 22.0, LY, ((PALE, "Concept means"), (DARK, "Bonding environment")), size=8, gap=14.0)

RY0 = LY + 26.0
cv.head(0.0, RY0, "D", "Observed and predicted", 192.0)
z = np.load(ACT, allow_pickle=False)
ids, mn = z["coordinate_ids"].tolist(), z["model_names"].tolist()
SX, SW, SH, SYS = (44.0, 120.0), 62.0, 46.0, (RY0 + 30.0, RY0 + 96.0)
LIMS = {"293": ((-0.6, 8.6), [0, 4, 8], (-0.7, 12.6), [0, 6, 12]), "454": ((-0.25, 3.2), [0, 1, 2, 3], (-0.4, 7.0), [0, 3, 6])}
n_atoms = 0
for r_, (lab, grp, coord, _) in enumerate(TWO):
    ci = ids.index(int(coord))
    obs, pos = z["actual"][:, ci], z["target_labels"][:, ci]
    n_atoms = len(obs)
    xl, xt, yl, yt = LIMS[coord]
    for k, cond in enumerate(("M1", "M2")):
        pred = z["predictions"][:, ci, mn.index(cond)]
        assert xl[0] < pred.min() and pred.max() < xl[1] and yl[0] < obs.min() and obs.max() < yl[1]
        d = cv.axes(SX[k], SYS[r_], SW, SH, xl, yl, xticks=xt, yticks=yt, left=(k == 0))
        d.plot([xl[0], xl[1]], [xl[0], xl[1]], color=MUTED, lw=0.6, ls=DASH, zorder=1)
        d.plot(pred[~pos], obs[~pos], marker="o", ms=1.2, mfc=GREY, mec="none", alpha=0.25, ls="none", zorder=2, rasterized=True)
        d.plot(pred[pos], obs[pos], marker="o", ms=1.2, mfc=GROUPC[grp], mec="none", alpha=0.14, ls="none", zorder=3, rasterized=True)
        if r_ == 0:
            ov.add_patch(Rectangle((SX[k] + SW / 2.0 - 3.2, SYS[0] - 12.0), 6.4, 6.4, facecolor=(PALE, DARK)[k], edgecolor="none", zorder=30))
txt(SX[0] + SW + 6.0, SYS[1] + SH + 22.0, "Predicted activation", 8, ha="center")
txt(10.0, (SYS[0] + SYS[1] + SH) / 2.0, "Observed activation", 8, rot=90, ha="center", va="center")

cv.head(204.0, RY0, "E", "Two molecules", 300.0)
geo = dict((m["name"], m) for m in G2D["molecules"])
stored = dict((str(e["molecule"]["coordinate"]), e["molecule"]) for e in D4["C_saved_atom_values"]["molecules"])
MX0, CELL = 262.0, 80.0
for k, col in enumerate((None, PALE, DARK)):
    cx = MX0 + (k + 0.5) * CELL
    if col is None:
        txt(cx, SYS[0] - 6.0, "Observed", 7.5, ha="center")
    else:
        ov.add_patch(Rectangle((cx - 3.2, SYS[0] - 12.0), 6.4, 6.4, facecolor=col, edgecolor="none", zorder=30))
for r_, (lab, grp, coord, tag) in enumerate(TWO):
    m, g = stored[coord], geo[tag]
    assert g["smiles"] == m["smiles"] and len(g["atoms"]) == m["n_atoms"] == len(m["observed"])
    top = max(max(m[key]) for key in ("observed", "M1", "M2"))
    yc = SYS[r_] + SH / 2.0
    txt(210.0, yc - 4.0, lab, 8)
    for q in range(24):
        ov.add_patch(Rectangle((218.0 + q * 1.1, yc + 3.0), 1.15, 5.0, facecolor=tint(ORANGE, (q + 0.5) / 24.0), edgecolor="none", zorder=30))
    ov.add_patch(Rectangle((218.0, yc + 3.0), 26.4, 5.0, facecolor="none", edgecolor=MUTED, linewidth=0.4, zorder=31))
    txt(215.5, yc + 8.2, "0", 7, color=MUTED, ha="right")
    txt(247.0, yc + 8.2, "%.2g" % top, 7, color=MUTED)
    y_top = yc - g["height_pt"] / 2.0
    for k, key in enumerate(("observed", "M1", "M2")):
        mx = MX0 + (k + 0.5) * CELL - g["width_pt"] / 2.0
        for a_ in sorted(g["atoms"], key=lambda q: m[key][q["index"]]):
            v = m[key][a_["index"]]
            if v > 0:
                ov.add_patch(Circle((mx + a_["x"], y_top + a_["y"]), 0.42 * g["bond_length_pt"], facecolor=tint(ORANGE, v / top), edgecolor="none", zorder=22))
        for ln in g["lines"]:
            ov.plot([mx + q[0] for q in ln], [y_top + q[1] for q in ln], color=INK, lw=0.6, solid_capstyle="round", solid_joinstyle="round", zorder=24)
        for l in g["labels"]:
            ov.text(mx + l["x"], y_top + l["y"], l["s"], fontsize=l.get("size", 7), color=INK, ha="left", va="baseline", zorder=25)

cv.finish(OUT, NAME, "coordinates that rise %d of 18 | means %s | atoms in D %d per scatter"
          % (RISE, " ".join("%s %.3f to %.3f" % (k, v[0], v[1]) for k, v in sorted(MEANS.items())), n_atoms))
