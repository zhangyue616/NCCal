import sys, os, math
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *

D2, SC6, D3, OUT = J(sys.argv[1]), J(sys.argv[2]), J(sys.argv[3]), sys.argv[4]
NAME = "Figure_2"
AB, CD = D2["panel_AB_frozen_source"], D2["panel_CD_frozen_source"]
W, H = 504.0, 398.0
cv = Canvas(W, H)
ov, txt = cv.ov, cv.txt
READ, BASIS, PASS = BLUE, ROSE, TAB
DASH = (0, (3, 2))

models = dict((m["model"], m["H4"]) for m in AB["audit"]["models"])
mf = AB["audit"]["molformer_primary"]
support = dict((r["checkpoint"], dict((c["metric"], c) for c in r["criteria"])) for r in AB["support_matrix_records"])
POINTS = (("chemberta_mlm", "ChemBERTa-77M-MLM", "smiles"), ("chemberta_mtr", "ChemBERTa-77M-MTR", "smiles"), ("molformer", "MoLFormer", "smiles"),
          ("chemeleon", "CheMeleon", "graph"), ("molclr_gin", "MolCLR GIN", "graph"), ("hu_masking", "Hu masking", "graph"),
          ("hu_contextpred", "Hu contextpred", "graph"), ("hu_edgepred", "Hu edgepred", "graph"), ("hu_infomax", "Hu infomax", "graph"))
assert set(n for _, n, _ in POINTS) == set(support) and len(POINTS) == 9


def fold(key, metric):
    if key == "molformer":
        return mf[metric]["fold"], mf[metric]["fold_ci_conditional_on_layer"]
    return models[key][metric + "_fold"], models[key][metric + "_fold_ci"]


both = [k for k, name, _ in POINTS if all(support[name][m]["Calibrated"] for m in ("J1", "J2"))]
SHOWN = {"MoLFormer": "MoLFormer-XL"}


def titled(x, y, s, color, rot=0):
    """An axis title with the key square of its colour in front of it."""
    t = txt(x, y, s, 8, ha="center", va="baseline" if rot == 0 else "center", rot=rot)
    wd = cv.width_of(t) if rot == 0 else None
    if rot == 0:
        t.set_x(x + 4.5)
        ov.add_patch(Rectangle((x + 4.5 - wd / 2.0 - 9.0, y - 6.2), 6.4, 6.4, facecolor=color, edgecolor="none", zorder=30))
    else:
        cv.fig.canvas.draw()
        ht = t.get_window_extent(cv.fig.canvas.get_renderer()).height * 72.0 / cv.fig.dpi
        t.set_y(y - 4.5)
        ov.add_patch(Rectangle((x - 3.2, y - 4.5 + ht / 2.0 + 2.6), 6.4, 6.4, facecolor=color, edgecolor="none", zorder=30))


cv.head(0.0, 11.0, "A", "Raw matching", 134.0)
cv.key(6.0, 29.0, ((DARK, "Trained"), (READ, "Untrained"), (BASIS, "Rotated")), size=7.5, gap=7.0)
AX, AY, AW, AH = 30.0, 40.0, 100.0, 156.0
a = cv.axes(AX, AY, AW, AH, (0.0, 3.0), (0.0, 100.0), yticks=[0, 25, 50, 75, 100])
PILOT = (("chemeleon", "CheMeleon"), ("molclr_gin", "MolCLR\nGIN"), ("hu_masking", "Hu\nmasking"))
n_ref, above = 0, 0
for i, (key, lab) in enumerate(PILOT):
    m = SC6["pilot"][key]
    for off, val, col, dots in ((-0.27, m["native"], DARK, None), (0.0, m["initialization_mean"], READ, m["initialization_dots_pct"]),
                                (0.27, m["rotation_mean"], BASIS, m["rotation_dots_pct"])):
        xc = i + 0.5 + off
        a.bar([xc], [val], width=0.24, color=col, linewidth=0, zorder=2)
        if dots:
            assert abs(sum(dots) / len(dots) - val) < 1e-9
            n_ref += len(dots)
            above += val > m["native"]
            for j, v in enumerate(sorted(dots)):
                a.plot([xc + (j - (len(dots) - 1) / 2.0) * (0.19 / max(len(dots) - 1, 1))], [v], marker="o", ms=2.1, mfc=INK, mec="none", alpha=0.55, ls="none", zorder=4)
    txt(AX + (i + 0.5) / 3.0 * AW, AY + AH + 5.0, lab, 7, ha="center", va="top")
txt(9.0, AY + AH / 2.0, "Matched neurons (%)", 8, rot=90, ha="center", va="center")

cv.head(140.0, 11.0, "B", "Calibration across checkpoints", 246.0)
BX, BY, BW, BH = 178.0, 40.0, 198.0, 156.0
XL, YL = (0.45, 4.0), (0.84, 2.45)
b = cv.axes(BX, BY, BW, BH, XL, YL, xticks=[0.5, 1, 2, 3], yticks=[1, 1.5, 2], xlog=True, ylog=True)
px = lambda v: BX + math.log10(v / XL[0]) / math.log10(XL[1] / XL[0]) * BW
py = lambda v: BY + math.log10(YL[1] / v) / math.log10(YL[1] / YL[0]) * BH
b.axvline(1, color=MUTED, lw=0.6, ls=DASH, zorder=1)
b.axhline(1, color=MUTED, lw=0.6, ls=DASH, zorder=1)
LAB = {"chemberta_mlm": "below_left", "chemberta_mtr": "above_right", "molformer": "after", "chemeleon": "after", "molclr_gin": "after",
       "hu_masking": (13.5, 7.5), "hu_contextpred": "below", "hu_edgepred": (4.0, -4.5), "hu_infomax": "above"}
for key, name, fam in POINTS:
    (f1, c1), (f2, c2) = fold(key, "J1"), fold(key, "J2")
    assert XL[0] < c1[0] and c1[1] < XL[1] and YL[0] < c2[0] and c2[1] < YL[1]
    cal = dict((m, support[name][m]["Calibrated"]) for m in ("J1", "J2"))
    b.plot(c1, [f2, f2], color=READ if cal["J1"] else pale(READ), lw=1.3, solid_capstyle="butt", zorder=3 if cal["J1"] else 2)
    b.plot([f1, f1], c2, color=BASIS if cal["J2"] else pale(BASIS), lw=1.3, solid_capstyle="butt", zorder=3 if cal["J2"] else 2)
    ok = key in both
    b.plot([f1], [f2], marker="o" if fam == "smiles" else "D", ms=6.4 if fam == "smiles" else 5.4, mfc=PASS if ok else "white", mec="white" if ok else INK,
           mew=0.6 if ok else 0.9, ls="none", zorder=4)
    how = LAB[key]
    x_, y_, ha = {"after": (px(c1[1]) + 3.5, py(f2) + 2.5, "left"),
                  "above_right": (px(f1) + 5.0, py(f2) - 4.5, "left"),
                  "below_left": (px(f1) - 6.0, py(f2) + 7.9, "right"),
                  "above": (px(f1), py(c2[1]) - 3.0, "center"),
                  "below": (BX + 3.0, py(c2[0]) + 6.4, "left")}.get(how) if not isinstance(how, tuple) else (px(f1) + how[0], py(f2) + how[1], "left")
    txt(x_, y_, SHOWN.get(name, name), 7, ha=ha)
t_ = txt(BX + BW - 2.0, BY + 11.0, "%d of %d pass both" % (len(both), len(POINTS)), 8, "bold", color=PASS, ha="right")
ov.plot([BX + BW - 2.0 - cv.width_of(t_) - 6.0], [BY + 8.2], marker="o", ms=6.0, mfc=PASS, mec="none", ls="none", zorder=30)
kx = BX + 2.0
for mk, ms, lab, smi in (("o", 6.0, "SMILES Transformers", True), ("D", 5.0, "Graph models", False)):
    oks = set(k_ in both for k_, _, fam_ in POINTS if (fam_ == "smiles") == smi)
    assert len(oks) == 1
    ok = oks.pop()
    ov.plot([kx + 3.2], [26.4], marker=mk, ms=ms, mfc=PASS if ok else "white", mec="none" if ok else INK, mew=0.0 if ok else 0.9, ls="none", zorder=30)
    kx = cv.right_edge(txt(kx + 9.0, 29.0, lab, 7.5)) + 9.0
titled(BX + BW / 2.0, BY + BH + 23.0, "Neuron readability (fold)", READ)
titled(BX - 27.0, BY + BH / 2.0, "Axis preference (fold)", BASIS, rot=90)

cv.head(392.0, 11.0, "C", "Depth profile", 112.0)
txt(424.0, 49.0, "MoLFormer-XL", 7.5, color=MUTED)
CX, CY, CW, CH = 420.0, 40.0, 78.0, 156.0
c = cv.axes(CX, CY, CW, CH, (0.4, 12.6), (0.9, 2.35), xticks=[1, 4, 7, 10, 12], yticks=[1, 1.5, 2])
depth = dict((r["metric"], r["values"]) for r in AB["depth"])
sel = int(AB["selected_natural_layer"])
assert all(len(v) == 12 for v in depth.values()) and all(abs(depth[m][sel - 1] - mf[m]["fold"]) < 1e-3 for m in ("J1", "J2"))
c.axvspan(sel - 0.5, sel + 0.5, color=LIGHT, lw=0, zorder=0)
c.axhline(1, color=MUTED, lw=0.6, ls=DASH, zorder=1)
for metric, col in (("J1", READ), ("J2", BASIS)):
    c.plot(range(1, 13), depth[metric], color=col, lw=1.3, marker="o", ms=3.4, mec="white", mew=0.5, zorder=3)
    c.plot([sel], [depth[metric][sel - 1]], marker="o", ms=5.4, mfc=col, mec=INK, mew=0.7, ls="none", zorder=4)
txt(CX + CW / 2.0, CY + CH + 23.0, "Layer", 8, ha="center")
txt(399.0, CY + CH / 2.0, "Fold", 8, rot=90, ha="center", va="center")

names = sorted(CD["approved_names"], key=lambda r: r["raw_theta_rank"])
raw = dict((int(r["exact_id"]), r) for r in CD["source_records"]["raw"])
sca = dict((int(r["concept_id"]), r) for r in CD["source_records"]["scaffold"])
ids = [r["id"] for r in names]
rank = dict((r["id"], r["raw_theta_rank"]) for r in names)
theta = dict((i, float(raw[i]["difference_Btrained_minus_BinitMedian"])) for i in ids)
theta_s = dict((i, float(sca[i]["Theta_point"])) for i in ids)
ginit = dict((i, float(raw[i]["Gconcept"])) for i in ids)
means = dict(((m["panel"], m["metric"], m["population"]), m["source_value"]) for m in CD["mean_lines"])
m_all, m_sca, m_g = means[("A", "Theta", "original_T")], means[("A", "Theta", "D_unseen_slice")], means[("B", "G_init", "original_T")]
group = dict((r["concept_id"], r["group"]) for r in D3["concept_rows"])
assert len(ids) == 71 and sorted(rank.values()) == list(range(1, 72)) and set(group) == set(CD["frozen_selection_record"]["panel_ids"])
assert all(theta[i] >= theta[j] for i, j in zip(ids, ids[1:]))
assert abs(sum(theta.values()) / 71.0 - m_all) < 1e-9 and abs(sum(ginit.values()) / 71.0 - m_g) < 1e-9
positive = sum(v > 0 for v in theta.values())
colour = lambda i: GROUPC[group[i]] if i in group else GREY
OTHER = tint(GREY, 0.55)
ORDER = sorted(ids, key=lambda i: i in group)
NAMED = ((24, "Carbonyl", -10.0, -38.0), (26, "Aliphatic OH", 9.0, -24.0))
assert all(group[i] == g for (i, _, _, _), g in zip(NAMED, ("Carbonyls", "Hydroxyls")))


def points(ax, xs, ys):
    for i in ORDER:
        hit = i in group
        ax.plot([xs[i]], [ys[i]], marker="o", ms=5.0 if hit else 3.2, mfc=colour(i), mec="white", mew=0.5 if hit else 0.3, ls="none", zorder=5 if hit else 3)


Y0 = 232.0
RA, RH = Y0 + 30.0, 88.0
XLAB = RA + RH + 21.0

cv.head(0.0, Y0 + 11.0, "D", "Transfer across 71 concepts", 244.0)
DX, DW = 36.0, 206.0
DXL, DYL = (0.2, 71.8), (-0.55, 2.95)
d = cv.axes(DX, RA, DW, RH, DXL, DYL, xticks=[1, 20, 40, 60, 71], yticks=[0, 1, 2])
dx_ = lambda v: DX + (v - DXL[0]) / (DXL[1] - DXL[0]) * DW
dy_ = lambda v: RA + (DYL[1] - v) / (DYL[1] - DYL[0]) * RH
d.axhline(0, color=INK, lw=0.6, zorder=1)
d.bar([rank[i] for i in ids], [theta[i] for i in ids], width=0.74, color=[GROUPC[group[i]] if i in group else OTHER for i in ids], linewidth=0, zorder=2)
d.axhline(m_all, color=MUTED, lw=0.7, ls=DASH, zorder=3)
txt(DX + DW, dy_(m_all) - 3.0, "mean %.2f" % m_all, 7.5, color=MUTED, ha="right")
txt(DX + DW, RA + 9.0, "%d of %d positive" % (positive, len(ids)), 8, "bold", color=TAB, ha="right")
for i, lab, ddx, ddy in NAMED:
    x_, top = dx_(rank[i]), dy_(theta[i])
    ov.plot([x_, x_ + ddx], [top - 2.0, top + ddy + 2.5], color=MUTED, lw=0.5, zorder=29)
    txt(x_ + ddx, top + ddy, lab, 7.5, ha="center")
txt(DX + DW / 2.0, XLAB, "Concept rank", 8, ha="center")
txt(10.0, RA + RH / 2.0, "Axis-preference transfer Θ", 8, rot=90, ha="center", va="center")

cv.head(250.0, Y0 + 11.0, "E", "Unseen scaffolds", 132.0)
EX = 292.0
LIM = (-0.6, 3.4)
e = cv.axes(EX, RA, RH, RH, LIM, LIM, xticks=[0, 1, 2, 3], yticks=[0, 1, 2, 3])
e.plot(LIM, LIM, color=MUTED, lw=0.6, ls=DASH, zorder=1)
points(e, theta, theta_s)
assert all(LIM[0] < v < LIM[1] for v in list(theta.values()) + list(theta_s.values()))
txt(EX + RH / 2.0, XLAB, "Θ, all 10,000", 8, ha="center")
txt(264.0, RA + RH / 2.0, "Θ, unseen 7,205", 8, rot=90, ha="center", va="center")

cv.head(388.0, Y0 + 11.0, "F", "Readability transfer", 116.0)
FX, FW = 424.0, 76.0
FXL, FYL = (-0.6, 3.0), (-2.4, 5.4)
f = cv.axes(FX, RA, FW, RH, FXL, FYL, xticks=[0, 1, 2], yticks=[-2, 0, 2, 4])
f.yaxis.set_major_formatter(FixedFormatter(["−2", "0", "2", "4"]))
f.axvline(m_all, color=MUTED, lw=0.6, ls=DASH, zorder=1)
f.axhline(m_g, color=MUTED, lw=0.6, ls=DASH, zorder=1)
points(f, theta, ginit)
assert all(FXL[0] < theta[i] < FXL[1] and FYL[0] < ginit[i] < FYL[1] for i in ids)
txt(FX + FW / 2.0, XLAB, "Θ", 8, ha="center")
gy = RA + RH / 2.0 + 9.0
t1 = ov.text(399.0, gy, "G", fontsize=8, fontstyle="italic", color=INK, rotation=90, ha="center", va="bottom", zorder=30)
cv.fig.canvas.draw()
gh = t1.get_window_extent(cv.fig.canvas.get_renderer()).height * 72.0 / cv.fig.dpi
ov.text(400.7, gy - gh - 0.4, "init", fontsize=5.8, color=INK, rotation=90, ha="center", va="bottom", zorder=30)
cv.texts.append((t1, "G init"))

cv.key(DX, XLAB + 17.0, [(GROUPC[g], GROUPN.get(g, g)) for g in ("Rings", "Carboxylic acids", "Carbonyls", "Hydroxyls", "Amine")] + [(OTHER, "Other")], size=7.5, gap=10.0)

cv.finish(OUT, NAME, "pass both %s | references above trained %d of 6, reference points %d | selected layer %d, folds %.3f %.3f | positive %d of %d | means %.4f %.4f %.4f | ranks of the twelve %s"
          % (both, above, n_ref, sel, mf["J1"]["fold"], mf["J2"]["fold"], positive, len(ids), m_all, m_sca, m_g, sorted(rank[i] for i in group)))
