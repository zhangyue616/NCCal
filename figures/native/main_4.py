import sys, os, math
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import numpy as np

D4, F6, F5, D3 = J(sys.argv[1]), J(sys.argv[2]), J(sys.argv[3]), J(sys.argv[4])
S2, H3 = np.load(sys.argv[5], allow_pickle=False), np.load(sys.argv[6], allow_pickle=False)
MECH, PD, OUT = J(sys.argv[7]), J(sys.argv[8]), sys.argv[9]
NAME = "Figure_4"
W, H = 504.0, 345.0
cv = Canvas(W, H)
ov, txt = cv.ov, cv.txt
DASH = (0, (3, 2))
TWO = (("Aliphatic OH", "Hydroxyls", "293"), ("Carbonyl", "Carbonyls", "454"))
ENC = (("chemberta_mtr", "ChemBERTa-\n77M-MTR", "ChemBERTa-77M-MTR"), ("molformer", "MoLFormer-\nXL", "MoLFormer-XL"))


def open_square(ax, x, y, color, ms=4.6):
    ax.plot([x], [y], marker="s", ms=ms, mfc="white", mec=color, mew=1.0, ls="none", zorder=7)


def open_diamond(ax, x, y, color, ms=3.9):
    """The dose-matched pairs of B: a diamond, so that the open square stays the residual permutation of the key."""
    ax.plot([x], [y], marker="D", ms=ms, mfc="white", mec=color, mew=1.0, ls="none", zorder=7)


cv.head(0.0, 11.0, "A", "Pretrained head", 291.0)
if "2" in OPT:
    txt(286.0, 11.0, "ChemBERTa-77M-MTR", 7.5, color=MUTED, ha="right")
out = dict(((r["source_row"]["coordinate"], r["condition"]), r["plot_values_display_units"]) for r in D4["output_rows"] if r["source_row"]["stratum"] == "all_T")
assert D4["display_unit_multiplier_B_D"] == 1000 and len(out) == 10
AY, AH = 38.0, 92.0
vals = S2["pretrained_values"]
FLOORED = []
conds, cids = S2["pretrained_conditions"].tolist(), S2["pretrained_coordinate_ids"].tolist()
for g, (lab, grp, coord) in enumerate(TWO):
    hue = GROUPC[grp]
    x0 = 36.0 + g * 88.0
    top = (26.0, [0, 10, 20]) if coord == "293" else (6.0, [0, 2, 4, 6])
    a = cv.axes(x0, AY, 58.0, AH, (0, 3), (0, top[0]), yticks=top[1])
    for k, (cond, col) in enumerate((("D_mean", REF), ("M1", pale(hue)), ("M2", hue))):
        est, lo, hi = out[(coord, cond)]
        a.bar(k + 0.5, est, width=0.72, color=col, linewidth=0, zorder=3)
        ov.add_patch(Rectangle((x0 + (k + 0.5) / 3.0 * 58.0 - 2.2, AY + AH + 3.0), 4.4, 4.4, facecolor=(REF, PALE, DARK)[k], edgecolor="none", zorder=30))
        a.plot([k + 0.5, k + 0.5], [lo, hi], color=ERR, lw=0.8, zorder=5)
        a.text(k + 0.5, hi + top[0] * 0.02, ("%.1f" % est), fontsize=7.5, color=INK, ha="center", va="bottom", zorder=6, fontweight="bold" if cond == "M2" else "normal")
        if cond != "D_mean":
            pe, pl, ph = out[(coord, cond + "_delta_perm")]
            a.plot([k + 0.5, k + 0.5], [pl, ph], color=hue, lw=0.8, zorder=6)
            open_square(a, k + 0.5, pe, hue)
    txt(x0 + 29.0, AY + AH + 21.0, lab, 8, ha="center")
    ci = cids.index(int(coord))
    xb, yl, pos = vals[:, ci, conds.index("M1")], vals[:, ci, conds.index("M2")], S2["pretrained_concept_positive"][:, ci]
    for cond in ("D_mean", "M1", "M2"):
        assert abs(vals[:, ci, conds.index(cond)].mean() * 1000.0 - out[(coord, cond)][0]) < 1e-6
    FLOOR, LIM = 0.001, (0.0006, 1500.0)
    FLOORED.append(int((xb * 1000.0 < FLOOR).sum() + (yl * 1000.0 < FLOOR).sum()))
    xs, ys = np.maximum(xb * 1000.0, FLOOR), np.maximum(yl * 1000.0, FLOOR)
    assert max(xs.max(), ys.max()) < LIM[1]
    sy = AY + g * 52.0
    s = cv.axes(232.0, sy, 44.0, 40.0, LIM, LIM, xticks=[0.01, 1, 100], yticks=[0.01, 1, 100], xlog=True, ylog=True)
    s.xaxis.set_major_formatter(FixedFormatter(["0.01", "", "100"]))
    s.yaxis.set_major_formatter(FixedFormatter(["0.01", "", "100"]))
    s.plot(LIM, LIM, color=MUTED, lw=0.6, ls=DASH, zorder=1)
    s.plot(xs[~pos], ys[~pos], marker="o", ms=1.3, mfc=GREY, mec="none", alpha=0.35, ls="none", zorder=2, rasterized=True)
    s.plot(xs[pos], ys[pos], marker="o", ms=1.3, mfc=hue, mec="none", alpha=0.40, ls="none", zorder=3, rasterized=True)
    if g == 0:
        s.set_xticklabels([])
txt(9.0, AY + AH / 2.0, "Output MAE (× 0.001)", 8, rot=90, ha="center", va="center")
txt(254.0, AY + AH + 21.0, "Concept means", 7.5, ha="center")
txt(199.0, AY + AH / 2.0, "Bonding environment", 7.5, rot=90, ha="center", va="center")

cv.head(300.0, 11.0, "B", "Clamping contrasts", 204.0)
HUE3 = {26: GROUPC["Hydroxyls"], 24: GROUPC["Carbonyls"], 25: GROUPC["Carbonyls"]}
NAME3 = {26: "Aliphatic OH", 24: "Carbonyl", 25: "Carbonyl (excl. COOH)"}
rows3 = sorted(D4["historical_contrast_rows"], key=lambda r: r["row_top_to_bottom"])
mech = dict((o["concept_id"], o) for o in MECH["objects"])
pair3 = dict((o["concept_id"], o) for o in PD["objects"])
for r in rows3:
    o, p = mech[r["concept_id"]], pair3[r["concept_id"]]
    assert abs(float(r["full_clamp_original_precision"][0]) - o["effect"]["H"]) < 1e-15 and abs(r["paired_dose_original_precision"][0] - p["H_pair1_existing"]) < 1e-15
    assert [k["name"] for k in o["conditions"][:6]] == ["selected_atom_full"] + ["control%d_atom_full" % i for i in range(1, 6)]
    assert [k["unit"] for k in o["conditions"][:6]] == p["units"] and p["pairs"] == 200 and o["n_positive_subset"] == o["n_negative_subset"] == 200
kx = 306.0
for filled, lab in ((True, "Full clamp"), (False, "Dose-matched")):
    ov.plot([kx + 3.0], [28.4], marker="D", ms=4.2, mfc=INK if filled else "white", mec=INK, mew=0.9 if not filled else 0.0, ls="none", zorder=30)
    kx = cv.right_edge(txt(kx + 8.5, 31.0, lab, 7.5)) + 8.0
ov.plot([kx + 3.0], [28.4], marker="o", ms=3.4, mfc=GREY, mec="none", ls="none", zorder=30)
txt(kx + 8.5, 31.0, "Controls", 7.5)
BY, SS, SX = 54.0, 76.0, 335.0
DLIM = (0.09, 40.0)
s = cv.axes(SX, BY, SS, SS, DLIM, DLIM, xticks=[0.1, 1, 10], yticks=[0.1, 1, 10], xlog=True, ylog=True)
s.plot(DLIM, DLIM, color=MUTED, lw=0.6, ls=DASH, zorder=1)
for c_ in (24, 25, 26):
    dc = mech[c_]["dose_comparison"]
    for k in range(1, 6):
        s.plot([dc[k]["negative"]["mean"]], [dc[k]["positive"]["mean"]], marker="o", ms=3.2, mfc=GREY, mec="white", mew=0.3, ls="none", zorder=2)
doses = {}
for c_ in (24, 25, 26):
    dc, tb, hue = mech[c_]["dose_comparison"], pair3[c_]["tau_box"], HUE3[c_]
    assert DLIM[0] < tb["min"] and all(DLIM[0] < dc[k][st]["mean"] < DLIM[1] for k in range(6) for st in ("positive", "negative"))
    s.plot([tb["min"], tb["max"]], [tb["min"], tb["max"]], color=hue, lw=1.0, zorder=3)
    s.plot([tb["median"]], [tb["median"]], marker="D", ms=3.2, mfc="white", mec=hue, mew=0.9, ls="none", zorder=4)
    s.plot([dc[0]["negative"]["mean"]], [dc[0]["positive"]["mean"]], marker="D", ms=4.4, mfc=hue, mec="white", mew=0.5, ls="none", zorder=5)
    doses[c_] = (dc[0]["negative"]["mean"], dc[0]["positive"]["mean"], min(dc[k]["positive"]["mean"] for k in range(1, 6)), max(dc[k]["positive"]["mean"] for k in range(1, 6)))
assert "%.1f %.2f %.2f" % doses[26][1:] == "14.9 1.24 2.32"
txt(SX + SS / 2.0, BY + SS + 21.0, "Dose, all-negative", 8, ha="center")
txt(308.5, BY + SS / 2.0, "Dose, positive", 8, rot=90, ha="center", va="center")
BX, BW, RHB = 421.0, 80.0, 76.0 / 3.0
b = cv.axes(BX, BY, BW, 3 * RHB, (-2.4, 320.0), (3.0, 0.0), left=False)
b.set_xscale("symlog", linthresh=1.0, linscale=0.8)
b.set_xlim(-2.4, 320.0)
b.xaxis.set_major_locator(FixedLocator([-1, 0, 1, 10, 100]))
b.xaxis.set_major_formatter(FixedFormatter(["−1", "0", "1", "10", "100"]))
b.xaxis.set_minor_locator(NullLocator())
b.axvline(0, color=MUTED, lw=0.6, ls=DASH, zorder=1)
signs = []
for r_, r in enumerate(rows3):
    full = [float(v) * 1000.0 for v in r["full_clamp_original_precision"]]
    pair = [float(v) * 1000.0 for v in r["paired_dose_original_precision"]]
    assert r["row_top_to_bottom"] == r_ + 1 and abs(r["original_ci_percent"] - 98.3333333333) < 1e-6 and -2.4 < pair[1] and full[2] < 320.0
    hue = HUE3[r["concept_id"]]
    b.plot(full[1:], [r_ + 0.50] * 2, color=hue, lw=1.0, zorder=3)
    b.plot([full[0]], [r_ + 0.50], marker="D", ms=4.4, mfc=hue, mec="white", mew=0.5, ls="none", zorder=4)
    b.plot(pair[1:], [r_ + 0.76] * 2, color=hue, lw=1.0, zorder=3)
    open_diamond(b, pair[0], r_ + 0.76, hue)
    signs.append((full[1] > 0, pair[1] < 0 < pair[2]))
    txt(BX + 1.0, BY + r_ * RHB + 5.4, NAME3[r["concept_id"]], 7).set_bbox(dict(facecolor="white", edgecolor="none", pad=0.5))
txt(BX + BW / 2.0, BY + 3 * RHB + 21.0, "Contrast (× 0.001)", 8, ha="center")

LY = 164.0
lx = cv.key(6.0, LY, ((REF, "Mean replacement"), (PALE, "Concept means"), (DARK, "Bonding environment")), size=8, gap=12.0)
ov.plot([lx + 15.2], [LY - 3.2], marker="s", ms=5.0, mfc="white", mec=INK, mew=1.0, ls="none", zorder=30)
txt(lx + 21.5, LY, "Shuffled displacement", 8)

RY = 186.0
cv.head(0.0, RY, "C", "Task heads", 291.0)
absol = dict(((r["model"], r["task"], r["condition"]), r["point"]) for r in F6["absolute"])
passed = sum(1 for r in F6["paired"] if r["Holm12_p"] <= 0.05)
assert len(F6["absolute"]) == 16 and len(F6["paired"]) == 12 and all(r["point"] > 0 for r in F6["paired"])
CYY, CH = RY + 30.0, 88.0
for t, (task, unit, top, ticks) in enumerate((("bace", "probability", 0.052, [0, 0.02, 0.04]), ("esol", "log S", 0.15, [0, 0.05, 0.1, 0.15]))):
    x0 = 44.0 + t * 146.0
    c = cv.axes(x0, CYY, 100.0, CH, (0, 2), (0, top), yticks=ticks)
    for e, (model, short, _) in enumerate(ENC):
        for k, (cond, col) in enumerate((("development_mean", REF), ("binary_mean", PALE), ("local_graph", DARK))):
            assert absol[(model, task, cond)] < top
            c.bar(e + 0.5 + (k - 1) * 0.25 - 0.09, absol[(model, task, cond)], width=0.22, color=col, linewidth=0, zorder=3)
        open_square(c, e + 0.5 + 0.385, absol[(model, task, "residual_permutation")], INK, ms=4.2)
        txt(x0 + (e + 0.5) / 2.0 * 100.0, CYY + CH + 5.0, short, 7, ha="center", va="top")
    txt(x0 + 3.0, CYY - 6.0, "%s, %s" % (task.upper(), unit), 7.5, color=MUTED)
txt(9.0, CYY + CH / 2.0, "Output deviation", 8, rot=90, ha="center", va="center")
txt(44.0 + (146.0 + 100.0) / 2.0, CYY + CH + 33.0, "%d of %d contrasts pass" % (passed, len(F6["paired"])), 8, "bold", color=TAB, ha="center")

cv.head(300.0, RY, "D", "Single coordinates", 204.0)
hv, hm, ht, hc = H3["values"], H3["models"].tolist(), H3["tasks"].tolist(), H3["conditions"].tolist()
assert hv.shape == (2, 2, 9, 4) and hc == ["development_mean", "binary_mean", "local_graph", "residual_permutation"]
group_of = dict((r["concept_id"], r["group"]) for r in D3["concept_rows"])
CWD, CHT, HY = 8.6, 8.4, RY + 40.0
for e, (model, _, full) in enumerate(ENC):
    mi = hm.index(model)
    order = [(ro["coordinate"], GROUPC[group_of[ro["representative_concept_id"]]]) for ro in F5["row_order"][model]]
    assert [c_ for c_, _ in order] == H3["coordinates"][mi].tolist()
    bx = 326.0 + e * 88.0
    for r_, (_, hue) in enumerate(order):
        ov.add_patch(Rectangle((bx - 7.0, HY + r_ * CHT + 1.6), 4.4, CHT - 3.2, facecolor=hue, edgecolor="none", zorder=24))
    for t, task in enumerate(("bace", "esol")):
        block = hv[mi, ht.index(task)]
        lo, hi = math.log(block.min()), math.log(block.max())
        tx = bx + t * (4 * CWD + 3.0)
        for r_ in range(9):
            for k in range(4):
                ov.add_patch(Rectangle((tx + k * CWD, HY + r_ * CHT), CWD - 0.5, CHT - 0.5, facecolor=tint(TAB, 0.06 + 0.94 * (math.log(block[r_, k]) - lo) / (hi - lo)),
                                       edgecolor="none", zorder=23))
        for k, col in enumerate((REF, PALE, DARK, None) if (e, t) == (0, 0) else ()):
            cx_ = tx + k * CWD + (CWD - 0.5) / 2.0
            if col is None:
                ov.plot([cx_], [HY + 9 * CHT + 5.0], marker="s", ms=3.6, mfc="white", mec=INK, mew=0.8, ls="none", zorder=30)
            else:
                ov.add_patch(Rectangle((cx_ - 2.2, HY + 9 * CHT + 2.8), 4.4, 4.4, facecolor=col, edgecolor="none", zorder=30))
        txt(tx + 2 * CWD - 0.25, HY - 4.0, task.upper(), 7, color=MUTED, ha="center")
    txt(bx + 4 * CWD + 1.25, HY - 14.0, full, 7, ha="center")
gx0, gy0 = 392.0, HY + 9 * CHT + 17.0
for q in range(30):
    ov.add_patch(Rectangle((gx0 + q * 1.4, gy0), 1.45, 5.0, facecolor=tint(TAB, 0.06 + 0.94 * (q + 0.5) / 30.0), edgecolor="none", zorder=30))
txt(gx0 - 3.5, gy0 + 4.8, "low", 7, color=MUTED, ha="right")
txt(gx0 + 45.5, gy0 + 4.8, "high", 7, color=MUTED)

cv.finish(OUT, NAME, "contrasts that pass %d of %d | clamp rows (full interval above zero, paired interval spans zero) %s | scatter molecules %d, values drawn at the floor %s"
                     " | full-clamp dose of the selected coordinate (all-negative, positive) %s | matched dose medians %s"
          % (passed, len(F6["paired"]), signs, len(vals), FLOORED, ["%d: %.2f, %.2f" % (c_, doses[c_][0], doses[c_][1]) for c_ in (26, 24, 25)],
             ["%.3f" % pair3[c_]["tau_box"]["median"] for c_ in (26, 24, 25)]))
