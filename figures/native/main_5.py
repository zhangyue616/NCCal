# -*- coding: utf-8 -*-
"""Draw Figure 5 from the molecular edits, SMILES rewrites and basis comparisons.

The stored activations and frozen rule supply the molecular example and the panel summaries.
Run from the repository root with python -B figures/render.py 5.
"""
import sys, os, io, json, gzip, math, csv
from pathlib import Path
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from style import *
from matplotlib.patches import Circle
from matplotlib.ticker import FixedLocator, FixedFormatter
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT, PAIRS, OUT, F5D = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
os.makedirs(OUT, exist_ok=True)



def table(n, delim="\t"):
    return list(csv.DictReader(io.StringIO((ROOT / n).read_bytes().decode("utf-8-sig")), delimiter=delim))


deltas = table("results/artwork/use/5a_and_s_r1_changed_atom_deltas.tsv")
metrics = [r for r in table("results/artwork/use/5a_and_s_r1_primary_metrics.tsv") if r["representation"] == "controlled"]
census = table("results/artwork/use/5b_and_s_q1_census.tsv")
basis = table("results/artwork/use/5c_and_s_s1_basis_summary.tsv")
raw = np.load(io.BytesIO((ROOT / "results/molecular/edits/raw_pairs.npz").read_bytes()))
U = 293
off, cls, okp, cids = raw["offsets"], raw["edit_classes"], raw["main_pass"], raw["cids"]
src, prod, pred, chg, sai = raw["source"][:, U], raw["controlled_product"][:, U], raw["predicted_delta"][:, U], raw["changed"], raw["source_atom_index"]
recs = {}
with gzip.open(PAIRS, "rt", encoding="utf-8") as f:
    for ln in f:
        r = json.loads(ln)
        recs[(int(r["cid"]), int(r["edit_class"]))] = r


def pick():
    """For each main hydroxyl-to-hydrogen pair, select the affected shared atom with the largest absolute rule-predicted change.
    Compute its absolute prediction error and the median over all 60 pairs. Among the 14 molecules with at most 16 heavy atoms,
    choose the pair nearest that median. The selected atom need not be the edited carbon in every pair."""
    c = []
    for p in range(len(cls)):
        if cls[p] != 1 or not okp[p]:
            continue
        s = slice(off[p], off[p + 1])
        m = chg[s]
        if not m.any():
            continue
        k = int(np.argmax(np.abs(pred[s]) * m))
        c.append((abs(float((prod[s] - src[s])[k] - pred[s][k])), p, k))
    med = float(np.median([e[0] for e in c]))
    small = [e for e in c if Chem.MolFromSmiles(recs[(int(cids[e[1]]), 1)]["source"]["smiles"]).GetNumAtoms() <= 16]
    e = min(small, key=lambda q: abs(q[0] - med))
    return e, med, len(c)


(ERR, P, K), MED, NPAIR = pick()
rec = recs[(int(cids[P]), 1)]
S = slice(off[P], off[P + 1])
A_SRC = dict((int(i), float(v)) for i, v in zip(sai[S], src[S]))
A_PROD = dict((int(i), float(v)) for i, v in zip(sai[S], prod[S]))
KATOM = int(sai[S][K])
D_MEAS, D_RULE = float((prod[S] - src[S])[K]), float(pred[S][K])
print("example pair: cid %d, %s, atoms %d | error %.2f (median of %d pairs %.2f) | measured %.2f rule %.2f" % (
    int(cids[P]), rec["source"]["smiles"], len(A_SRC), ERR, NPAIR, MED, D_MEAS, D_RULE))

W, H = 504.0, 410.0
cv = Canvas(W, H)
plt.rcParams["mathtext.default"] = "regular"
ov = cv.ov
MINUS = lambda v: ("%g" % v).replace("-", "−")


def line(xs, ys, color=INK, lw=0.8, z=24, ls="-"):
    ov.plot(xs, ys, color=color, lw=lw, ls=ls, solid_capstyle="round", zorder=z)


def arrow(x1, y1, x2, y2, color=INK, lw=0.9):
    ov.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=27,
                arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.25", color=color, lw=lw, mutation_scale=6.5, shrinkA=0, shrinkB=0))


def tick(x, y, s, color):
    line([x - 0.36 * s, x - 0.08 * s, x + 0.40 * s], [y + 0.02 * s, y + 0.30 * s, y - 0.32 * s], color=color, lw=1.0, z=31)


def minus_ticks(a, xt, yt):
    a.xaxis.set_major_formatter(FixedFormatter([MINUS(v) for v in xt]))
    a.yaxis.set_major_formatter(FixedFormatter([MINUS(v) for v in yt]))


# ------------------------------------------------------------------ a  the molecule pair
mol = Chem.MolFromSmiles(rec["source"]["smiles"])
AllChem.Compute2DCoords(mol)
kek = Chem.Mol(mol)
Chem.Kekulize(kek, clearAromaticFlags=True)
conf = mol.GetConformer()
XY = dict((a.GetIdx(), (conf.GetAtomPosition(a.GetIdx()).x, -conf.GetAtomPosition(a.GetIdx()).y)) for a in mol.GetAtoms())
REMOVED = set(int(i) for i in rec["removed_source_atoms"])
VMAX = 10.0
NB = dict((a.GetIdx(), [n.GetIdx() for n in a.GetNeighbors()]) for a in mol.GetAtoms())


def label_of(a):
    if a.GetSymbol() == "C":
        return "", ""
    h = a.GetTotalNumHs()
    return a.GetSymbol(), ("" if h == 0 else "H" if h == 1 else "H$_{%d}$" % h)


LAB = dict((a.GetIdx(), label_of(a)) for a in mol.GetAtoms())


def draw_mol(cx, cy, bl, present, values, mark):
    xs = [XY[i][0] for i in XY]
    ys = [XY[i][1] for i in XY]
    x0, y0 = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    f = bl / 1.5
    pos = dict((i, (cx + (XY[i][0] - x0) * f, cy + (XY[i][1] - y0) * f)) for i in XY)
    for i in sorted(values, key=lambda q: values[q]):
        if i in present:
            ov.add_patch(Circle(pos[i], 0.46 * bl, facecolor=tint(ORANGE, min(max(values[i], 0.0) / VMAX, 1.0)), edgecolor="none", zorder=22))
    for b in kek.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i not in present or j not in present:
            continue
        (xa, ya), (xb, yb) = pos[i], pos[j]
        d = math.hypot(xb - xa, yb - ya)
        ux, uy = (xb - xa) / d, (yb - ya) / d
        ga, gb = (0.33 * bl if LAB[i][0] else 0.0), (0.33 * bl if LAB[j][0] else 0.0)
        xa, ya, xb, yb = xa + ux * ga, ya + uy * ga, xb - ux * gb, yb - uy * gb
        if b.GetBondTypeAsDouble() >= 2:
            for o in (-0.09 * bl, 0.09 * bl):
                line([xa - uy * o, xb - uy * o], [ya + ux * o, yb + ux * o], lw=0.7)
        else:
            line([xa, xb], [ya, yb], lw=0.7)
    for i in present:
        sym, hs = LAB[i]
        if not sym:
            continue
        px, py = pos[i]
        nb = [pos[n] for n in NB[i] if n in present]
        right = (sum(q[0] for q in nb) / len(nb) - px) > 0 if nb else False       # neighbours to the right: hydrogens go left
        if hs and right:
            cv.txt(px + 2.5, py, hs + sym, 6.5, ha="right", va="center", count=False)
        elif hs:
            cv.txt(px - 2.5, py, sym + hs, 6.5, ha="left", va="center", count=False)
        else:
            cv.txt(px, py, sym, 6.5, ha="center", va="center", count=False)
    if mark in present:
        ov.add_patch(Circle(pos[mark], 0.60 * bl, facecolor="none", edgecolor=INK, lw=0.8, zorder=26))
    return pos


cv.head(4, 11, "a", "one edited pair", 0)
span = max(max(v[0] for v in XY.values()) - min(v[0] for v in XY.values()), 1.0)
tall = max(max(v[1] for v in XY.values()) - min(v[1] for v in XY.values()), 1.0)
BL = min(13.5, 1.5 * 140.0 / span, 1.5 * 38.0 / tall)
ALL = set(XY)
draw_mol(96, 48, BL, ALL, A_SRC, KATOM)
draw_mol(96, 112, BL, ALL - REMOVED, A_PROD, KATOM)
arrow(96, 73, 96, 88)
cv.txt(101, 83, "OH→H", 6.5, color=MUTED)
cv.txt(24, 26, "Parent", 7.5, color=MUTED)
cv.txt(24, 92, "Edited", 7.5, color=MUTED)
cv.txt(8, 148, "Change at the circled carbon", 7.5)
for k, (lab_, val, col) in enumerate((("Measured", D_MEAS, ORANGE), ("Rule", D_RULE, DARK))):
    y = 159 + 11 * k
    cv.txt(8, y + 2.6, lab_, 7.5, color=INK)
    ov.add_patch(Rectangle((48, y - 3.4), abs(val) * 7.0, 6.4, facecolor=col, edgecolor="none", zorder=23))
    cv.txt(48 + abs(val) * 7.0 + 3, y + 2.6, ("%.1f" % val).replace("-", "−"), 7.5)
for k in range(40):                                              # activation key
    ov.add_patch(Rectangle((122 + k * 1.1, 184.0), 1.15, 5.0, facecolor=tint(ORANGE, k / 39.0), edgecolor="none", zorder=23))
cv.txt(119, 188.7, "0", 6.5, ha="right", color=MUTED)
cv.txt(168, 188.7, "10", 6.5, color=MUTED)
cv.txt(8, 188.7, "Disc: activation of neuron 293", 6.5, color=MUTED)

# ------------------------------------------------------------------ b  predicted against measured
cv.head(186, 11, "b", "predicted against measured", 0)
pts = [(float(r["predicted_delta"]), float(r["measured_delta"])) for r in deltas if r["unit"] == str(U) and r["edit_class"] == "1"]
mrow = [r for r in metrics if r["unit"] == str(U) and r["edit_class"] == "1"][0]
B = cv.axes(214, 26, 112, 124, (-12.5, 7.0), (-12.5, 7.0), [-10, -5, 0, 5], [-10, -5, 0, 5])
minus_ticks(B, [-10, -5, 0, 5], [-10, -5, 0, 5])
assert all(-12.5 < p[0] < 7.0 and -12.5 < p[1] < 7.0 for p in pts)
B.plot([-12.5, 7], [-12.5, 7], color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
B.scatter([p[0] for p in pts], [p[1] for p in pts], s=9, color=ROSE, alpha=0.55, linewidths=0, zorder=3)
B.scatter([D_RULE], [D_MEAS], s=26, facecolor="none", edgecolor=INK, linewidths=0.8, zorder=4)
B.set_xlabel("Rule-predicted change", fontsize=7.5, labelpad=2)
B.set_ylabel("Measured change", fontsize=7.5, labelpad=1)
cv.txt(220, 36, "R$^2$ = %.2f" % float(mrow["R2"]), 7.5)
cv.txt(220, 46, "%d atoms, %s pairs" % (len(pts), mrow["molecules"]), 6.5, color=MUTED)

# ------------------------------------------------------------------ c  nine neurons by five edits
cv.head(338, 11, "c", "neurons by edits", 0)
units = [293, 454] + sorted(int(u) for u in set(r["unit"] for r in metrics) if int(u) not in (293, 454))
EDITS = ["OH→H", "OH→OMe", "C=O→CH$_2$", "CO$_2$H→CO$_2$Me", "Ar–H→F"]
NAMES = {293: ("Aliphatic hydroxyl", "Hydroxyls"), 454: ("Carbonyl", "Carbonyls"), 99: ("Aliphatic acid", "Carboxylic acids"),
         121: ("6-membered ring", "Rings"), 188: ("3-membered ring", "Rings"), 266: ("Aromatic acid", "Carboxylic acids"),
         349: ("5-membered ring", "Rings"), 382: ("Primary amine", "Amine"), 388: ("Aromatic hydroxyl", "Hydroxyls")}      # worksheet S-M1
CX, CY, CWd, CHt = 412.0, 62.0, 13.5, 12.0
npass = 0
for i, u in enumerate(units):
    y = CY + i * CHt
    cv.txt(CX - 4, y + CHt / 2.0 + 2.6, "%d" % u, 7.5, ha="right", weight="bold", color=GROUPC[NAMES[u][1]])
    cv.txt(CX - 21, y + CHt / 2.0 + 2.4, NAMES[u][0], 6.5, ha="right", color=INK if u in (293, 454) else MUTED)
    for j in range(5):
        r = [q for q in metrics if int(q["unit"]) == u and q["edit_class"] == str(j + 1)][0]
        r2 = float(r["R2"])
        x = CX + j * CWd + (4.0 if j == 4 else 0.0)
        k = min(max(r2, 0.0) / 0.8, 1.0)
        face = LIGHT if r2 <= 0 else tint(TAB, 0.10 + 0.90 * k)
        ov.add_patch(Rectangle((x + 0.5, y + 0.5), CWd - 1.0, CHt - 1.0, facecolor=face, edgecolor="none", zorder=22))
        if j < 4 and r["better_than_both"] == "True" and r["unchanged_small"] == "True":
            tick(x + CWd / 2.0, y + CHt / 2.0, 6.6, "white" if k > 0.45 else INK)
            npass += 1
for j, e in enumerate(EDITS):
    x = CX + j * CWd + (4.0 if j == 4 else 0.0) + CWd / 2.0
    cv.txt(x - 2.5, CY - 3.0, e, 6.5, rot=60, ha="left", va="bottom", color=MUTED if j == 4 else INK, count=False)
cv.txt(CX + 4 * CWd + 4.0 + CWd / 2.0, CY + 9 * CHt + 7.5, "control", 6.5, ha="center", color=MUTED)
ky = CY + 9 * CHt + 17
for k in range(40):
    ov.add_patch(Rectangle((CX + k * 1.2, ky), 1.25, 5.0, facecolor=tint(TAB, 0.10 + 0.90 * k / 39.0), edgecolor="none", zorder=23))
cv.txt(CX - 3, ky + 4.7, "R$^2$  0", 6.5, ha="right", color=MUTED)
cv.txt(CX + 50.5, ky + 4.7, "0.8", 6.5, color=MUTED)
ov.add_patch(Rectangle((CX + 66, ky), 5.0, 5.0, facecolor=LIGHT, edgecolor="none", zorder=23))
cv.txt(CX + 73.5, ky + 4.7, "≤ 0", 6.5, color=MUTED)
tick(CX - 44, ky + 13.5, 6.5, INK)
cv.txt(CX - 38, ky + 16, "passes (%d of 36)" % npass, 6.5, color=MUTED)

# ------------------------------------------------------------------ d  three frozen notations
cv.head(4, 216, "d", "one molecule, three notations", 0)
J5 = json.load(io.open(F5D, encoding="utf-8"))
assert int(J5["cid"]) == int(cids[P])                              # the molecule of panel a
HI = int(J5["highlighted_site_from_figure_5a_edit"]["canonical_carbon_atom_index_0"])
CELL, DX0, BARK = 7.3, 14.0, 13.0
rule_v, meas_v = None, []
for k, enc in enumerate(J5["encodings"]):
    y = 246.0 + k * 40.0
    cv.txt(DX0, y - 4.0, "Canonical" if enc["encoding_id"] == "canonical" else enc["encoding_id"].replace("rewrite_", "Rewrite "), 6.5, color=MUTED)
    val, mark = {}, None
    for a in enc["atoms_in_canonical_order"]:
        st, en = a["source_element_character_span"]["zero_based_half_open"]
        v = float(a["neurons"]["293"]["measured_post_gelu"])
        for c in range(st, en):
            val[c] = v
        if int(a["canonical_atom_index_0"]) == HI:
            mark = (st, en, v, float(a["neurons"]["293"]["frozen_rule_prediction"]))
    for c, ch_ in enumerate(enc["smiles"]):
        face = tint(ORANGE, min(max(val[c], 0.0) / VMAX, 1.0)) if c in val else "white"
        ov.add_patch(Rectangle((DX0 + c * CELL, y), CELL - 0.6, 10.0, facecolor=face, edgecolor=GREY if c in val else LIGHT, linewidth=0.4, zorder=23))
        ov.text(DX0 + c * CELL + (CELL - 0.6) / 2.0, y + 7.4, ch_, fontsize=6.5, color=INK, ha="center", va="baseline", zorder=30)
    cv.texts.append((ov.texts[-1], enc["smiles"]))
    ov.add_patch(Rectangle((DX0 + mark[0] * CELL - 0.5, y - 0.5), (mark[1] - mark[0]) * CELL + 0.4, 11.0, facecolor="none", edgecolor=INK, linewidth=0.9, zorder=31))
    by = y + 14.5
    ov.add_patch(Rectangle((DX0, by), mark[2] * BARK, 5.0, facecolor=ORANGE, edgecolor="none", zorder=23))
    line([DX0 + mark[3] * BARK] * 2, [by - 1.6, by + 6.6], color=DARK, lw=1.1, z=25)
    cv.txt(DX0 + mark[2] * BARK + 3.0, by + 4.6, "%.1f" % mark[2], 6.5)
    rule_v = mark[3]
    meas_v.append(mark[2])
assert abs(meas_v[0] - A_SRC[KATOM]) < 1e-4                       # the canonical reading is the one drawn in panel a
ly = 246.0 + 3 * 40.0 - 8.0
ov.add_patch(Rectangle((DX0, ly), 9.0, 5.0, facecolor=ORANGE, edgecolor="none", zorder=23))
cv.txt(DX0 + 12.0, ly + 4.6, "Activation at the marked carbon", 6.5, color=MUTED)
line([DX0 + 4.5] * 2, [ly + 9.4, ly + 17.6], color=DARK, lw=1.1, z=25)
cv.txt(DX0 + 12.0, ly + 15.8, "Rule prediction, %.1f in every notation" % rule_v, 6.5, color=MUTED)
print("panel d: marked carbon %s | measured %s | rule %.2f" % (HI, [round(v, 2) for v in meas_v], rule_v))

# ------------------------------------------------------------------ e  census under rewriting
cv.head(186, 216, "e", "census under rewriting", 0)
us = [r for r in census if r["canonical_usable"] == "True"]
COL = {"retained": SAGE, "failed": ROSE, "undetermined": GREY}
MK = {"retained": "o", "failed": "X", "undetermined": "s"}
E = cv.axes(218, 264, 112, 108, (0.55, 1.0), (-6.5, 13.5), [0.6, 0.7, 0.8, 0.9, 1.0], [-5, 0, 5, 10])
minus_ticks(E, [0.6, 0.7, 0.8, 0.9, 1.0], [-5, 0, 5, 10])
assert all(0.55 < float(r["Pearson"]) < 1.0 and -6.5 < 1000 * float(r["rewritten_mean_gain"]) < 13.5 for r in us)
E.axhline(0, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
E.axvline(0.8, color=MUTED, lw=0.7, ls=(0, (1, 2)), zorder=1)
for st in ("undetermined", "retained", "failed"):
    q = [r for r in us if r["rewritten_status"] == st]
    E.scatter([float(r["Pearson"]) for r in q], [1000 * float(r["rewritten_mean_gain"]) for r in q], s=13, marker=MK[st], color=COL[st],
              edgecolors="white", linewidths=0.3, zorder=3 if st != "undetermined" else 2)
E.set_xlabel("Same-atom correlation across notations", fontsize=7.5, labelpad=2)
E.set_ylabel("Advantage over mean\nreplacement (10$^{-3}$)", fontsize=7.5, labelpad=1)
for u, lx, ly, ha in ((293, 0.755, 7.4, "right"), (454, 0.79, -4.2, "left")):
    r = [q for q in us if q["unit"] == str(u)][0]
    px, py = float(r["Pearson"]), 1000 * float(r["rewritten_mean_gain"])
    E.plot([px, lx + (0.004 if ha == "right" else -0.004)], [py, ly], color=INK, lw=0.5, zorder=4)
    E.text(lx, ly, "%d" % u, fontsize=7.5, fontweight="bold", ha=ha, va="center", color=INK, zorder=5)
Hh = cv.axes(218, 240, 112, 19, (0.55, 1.0), (0, 18), None, None, left=False)
Hh.spines["bottom"].set_visible(False)
edges = np.arange(0.55, 1.0001, 0.05)
bot = np.zeros(len(edges) - 1)
for st in ("retained", "undetermined", "failed"):
    h = np.histogram([float(r["Pearson"]) for r in us if r["rewritten_status"] == st], bins=edges)[0]
    Hh.bar(edges[:-1] + 0.025, h, width=0.044, bottom=bot, color=COL[st], linewidth=0, zorder=2)
    bot += h
assert bot.max() <= 18 and bot.sum() == len(us), (bot.max(), bot.sum())          # no bar is cut, every neuron is counted
kx = 204.0
for st, lab_ in (("retained", "Retained"), ("failed", "Failed"), ("undetermined", "Unresolved")):
    n = sum(1 for r in us if r["rewritten_status"] == st)
    ov.scatter([kx + 3], [231.5], s=13, marker=MK[st], color=COL[st], edgecolors="none", zorder=30)
    kx = cv.right_edge(cv.txt(kx + 8, 234, "%s %d" % (lab_, n), 6.5)) + 6

# ------------------------------------------------------------------ f  explained, replaceable, kept
cv.head(344, 216, "f", "explained, replaceable, kept", 0)
sq = lambda v: math.sqrt(float(v))
FX, FY, FW, FH = 368.0, 240.0, 90.0, 132.0
F = cv.axes(FX, FY, FW, FH, (-0.38, 2.2), (0, sq(430)), None, None)
tk = [0, 10, 50, 150, 400]
F.yaxis.set_major_locator(FixedLocator([sq(v) for v in tk]))
F.yaxis.set_major_formatter(FixedFormatter(["%d" % v for v in tk]))
F.xaxis.set_major_locator(FixedLocator([0, 1, 2]))
F.xaxis.set_major_formatter(FixedFormatter(["Explained", "Replaceable", "Kept"]))
F.tick_params(axis="x", labelsize=6.5, length=0, pad=3)
F.set_ylabel("Directions per 464", fontsize=7.5, labelpad=1)
row = lambda b, sub="full": [r for r in basis if r["basis"] == b and r["subset"] == sub][0]
tri = lambda r: [sq(r["R2_ge05_n"]), sq(r["canonical_usable_n"]), sq(r["retained"])]
RR = [r for r in basis if r["basis"].startswith("rot_")]
SS = [r for r in basis if r["basis"].startswith("sae_") and r["subset"] != "full"]
rots, saes = [tri(r) for r in RR], [tri(r) for r in SS]
for t in rots:
    F.plot([0, 1, 2], t, color=REF, lw=0.6, alpha=0.75, zorder=2)
for t in saes:
    F.plot([0, 1, 2], t, color=LILAC, lw=0.8, alpha=0.9, zorder=3)
F.plot([0, 1, 2], tri(row("pca")), color=GREY, lw=0.9, ls=(0, (3, 1.5)), zorder=2)
F.plot([0, 1, 2], tri(row("native")), color=INK, lw=1.6, marker="o", ms=3.2, zorder=5)
rng = lambda key, rows: (min(int(r[key]) for r in rows), max(int(r[key]) for r in rows))
nat, pca = row("native"), row("pca")
lab_r = [("Dictionary", "%d–%d" % rng("retained", SS), LILAC, np.mean([t[2] for t in saes])),
         ("Neurons", nat["retained"], INK, sq(nat["retained"])),
         ("Principal", pca["retained"], MUTED, sq(pca["retained"])),
         ("Rotated", "%d–%d" % rng("retained", RR), REF, np.mean([t[2] for t in rots]))]
ypt = lambda v: FY + FH * (1 - v / sq(430))
prev = -99.0
for name, val, col, v in sorted(lab_r, key=lambda q: -q[3]):
    y = max(ypt(v) - 6.0, prev + 17.0)
    cv.txt(FX + FW + 4, y, name, 6.5, color=col, weight="bold" if name == "Neurons" else "normal")
    cv.txt(FX + FW + 4, y + 7.6, "%s kept" % val, 6.5, color=col)
    prev = y
cv.txt(FX + 27, ypt(np.mean([t[0] for t in rots])) + 6, "%d–%d" % rng("R2_ge05_n", RR), 6.5, color=REF)
print("panel c ticks: %d of 36 | rotations explained %s replaceable %s kept %s | dictionaries %s %s %s" % (
    npass, rng("R2_ge05_n", RR), rng("canonical_usable_n", RR), rng("retained", RR), rng("R2_ge05_n", SS), rng("canonical_usable_n", SS), rng("retained", SS)))
cv.finish(OUT, "Figure_5", write_labels=True)
