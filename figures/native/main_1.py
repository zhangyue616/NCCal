import sys, os, math, gzip, json
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.colors import to_rgb
from PIL import Image
import numpy as np

VALUES = "--values" in sys.argv
ARG = [a for a in sys.argv[1:] if not a.startswith("--")]
G1, HEAT, G4, D4, D2, D3 = [J(p) for p in ARG[:6]]
ACT, ANCH, PROT, MAN, OUT, EXPO = ARG[6], ARG[7], ARG[8], J(ARG[9]), ARG[10], J(ARG[11])
NAME = "Figure_1"
W, H = 504.0, 287.0
cv = Canvas(W, H)
ov, txt = cv.ov, cv.txt
PINK, PLUM = "#CC6A8C", TAB
ENC, ENCP = "#2F5D8C", PLUM
GROUND, BAND = 0.10, 0.50

mol = dict((m["name"], m) for m in G1["molecules"])["ketone2"]
assert mol["smiles"] == HEAT["molecule"]["smiles"]
n = len(mol["atoms"])
row = HEAT["molecule"]["row_zero_based"]
a0, a1 = HEAT["molecule"]["atom_slice"]
z = np.load(ACT, allow_pickle=False)
col, mn = z["coordinate_ids"].tolist().index(454), z["model_names"].tolist()
raw = {"obs": z["actual"][a0:a1, col].astype(float), "bin": z["predictions"][a0:a1, col, mn.index("M1")], "loc": z["predictions"][a0:a1, col, mn.index("M2")]}
assert a1 - a0 == n and set(z["mol_idx"][a0:a1].tolist()) == {row}
assert np.allclose(raw["obs"], HEAT["raw_observed"], atol=2e-6) and np.allclose(raw["loc"], HEAT["raw_reconstructed"], atol=2e-6)
positive = z["target_labels"][a0:a1, col]
assert len(set(np.round(raw["bin"][positive], 9))) == 1 and len(set(np.round(raw["bin"][~positive], 9))) == 1
SCALE = max(float(v.max()) for v in raw.values())
shade = lambda v: dict((str(i), max(float(q), 0.0) / SCALE) for i, q in enumerate(v))
OBS, BIN, LOC = shade(raw["obs"]), shade(raw["bin"]), shade(raw["loc"])
anchor = J(ANCH)
assert str(anchor["cid"]) == str(HEAT["molecule"]["cid"]) and len(anchor["atoms"]) == n
N_TOK, ANCHORS = len(anchor["input_ids"]), set(a["anchor_token_index"] for a in anchor["atoms"])
assert len(ANCHORS) == n
from rdkit import Chem
rd = Chem.MolFromSmiles(mol["smiles"])
BONDS = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in rd.GetBonds()]
XY = dict((a["index"], (a["x"], a["y"])) for a in mol["atoms"])
assert rd.GetNumAtoms() == n and all(0.75 < math.hypot(XY[i][0] - XY[j][0], XY[i][1] - XY[j][1]) / mol["bond_length_pt"] < 1.25 for i, j in BONDS)
CONCEPT = set(int(i) for i in np.flatnonzero(positive))
AB, CD = D2["panel_AB_frozen_source"], D2["panel_CD_frozen_source"]
mtr = [m["H4"] for m in AB["audit"]["models"] if m["model"] == "chemberta_mtr"][0]
theta = [m["source_value"] for m in CD["mean_lines"] if m["metric"] == "Theta" and m["population"] == "original_T"][0]
r2 = dict((r["condition"], r["plot_values"][0]) for r in D4["activation_rows"] if r["label"] == "Carbonyl")
mae = dict((r["source_row"]["condition"], float(r["source_row"]["estimate"])) for r in D4["output_rows"]
           if r["source_row"]["coordinate"] == "454" and r["source_row"]["stratum"] == "all_T")
brier = dict((r["model"], float(r["paired_Brier_mean"])) for r in D3["absolute_losses"])
site = [m for m in MAN if m["file"] == os.path.basename(PROT)][0]
ex = D3["sequence_example"]
assert ex["PDB"] == site["pdb_id"] and set(brier) >= {"X", "Xj", "S", "Sj", "L", "Lj"}
windows = []
for key, res in (("A164", ex["HELIX_site_1based"]), ("A178", ex["STRAND_site_1based"])):
    i = res - ex["fragment_start_1based"]
    assert ex["sequence_fragment"][i] == "A" and site["sites"][key]["residue"] == "ALA%d" % res
    windows.append(ex["sequence_fragment"][i - 3:i + 4])


def rbox(x, y, w, h, face="white", edge="none", lw=0.0, r=3.0, z=22):
    ov.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%g" % r, facecolor=face, edgecolor=edge, linewidth=lw, zorder=z))


def arrow(x1, y1, x2, y2, lw=0.9, color=INK, z=27, ms=6.5, both=False):
    ov.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=z,
                arrowprops=dict(arrowstyle=("<|-|>" if both else "-|>") + ",head_length=0.5,head_width=0.25", color=color, lw=lw, mutation_scale=ms, shrinkA=0, shrinkB=0))


def line(xs, ys, color=INK, lw=0.9, z=26, cap="butt"):
    ov.plot(xs, ys, color=color, lw=lw, solid_capstyle=cap, solid_joinstyle="miter", zorder=z)


def snow(x, y, r=4.0, color=INK):
    for k in range(6):
        a = math.radians(60 * k + 30)
        ux_, uy_ = math.cos(a), math.sin(a)
        ov.plot([x, x + r * ux_], [y, y + r * uy_], color=color, lw=0.7, solid_capstyle="round", zorder=31)
        bx, by = x + 0.58 * r * ux_, y + 0.58 * r * uy_
        for s_ in (1.0, -1.0):
            b = a + s_ * math.radians(55)
            ov.plot([bx, bx + 0.36 * r * math.cos(b)], [by, by + 0.36 * r * math.sin(b)], color=color, lw=0.7, solid_capstyle="round", zorder=31)


def mix(base, color, k):
    b, c = to_rgb(base), to_rgb(color)
    return tuple(b[i] + k * (c[i] - b[i]) for i in range(3))


ground = lambda hue: tint(hue, GROUND)
UNIT = tint(BLUE, 0.70)
LIGHT_B = tint(BLUE, 0.25)
YA = [a["y"] for a in mol["atoms"]]


def molecule(g, cx, y_top, values=None, f=1.0, color=INK, lw=0.6, base="white"):
    x0 = cx - g["width_pt"] * f / 2.0
    if values is not None:
        for a in sorted(g["atoms"], key=lambda a: values[str(a["index"])]):
            v = values[str(a["index"])]
            if v > 0:
                ov.add_patch(Circle((x0 + a["x"] * f, y_top + a["y"] * f), 0.42 * g["bond_length_pt"] * f, facecolor=mix(base, ORANGE, v), edgecolor="none", zorder=23))
    for ln in g["lines"]:
        ov.plot([x0 + q[0] * f for q in ln], [y_top + q[1] * f for q in ln], color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=24)
    for l in g["labels"]:
        ov.text(x0 + l["x"] * f, y_top + l["y"] * f, l["s"], fontsize=l.get("size", 7) * f, color=color, ha="left", va="baseline", zorder=25)


def mol_at(cx, cy, values, f, base):
    """The molecule with its heavy atoms centred on (cx, cy). Returns its half width."""
    molecule(mol, cx, cy - f * (min(YA) + max(YA)) / 2.0, values, f=f, lw=0.6 * max(f, 1.0), base=base)
    return mol["width_pt"] * f / 2.0


ATOM = dict((a["index"], a) for a in mol["atoms"])
TOPC = min(CONCEPT, key=lambda i: ATOM[i]["y"])


def atom_at(cx, cy, f, i):
    """Where mol_at draws atom i, and the radius of its disc."""
    return cx - mol["width_pt"] * f / 2.0 + ATOM[i]["x"] * f, cy - f * (min(YA) + max(YA)) / 2.0 + ATOM[i]["y"] * f, 0.42 * mol["bond_length_pt"] * f


def module(x, y, w, h, label, face, size=6.0, color="white", z=23):
    rbox(x, y, w, h, face, r=2.0, z=z)
    if label:
        txt(x + w / 2.0, y + h / 2.0 + 0.36 * size, label, size, "bold", color=color, ha="center")


def graph(cx, y_top, f, marked):
    """The molecule as its graph: a node per heavy atom, the atoms of the concept filled."""
    x0 = cx - (max(p[0] for p in XY.values()) + min(p[0] for p in XY.values())) * f / 2.0
    P = dict((i, (x0 + p[0] * f, y_top + p[1] * f)) for i, p in XY.items())
    for i, j in BONDS:
        line([P[i][0], P[j][0]], [P[i][1], P[j][1]], lw=0.7, z=23)
    for i, p in P.items():
        if i in marked:
            ov.add_patch(Circle(p, 2.5, facecolor=INK, edgecolor="none", zorder=25))
        else:
            ov.add_patch(Circle(p, 1.9, facecolor="white", edgecolor=INK, linewidth=0.6, zorder=24))
    return max(p[0] for p in P.values()), min(p[1] for p in P.values()), max(p[1] for p in P.values())


def unit(x, y, r=2.9):
    """The studied unit."""
    ov.add_patch(Circle((x, y), r, facecolor=ORANGE, edgecolor="white", linewidth=0.5, zorder=28))


def sq(x, y, s, col, w=None):
    """A key square or a bar of one of the three replacements; the pale one gets a hairline so that it holds on a tinted ground."""
    ov.add_patch(Rectangle((x, y), s if w is None else w, s, facecolor=col, edgecolor=MUTED if col == PALE else "none", linewidth=0.35, zorder=24))


def header(x, y, w, letter, title, hue, size=7.0):
    ov.add_patch(Rectangle((x, y), w, 15.0, facecolor=tint(hue, BAND), edgecolor="none", zorder=3))
    ov.add_patch(Rectangle((x, y), 14.0, 15.0, facecolor=hue, edgecolor="none", zorder=4))
    txt(x + 7.0, y + 11.0, letter.lower() if "6" in OPT else letter, 9, "bold", color="white", ha="center")
    if title:
        txt(x + 19.0, y + 10.5, title, size, "bold")


def result(x, y, w, parts, hue):
    """The closing box of a station: what is measured (and, with --values, what was found)."""
    rbox(x, y, w, 18.0, "white", hue, 1.0)
    objs = [ov.text(0, y + 11.7, s, fontsize=7.5 if len(parts) == 1 else 6.6, fontweight="bold" if b else "normal", color=INK, ha="left", va="baseline", zorder=30) for s, b in parts]
    ws = [cv.width_of(o) for o in objs]
    xx = x + (w - sum(ws) - 2.5 * (len(ws) - 1)) / 2.0
    for o, wd in zip(objs, ws):
        o.set_x(xx)
        xx += wd + 2.5
    cv.texts.append((objs[0], " ".join(s for s, _ in parts)))


BUS, TOP, LB = 23.0, 31.0, 202.0
XS = {"A": (0.0, 176.0), "B": (190.0, 60.0), "C": (254.0, 70.0), "D": (328.0, 78.0), "E": (410.0, 94.0)}
HUE = {"A": ENC, "B": LILAC, "C": SAGE, "D": ROSE, "E": PINK}
YT, YL, MBY = 66.0, 130.0, 176.0
FX0, FX1, BX0, BX1 = 68.0, 166.0, 74.0, 172.0
for k, title in (("B", "Calibration"), ("C", "Transfer"), ("D", "Reconstruction"), ("E", "Output preservation")):
    header(XS[k][0], 0.0, XS[k][1], k, title, HUE[k])
header(0.0, 0.0, 176.0, "A", "Input and frozen encoders", ENC)
for k in "ABCDE":
    rbox(XS[k][0], TOP, XS[k][1], LB - TOP, ground(HUE[k]), r=4.0, z=2)

txt(4.0, 40.5, "Molecule", 6.5, "bold")
molecule(mol, 31.0, 44.0, f=1.25, lw=0.7)
YP = 95.0
txt(31.5, YP - 9.0, mol["smiles"], 5.5, ha="center")
for k in range(N_TOK):
    face = tint(SAGE, 0.95) if k == 0 else (tint(BLUE, 0.80) if k in ANCHORS else GREY)
    ov.add_patch(Rectangle((3.4 + k * 2.55, YP - 4.0), 2.1, 8.0, facecolor=face, edgecolor="none", zorder=23))
line([59.9, 63.0], [YP, YP])
arrow(63.0, YP, 69.4, YP, ms=5.0)
rbox(BX0, 39.0, BX1 - BX0, 58.0, ENC, r=3.0, z=19)
txt(BX0 + 6.0, 50.4, "MoLFormer-XL", 6.8, "bold", color="white")
snow(BX1 - 7.5, 47.2, r=3.0, color="white")
rbox(FX0, 55.0, FX1 - FX0, 58.0, ENC, "white", 0.9, r=3.0, z=21)
txt(113.0, 64.6, "ChemBERTa-77M-MTR", 6.8, "bold", color="white", ha="center")
snow(FX1 - 7.5, 62.2, r=3.0, color="white")
MT, MH = YP - 12.5, 25.0
module(69.75, MT, 5.5, MH, "", LIGHT_B)
rbox(77.75, 70.0, 78.5, 40.0, tint(BLUE, 0.10), "white", 0.6, r=2.5, z=22)
txt(80.5, 78.8, "Block 2 FFN", 5.8, "bold")
module(80.5, MT, 19.5, MH, "Linear", ENC, 5.8)
arrow(100.0, YP, 103.8, YP, lw=0.7, ms=4.0)
module(104.0, MT, 18.0, MH, "GELU", ENC, 5.8)
UX, UY = 128.0, [YP - 9.0 + 3.0 * i for i in range(7)]
for yy in UY:
    ov.plot([122.0, UX], [YP, yy], color=GREY, lw=0.3, zorder=22)
    ov.plot([UX, 134.0], [yy, YP], color=GREY, lw=0.3, zorder=22)
    ov.add_patch(Circle((UX, yy), 1.25, facecolor=UNIT, edgecolor="none", zorder=24))
module(134.0, MT, 19.5, MH, "Linear", ENC, 5.8)
module(158.75, MT, 5.5, MH, "", LIGHT_B)
XA = (FX0 + FX1) / 2.0
arrow(XA, 113.0, XA, 129.6)
txt(XA + 4.5, 123.3, "Activations", 6.5, "bold")
EDGE = tint(BLUE, 0.50)
rbox(BX0, 125.0, BX1 - BX0, 66.0, "white", EDGE, 0.5, r=2.0, z=20)
rbox(FX0, 130.0, FX1 - FX0, 66.0, "white", EDGE, 0.5, r=2.0, z=22)
GX, GY, CW_, CH_, NCOL = 79.65, 143.0, 5.4, 4.3, 15
txt(GX + NCOL * CW_ / 2.0, 139.6, "All units" if "5" in OPT else "464 units", 6.0, ha="center")
for r_ in range(n):
    ov.add_patch(Rectangle((GX - 6.3, GY + r_ * CH_), 3.6, CH_ - 0.6, facecolor=INK if r_ in CONCEPT else "white", edgecolor=INK, linewidth=0.4, zorder=23))
    for c_ in range(NCOL):
        ov.add_patch(Rectangle((GX + c_ * CW_, GY + r_ * CH_), CW_ - 0.5, CH_ - 0.5, facecolor=tint(BLUE, 0.10 + 0.40 * (((r_ * 7 + c_ * 13 + r_ * c_ + (r_ % 3) * 5) % 7) / 6.0)), edgecolor="none", zorder=23))
txt(4.0, 149.0, "Concept", 6.5, "bold")
gx1, gy0, gy1 = graph(32.0, 155.0, 1.3, CONCEPT)
MIDY = GY + n * CH_ / 2.0
line([gx1 + 4.5, GX - 8.8], [MIDY, MIDY], lw=0.6, z=23)
line([GX - 7.2, GX - 8.8, GX - 8.8, GX - 7.2], [GY - 0.3, GY - 0.3, GY + n * CH_ - 0.3, GY + n * CH_ - 0.3], lw=0.6, z=23)
assert gx1 + 4.5 < GX - 12.0 and gy1 < 196.0 and GY + n * CH_ < 193.0

XB, XC, XR = XS["B"][0] + XS["B"][1] / 2.0, XS["C"][0] + XS["C"][1] / 2.0, 183.0
line([FX1, XR, XR, XC, XC], [MIDY, MIDY, BUS, BUS, BUS + 3.0], lw=1.0)
ov.add_patch(Circle((XB, BUS), 1.5, facecolor=INK, edgecolor="none", zorder=27))
arrow(XB, BUS, XB, TOP + 3.8, lw=1.0)
arrow(XC, BUS + 2.0, XC, TOP + 3.8, lw=1.0)

x, w = XS["B"]
cx = x + w / 2.0
hx = cx - 23.0
for j, (name, ref, pin, y0) in enumerate((("Neuron\nreadability", "199 untrained", "Trained", 38.0), ("Axis\npreference", "9 rotated", "Model axes", 102.0))):
    txt(cx, y0 + 7.0, name, 6.5, "bold", ha="center", va="center")
    hy = y0 + 50.0
    if j == 0:
        for i, hh in enumerate((2, 4, 8, 13, 16, 12, 7, 3, 2)):
            ov.add_patch(Rectangle((hx + i * 3.0, hy - hh), 2.6, hh, facecolor=BLUE, edgecolor="none", zorder=23))
    else:
        for i in range(9):
            ov.add_patch(Circle((hx + 1.7 + i * 2.9, hy - 2.4), 1.25, facecolor=ROSE, edgecolor="none", zorder=23))
    line([hx - 1.5, hx + 42.0], [hy, hy], lw=0.6, z=24)
    line([hx + 37.0, hx + 37.0], [hy, hy - 20.0], color=DARK, lw=1.4, z=24)
    ov.plot([hx + 37.0], [hy - 20.0], marker="o", ms=4.8, mfc=DARK, mec="white", mew=0.5, ls="none", zorder=25)
    txt(cx + 26.0, hy - 26.0, pin, 6.0, ha="right")
    txt(hx + 13.3, hy + 7.4, ref, 6.0, ha="center")
arrow(cx, 163.5, cx, MBY - 1.5)
result(x + 6.0, MBY, w - 12.0, ((("%.2f×" % mtr["J1_fold"]), True), ("·", False), (("%.2f×" % mtr["J2_fold"]), True)) if VALUES else (("Fold", True),), LILAC)

x, w = XS["C"]
cx = x + w / 2.0
txt(cx, 46.5, "Development 5,000", 6.0, ha="center")
NU, UP, HOT = 15, 4.2, 10
sx = cx - (NU - 1) * UP / 2.0
for c_ in range(NU):
    if c_ != HOT:
        ov.add_patch(Circle((sx + c_ * UP, 55.5), 1.35, facecolor=UNIT, edgecolor="none", zorder=24))
xw = sx + HOT * UP
unit(xw, 55.5, 2.4)
arrow(xw, 58.6, xw, 100.5, color=ORANGE, lw=1.2, ms=7.0)
for q, word in enumerate(("Chosen", "neuron")):
    txt(xw - 4.5, 77.5 + 7.6 * q, word, 6.5, "bold", ha="right")
txt(cx, 109.5, "Confirmation 10,000", 6.0, ha="center")
hw = mol_at(cx - 3.0, YL, OBS, 1.2, ground(SAGE))
txt(cx, YL + 26.0, "Scaffold-unseen 7,205", 6.0, ha="center")
x_c = cx - 3.0 + hw + 2.5
arrow(cx, YL + 31.0, cx, MBY - 1.5)
result(x + 6.0, MBY, w - 12.0, (("Θ", False), ("%.2f" % theta, True)) if VALUES else (("Transfer Θ", True),), SAGE)

x, w = XS["D"]
cx = x + w / 2.0
txt(cx, 46.5, "Predicted", 6.0, ha="center")
STEM = []
for px_, vals, col_, lab in ((x + 21.0, BIN, PALE, ("Concept", "means")), (x + 57.0, LOC, DARK, ("Bonding", "environment"))):
    hw_p = mol_at(px_, YT, vals, 0.92, ground(ROSE))
    STEM.append(atom_at(px_, YT, 0.92, TOPC))
    sq(px_ - 2.8, YT + 14.5, 5.6, col_)
    for q, word in enumerate(lab):
        txt(px_, YT + 27.6 + 7.9 * q, word, 6.0, ha="center")
YR = min(sy_ - sr_ for _, sy_, sr_ in STEM) - 6.0
assert 46.5 + 2.5 < YR
for sx_, sy_, sr_ in STEM:
    line([sx_, sx_], [YR, sy_ - sr_ - 1.0], color=ORANGE, lw=1.0)
assert x + 21.0 - hw_p > x + 3.0 and x + 57.0 + hw_p < x + w - 3.0
arrow(cx, 105.5, cx, 116.0, lw=0.8, ms=4.5, both=True)
hw = mol_at(cx, YL, OBS, 1.2, ground(ROSE))
txt(cx, YL + 26.0, "Observed", 6.0, ha="center")
arrow(cx, YL + 31.0, cx, MBY - 1.5)
arrow(x_c, YL, cx - hw - 2.5, YL, color=ORANGE, lw=1.2, ms=7.0)
result(x + 6.0, MBY, w - 12.0, (("R²", False), ("%.2f" % r2["M1"], True), ("→", False), ("%.2f" % r2["M2"], True)) if VALUES else (("R²", True),), ROSE)

x, w = XS["E"]
ux = x + 16.0
txt(x + 5.0, 46.5, "Written back", 6.0)
line([STEM[0][0], ux], [YR, YR], color=ORANGE, lw=1.2, cap="projecting")
arrow(ux, YR, ux, YT - 3.6, color=ORANGE, lw=1.2, ms=7.0)
for i in range(5):
    ov.plot([ux, x + 26.0], [YT, YT - 11.0 + i * 5.5], color=ORANGE, lw=0.45, zorder=23)
unit(ux, YT)
if "1" in OPT:
    module(x + 26.0, YT - 14.0, 34.0, 28.0, "", ENC)
    snow(x + 43.0, YT - 7.4, r=3.0, color="white")
    for q, word in enumerate(("Remaining", "layers")):
        txt(x + 43.0, YT + 3.8 + 6.5 * q, word, 5.5, "bold", color="white", ha="center")
    line([x + 60.0, x + 63.0], [YT, YT])
else:
    module(x + 26.0, YT - 14.0, 19.0, 28.0, "", ENC)
    snow(x + 35.5, YT, r=3.4, color="white")
    arrow(x + 45.4, YT, x + 52.6, YT, ms=5.0)
    module(x + 53.0, YT - 14.0, 6.5, 28.0, "", LIGHT_B)
    line([x + 59.5, x + 63.0], [YT, YT])
line([x + 63.0, x + 63.0], [YT - 8.0, YT + 8.0], cap="projecting")
arrow(x + 63.0, YT - 8.0, x + 67.7, YT - 8.0, ms=4.5)
arrow(x + 63.0, YT + 8.0, x + 67.7, YT + 8.0, ms=4.5)
module(x + 68.0, YT - 13.5, 9.0, 11.0, "", PINK)
module(x + 79.0, YT - 13.5, 9.0, 11.0, "", PINK)
module(x + 68.0, YT + 2.5, 20.0, 11.0, "", ENC)
txt(x + 90.0, 46.5, "BACE, ESOL", 5.8, ha="right")
txt(x + 90.0, YT + 22.5, "Pretrained head", 5.8, ha="right")
arrow(x + 78.0, YT + 27.5, x + 78.0, 119.5)
by0 = 125.0
line([x + 26.0, x + 26.0], [by0 - 2.5, by0 + 33.0], lw=0.6, z=25)
for j, (key, col_) in enumerate((("D_mean", REF), ("M1", PALE), ("M2", DARK))):
    sq(x + 26.0, by0 + j * 11.5, 7.5, col_, w=64.0 * mae[key] / mae["D_mean"])
txt(x + 23.0, by0 + 5.8, "Mean", 5.8, ha="right")
assert by0 + 33.0 < MBY - 4.0
arrow(x + w / 2.0, by0 + 36.0, x + w / 2.0, MBY - 1.5)
result(x + 6.0, MBY, w - 12.0, (("MAE", False), ("%.4f" % mae["D_mean"], True), ("→", False), ("%.4f" % mae["M2"], True)) if VALUES else (("Output deviation", True),), PINK)

FY0, FL0, FL1 = 207.0, 226.0, 284.0
FM = 255.0
ov.add_patch(Rectangle((0.0, FY0), W, 15.0, facecolor=tint(PLUM, 0.40), edgecolor="none", zorder=3))
ov.add_patch(Rectangle((0.0, FY0), 14.0, 15.0, facecolor=PLUM, edgecolor="none", zorder=4))
txt(7.0, FY0 + 11.0, "f" if "6" in OPT else "F", 9, "bold", color="white", ha="center")
txt(19.0, FY0 + 10.5, "Proteins", 7.0, "bold")
rbox(0.0, FL0, W, FL1 - FL0, ground(PLUM), r=4.0, z=2)
im = np.asarray(Image.open(PROT).convert("RGBA"))
pw = 25.0
ph = pw * im.shape[0] / im.shape[1]
px0, py0 = 1.5, FM - ph / 2.0
ov.imshow(im, extent=(px0, px0 + pw, py0 + ph, py0), zorder=22, interpolation="lanczos", aspect="auto")
ov.set_xlim(0, W)
ov.set_ylim(H, 0)
assert FL0 + 1.0 < py0 and py0 + ph < FL1 - 1.0
WX, WP, WC, WH = 29.0, 4.2, 3.8, 7.6
for seq, lab, wy, ly in zip(windows, ("Helix", "Strand"), (FM - 8.6, FM + 1.0), (FM - 11.4, FM + 15.6)):
    for k, ch_ in enumerate(seq):
        masked = (k == 3)
        ov.add_patch(Rectangle((WX + k * WP, wy), WC, WH, facecolor=ORANGE if masked else "white", edgecolor=ORANGE if masked else GREY, linewidth=0.45, zorder=23))
        if not masked or "3" in OPT:
            ov.text(WX + k * WP + WC / 2.0, wy + 5.65, ch_, fontsize=5.3, family="DejaVu Sans Mono", color="white" if masked else INK,
                    fontweight="bold" if masked else "normal", ha="center", va="baseline", zorder=30)
    cv.texts.append((ov.texts[-1], seq))
    txt(WX, ly, lab, 6.0)
WR = WX + 6 * WP + WC
assert WR < 58.4
XT = 211.5
line([WR + 0.5, 61.5], [FM, FM])
line([61.5, 61.5, 190.0, 190.0], [FM, FM + 21.0, FM + 21.0, FM + 9.0], color=MUTED, lw=1.0, cap="projecting")
arrow(190.0, FM + 9.0, XT, FM + 9.0, color=MUTED, lw=1.0)
arrow(61.5, FM, 67.6, FM, ms=5.0)
rbox(FX0, FM - 15.0, BX1 - FX0, 30.0, ENCP, r=3.0, z=21)
snow(104.5, FM, r=3.4, color="white")
txt(111.5, FM + 2.7, "ESM-2", 7.5, "bold", color="white")
unit(BX1, FM)
arrow(BX1 + 3.6, FM, XT, FM, color=ORANGE, lw=1.2, ms=7.0)
for q, word in enumerate(("Chosen", "neuron")):
    txt((BX1 + XT) / 2.0 + 1.0, FM - 12.8 + 7.4 * q, word, 6.2, "bold", ha="center")
zy, sc, BW = FM + 13.0, 115.0, 15.0
cv.key(217.0, FL0 + 9.2, ((GREY, "Sequence alone"), (ORANGE, "With neuron")), size=6.0, gap=8.0)
line([217.0, 387.0], [zy, zy], lw=0.6, z=25)
pe = EXPO["buried_exposed_pairs"]
assert EXPO["selection"]["axis_zero_based"] == 814 and pe["T_groups"] == 967
PAIRB = ((brier["S"], brier["Sj"], "Helix or strand"), (float(pe["sequence_brier"]), float(pe["augmented_brier"]), "Buried or exposed"))
for gc, (b0, b1, lab) in zip((258.0, 346.0), PAIRB):
    for q, (val, col_) in enumerate(((b0, GREY), (b1, ORANGE))):
        hgt = val * sc
        assert hgt < zy - FL0 - 12.0
        ov.add_patch(Rectangle((gc - BW - 1.2 + q * (BW + 2.4), zy - hgt), BW, hgt, facecolor=col_, edgecolor="none", zorder=24))
    txt(gc, zy + 8.4, lab, 6.0, ha="center")
arrow(391.0, FM, 411.6, FM)
result(XS["E"][0] + 6.0, FM - 9.0, XS["E"][1] - 12.0, (("Brier", False), ("%.3f" % brier["X"], True), ("→", False), ("%.3f" % brier["Xj"], True)) if VALUES else (("Brier loss", True),), PLUM)

cv.finish(OUT, NAME, "tokens %d, anchors %d | scale %.4f | binary levels %.4f %.4f | fold %.3f %.3f | theta %.4f | R2 %.4f %.4f | MAE %.5f %.5f %.5f | Brier %s | windows %s"
          % (N_TOK, n, SCALE, float(raw["bin"][positive][0]), float(raw["bin"][~positive][0]), mtr["J1_fold"], mtr["J2_fold"], theta, r2["M1"], r2["M2"],
             mae["D_mean"], mae["M1"], mae["M2"], " ".join("%s %.4f" % (k, brier[k]) for k in ("S", "Sj")) + " exposure %.4f %.4f" % (PAIRB[1][0], PAIRB[1][1]), " ".join(windows)))
