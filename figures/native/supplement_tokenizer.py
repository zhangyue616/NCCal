import sys,csv,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import supplement_style as common
ROOT=Path(__file__).resolve().parents[2]
with (ROOT/'results/artwork/supplement/token_embedding_rows.csv').open(encoding='utf-8')as f:rows=list(csv.DictReader(f))
for r in rows:
    r['corpus_count']=int(r['corpus_count']);r['norm_mlm']=float(r['norm_mlm']);r['norm_mtr']=float(r['norm_mtr'])
summary=json.loads((ROOT/'results/artwork/supplement/figure_s_c1.json').read_text())
CKPT=['chemberta_mlm','chemberta_mtr'];per_model={m:{'init_expected_norm':summary['init_expected_norm'][m]}for m in CKPT}
emitted=lambda r:r['corpus_count']>0
common.setup_style()
fig = plt.figure(figsize=(7.0, 2.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.35)
cats = [("emitted", lambda r: r["kind"] == "single-character" and emitted(r)),
        ("unused\n1-char.", lambda r: r["kind"] == "single-character" and not emitted(r)),
        ("unused\nmulti-char.", lambda r: r["kind"] == "multi-character" and not emitted(r)),
        ("special", lambda r: r["kind"] == "special")]
cols = [common.OI["blue"], common.OI["grey"], common.OI["vermillion"], common.OI["grey"]]
rng = np.random.default_rng(0)
stereo = {"/", "\\"}
for j, (m, letter) in enumerate(zip(CKPT, "ab")):
    ax = fig.add_subplot(gs[0, j])
    key = f"norm_{m.split('_')[1]}"
    for i, ((lab, sel), col) in enumerate(zip(cats, cols)):
        v = [(r[key], r["token"]) for r in rows if sel(r) and r[key] > 0]
        x = i + rng.uniform(-0.25, 0.25, len(v))
        ax.scatter(x, [a for a, _ in v], s=5, c=col, alpha=0.6, linewidths=0)
        for xi, (a, tkn) in zip(x, v):
            if tkn in stereo:
                ax.scatter([xi], [a], s=14, facecolors="none", edgecolors="k", linewidths=0.6)
    ax.axhline(per_model[m]["init_expected_norm"], color="k", ls="--", lw=0.6)
    ax.set_yscale("log")
    ax.set_ylim(0.2, 20)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([c[0] for c in cats], fontsize=6)
    ax.set_title(common.NAMES[m])
    common.panel(ax, letter, x=-0.2)
    if j == 0:
        ax.set_ylabel("input-embedding L2 norm")
        ax.text(3.45, per_model[m]["init_expected_norm"] * 0.72, "initialization\nscale", fontsize=5.5,
                ha="right", va="top")
ax = fig.add_subplot(gs[0, 2])
ax.axis("off")
common.panel(ax, "c", x=-0.02)
lines = ["Released tokenizer (0 BPE merges):", ""]
for s in ("Clc1ccc(Br)cc1", "C[C@@H](O)c1ccccc1", "c1cc[nH]c1", "C[N+](=O)[O-]"):
    toks = [x for x in summary["examples"][s] if x not in ("[CLS]", "[SEP]")]
    lines += [s, "  → " + " ".join(toks), ""]
n_multi = sum(1 for r in rows if r["kind"] == "multi-character")
lines += ["Cl, Br, [nH], [C@@H], [N+], [O-] are among", f"the {n_multi} multi-character tokens; none",
          "is ever emitted (unused).", "",
          "Circled: / and \\ (bond stereo marks),", "emitted in this corpus but at",
          "initialization scale in MTR."]
ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=5.8, family="DejaVu Sans Mono",
        transform=ax.transAxes)
common.save(fig, "figS1_tokenizer")
