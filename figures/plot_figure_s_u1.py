"""Render Supplementary Figure S-U1 from its archived plotting values."""
import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

INK = "#20252A"
BLUE = "#3A77B5"
ROSE = "#A4395C"
SAGE = "#589564"
LILAC = "#7B62A8"
GREY = "#939CA6"
GRID = "#E3E7EB"


def read_data(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source, delimiter="\t"))
    if Counter(row["panel"] for row in rows) != {"A": 16, "B": 32, "C": 63}:
        raise ValueError("Expected values for 16 bases, two validation cohorts and 63 donors.")
    return rows


def basis_label(basis):
    if basis == "release":
        return "Released"
    if basis == "PCA128":
        return "RNA principal axes"
    group, seed = basis.split("_")
    if group == "rotation":
        return f"Rotation {int(seed) - 14000}"
    return f"{group.capitalize()} {int(seed) - 1400}"


def basis_colour(basis):
    if basis == "release":
        return INK
    if basis == "PCA128":
        return LILAC
    return {"rotation": ROSE, "untrained": BLUE, "retrained": SAGE}[basis.split("_")[0]]


def render(data, output):
    rows = read_data(data)
    same_name = {r["basis"]: float(r["value"]) for r in rows if r["panel"] == "A"}
    rates = {(r["cohort"], r["basis"]): float(r["value"]) for r in rows if r["panel"] == "B"}
    bases = list(same_name)
    if len(bases) != 16 or len(rates) != 32:
        raise ValueError("Repeated basis identifier.")
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 8.6, "axes.labelsize": 8.6,
        "xtick.labelsize": 8.1, "ytick.labelsize": 8.1, "axes.linewidth": .65,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.size": 2.5, "ytick.major.size": 0,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "svg.fonttype": "none", "svg.hashsalt": "NCCal-SU1", "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.8, 4.55))
    a = fig.add_axes([.148, .25, .224, .576])
    b = fig.add_axes([.437, .25, .205, .576])
    c = fig.add_axes([.763, .25, .215, .576])
    for ax, letter, title in [(a, "a", "Same-name pairs"), (b, "b", "Pair recovery"),
                              (c, "c", "Measured-protein scores")]:
        x = ax.get_position().x0
        fig.text(x - .029, .937, letter, weight="bold", fontsize=12, va="top")
        fig.text(x, .934, title, weight="bold", fontsize=9.3, va="top")
    y = list(range(len(bases)))
    a.barh(y, [same_name[v] for v in bases], color=[basis_colour(v) for v in bases], height=.65)
    a.set_yticks(y, [basis_label(v) for v in bases])
    a.set(xlim=(0, 545), xticks=[0, 250, 500], xlabel="Pairs with the same name")
    for group, marker, offset, colour in [("JDM", "o", -.15, BLUE), ("Wells", "s", .15, ROSE)]:
        b.scatter([rates[(group, v)] for v in bases], [i + offset for i in y],
                  marker=marker, color=colour, s=13, label=group, linewidths=0, zorder=3)
    b.set(xlim=(-.8, 17.3), xticks=[0, 5, 10, 15], yticks=[], xlabel="Pairs recovered (%)")
    b.legend(loc="lower center", bbox_to_anchor=(.48, 1.01), ncol=2, frameon=False,
             columnspacing=.75, handlelength=.8, handletextpad=.35, fontsize=7.8, borderpad=0)
    for ax in [a, b]:
        ax.set_ylim(len(bases) - .37, -.63)
        for line in (.5, 8.5, 11.5, 12.5):
            ax.axhline(line, color=GRID, linewidth=.65, zorder=0)
    cohorts = [("UCSF", 20, BLUE), ("JDM", 20, SAGE), ("GSE313642", 23, ROSE)]
    c.axhline(0, color=GREY, linestyle=(0, (3, 2)), linewidth=.7, zorder=0)
    for x, (cohort, count, colour) in enumerate(cohorts):
        cr = [r for r in rows if r["panel"] == "C" and r["cohort"] == cohort]
        if len(cr) != count or len({r["donor_order"] for r in cr}) != count:
            raise ValueError("Unexpected donor count in " + cohort)
        values = sorted(float(r["value"]) for r in cr)
        xpos = [x + (i % 5 - 2) * .065 for i in range(count)]
        c.scatter(xpos, values, color=colour, s=13, linewidths=0, alpha=.8, zorder=2)
        c.hlines(mean(values), x - .24, x + .24, color=INK, linewidth=1.55, zorder=3)
    c.set(xlim=(-.5, 2.5), xticks=range(3), xticklabels=["UCSF", "JDM", "GSE313642"],
          ylabel="Native − rotated score", xlabel="External cohort")
    c.tick_params(axis="x", labelsize=7.3)
    c.yaxis.set_major_locator(MaxNLocator(nbins=5))
    fig.text(.035, .108, "a, b  Rows correspond. Recovery is the median within-donor rate. Each unit must beat its partner and rank in the top five.", fontsize=8)
    fig.text(.035, .062, "c  Each point is one donor. Differences use the median over eight rotations. Bars mark cohort means. GSE313642 was previously unused.", fontsize=8)
    output.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        meta = {"CreationDate": None, "ModDate": None} if ext == "pdf" else {"Date": None} if ext == "svg" else {}
        fig.savefig(output / ("Figure_S-U1." + ext), dpi=400, metadata=meta)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render(args.data, args.output)
