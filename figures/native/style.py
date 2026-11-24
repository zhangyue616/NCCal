import io, json, os
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

INK, MUTED, GREY, LIGHT = "#20252A", "#6F7780", "#B8C0C8", "#E3E7EB"
GROUPC = {"Rings": "#4F3A70", "Carboxylic acids": "#4E8A5A", "Carbonyls": "#3A77B5", "Hydroxyls": "#B04466", "Amine": "#C48E1C"}
GROUPN = {"Amine": "Amines"}
PALE, DARK, ERR = "#D6D6DB", "#2B2B30", "#2B2B30"
TAB, BANDC, TITLEC = "#4F3A70", "#ECE8F3", "#3B2C55"
ORANGE = "#EE7733"
ROSE, SAGE, BLUE, LILAC = "#B04466", "#4E8A5A", "#3A77B5", "#7B62A8"
REF = "#939CA6"
OPT = set(os.environ.get("FIGSW", "146"))
if "4" in OPT:
    GROUPC["Carboxylic acids"], GROUPC["Hydroxyls"] = "#589564", "#A4395C"
    ROSE, SAGE = GROUPC["Hydroxyls"], GROUPC["Carboxylic acids"]

J = lambda p: json.load(io.open(p, encoding="utf-8"))


def tint(color, k):
    """White at k = 0, the colour at k = 1, linear in between."""
    return tuple(1 - k * (1 - c) for c in to_rgb(color))


def pale(color):
    """The light shade of a hue: 45% of the way from white."""
    return tint(color, 0.45)


class Canvas(object):
    def __init__(self, width, height):
        plt.rcParams.update({"font.family": "Arial", "font.size": 7.5, "svg.fonttype": "none", "pdf.fonttype": 42,
                             "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
                             "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5})
        self.W, self.H = float(width), float(height)
        self.fig = plt.figure(figsize=(self.W / 72.0, self.H / 72.0), dpi=600)
        self.ov = self.fig.add_axes([0, 0, 1, 1])
        self.ov.set_xlim(0, self.W)
        self.ov.set_ylim(self.H, 0)
        self.ov.axis("off")
        self.ov.set_zorder(20)
        self.ov.patch.set_alpha(0)
        self.ov.set_autoscale_on(False)
        self.bg = self.fig.add_axes([0, 0, 1, 1])
        self.bg.set_xlim(0, self.W)
        self.bg.set_ylim(self.H, 0)
        self.bg.axis("off")
        self.bg.set_zorder(-5)
        self.bg.set_autoscale_on(False)
        self.texts = []

    def txt(self, x, y, s, size=7.5, weight="normal", color=INK, ha="left", va="baseline", rot=0, style="normal", count=True):
        t = self.ov.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, rotation=rot, fontstyle=style, zorder=30, linespacing=1.2)
        if count:
            self.texts.append((t, s))
        return t

    def right_edge(self, t):
        self.fig.canvas.draw()
        return t.get_window_extent(self.fig.canvas.get_renderer()).x1 * 72.0 / self.fig.dpi

    def width_of(self, t):
        self.fig.canvas.draw()
        return t.get_window_extent(self.fig.canvas.get_renderer()).width * 72.0 / self.fig.dpi

    def head(self, x, y, letter, title, w):
        """Panel label: the letter alone, bold, at the upper left corner of the panel. The title is the working name of
        the panel; it is not drawn (it belongs in the legend), and the width w is no longer used."""
        self.txt(x + 1.0, y + 1.0, letter.lower() if "6" in OPT else letter, 11, "bold")

    def axes(self, x, y, w, h, xlim, ylim, xticks=None, yticks=None, left=True, xlog=False, ylog=False):
        a = self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H])
        if xlog:
            a.set_xscale("log")
        if ylog:
            a.set_yscale("log")
        a.set_xlim(*xlim)
        a.set_ylim(*ylim)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.spines["left"].set_visible(left)
        a.patch.set_alpha(0)
        a.tick_params(pad=2)
        a.xaxis.set_minor_locator(NullLocator())
        a.yaxis.set_minor_locator(NullLocator())
        if xticks is not None:
            a.xaxis.set_major_locator(FixedLocator(xticks))
            a.xaxis.set_major_formatter(FixedFormatter(["%g" % v for v in xticks]))
        else:
            a.xaxis.set_major_locator(NullLocator())
        if yticks is not None and left:
            a.yaxis.set_major_locator(FixedLocator(yticks))
            a.yaxis.set_major_formatter(FixedFormatter(["%g" % v for v in yticks]))
        else:
            a.yaxis.set_major_locator(NullLocator())
        return a

    def key(self, x, y, entries, size=8, gap=10.0):
        """A row of key squares with their labels; entries are (colour, label). Returns the right end."""
        for col, lab in entries:
            self.ov.add_patch(Rectangle((x, y - 6.4), 6.4, 6.4, facecolor=col, edgecolor="none", zorder=30))
            x = self.right_edge(self.txt(x + 9.0, y, lab, size)) + gap
        return x - gap

    def finish(self, out, name, extra="", write_labels=False):
        self.ov.set_xlim(0, self.W)
        self.ov.set_ylim(self.H, 0)
        self.fig.canvas.draw()
        rend = self.fig.canvas.get_renderer()
        boxes = [(t.get_window_extent(rend), s) for t, s in self.texts]
        outside = [s for b, s in boxes if b.x0 < 0 or b.y0 < 0 or b.x1 > self.fig.bbox.width or b.y1 > self.fig.bbox.height]
        clash = [(boxes[i][1], boxes[j][1]) for i in range(len(boxes)) for j in range(i + 1, len(boxes))
                 if min(boxes[i][0].x1, boxes[j][0].x1) - max(boxes[i][0].x0, boxes[j][0].x0) > 0.5
                 and min(boxes[i][0].y1, boxes[j][0].y1) - max(boxes[i][0].y0, boxes[j][0].y0) > 0.5]
        words = sum(len(s.split()) for _, s in self.texts)
        print("%s | %g x %g pt | text objects %d, words %d | text outside canvas %s | overlapping label pairs %d %s%s"
              % (name, self.W, self.H, len(self.texts), words, outside, len(clash), clash, (" | " + extra) if extra else ""))
        out = out.replace("\\", "/").rstrip("/") + "/"
        self.fig.savefig(out + name + ".png", dpi=600)
        self.fig.savefig(out + name + ".svg")
        self.fig.savefig(out + name + ".pdf")
        if write_labels:
            with io.open(out + name + "_strings.txt", "w", encoding="utf-8") as labels:
                labels.write("\n".join(text for _, text in self.texts) + "\n")
        
