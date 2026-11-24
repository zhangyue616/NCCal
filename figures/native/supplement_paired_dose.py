from pathlib import Path
import sys,json,csv,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle,Circle,FancyArrowPatch,Arc
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'results/artwork/supplement'
OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=True)
NAMES=['Carbonyl','Carbonyl (noCOO)','Aliphatic OH']
PANEL_BOUNDS={'A':None,'B':None,'C':None,'D':None}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(fig,name,*args,**kwargs):
    name={'figure_S_B1':'Figure_S_B1','figure_S_I1':'Figure_S_I1','figure_6':'Figure_S_E1','figure_2':'Figure_S_C6'}.get(name,name)
    for ext in ('png','svg','pdf'):fig.savefig(OUT/(name+'.'+ext),dpi=400,facecolor='white')
    plt.close(fig)
INK = '#000000'
ORANGE = '#D55E00'
BLUE = '#0072B2'
GRAY = '#7F7F7F'
WHITE = '#FFFFFF'
COLORS = {'selected': ORANGE, 'baseline': BLUE, 'control': GRAY, 'ink': INK}
FONT_LARGE, FONT_NORMAL, FONT_SMALL = 9, 8, 7
STYLE = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
         'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
         'xtick.labelsize': 7, 'ytick.labelsize': 8, 'legend.fontsize': 7,
         'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK,
         'xtick.color': INK, 'ytick.color': INK, 'axes.linewidth': .6,
         'xtick.major.width': .6, 'ytick.major.width': .6,
         'xtick.direction': 'out', 'ytick.direction': 'out',
         'axes.grid': False, 'axes.unicode_minus': True,
         'figure.facecolor': WHITE, 'axes.facecolor': WHITE, 'savefig.facecolor': WHITE,
         'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
         'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
         'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold'}

def new_figure(width,height):
    plt.rcParams.update(STYLE);return plt.figure(figsize=(width,height))
def clean_axes(ax):
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out',width=.6,length=2.5,colors=INK)
def title(fig, panel, text, x, y):
    fig.text(x, y, panel.lower(), fontsize=9, weight='bold', va='top', color=INK)
    if text:
        fig.text(x + .034, y, text, fontsize=9, weight='bold', va='top', color=INK)

def formal_text(fig, x, y, pieces, fontsize=8, rotation=0):
    renderer = fig.canvas.get_renderer()
    widths = [renderer.get_text_width_height_descent(
        text, FontProperties(family='Arial', size=size), False)[0]/fig.dpi
        for text, size, offset in pieces]
    cursor = -sum(widths)/2
    fw, fh = fig.get_size_inches()
    for (text, size, offset), width in zip(pieces, widths):
        ox, oy = cursor, offset/72
        if rotation == 90:
            ox, oy = -oy, ox
        fig.text(x+ox/fw, y+oy/fh, text, fontsize=size, family='Arial',
                 color=INK, va='baseline', ha='left', rotation=rotation,
                 rotation_mode='anchor')
        cursor += width

def power_unit(fig, x, y, fontsize=8, rotation=0):
    formal_text(fig, x, y, [('×10', fontsize, 0), ('−3', 7, 3.2)],
                fontsize=fontsize, rotation=rotation)

def interval(ax, point, bounds, y):
    ax.plot([float(bounds[0]) * 1000, float(bounds[1]) * 1000],
            [y, y], color=ORANGE, lw=1.5, solid_capstyle='butt', zorder=2)
    ax.plot(float(point) * 1000, y, 'o', color=ORANGE, ms=4.5,
            markeredgewidth=.5, zorder=3)

def cut_marks(ax, side):
    x = 0 if side == 'left' else 1
    dx, dy = .045, .014
    for y in [0, 1]:
        ax.plot([x-dx, x+dx], [y-dy, y+dy], transform=ax.transAxes,
                color=INK, lw=1, clip_on=False)

def packed_control_offsets(ax, controls, selected_x, y):
    px_to_pt = 72/ax.figure.dpi
    sx, sy = ax.transData.transform((selected_x, y))
    placed = [(sx*px_to_pt, sy*px_to_pt, 2.25)]
    result = []
    for k, row in controls:
        choices = []
        for off in [0, .28, -.28, .14, -.14, .32, -.32]:
            px, py = ax.transData.transform((float(row['mean']), y+off))
            px, py = px*px_to_pt, py*px_to_pt
            slack = min(math.hypot(px-qx, py-qy)-(2+radius)
                        for qx, qy, radius in placed)
            choices.append((slack, off, px, py))
        chosen = next((v for v in choices if v[0] >= 0), max(choices, key=lambda v:v[0]))
        slack, off, px, py = chosen
        placed.append((px, py, 2))
        result.append((off, k, row, slack))
    return result

def build(version):
    data=read(DATA/'figure_s_j1.json')
    assert len(data['historical_objects']) == len(data['paired_objects']) == 3
    assert len(data['historical_full_dose_rows']) == 36

    fig = new_figure(width=7, height=5.8)
    axes_by_panel = {p: [] for p in PANEL_BOUNDS}
    display = []
    orange = []
    control_layout = []

    def record(panel, name, source_field, raw, display_value=None, units='original units'):
        display.append({'panel': panel, 'name': name, 'source_file': 'results/artwork/supplement/figure_s_j1.json',
                        'source_field': source_field, 'source_value_original_precision': raw,
                        'display_position': raw if display_value is None else display_value,
                        'display_units': units,
                        'display_label': format(float(raw), '.3g')})

    title(fig, 'A', 'Responses at matched scalar dose', .025, .98)
    handles = [
        Line2D([], [], color=ORANGE, marker='o', mfc=ORANGE, mec=ORANGE,
               lw=1.5, ms=4, label='Selected unit'),
        Line2D([], [], color=GRAY, lw=1, ls='--', label='Control median'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .948),
               frameon=False, ncol=2, fontsize=7, handlelength=1.8,
               handletextpad=.5, columnspacing=1.5)
    formal_text(fig, .025, .755,
                [('Response ', 9, 0), ('D', 9, 0), ('c', 7, -2), ('(s)', 9, 0)],
                fontsize=9, rotation=90)
    power_unit(fig, .048, .755, fontsize=8, rotation=90)
    for i, (obj, name) in enumerate(zip(data['paired_objects'], NAMES)):
        ax = fig.add_axes([.100+i*.307, .635, .258, .230])
        axes_by_panel['A'].append(ax)
        clean_axes(ax)
        ax.axhline(0, color=GRAY, lw=.5)
        selected = [v*1000 for v in obj['selected_contrast_by_scale']]
        median = [v*1000 for v in obj['median_five_control_contrasts_by_scale']]
        ax.plot(obj['scales'], selected, 'o-', color=ORANGE, lw=1.5, ms=4,
                markeredgewidth=.5, zorder=3)
        ax.plot(obj['scales'], median, color=GRAY, lw=1, ls='--', zorder=2)
        ax.set(xlim=(.20, 1.05), ylim=(-.15, 2.2), xticks=[.25, .5, 1],
               xticklabels=['0.25', '0.5', '1'], yticks=[0, 1, 2])
        ax.set_title(name, fontsize=8, pad=6, color=INK)
        if i:
            ax.set_yticklabels([])
        for j, s in enumerate(obj['scales']):
            record('A', name+' dose fraction', 'paired_objects[%d].scales[%d]' % (i,j), s)
            for field in ['selected_contrast_by_scale', 'median_five_control_contrasts_by_scale']:
                raw = obj[field][j]
                record('A', name+' '+field, 'paired_objects[%d].%s[%d]' % (i,field,j),
                       raw, raw*1000, 'original response × 1000; axis unit 10^-3')
        orange.append({'where': 'A / '+name, 'visible_objects': 'one three-point curve',
                       'meaning': 'selected coordinate response contrast at the three frozen dose fractions'})
    fig.text(.535, .573, 'Dose fraction s', ha='center', va='center', fontsize=8, color=INK)

    title(fig, 'B', 'Intervention contrasts', .025, .505)
    fig.text(.285, .457, 'Full clamp H', ha='center', va='center', fontsize=8)
    formal_text(fig, .485, .452,
                [('Paired dose H', 8, 0), ('pair', 7, -2), ('(1)', 8, 0)])
    low = fig.add_axes([.205, .135, .070, .275])
    high = fig.add_axes([.292, .135, .085, .275], sharey=low)
    pair = fig.add_axes([.428, .135, .135, .275], sharey=low)
    axes_by_panel['B'].extend([low, high, pair])
    for ax in [low, high, pair]:
        clean_axes(ax)
        ax.spines['left'].set_visible(False)
        ax.set_ylim(-.55, 2.55)
        ax.set_yticks([])
    low.set(xlim=(-.2, 5), xticks=[0, 2, 4])
    high.set(xlim=(120, 170), xticks=[130, 160])
    pair.set(xlim=(-1.4, 1.05), xticks=[-1, 0, 1])
    pair.axvline(0, color=GRAY, lw=.5, ls='--', zorder=0)
    low.axvline(0, color=GRAY, lw=.5, ls='--', zorder=0)
    cut_marks(low, 'right')
    cut_marks(high, 'left')
    break_y = [.135+i*.0275 for i in range(11)]
    fig.lines.append(Line2D([.281 if i%2==0 else .286 for i in range(11)],
                            break_y, transform=fig.transFigure, color=GRAY, lw=.5))
    fig.lines.append(Line2D([.402,.402], [.130,.437], transform=fig.transFigure,
                            color=INK, lw=.5))
    for i, (old, obj, name) in enumerate(zip(data['historical_objects'], data['paired_objects'], NAMES)):
        y = 2-i
        ci = [old['H_CI_lower'], old['H_CI_upper']]
        assert not (5 < float(old['H'])*1000 < 120)
        assert float(ci[1])*1000 <= 5 or float(ci[0])*1000 >= 120
        interval(low, old['H'], ci, y)
        interval(high, old['H'], ci, y)
        interval(pair, obj['H_pair1_existing'], obj['H_pair1_CI98_3333333333_existing'], y)
        label_y = fig.transFigure.inverted().transform(low.transData.transform((0, y)))[1]
        fig.text(.191, label_y, name, fontsize=8, va='center', ha='right', color=INK)
        for field in ['H', 'H_CI_lower', 'H_CI_upper']:
            raw = old[field]
            record('B', name+' '+field, 'historical_objects[%d].%s' % (i,field),
                   raw, float(raw)*1000, 'original H × 1000; full-clamp axis unit 10^-3')
        record('B', name+' H_pair', 'paired_objects[%d].H_pair1_existing' % i,
               obj['H_pair1_existing'], obj['H_pair1_existing']*1000,
               'original H_pair × 1000; paired-dose axis unit 10^-3')
        for j, raw in enumerate(obj['H_pair1_CI98_3333333333_existing']):
            record('B', name+' H_pair interval',
                   'paired_objects[%d].H_pair1_CI98_3333333333_existing[%d]' % (i,j),
                   raw, raw*1000, 'original H_pair × 1000; paired-dose axis unit 10^-3')
    for center in [.286, .495]:
        formal_text(fig, center, .073,
                    [('Contrast (×10',8,0), ('−3',7,3.2), (')',8,0)])
    orange.append({'where': 'B / full clamp', 'visible_objects': 'three points and three CI segments',
                   'meaning': 'selected-minus-control-median contrast in the full-clamp design'})
    orange.append({'where': 'B / paired dose', 'visible_objects': 'three points and three CI segments',
                   'meaning': 'selected-minus-control-median contrast in the paired scalar-dose design'})

    title(fig, 'C', 'Full-clamp doses', .605, .505)
    selected_handle = Line2D([], [], color=ORANGE, marker='o', mfc=ORANGE,
                             mec=ORANGE, lw=0, ms=4, label='Selected unit')
    controls_handle = Line2D([], [], color=GRAY, marker='o', mfc=WHITE,
                             mec=GRAY, lw=0, ms=4, label='Five controls')
    fig.legend(handles=[selected_handle,controls_handle], loc='center',
               bbox_to_anchor=(.805, .470), frameon=False, ncol=2,
               fontsize=7, handlelength=.8, handletextpad=.35, columnspacing=.8)
    dose = fig.add_axes([.800, .265, .167, .182])
    axes_by_panel['C'].append(dose)
    clean_axes(dose)
    dose.spines['left'].set_visible(True)
    dose.set_xscale('log')
    dose.set(xlim=(.5, 20), ylim=(-.55, 5.55), xticks=[.5, 2, 10, 20],
             xticklabels=['0.5', '2', '10', '20'], yticks=[5,4,3,2,1,0],
             yticklabels=['']*6)
    dose.minorticks_off()
    for i, (cid, name) in enumerate(zip(['24','25','26'], NAMES)):
        for j, (stratum, sign) in enumerate([('positive','+'),('negative','−')]):
            y = 5-(2*i+j)
            rows = [(k,r) for k,r in enumerate(data['historical_full_dose_rows'])
                    if str(r['exact_id']) == cid and r['stratum'] == stratum]
            controls = sorted([(k,r) for k,r in rows if r['condition'] != 'selected_atom_full'],
                              key=lambda kr:kr[1]['condition'])
            k_selected, selected = next((k,r) for k,r in rows if r['condition'] == 'selected_atom_full')
            packed = packed_control_offsets(dose, controls, float(selected['mean']), y)
            for off, k, row, slack in packed:
                dose.plot(float(row['mean']), y+off, 'o', mfc=WHITE, mec=GRAY,
                          mew=.5, ms=3.5, zorder=2)
                control_layout.append({'concept':name,'stratum':stratum,
                                       'condition':row['condition'],
                                       'fixed_dose_x':row['mean'],
                                       'categorical_y_offset':off,
                                       'clearance_from_earlier_marks_pt':slack})
                record('C', name+' '+sign+' '+row['condition'],
                       'historical_full_dose_rows[%d].mean' % k, row['mean'])
            dose.plot(float(selected['mean']), y, 'o', mfc=ORANGE, mec=ORANGE,
                      mew=.5, ms=4, zorder=3)
            record('C', name+' '+sign+' selected',
                   'historical_full_dose_rows[%d].mean' % k_selected, selected['mean'])
            label_y = fig.transFigure.inverted().transform(dose.transData.transform((1,y)))[1]
            fig.text(.790, label_y, name+' '+sign, fontsize=7, va='center', ha='right', color=INK)
            orange.append({'where': 'C / '+name+' '+sign, 'visible_objects': 1,
                           'meaning': 'selected coordinate mean full-clamp dose in this stratum'})
    dose.set_xlabel('Mean dose (log)', fontsize=7, labelpad=2)

    title(fig, 'D', 'Paired-dose rule', .605, .155)
    for sign, row_y in [('+',.1225),('−',.1025)]:
        fig.text(.639, row_y, sign+' molecule', fontsize=7, va='center', color=INK)
        for i in range(6):
            fig.patches.append(Rectangle((.762+i*.032, row_y-.0075), .026, .015,
                                         transform=fig.transFigure,
                                         facecolor=ORANGE if i==0 else GRAY,
                                         edgecolor=INK, lw=.5))
    fig.text(.639, .076, '2 molecules × 6 coordinates', fontsize=7, va='center', color=INK)
    fig.text(.639, .049, 'dose = min of 12 full doses', fontsize=7, va='center', color=INK)
    orange.append({'where': 'D / first column of both molecule rows', 'visible_objects': 2,
                   'meaning': 'the same selected coordinate applied to each of the paired positive/negative molecules; six coordinate columns total'})

    save(fig,'Figure_S_J1')

build('portable')
