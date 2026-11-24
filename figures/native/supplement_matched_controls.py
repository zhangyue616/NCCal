from pathlib import Path
import importlib.util
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from matplotlib.colors import to_rgba
from matplotlib.ticker import MaxNLocator
import numpy as np

import sys
from types import SimpleNamespace
from matplotlib.ticker import FuncFormatter
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'results/artwork/supplement'
OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=True)
FROZEN=json.loads((DATA/'matched_control_panels.json').read_text())
PLOTTED={'reconstruction':[],'singles':[],'joint_controls':[]}
def export(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    w, h = fig.canvas.get_width_height()
    outside = []
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            assert text.get_fontsize() >= 9
            box = text.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > w + 1 or box.y1 > h + 1:
                outside.append(text.get_text())
    assert not outside, outside
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(OUT / (name + '.' + ext), dpi=600, facecolor='white')
    fig.savefig(OUT / (name + '_preview.png'), dpi=160, facecolor='white')
    size = list(fig.get_size_inches())
    plt.close(fig)
    return {'name': name, 'inches': size, 'minimum_font_pt': 9, 'clipped_text': outside}

def reconstruction(r):
    groups, _, _, _ = r.load_sources()
    fig = plt.figure(figsize=(7.2, 5.6), dpi=160)
    H = 5.6 * 72
    handles = [Line2D([], [], marker=m, color=c, ls='none', ms=3.7,
               mfc='white' if i == 1 else c, mew=.8, label=label)
               for i, (m, c, label) in enumerate(zip(r.MARKERS, r.COLORS,
                   ('Development mean', 'Paired-concept means', 'Local graph')))]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, .994),
               ncol=3, frameon=False, fontsize=9, handletextpad=.45, columnspacing=1.3)
    fig.text(.5, 1 - 28 / H, 'Bars: descriptive 95% molecule-bootstrap CI (2,000 draws).',
             ha='center', va='center', fontsize=9, color='#46515A')
    for model, top, title, head, letter in zip(r.MODELS, (85, 268), (52, 235), (69, 252), 'ab'):
        records = groups[(model, 'matched_control')]
        ax = fig.add_axes([.36, 1 - (top + 105) / H, .605, 105 / H])
        y = np.arange(9)[::-1].astype(float)
        trans = blended_transform_factory(fig.transFigure, ax.transData)
        for level in y:
            ax.plot([.04, .965], [level - .5] * 2, transform=trans, color='#E6E9EC',
                    lw=.4, zorder=0, clip_on=False)
        ax.axvline(0, color='#AEB7BE', lw=.55, ls=(0, (2, 2)), zorder=1)
        for yy, row in zip(y, records):
            ax.text(.04, yy, row['label'], transform=trans, va='center', fontsize=9)
            ax.text(.335, yy, str(row['coordinate']), transform=trans, va='center',
                    ha='right', fontsize=9, color='#36414A')
            for i, (color, marker, dy) in enumerate(zip(r.COLORS, r.MARKERS, (.24, 0., -.24))):
                point = row['R2_global_binary_local'][i]
                lo, hi = row['intervals'][i]
                ci = ax.plot([lo, hi], [yy + dy] * 2, color=color, lw=.8, zorder=2)[0]
                ax.plot([lo, hi], [yy + dy] * 2, ls='none', marker='|', color=color, ms=2.2, zorder=2)
                pt = ax.plot(point, yy + dy, marker=marker, ms=3.2, ls='none', color=color,
                             mfc='white' if i == 1 else color, mew=.75, zorder=3)[0]
                PLOTTED['reconstruction'].append({'model': model, 'role': 'matched_control',
                    'coordinate': row['coordinate'], 'representative_concept_id': row['representative_concept_id'],
                    'predictor': r.METHODS[i], 'point': float(pt.get_xdata()[0]),
                    'CI95': [float(x) for x in ci.get_xdata()]})
        ax.set(xlim=(-.1, 1.02), ylim=(-.65, 8.65))
        ax.set_xticks([-.1, 0, .25, .5, .75, 1], ['−0.10', '0', '0.25', '0.50', '0.75', '1.00'])
        ax.set_yticks([])
        ax.tick_params(axis='x', labelsize=9, length=3, width=.6)
        ax.spines[['left', 'right', 'top']].set_visible(False)
        ax.spines['bottom'].set_color('#8B9298')
        ax.spines['bottom'].set_linewidth(.6)
        fig.text(.015, 1 - title / H, letter, fontsize=11.5, fontweight='bold')
        fig.text(.05, 1 - title / H, r.MODEL_NAMES[model] + ' · matched controls', fontsize=10.5, fontweight='bold')
        fig.text(.04, 1 - head / H, 'Paired reference\nconcept', fontsize=9, fontweight='bold', va='center')
        fig.text(.335, 1 - head / H, 'Control\ncoordinate', fontsize=9, fontweight='bold', va='center', ha='right')
        if model == 'molformer':
            ax.set_xlabel(r'Confirmation-set $\mathit{R}^{2}$', fontsize=9.5, labelpad=4)
    assert len(PLOTTED['reconstruction']) == 54
    return export(fig, 'Figure_S_M1')

def fidelity(f):
    summary, scopes, _ = f.complete_sources(None)
    paired, _ = f.normalize(summary, scopes)
    fig = plt.figure(figsize=(7.2, 7.2), dpi=160)
    H = 7.2 * 72
    fig.text(.39, 1 - 16 / H, 'Single-coordinate contrasts', ha='center', fontsize=10, fontweight='bold')
    fig.text(.795, 1 - 16 / H, 'Joint matched-control contrasts', ha='center', fontsize=9.5, fontweight='bold')
    fig.text(.39, 1 - 31 / H, '9 selected + 9 controls per reference', ha='center', fontsize=9)
    fig.text(.795, 1 - 31 / H, 'Mean difference; unadjusted 95% CI', ha='center', fontsize=9)
    handles = [Line2D([], [], marker=m, ls='none', mfc=to_rgba(c, .6), mec=c, mew=.45,
               ms=3, label=label) for c, m, label in [('#0072B2', 'o', 'Selected singles'),
                                                    ('#6F7478', 's', 'Control singles')]]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(.385, 1 - 49 / H),
               ncol=2, frameon=False, fontsize=9, handletextpad=.4, columnspacing=1.)
    fig.legend(handles=[Line2D([], [], marker='D', color='#30363B', lw=1, ms=3.1, label='Joint controls')],
               loc='center', bbox_to_anchor=(.795, 1 - 49 / H), frameon=False, fontsize=9)
    fig.text(.5, 1 - 68 / H, 'Reference deviation − local deviation; >0 favors local output fidelity.',
             ha='center', fontsize=9, color='#46515A')
    order = (('chemberta_mtr', 'bace'), ('molformer', 'bace'),
             ('chemberta_mtr', 'esol'), ('molformer', 'esol'))
    fig.text(.04, 1 - 87 / H, 'BACE · predicted class-1 probability · n = 1,496', fontsize=10, fontweight='bold')
    fig.text(.04, 1 - 282 / H, 'ESOL · predicted log-solubility units · n = 939', fontsize=10, fontweight='bold')
    axis_ranges = []
    for i, ((model, task), top, title) in enumerate(zip(order, (115, 203, 310, 398), (102, 190, 297, 385))):
        fig.text(.04, 1 - title / H, 'abcd'[i] + '  ' + f.MODEL_LABEL[model], fontsize=10, fontweight='bold')
        singles = [r for k, r in paired.items() if k[:2] == (model, task) and r['kind'] == 'single']
        controls = [paired[(model, task, 'matched_control/joint', ref)] for ref in f.REFERENCES]
        left = fig.add_axes([.24, 1 - (top + 46) / H, .30, 46 / H])
        right = fig.add_axes([.64, 1 - (top + 46) / H, .31, 46 / H])
        left_values = [r['effect_reference_error_minus_local_error'] for r in singles]
        right_values = [v for r in controls for v in [r['effect_reference_error_minus_local_error'], *r['CI95_percentile']]]
        f.style_axis(left, f.limits(left_values), [f.LABEL[r] for r in reversed(f.REFERENCES)] + [''])
        f.style_axis(right, f.limits(right_values))
        for ax, values, bins, steps, padding in ((left, left_values, 3 if (model, task) == ('molformer', 'bace') else 5, [1, 2, 2.5, 5, 10], .03),
                                               (right, right_values, 4, [1, 2.5, 5, 10], 0.)):
            values = [float(v) for v in values if v is not None and np.isfinite(v)]
            lo, hi = min(0., min(values)), max(0., max(values))
            pad = padding * (hi - lo)
            ticks = MaxNLocator(nbins=bins, steps=steps, min_n_ticks=2).tick_values(lo - pad, hi + pad)
            ax.set_xlim(ticks[0], ticks[-1])
            ax.set_xticks(ticks)
            ax.set_ylim(-.45, 2.45)
            ax.set_yticks((0, 1, 2), [f.LABEL[r] for r in reversed(f.REFERENCES)] if ax is left else [''] * 3)
            ax.patch.set_alpha(0)
        guide = blended_transform_factory(fig.transFigure, left.transData)
        for y in (0, 1, 2):
            left.plot([.04, .95], [y - .42] * 2, transform=guide, color='#E7EAED', lw=.4, clip_on=False, zorder=0)
        for ref, y in zip(f.REFERENCES, (2, 1, 0)):
            for role, offset, color, marker in (('selected', .20, '#0072B2', 'o'), ('matched_control', -.20, '#6F7478', 's')):
                rows = sorted((r for r in singles if r['role'] == role and r['reference'] == ref), key=lambda r: r['coordinates'][0])
                assert len(rows) == 9
                for row, dy in zip(rows, np.linspace(-.10, .10, 9)):
                    point = row['effect_reference_error_minus_local_error']
                    assert point is not None
                    artist = left.plot(point, y + offset + dy, marker=marker, mfc=to_rgba(color, .6),
                                       mec=color, mew=.45, ms=2.6, ls='none')[0]
                    PLOTTED['singles'].append({'model': model, 'task': task, 'scope': row['scope'],
                        'reference': ref, 'role': role, 'point': float(artist.get_xdata()[0])})
            row = paired[(model, task, 'matched_control/joint', ref)]
            lo, hi = row['CI95_percentile']
            assert lo is not None and hi is not None
            ci = right.plot([lo, hi], [y] * 2, color='#30363B', lw=1., zorder=2)[0]
            right.plot([lo, hi], [y] * 2, ls='none', marker='|', color='#30363B', ms=4, zorder=2)
            artist = right.plot(row['effect_reference_error_minus_local_error'], y, marker='D',
                                color='#30363B', ms=3.1, ls='none', zorder=3)[0]
            PLOTTED['joint_controls'].append({'model': model, 'task': task, 'scope': row['scope'],
                'reference': ref, 'point': float(artist.get_xdata()[0]), 'CI95': [float(v) for v in ci.get_xdata()]})
        axis_ranges.append({'model': model, 'task': task, 'single': list(left.get_xlim()), 'joint_controls': list(right.get_xlim())})
    for x in (.39, .795):
        fig.text(x, 1 - 486 / H, 'Reference deviation − local deviation', ha='center', fontsize=9)
    fig.text(.5, 1 - 505 / H, 'Axes use independent ranges; vertical offsets identify points in fixed coordinate order.', ha='center', fontsize=9, color='#46515A')
    assert len(PLOTTED['singles']) == 216 and len(PLOTTED['joint_controls']) == 12
    PLOTTED['fidelity_panel_order'] = [list(x) for x in order]
    PLOTTED['fidelity_axis_ranges'] = axis_ranges
    return export(fig, 'Figure_S_N1')

LABELS={0:'3-membered ring',2:'5-membered ring',3:'6-membered ring',15:'Aromatic carboxyl',16:'Aliphatic carboxyl',18:'Primary aliphatic amine',24:'Carbonyl',25:'Carbonyl excluding COOH',26:'Aliphatic hydroxyl',27:'Aromatic hydroxyl'}
MODELS=('chemberta_mtr','molformer');MODEL_NAMES={'chemberta_mtr':'ChemBERTa-77M-MTR','molformer':'MoLFormer-XL'}
METHODS=('Development mean','Binary concept means','Local graph')
def load_sources():
    groups={}
    for model in MODELS:
        raw=[r for r in FROZEN['reconstruction']if r['model']==model];records=[]
        for start in range(0,len(raw),3):
            cells=raw[start:start+3];r=cells[0]
            assert tuple(x['predictor']for x in cells)==METHODS
            records.append(dict(label=LABELS[r['representative_concept_id']],coordinate=r['coordinate'],representative_concept_id=r['representative_concept_id'],R2_global_binary_local=[x['point']for x in cells],intervals=[x['CI95']for x in cells]))
        assert len(records)==9
        groups[model,'matched_control']=records
    return groups,None,None,None
def paired_records():
    records={}
    for row in FROZEN['singles']+FROZEN['joint_controls']:
        r=dict(row);single='single'in r['scope']
        r.update(kind='single'if single else'joint',coordinates=[int(r['scope'].split('/')[-1])]if single else[],effect_reference_error_minus_local_error=r['point'],CI95_percentile=r.get('CI95'))
        records[r['model'],r['task'],r['scope'],r['reference']]=r
    return records
def require(ok,message):assert ok,message
def limits(values, *, nonnegative=False):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    require(bool(values), "No defined values available for the predefined axis")
    low,high = min(0.,min(values)),max(0.,max(values))
    span = high-low
    pad = .12*(span if span else max(abs(high),1e-3))
    return (0. if nonnegative else low-pad,high+pad)

def style_axis(ax, xlim, ylabels=None):
    ax.set(xlim=xlim,ylim=(-.45,3.45))
    ax.spines[["top","right","left"]].set_visible(False)
    ax.spines["bottom"].set_color("#8C9399")
    ax.set_yticks(range(4), ylabels if ylabels is not None else [""]*4)
    ax.tick_params(axis="both",labelsize=9,length=2,pad=3)
    ax.tick_params(axis="y",length=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f"{x:.3g}"))
    ax.axvline(0,color="#B4BABF",linewidth=.7,linestyle=(0,(3,2)),zorder=0)

r=SimpleNamespace(MODELS=MODELS,MODEL_NAMES=MODEL_NAMES,COLORS=('#6F7478','#C87900','#0072B2'),MARKERS=('^','o','s'),METHODS=METHODS,load_sources=load_sources)
f=SimpleNamespace(MODEL_LABEL=MODEL_NAMES,REFERENCES=('development_mean','binary_mean','residual_permutation'),LABEL={'development_mean':'Development mean','binary_mean':'Binary-concept means','residual_permutation':'Shuffled displacement'},complete_sources=lambda x:(None,None,None),normalize=lambda x,y:(paired_records(),None),style_axis=style_axis,limits=limits)
with plt.rc_context({'font.family':'Arial','font.size':9,'svg.fonttype':'none','pdf.fonttype':42,'axes.labelsize':9,'xtick.labelsize':9,'ytick.labelsize':9,'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic'}):
    reconstruction(r);fidelity(f)
assert PLOTTED['reconstruction']==FROZEN['reconstruction']
assert PLOTTED['singles']==FROZEN['singles']
assert PLOTTED['joint_controls']==FROZEN['joint_controls']
assert PLOTTED['fidelity_axis_ranges']==FROZEN['fidelity_axis_ranges']
print('54 reconstruction points, 216 single contrasts, 12 joint-control intervals and all axis ranges match frozen inputs.')
