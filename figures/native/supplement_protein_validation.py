
from pathlib import Path
import sys
import csv
import json

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np

HERE=Path(sys.argv[1]);HERE.mkdir(parents=True,exist_ok=True)
ROOT=Path(__file__).resolve().parents[2]/'results/artwork/supplement'
INK, MUTED, GRID = '#222731', '#5e6872', '#dce0e4'
BLUE, ORANGE = '#386995', '#ac501b'
mpl.rcParams.update({
    'font.family': 'Arial', 'font.size': 8.5, 'axes.labelsize': 8.5,
    'axes.labelcolor': INK, 'axes.edgecolor': INK, 'text.color': INK,
    'xtick.color': INK, 'ytick.color': INK, 'axes.linewidth': .75,
    'xtick.major.width': .7, 'ytick.major.width': .7, 'xtick.major.size': 2.8,
    'ytick.major.size': 2.8, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'svg.hashsalt': 'NCCal-S-G4', 'axes.spines.top': False, 'axes.spines.right': False,
})

def rows(name):
    with (ROOT / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))

cycle_rows = rows('protein_null_selection.tsv')
null_results = {r['method']: r for r in cycle_rows
                if r['condition'] == 'null_pair_flips'}
null_report=json.loads((ROOT/'protein_null_summary.json').read_text(encoding='utf-8-sig'))
basis = {r['arm']: r for r in rows('protein_basis_diagnostics.tsv')}
structures = [r for r in rows('protein_repeated_structures.tsv')
              if r['quality_atom_match'] == 'True']
decisions = next(r for r in rows('protein_decision_policies.tsv') if r['frame'] == '814 original')

fig = plt.figure(figsize=(7.04, 6.43), facecolor='white')
layout = fig.add_gridspec(3, 2, left=.13, right=.97, top=.91, bottom=.06,
                          height_ratios=[1,1,.32], hspace=.91, wspace=.62)
axes = [fig.add_subplot(layout[i, j]) for i in range(2) for j in range(2)]
for ax, letter, heading in zip(axes, 'abcd',
        ['Screening random labels', 'Finding a planted signal',
         'Recovering an activation', 'Repeating an exposure label']):
    ax.text(-.21, 1.20, letter, weight='bold', fontsize=12, transform=ax.transAxes,
            ha='left', va='bottom')
    ax.text(0, 1.21, heading, weight='bold', fontsize=9.2, transform=ax.transAxes,
            ha='left', va='bottom')

ax = axes[0]
null_methods = ['D_unadjusted', 'D_max_calibrated', 'D_then_E_locked']
labels = ['Unadjusted\nsearch', 'Search-max\ncalibration', 'Locked axis,\nnew groups']
chart = {'a': [], 'b': [], 'c': [], 'd': [], 'e': []}
for i, (method, label) in enumerate(zip(null_methods, labels)):
    row = null_results[method]
    percent = 100 * float(row['retention_rate']); n = int(row['retained'])
    low = 100 * float(row['retention_mc95_low']); high = 100 * float(row['retention_mc95_high'])
    color = MUTED if i == 0 else BLUE
    ax.errorbar(percent, 2-i, xerr=[[percent-low], [high-percent]], fmt='o', color=color,
                ms=4.8, capsize=2.4, lw=1.25, zorder=3)
    ax.annotate(f'{n:,}/1,000', (percent, 2-i), xytext=(-4 if i==0 else 6, 8),
                textcoords='offset points', ha='right' if i==0 else 'left', color=color, fontsize=8)
    chart['a'].append({'method':method,'retained':n,'trials':int(row['rounds']),
                       'percent':percent,'mc95_percent':[low,high]})
ax.set_xscale('log'); ax.set_xlim(.1,155); ax.set_ylim(-.62,2.55)
ax.xaxis.set_major_locator(FixedLocator([.1,1,10,100])); ax.xaxis.set_major_formatter(FixedFormatter(['0.1','1','10','100']))
ax.minorticks_off(); ax.set_yticks([2,1,0], labels); ax.tick_params(axis='y',length=0)
ax.set_xlabel('Random-label trials retained (%)')

ax = axes[1]
variants=[('single_4','One signal axis',BLUE,'o'),('distributed_4','Eight signal axes',ORANGE,'D')]
for i,(condition,lab,col,marker) in enumerate(variants):
    planted=next(r for r in null_report['conditions'] if r['condition']==condition)
    ret=next(r for r in cycle_rows if r['condition']==condition and r['method']=='E_locked')
    nn=int(ret['rounds']); counts=[planted['oracle_confirmation_retained'],int(ret['retained']),int(ret['recovered_generator_member_and_sign'])]
    offset=.13 if i==0 else -.13
    for pos,v in zip([2,1,0],counts):
        ax.scatter(v/nn*100,pos+offset,s=21,marker=marker,edgecolor=col,facecolor=col if i==0 else 'white',lw=1.05,zorder=2)
        side=-6 if v>nn*.78 else 6
        ax.annotate(str(v),(v/nn*100,pos+offset),xytext=(side,0),textcoords='offset points',
                    ha='right' if side<0 else 'left',va='center',fontsize=8,color=col)
    chart['b'].append(dict(condition=condition,rounds=nn,known_signal_confirmed=counts[0],selected_axis_confirmed=counts[1],generator_member_and_sign_selected=counts[2]))
ax.set_ylim(-.36,2.53);ax.set_xlim(-2,104);ax.set_xticks([0,25,50,75,100]);ax.set_yticks([2,1,0],['Known signal\nconfirmed','Selected axis\nconfirmed','Contributing\naxis identified']);ax.tick_params(axis='y',length=0)
ax.set_xlabel('Trials meeting criterion (%)'); ax.legend(handles=[Line2D([],[],ls='',marker=m,ms=4.4,color=c,markerfacecolor=c if j==0 else 'white',label=l) for j,(_,l,c,m) in enumerate(variants)],loc='upper left',bbox_to_anchor=(.005,1.04),frameon=False,fontsize=8,handlelength=.7,handletextpad=.32,labelspacing=.25)

ax = axes[2]
arms = ['signed_swap','rotation45','hadamard4','hadamard16']; x=np.arange(4)
single = [float(basis[a]['single_R2']) for a in arms]; fit = [float(basis[a]['OMP_R2']) for a in arms]
ax.plot(x, single, color=BLUE, lw=1.25, marker='o', ms=5, label='Selected single axis')
ax.plot(x, fit, color=ORANGE, lw=1.0, marker='s', markerfacecolor='white', ms=5,
        label='Calibrated combination')
ax.set_xlim(-.35,3.36); ax.set_ylim(-.03,1.12); ax.set_yticks([0,.25,.5,.75,1],['0','.25','.50','.75','1'])
ax.set_xticks(x,['1','2','4','16']); ax.set_xlabel('Axes carrying the original readout'); ax.set_ylabel('Held-out $R^2$')
ax.legend(loc='center left', bbox_to_anchor=(0,.39), frameon=False, handlelength=1.15,
          handletextpad=.45, fontsize=8, labelspacing=.29, borderpad=.1)
ax.annotate(f'{single[-1]:.3f}',(3,single[-1]),xytext=(-3,-13),textcoords='offset points', ha='right',color=BLUE)
ax.annotate(f'{fit[-1]:.3f}',(3,fit[-1]),xytext=(-4,-12),textcoords='offset points',ha='right',color=ORANGE)
chart['c']=[{'arm':arm,'single_R2':s,'calibrated_R2':f,'support_retained':float(basis[arm]['OMP_support_recall'])} for arm,s,f in zip(arms,single,fit)]

ax=axes[3]
for r in structures:
    sign = 2*int(r['original_target'])-1
    before = sign * (float(r['late_rsa_original']) - float(r['early_rsa_original']))
    after = sign * (float(r['late_rsa_second']) - float(r['early_rsa_second']))
    kept = r['pair_state'] == 'retained_extremes'
    ax.scatter(before,after,marker='o' if kept else 'D',s=25 if kept else 27,
               facecolor=BLUE if kept else 'white', edgecolor='white' if kept else ORANGE,
               linewidth=.6 if kept else 1.1,zorder=3)
    chart['d'].append({'group':r['group'],'PDB1':r['original_pdb'],'PDB2':r['second_pdb'],
                       'RSA_gap_original':before,'RSA_gap_second':after,'label':r['pair_state']})
limit=.25*np.ceil(4*max(max(v['RSA_gap_original'],v['RSA_gap_second']) for v in chart['d']))
ax.plot([0,limit],[0,limit],color='#8d9398',ls='--',lw=.8,zorder=1)
ticks=np.arange(0,limit+.01,.25); mark=['0' if t==0 else f'{t:.2f}'.lstrip('0') for t in ticks]
ax.set_xticks(ticks,mark);ax.set_yticks(ticks,mark);ax.set_xlim(0,limit*1.03);ax.set_ylim(0,limit*1.03)
ax.set_xlabel('Exposure gap, first PDB entry');ax.set_ylabel('Exposure gap, second PDB entry')
kept_count=sum(r['label']=='retained_extremes' for r in chart['d'])
handles=[Line2D([],[],ls='',marker='o',ms=5,color=BLUE,label=f'Both cutoffs retained ({kept_count})'),
         Line2D([],[],ls='',marker='D',ms=5,markerfacecolor='white',color=ORANGE,label=f'Intermediate exposure ({len(structures)-kept_count})')]
ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(.0,1.02),fontsize=8,frameon=False,handlelength=.9,handletextpad=.35,labelspacing=.25)

ax=fig.add_subplot(layout[2,:]);ax.set_axis_off()
ax.text(-.078,1.12,'e',transform=ax.transAxes,va='bottom',weight='bold',fontsize=12)
ax.text(0,1.16,'Adding neuron 814 to local sequence',transform=ax.transAxes,va='bottom',weight='bold',fontsize=9.2)
colpos=[0,.45,.62,.78,.96]
for j,lab in enumerate(['Use of neuron 814','Pairs scored','Corrected','New errors','Net gain']):
    ax.text(colpos[j],.87,lab,transform=ax.transAxes,ha='left' if j==0 else 'right',va='center',color=MUTED,fontsize=8.2)
ax.plot([0,.97],[.70,.70], transform=ax.transAxes,color=GRID,lw=.8,clip_on=False)
groups=[('All pairs','fixed_incorrect_all','spoiled_correct_all',int(decisions['n'])),
        ('Uncertain third','routed_fixed_incorrect','routed_spoiled_correct', int(decisions['selected']))]
for i,(group,corrected,spoiled,n) in enumerate(groups):
    correct=int(decisions[corrected]); wrong=int(decisions[spoiled]);
    for j,value in enumerate([group,n,correct,wrong,f'+{correct-wrong}']):
        ax.text(colpos[j],.39-i*.44,str(value),transform=ax.transAxes,ha='left' if j==0 else 'right',va='center',weight='bold' if j==4 else 'normal',color=BLUE if j==4 else INK)
    chart['e'].append({'policy':group,'pairs_receiving_readout':n,'corrected':correct,'new_errors':wrong})

for suffix in ['png','svg','pdf']:
    meta={'Creator':'NCCal','Date':None} if suffix=='svg' else {'Creator':'NCCal','CreationDate':None,'ModDate':None} if suffix=='pdf' else {'Software':'NCCal'}
    fig.savefig(HERE/f'Figure_S_G4.{suffix}',dpi=400,metadata=meta)
fig.savefig(HERE/'preview.png',dpi=145)
(HERE/'figure_data.json').write_text(json.dumps(chart,indent=2)+'\n',encoding='utf-8')

