"""Draw supplementary Figures S-Q1, S-R1 and S-T1 from the packaged result tables."""
import sys, csv
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0,str(Path(__file__).resolve().parent/'native'))
from style import INK,MUTED,GREY,LIGHT,ORANGE,BLUE,ROSE,SAGE,LILAC
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/artwork/use'
OUT=Path(sys.argv[2])if len(sys.argv)>2 else ROOT/'figures/rendered'
OUT.mkdir(parents=True,exist_ok=True)
TARGET=sys.argv[1]if len(sys.argv)>1 else 'all'
if TARGET not in ('all', 'Figure_S_Q1', 'Figure_S_R1', 'Figure_S_T1'):
    raise SystemExit('Choose Figure_S_Q1, Figure_S_R1 or Figure_S_T1. Use figures/render.py for the main figures.')
plt.rcParams.update({'font.family':'Arial','font.size':8.8,'axes.labelsize':8.8,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42,'xtick.major.size':2.5,'ytick.major.size':2.5,'xtick.major.width':.7,'ytick.major.width':.7})
def finish(f,name,caption):
    minimum=8.7 if name.startswith('Figure_S_')else 8.0
    for t in f.findobj(matplotlib.text.Text):
        if t.get_fontsize()<minimum:t.set_fontsize(minimum)
    for ext in ['png','svg','pdf']:f.savefig(str(OUT/(name+'.'+ext)),dpi=600)
    plt.close(f)
def read(path, delim='\t'):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f,delimiter=delim))

def fl(r,k): return float(r[k])

def yes(r,k): return str(r[k]).lower()=='true'

def fig(w,h): return plt.figure(figsize=(w/72.,h/72.))

def ax(f, x,y,w,h):
    W,H=f.get_size_inches()*72
    a=f.add_axes([x/W,1-(y+h)/H,w/W,h/H])
    a.spines['top'].set_visible(False);a.spines['right'].set_visible(False)
    a.tick_params(pad=2)
    return a

def letter(f,x,y,s):
    W,H=f.get_size_inches()*72
    return f.text(x/W,1-y/H,s,fontsize=11,fontweight='bold',va='top',color=INK)

def text(f,x,y,s,size=8.8,**kw):
    W,H=f.get_size_inches()*72
    return f.text(x/W,1-y/H,s,fontsize=size,va='top',color=kw.pop('color',INK),**kw)


def edit_points(a,u,k,compact=False):
    rr=[r for r in EDIT if r['unit']==u and r['edit_class']==k]
    met=next(r for r in PRIMARY if r['representation']=='controlled' and int(r['unit'])==u and int(r['edit_class'])==k)
    x=np.array([r['predicted_delta'] for r in rr]); y=np.array([r['measured_delta'] for r in rr])
    lo=min(x.min(),y.min(),0.); hi=max(x.max(),y.max(),0.); pad=(hi-lo)*.065
    lim=(lo-pad,hi+pad)
    a.plot(lim,lim,color=MUTED,ls='--',lw=.7,zorder=1)
    a.scatter(x,y,s=9 if compact else 15,c=ORANGE if u==293 else BLUE,alpha=.52,edgecolors='none',zorder=2)
    a.set_xlim(*lim);a.set_ylim(*lim)
    a.tick_params(labelsize=8.6 if compact else 8.5)
    a.locator_params(axis='both',nbins=3 if compact else 4)
    a.text(.04,.96,'R² = %.2f\nn = %s pairs'%(fl(met,'R2'),met['molecules']),
        transform=a.transAxes,va='top',fontsize=8.6 if compact else 8.8)
    return met

def census_scatter(a,rows,xkey='Pearson',all_units=False):
    if all_units:
        other=[r for r in rows if not yes(r,'canonical_usable')]
        a.scatter([fl(r,xkey)for r in other],[fl(r,'rewritten_mean_gain')*1000 for r in other],
            s=10,c=LIGHT,edgecolors=MUTED,linewidths=.15,alpha=.8,label='Not canonically replaceable',zorder=1)
    for status,(col,mk,lab) in STATUS.items():
        rr=[r for r in rows if yes(r,'canonical_usable') and r['rewritten_status']==status]
        a.scatter([fl(r,xkey)for r in rr],[fl(r,'rewritten_mean_gain')*1000 for r in rr],
            s=25,marker=mk,c=col,edgecolors='white',linewidths=.35,label=lab,zorder=3)
    a.axhline(0,color=MUTED,lw=.7,ls='--',zorder=0)
    if xkey=='Pearson': a.axvline(.8,color=MUTED,lw=.7,ls=':',zorder=0)
    for u in [293,454]:
        r=next(r for r in rows if int(r['unit'])==u)
        a.annotate(str(u),(fl(r,xkey),fl(r,'rewritten_mean_gain')*1000),xytext=(5,8 if u==293 else -13),
            textcoords='offset points',fontsize=8.8,fontweight='bold',arrowprops={'arrowstyle':'-','lw':.55,'color':MUTED})
    a.set_ylabel('Advantage over mean replacement\n(output RMS deviation, in 0.001 units)')

def public_scatter(a,frame):
    rr=[r for r in public if r['frame']==frame]
    for key,(color,marker,label) in CATEGORIES.items():
        for filled in [False,True]:
            qs=[r for r in rr if r['annotation_category']==key and yes(r,'strong_increment_flag')==filled]
            a.scatter([fl(r,'auc_mean')for r in qs],[fl(r,'pair_accuracy')for r in qs],
                s=22,marker=marker,facecolors=color if filled else 'none',edgecolors=color,linewidths=.8,alpha=.8)
    a.axhline(.5,color=MUTED,lw=.7,ls='--');a.axvline(.7,color=MUTED,lw=.7,ls=':')
    a.set_xlim(.36,1.02);a.set_ylim(.35,1.01)
    a.set_xticks([.4,.6,.8,1.]);a.set_yticks([.4,.6,.8,1.])
    a.set_xlabel('Annotation-vs-unannotated AUROC')
    a.set_ylabel('Same-amino-acid pair accuracy')
    return rr

CENSUS=read(DATA/'5b_and_s_q1_census.tsv')
PRIMARY=read(DATA/'5a_and_s_r1_primary_metrics.tsv')
EDIT=read(DATA/'5a_and_s_r1_changed_atom_deltas.tsv')
for r in EDIT:
    for k in ['unit','edit_class']:r[k]=int(r[k])
    for k in ['predicted_delta','measured_delta']:r[k]=float(r[k])
PROTEIN=read(DATA/'6b_and_s_t1_public_labels.tsv')
public=[r for r in PROTEIN if r['status']=='EVALUABLE']
STATUS={'retained':(SAGE,'o','Retained'),'failed':(ROSE,'X','Failed'),'undetermined':(GREY,'s','Unresolved')}
CATEGORIES={'alpha_helix':(ROSE,'o','Helix'),'beta_strand':(BLUE,'s','Strand'),'disorder':(SAGE,'^','Disorder'),'turn':(LILAC,'D','Turn')}


if TARGET in ["all","Figure_S_Q1"]:
    f=fig(504,250)
    for n,key,x in [('a','Pearson',49),('b','canonical_string_increment_T',307)]:
        letter(f,x-47,3,n);a=ax(f,x,30,180,157);census_scatter(a,CENSUS,key,True)
        a.set_xlabel('Same-atom Pearson correlation' if key=='Pearson' else 'Canonical string-dependence score')
        if key=='Pearson':a.set_xlim(-.04,1.04)
    handles=[Line2D([0],[0],marker='o',color=LIGHT,mec=MUTED,ls='None',label='Not canonically replaceable')]
    handles += [Line2D([0],[0],marker=m,color=c,ls='None',label=l)for c,m,l in STATUS.values()]
    f.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.012),ncol=2,frameon=False,fontsize=8.1)
    finish(f,'Figure_S_Q1','Figure S-Q1. Rewriting census of all 464 ChemBERTa-77M-MTR neurons. The reduction in output RMS deviation from rule replacement relative to mean replacement is plotted against (a) same-atom Pearson correlation across canonical and five rewritten strings and (b) the prespecified canonical-string dependence score (canonical_string_increment_T). RMS deviation is computed across the 199 normalized descriptor outputs before averaging. Nonreplaceable neurons in canonical notation are pale grey; the 54 canonically replaceable neurons retain their outcome colours and shapes. Neurons 293 and 454 are labelled. The horizontal dashed line is zero advantage and the vertical dotted line in a is correlation 0.8. Each output advantage averages the five rewrites within each of 2,000 molecules. The canonical-only score predicted failure with AUROC 0.315, versus 0.831 using negative same-atom correlation, among the 31 neurons with resolved outcomes. These AUROCs refer to the resolved subset, whereas both panels display all 464 neurons.')

if TARGET in ["all","Figure_S_R1"]:
    f=fig(504,314)
    names=['OH → H','OH → OMe','Ketone → CH2','Acid → ester','Remote aryl F']
    for row,u in enumerate([293,454]):
        for col,k in enumerate(range(1,6)):
            x=52+col*89;y=30+row*139
            a=ax(f,x,y,66,92);edit_points(a,u,k,True)
            letter(f,x-21,y-20,chr(97+row*5+col))
            if row==0:a.set_title(names[col],fontsize=8.2,pad=5)
            if col==0:a.set_ylabel('Measured Δactivation\nNeuron %d'%u,fontsize=8.4)
    text(f,252,298,'Rule-predicted activation change (native post-GELU scale)',8.8,ha='center')
    finish(f,'Figure_S_R1','Figure S-R1. Frozen-rule predictions for five molecular edits. Rows show neurons 293 (hydroxyl; a–e) and 454 (carbonyl; f–j). Columns show aliphatic hydroxyl to hydrogen, hydroxyl to methoxy, ketone to methylene, carboxylic acid to methyl ester, and remote aromatic fluorination. The corresponding main-analysis pair counts are 60, 61, 62, 91 and 193. Points are changed-environment shared atoms; added and deleted atoms have no paired activation. Each dashed line is identity. Activation differences use the native post-GELU scale; each panel shows its own full data range, since ranges differ strongly across edits and neurons. Displayed R² values use the frozen molecule-equal weighting from primary_results.csv, so molecules with more changed atoms receive no extra weight. No line was fitted to the points. The full metrics and both frozen reference comparisons are in Worksheets S-R1 and S-R2.')

if TARGET in ["all","Figure_S_T1"]:
    f=fig(504,268)
    for x,fr,lab,n in [(49,'T','Confirmation frame','a'),(307,'NEW','Reserved + newly collected','b')]:
        letter(f,x-47,4,n);a=ax(f,x,37,180,155);rr=public_scatter(a,fr)
        a.set_title(lab,fontsize=9,pad=8)
        text(f,x,230,'%d / 98 same-AA; %d / 98 sequence increment'%(sum(yes(r,'same_AA_flag')for r in rr),sum(yes(r,'strong_increment_flag')for r in rr)),8.2)
    f.legend(handles=[Line2D([0],[0],marker=m,color=c,ls='None',label=l)for c,m,l in CATEGORIES.values()],
        ncol=4,loc='lower center',frameon=False,fontsize=8.2,bbox_to_anchor=(.5,.005))
    finish(f,'Figure_S_T1','Figure S-T1. Published protein labels in two protein frames. Each point is one of 98 evaluable public labels, coloured and shaped by annotation class. The x axis is group-mean annotation-vs-unannotated AUROC and the y axis is accuracy on same-amino-acid residue pairs. Filled symbols have a positive Brier increment over the frozen gradient-boosted sequence predictor at BH q < 0.05; hollow symbols do not establish that increment. (a) The 1,099-group confirmation mother frame: 88 labels pass the same-amino-acid test and 61 pass the sequence-increment test. (b) Reserved plus newly collected structure groups: corresponding counts are 77 and 52. Actual usable group counts vary by annotation and model and are retained in Worksheet S-T1. The dashed horizontal line is accuracy 0.5 and the dotted vertical line is AUROC 0.70. These labels retain the authors’ unit identities and signs; unannotated residues are not experimentally verified negatives.')
