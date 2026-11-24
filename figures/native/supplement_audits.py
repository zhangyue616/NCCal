from pathlib import Path
import sys,json,csv,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Rectangle,Circle,FancyArrowPatch,Arc
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'results/artwork/supplement'
OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=True)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(fig,name,*args,**kwargs):
    name={'figure_S_B1':'Figure_S_B1','figure_S_I1':'Figure_S_I1','figure_6':'Figure_S_E1','figure_2':'Figure_S_C6'}.get(name,name)
    for ext in ('png','svg','pdf'):fig.savefig(OUT/(name+'.'+ext),dpi=400,facecolor='white')
    plt.close(fig)
INK, BLUE, ORANGE, GREEN, MUTED, LINE = '#243541', '#286B95', '#B76A2A', '#237D6B', '#62727B', '#D7E0E5'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial'],
    'font.size':8.3,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,
    'axes.titlesize':9,'text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':MUTED,
    'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,
    'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white',
    'axes.facecolor':'white','savefig.facecolor':'white'})
def panel(fig, letter, title, x, y):
    fig.text(x,y,letter.lower(),fontsize=11.2,weight='bold',va='top')
    fig.text(x+.040,y-.003,title,fontsize=9,weight='bold',va='top')

def fold_axis(ax,ticks,limits):
    ax.set_xscale('log');ax.set_xlim(*limits);ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter());ax.minorticks_off()
    ax.axvline(1,color=MUTED,lw=.7,ls='--')
    for side in ['top','right','left']:ax.spines[side].set_visible(False)
    ax.tick_params(axis='y',length=0)

def scope_checks():
    d=read(DATA/'figure_s_b1.json')
    fig=plt.figure(figsize=(180/25.4,101/25.4))
    panel(fig,'A','H3 readout checks',.025,.965)
    panel(fig,'B','CheMeleon scope',.650,.965)
    fig.text(.068,.900,'39 initialization twins; all 173 concepts',fontsize=8.1,color=MUTED)
    fig.text(.690,.900,'39 twins; main atom output',fontsize=8.1,color=MUTED)
    ys=[9,8,7,5.5,4.5,3.5,1.5,.5]
    for i,(metric,color,limits,ticks) in enumerate([('J1',BLUE,(.65,14),[1,2,5,10]),('J2',ORANGE,(.95,2.2),[1,1.5,2])]):
        ax=fig.add_axes([.245+i*.185,.225,.155,.570]);fold_axis(ax,ticks,limits)
        vals=[r['saved_folds'][metric+'_fold'] for r in d['H3_readouts_all_concepts']]
        ax.scatter(vals,ys,s=25,color=color,marker='o' if i==0 else 's')
        for line in [6.25,2.5]:ax.axhline(line,color=LINE,lw=.6)
        ax.set_ylim(-.2,9.7);ax.set_yticks(ys)
        ax.set_yticklabels([r['display_label'] for r in d['H3_readouts_all_concepts']] if i==0 else ['']*8)
        ax.set_title(metric,color=color,pad=6);ax.set_xlabel('Fold (log)')
    ax=fig.add_axes([.757,.225,.145,.570]);fold_axis(ax,[1,1.5,2],(.93,2.5))
    for i,row in enumerate(d['CheMeleon_main_subsets']):
        v=row['display'];y=3-i
        ax.plot(v['J1_fold'],y+.10,'o',ms=4,color=BLUE)
        ax.plot(v['J2_fold'],y-.10,'s',ms=4,color=ORANGE)
        ax.text(1.12,y,'pass' if v['joint_pass'] else 'fail',transform=ax.get_yaxis_transform(),
            fontsize=8,va='center',color=GREEN if v['joint_pass'] else ORANGE)
    ax.set_ylim(-.6,3.6);ax.set_yticks([3,2,1,0]);ax.set_yticklabels(['All','Common','Curated','Rare'])
    ax.set_xlabel('Fold (log)')
    fig.text(.715,.835,'● J1   ■ J2',fontsize=8)
    fig.text(.945,.835,'Nominal\njoint',fontsize=8,ha='center',color=MUTED)
    fig.text(.068,.090,'Distinct MLM/MTR blocks and CheMeleon main/mp3 readouts; no shared depth trajectory.',fontsize=8)
    data={'source':'results/artwork/supplement/figure_s_b1.json','source_panels':['B','C'],
        'H3_readouts_all_concepts':d['H3_readouts_all_concepts'],'CheMeleon_main_subsets':d['CheMeleon_main_subsets'],
        'new_statistics':False}
    caption='**Figure S-B1.** Descriptive H3 readout and concept-scope checks. (A) J1 (blue circles) and J2 (orange squares) folds for three ChemBERTa-2 MLM blocks, three MTR blocks, and distinct CheMeleon main and mp3 readouts, using 39 initialization twins and all 173 concepts. These readouts do not define a common depth trajectory. (B) CheMeleon main-readout folds for all, common, curated, and rare concept subsets, with the original nominal joint decisions. The rare subset fails the joint rule. Dashed lines mark fold = 1; all horizontal axes are logarithmic. These historical checks are descriptive and separate from the primary H4 gates (Tables S-B1–S-B3).'
    save(fig,'figure_S_B1',180,101,data,caption)

def lineage_audit():
    d=read(DATA/'figure_s_i1.json')
    fig=plt.figure(figsize=(180/25.4,105/25.4))
    panel(fig,'A','Sample lineage',.025,.965)
    panel(fig,'B','H4 element-anchor audit',.585,.965)
    fig.text(.068,.883,'H4: 10,000; D: 5,000; T: 10,000 molecules',fontsize=8.1)
    ax=fig.add_axes([.035,.455,.505,.385]);ax.axis('off')
    definitions=[('CID','cid_identity'),('Isomeric identity','canonical_isomeric_smiles'),
        ('No stereochemistry','canonical_nonisomeric_smiles'),('Nonempty scaffold','murcko_scaffold_smiles')]
    xs=[.01,.61,.77,.93]
    for x,label in zip(xs,['Unique shared identities','H4–D','H4–T','D–T']):
        ax.text(x,.96,label,ha='left' if x==xs[0] else 'center',va='top',weight='bold',fontsize=8)
    ax.axhline(.82,color=LINE,lw=.7)
    for y,(label,key) in zip([.69,.49,.29,.09],definitions):
        ax.text(xs[0],y,label,va='center')
        for x,pair in zip(xs[1:],['H4-D','H4-T','D-T']):
            ax.text(x,y,str(d['lineage']['intersections'][key][pair]['unique_identities']),ha='center',va='center')
    audit=fig.add_axes([.668,.665,.292,.158])
    for y,key in zip([1,0],['chemberta_mlm','chemberta_mtr']):
        counts=d['mapping']['models'][key]['category_counts'];v=counts['verified_unique_anchor_match'];u=counts['ambiguous/unverifiable']
        audit.barh(y,v,color=BLUE,height=.46);audit.barh(y,u,left=v,color=ORANGE,height=.46,hatch='///',edgecolor=ORANGE)
        audit.text(v*.51,y,'{:,} / 10,000'.format(v),ha='center',va='center',color='white',fontsize=8)
    audit.set(xlim=(0,10000),ylim=(-.6,1.6),yticks=[1,0],yticklabels=['MLM','MTR'],xticks=[])
    for spine in audit.spines.values():spine.set_visible(False)
    audit.tick_params(axis='y',length=0)
    fig.text(.605,.590,'165 unverifiable per tokenizer:',color=ORANGE)
    fig.text(.605,.535,'114 graph/stereo; 51 element-anchor',color=ORANGE,fontsize=8)
    fig.text(.605,.445,'0 disagreements among 241,267',fontsize=8.2)
    fig.text(.605,.395,'verified atoms per tokenizer.',fontsize=8.2)
    fig.text(.060,.340,'T slice: 7,205 nonempty scaffolds absent from D',fontsize=8.2)
    ax=fig.add_axes([.060,.265,.48,.038]);left=0
    for count,color in [(7205,GREEN),(2399,MUTED),(396,LINE)]:
        ax.barh(0,count,left=left,height=.6,color=color,edgecolor='white',linewidth=.6);left+=count
    ax.set(xlim=(0,10000),ylim=(-.5,.5));ax.axis('off')
    fig.text(.060,.215,'Excluded: 2,399 D-shared; 396 empty-scaffold rows',fontsize=8)
    fig.text(.060,.155,'622 retained rows share a scaffold with H4.',fontsize=8)
    fig.text(.605,.275,'Unverified cases were retained;',fontsize=8)
    fig.text(.605,.215,'no historical H4 p-values recalculated.',fontsize=8)
    fig.text(.060,.065,'Post hoc audit. The slice is nested within T and is not an independent scaffold split.',fontsize=8)
    data={'source':'results/artwork/supplement/figure_s_i1.json','source_panels':['a','b'],
        'lineage':d['lineage'],'mapping':d['mapping'],'new_statistics':False}
    caption='**Figure S-I1.** Post hoc molecular lineage and H4 anchor verification. (A) Intersections count distinct shared identities, rather than molecule rows. Scaffolds are nonempty, achiral Bemis–Murcko scaffolds. The fixed sensitivity slice retains 7,205 T molecules whose nonempty scaffold is absent from D, excluding 2,399 D-shared and 396 empty-scaffold rows; 622 retained molecules share a scaffold with H4. (B) Both original ChemBERTa tokenizers give 9,835 verified molecules and 241,267 verified atoms, with zero position disagreements within the verified portion. The remaining 165 molecules per tokenizer are unverifiable: 114 auxiliary graph/stereochemistry cases and 51 element-anchor cases. They are neither verified passes nor proven mismatches, and were not removed from the historical H4 analysis. These audits do not change historical H4 significance or establish disjointness from pretraining data.'
    save(fig,'figure_S_I1',180,105,data,caption)

scope_checks();lineage_audit()
