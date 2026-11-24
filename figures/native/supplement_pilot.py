from pathlib import Path
import sys,json,csv,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle,Circle,FancyArrowPatch,Arc
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'results/artwork/supplement'
OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=True)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(fig,name,*args,**kwargs):
    name={'figure_S_B1':'Figure_S_B1','figure_S_I1':'Figure_S_I1','figure_6':'Figure_S_E1','figure_2':'Figure_S_C6'}.get(name,name)
    for ext in ('png','svg','pdf'):fig.savefig(OUT/(name+'.'+ext),dpi=400,facecolor='white')
    plt.close(fig)
INK, MUTED, LINE = '#243541', '#62727B', '#D7E0E5'
BLUE, ORANGE, GREEN = '#286B95', '#B76A2A', '#237D6B'
PALE_BLUE, PALE_ORANGE, PALE_GREEN = '#F0F6FA', '#FCF5EE', '#EFF7F4'
plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':8.3,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,
 'axes.titlesize':9,'text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':MUTED,
 'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,
 'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white',
 'axes.facecolor':'white','savefig.facecolor':'white'})
def fig_mm(w,h):
    return plt.figure(figsize=(w/25.4,h/25.4))

def plain(ax, grid=True):
    for s in ['top','right']:
        ax.spines[s].set_visible(False)
    ax.tick_params(length=3,pad=3)
    if grid:
        ax.set_axisbelow(True)
        ax.grid(axis='y',color='#EAF0F3',lw=.55)

def panel(fig, letter, x, y, title=None):
    fig.text(x,y,letter,fontsize=11.2,fontweight='bold',va='top')
    if title:
        fig.text(x+.035,y-.002,title,fontsize=9,fontweight='bold',va='top')

def interval(ax,p,lo,hi,y,col,marker='o',lw=1.6,ms=4.5):
    ax.plot([lo,hi],[y,y],color=col,lw=lw,zorder=3,solid_capstyle='butt')
    for xx in [lo,hi]:ax.plot([xx,xx],[y-.055,y+.055],color=col,lw=.7,zorder=3)
    ax.plot(p,y,marker=marker,ms=ms,color=col,zorder=4)

def zero_axis(ax):
    plain(ax,False);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    ax.axvline(0,color=MUTED,ls='--',lw=.75,zorder=0)


from rdkit import Chem
from rdkit.Chem import Draw,AllChem
def depict(s):
    mol=Chem.MolFromSmiles(s['smiles']);AllChem.Compute2DCoords(mol)
    target=set(s['concept_atom_indices']);response=set(s['responding_atom_indices']);colors={i:(1.,.63,.30)if i in target else(.72,.86,.97)for i in target|response}
    return Draw.MolToImage(mol,size=(800,400),highlightAtoms=sorted(colors),highlightAtomColors=colors)
def figure2():
    d=read(DATA/'figure_s_c6.json'); f=fig_mm(170,143)
    keys=['chemeleon','molclr_gin','hu_masking']
    names=['CheMeleon','MolCLR GIN','Hu masking']
    for ix,(mode,title,col) in enumerate([('initialization','Initialization reference',MUTED),
                                         ('rotation','Rotation reference',ORANGE)]):
        x=.155 if ix==0 else .605
        ax=f.add_axes([x,.66,.355,.245])
        for i,k in enumerate(keys):
            yy=2-i;r=d['pilot'][k]
            ax.plot([r['native'],r[mode+'_mean']],[yy,yy],color=LINE,lw=1.5,zorder=1)
            dots=r[mode+'_dots_pct']
            ax.scatter(dots,yy+np.linspace(-.105,.105,len(dots)),s=12,
                       facecolors='white',edgecolors=col,linewidths=.75,zorder=3)
            ax.scatter([r[mode+'_mean']],[yy],s=27,marker='D',color=col,zorder=4)
            ax.scatter([r['native']],[yy],s=32,color=BLUE,edgecolors='white',linewidths=.4,zorder=5)
        ax.set_xlim(0,100);ax.set_ylim(-.55,2.55)
        ax.set_xticks([0,25,50,75,100]);ax.set_yticks([2,1,0]);ax.set_yticklabels(names if ix==0 else ['']*3)
        ax.set_xlabel('Matched coordinates (%)');plain(ax,False)
        ax.grid(axis='x',color='#EAF0F3',lw=.6)
        panel(f,'ab'[ix],.025+ix*.53,.975,title)
        handles=[Line2D([],[],ls='None',marker='o',color=BLUE,label='Native'),
                 Line2D([],[],ls='None',marker='D',color=col,label='Reference mean')]
        ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(-.1,1.16),frameon=False,
                  fontsize=7.9,ncol=2,handletextpad=.4,columnspacing=1.0)
    f.text(.065,.585,'20,000 molecules · 173 concepts · 3 initialization twins · 10 rotations',fontsize=8.1)
    f.text(.065,.554,'Match: any pole / concept; lift ≥3; permutation-null threshold; BH q = .05',fontsize=8.1,color=MUTED)
    f.lines.append(Line2D([.04,.97],[.52,.52],transform=f.transFigure,color=LINE,lw=.8))
    molecular=[]
    for i,row in enumerate(d['H3_examples']):
        x=.035+i*.49
        s=row['depiction']
        panel(f,'cd'[i],x,.495,'ChemBERTa-77M-MTR' if i==0 else 'CheMeleon')
        f.text(x+.035,.449,'Final-block pre-GELU j148 · GELU(z)' if i==0 else
               'Main-output pre-ReLU j783 · ReLU(z)',fontsize=8.1)
        f.text(x+.035,.419,'Aromatic-nitro concept atom' if i==0 else
               'Sulfonyl-S response example',fontsize=8,color=MUTED)
        ax=f.add_axes([x+.008,.181,.455,.224]);ax.imshow(depict(s));ax.axis('off')
        ax=f.add_axes([x+.025,.145,.425,.035]);left=0
        for label,n,col in row['display_composition']:
            ax.barh(0,n,left=left,height=.85,color=col,edgecolor='white',linewidth=.7)
            ax.text(left+n/2,0,str(n),ha='center',va='center',color='white',fontsize=8.2,fontweight='bold')
            left+=n
        ax.set_xlim(0,100);ax.set_ylim(-.6,.6);ax.axis('off')
        f.text(x+.025,.103,'Top 100: 85 nitro + 15 other; 92 molecules' if i==0 else
               'Top 100: 56 ketone C + 32 sulfonyl S\n+ 12 other; 88 molecules',fontsize=8,va='center')
    f.text(.06,.040,'Orange atoms: concept labels. Pale blue: other responses ≥30% of within-molecule maximum.',fontsize=7.9)
    f.text(.06,.016,'H3 examples are curated; these coordinates are distinct from the fixed post-GELU j454.',fontsize=7.9,color=MUTED)
    save(f,'Figure_S_C6')

figure2()
