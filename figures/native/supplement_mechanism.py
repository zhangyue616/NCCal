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

def figure6():
    d=read(DATA/'figure_s_e1.json');f=fig_mm(170,167)
    colors={24:ORANGE,25:BLUE,26:GREEN}
    panel(f,'a',.025,.98,'A fixed carbonyl object: separate descriptive measurements')
    a=f.add_axes([.055,.818,.91,.105]);a.axis('off');a.set_xlim(0,1);a.set_ylim(0,1)
    a.text(0,.94,'id24 · j454+ · q124',color=ORANGE,weight='bold',fontsize=8.5)
    for x,label in [(.035,'C'),(.135,'O')]:
        a.text(x,.47,label,ha='center',va='center',fontsize=11,color=ORANGE,
               bbox=dict(boxstyle='circle,pad=.25',fc=PALE_ORANGE,ec=ORANGE,lw=.8))
    for y in [.44,.50]:a.plot([.061,.108],[y,y],color=INK,lw=.85)
    a.text(.20,.62,'[CX3]=[OX1]',fontsize=8.4)
    a.text(.20,.29,'Concept definition',fontsize=8,color=MUTED)
    for x,label,value in [(.47,'Native AP','%.6f'%d['carbonyl']['native_AP']),
                           (.66,'L AP','%.6f'%d['carbonyl']['AP_L']),
                           (.84,'ΔBrier','≈0')]:
        a.text(x,.72,label,fontsize=8.2,color=MUTED)
        a.text(x,.33,value,fontsize=10,fontweight='bold')
    f.lines.append(Line2D([.04,.97],[.795,.795],transform=f.transFigure,color=LINE,lw=.8))
    panel(f,'b',.025,.773,'Sensitivity contrast H')
    panel(f,'c',.64,.773,'Count–output association')
    f.text(.34,.731,'Row-specific linear scales',fontsize=8,color=MUTED)
    f.text(.70,.731,'All T · 10,000 molecules',fontsize=8,color=MUTED)
    names=['Carbonyl (id24)','Carbonyl, noCOO (id25)','Aliphatic OH (id26)']
    for i,(r,name) in enumerate(zip(d['objects'],names)):
        cid=int(r['exact_id']);col=colors[cid];cy=.43+(2-i+.5)*.29/3
        f.text(.045,cy+.018,name,fontsize=8.3,color=col,fontweight='bold',va='center')
        f.text(.045,cy-.007,'j%s / q%s'%(r['D_selected_unit'],r['D_anonymous_output_q']),fontsize=8.1,va='center')
        f.text(.045,cy-.031,'H = %.6f'%float(r['H']),fontsize=8,color=MUTED,va='center')
        ax=f.add_axes([.34,cy-.025,.245,.055]);p,lo,hi=[float(r[k]) for k in ['H','H_CI_lower','H_CI_upper']]
        interval(ax,p,lo,hi,.65,col)
        ax.set_xlim(*((-.00015,.004) if cid!=26 else (-.006,.18)))
        ax.set_xticks([0,.002,.004] if cid!=26 else [0,.08,.16]);ax.set_ylim(0,1.4);ax.set_yticks([])
        zero_axis(ax)
    ax=f.add_axes([.70,.43,.26,.29]);ax.set_ylim(-.5,2.5)
    for r,yy in zip(d['objects'],[2,1,0]):
        col=colors[int(r['exact_id'])];p,lo,hi=[float(r[k]) for k in ['oriented_rho','oriented_rho_CI_lower','oriented_rho_CI_upper']]
        interval(ax,p,lo,hi,yy,col,ms=3.5)
        ax.text(.99,yy+.24,'%.4f [%.4f, %.4f]'%(p,lo,hi),ha='right',fontsize=7.9,color=col)
    ax.set_xlim(-.035,1.02);ax.set_xticks([0,.5,1]);ax.set_yticks([])
    ax.set_xlabel('D-sign-oriented Spearman ρ');zero_axis(ax)
    f.text(.067,.387,'H and ρ: separate three-object 98.3333% CI families. H uses absolute output changes.',fontsize=8)
    panel(f,'d',.025,.350,'Realized perturbation dose by molecule stratum')
    handles=[Line2D([],[],ls='None',marker='D',color=INK,label='Selected coordinate'),
             Line2D([],[],ls='None',marker='o',mfc='white',mec=MUTED,label='Five D-matched controls')]
    f.legend(handles=handles,loc='upper left',bbox_to_anchor=(.055,.333),ncol=2,frameon=False,
             fontsize=8,handletextpad=.5,columnspacing=1.6)
    for si,(stratum,title) in enumerate([('positive','Concept-positive'),('negative','All-negative')]):
        ax=f.add_axes([.16+si*.44,.14,.36,.125]);ax.set_xscale('log')
        for cid,yy in zip([24,25,26],[2,1,0]):
            rows=[r for r in d['full_dose_rows'] if int(r['exact_id'])==cid and r['stratum']==stratum]
            controls=sorted([r for r in rows if r['condition']!='selected_atom_full'],key=lambda r:r['condition'])
            for k,r in enumerate(controls):
                ax.plot(float(r['mean']),yy+(k-2)*.065,'o',mfc='white',mec=MUTED,ms=3.7,mew=.8)
            selected=next(r for r in rows if r['condition']=='selected_atom_full')
            val=float(selected['mean']);ax.plot(val,yy,'D',color=colors[cid],ms=5)
            offset=(9,-2) if cid==25 and stratum=='positive' else (0,8)
            ax.annotate('%.3f'%val,(val,yy),xytext=offset,textcoords='offset points',
                        ha='left' if cid==25 and stratum=='positive' else 'center',fontsize=8,color=colors[cid])
        ax.set_xlim(.4,22);ax.set_ylim(-.5,2.7);ax.set_xticks([.5,1,2,5,10,20])
        ax.set_xticklabels(['0.5','1','2','5','10','20']);ax.minorticks_off()
        ax.set_yticks([2,1,0]);ax.set_yticklabels(['id24','id25','id26'] if si==0 else ['']*3)
        ax.set_xlabel('Mean realized dose (log scale)');ax.set_title(title,fontsize=8.8,pad=6)
        plain(ax,False);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
    f.text(.067,.060,'Dose = mean[RMS(activation change) × FFN outgoing-projection column norm] within stratum.',fontsize=7.9)
    f.text(.067,.030,'200 positive + 200 all-negative T molecules per object; control dispersion is not a confidence interval.',fontsize=7.9,color=MUTED)
    save(f,'figure_6',170,167)

figure6()
