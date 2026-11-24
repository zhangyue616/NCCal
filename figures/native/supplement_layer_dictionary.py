import os
import sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(Path(sys.argv[1])/'runtime_cache'/'matplotlib')
from supplement_layer_data import *
from PIL import Image, ImageDraw, ImageFont
import xml.etree.ElementTree as ET

W=504.0 
INK='#20252A'; MUTED='#6F7780'; GREY='#A3ACB5'; LIGHT='#E3E7EB'
ENV='#0077BB'; ORANGE='#EE7733'; MTR='#0077BB'; MLM='#EE7733'
METHOD_COLORS=(GREY,'#77AADD',ENV,ORANGE)
rc={'font.family':'Arial','font.size':7,'axes.labelsize':7,
    'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.6,
    'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.5,
    'ytick.major.size':2.5,'svg.fonttype':'none','pdf.fonttype':42,
    'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
    'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'}

def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=2.5,width=.6,pad=3)
    ax.grid(axis='x',color=LIGHT,linewidth=.45,zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():spine.set_color(INK)

TEXT=[]
def text(fig,x,y,s,size=7,weight='normal',**kwargs):
    obj=fig.text(x,y,s,fontsize=size,fontweight=weight,**kwargs)
    TEXT.append((obj,s));return obj

def export(fig,name,height_mm):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    records=[]
    alltexts=list(fig.texts)
    for ax in fig.axes:
        alltexts.extend(ax.texts+[ax.title,ax.xaxis.label,ax.yaxis.label])
        alltexts.extend(ax.get_xticklabels()+ax.get_yticklabels())
    for obj in alltexts:
        if not obj.get_visible() or not obj.get_text():continue
        bb=obj.get_window_extent(renderer)
        records.append(dict(text=obj.get_text(),fontsize_pt=obj.get_fontsize(),
            bounds_pt=[float(v*72/fig.dpi) for v in (bb.x0,bb.y0,bb.x1,bb.y1)]))
    assert all(r['bounds_pt'][0]>=-0.5 and r['bounds_pt'][2]<=W+.5 for r in records),[r for r in records if r['bounds_pt'][0]<-.5 or r['bounds_pt'][2]>W+.5]
    for ext in ('svg','pdf','png'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=600)
    fig.savefig(OUT/f'{name}_printsize_150dpi.png',dpi=150)
    (OUT/f'{name}_TEXT_BOUNDS.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf8')
    plt.close(fig)
    return dict(name=name,dimensions_mm=[177.8,height_mm],png_dpi=600,
        font_min_pt=min(r['fontsize_pt'] for r in records),text_bounds_checked=True)

with plt.rc_context(rc):
    fig=plt.figure(figsize=(7,180/25.4),dpi=600)
    text(fig,.025,.976,'a',size=9,weight='bold')
    text(fig,.058,.976,'Training shifts the composition of explained variance',size=8,weight='bold')
    handles=[Patch(facecolor=GREY,label='Atom type'),Patch(facecolor=ENV,hatch='//',label='+ bonding environment')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.20,.974),frameon=False,
        ncol=2,fontsize=7,handlelength=1.7,columnspacing=1.8)
    for mi,model in enumerate(models):
        left=.21 if mi==0 else .71
        ax=fig.add_axes([left,.692,.275,.213])
        nets=['trained','init21000','init21001','init21002'] if mi==0 else ['trained','init4000','init4001','init4002']
        ys=np.array([6,4,2,0])
        for yi,network in zip(ys,nets):
            row=layer[model,network]
            atom=row['VE_radius0']*100; delta=row['VE_local_minus_radius0']*100
            total=row['VE_local_graph']*100
            ax.barh(yi,atom,color=GREY,height=.70,edgecolor='white',linewidth=.5)
            ax.barh(yi,delta,left=atom,color=ENV,height=.70,edgecolor='white',linewidth=.5,hatch='//')
            ax.text(0,yi+.52,f'{atom:.1f} | {delta:.1f} | {total:.1f}%',fontsize=7,va='bottom')
        ax.set_yticks(ys,['Trained','Untrained 1','Untrained 2','Untrained 3'])
        ax.set_ylim(-.9,7.2);ax.set_xlim(0,100);ax.set_xticks([0,50,100])
        ax.set_xlabel('Activation variance explained (%)',labelpad=4)
        ax.set_title('ChemBERTa-MTR' if mi==0 else 'MoLFormer',fontsize=7.5,pad=10)
        clean(ax)

    text(fig,.025,.608,'b',size=9,weight='bold')
    text(fig,.058,.608,'Whole-layer replacement beats mean and atom-type replacement',size=8,weight='bold')
    text(fig,.058,.585,'but not shuffled displacements',size=8,weight='bold')
    text(fig,.21,.552,'Mean absolute deviation from the original output (smaller is closer)',size=7)
    for mi,model in enumerate(models):
        for ti,task in enumerate(tasks):
            left=.21 if ti==0 else .71
            bottom=.340 if mi==0 else .092
            ax=fig.add_axes([left,bottom,.275,.167])
            vals=[errors[model,task,method] for method in methods]
            ys=np.arange(4)[::-1]
            for yi,val,color in zip(ys,vals,METHOD_COLORS):
                ax.barh(yi,val,color=color,height=.56)
                ax.text(val+(.009 if task=='bace' else .020),yi,f'{val:.4f}',va='center',fontsize=7)
            ax.set_yticks(ys,['Mean','Atom type','Bonding environment','Shuffled displacement'])
            ax.set_xlim(0,.56 if task=='bace' else 1.20);ax.set_ylim(-.6,3.6)
            ax.set_xticks([0,.25,.5] if task=='bace' else [0,.5,1.0])
            ax.set_title(('MTR' if mi==0 else 'MoLFormer')+' / '+task.upper(),fontsize=7.5,pad=7,loc='left')
            if mi==1:ax.set_xlabel('Class-1 probability' if task=='bace' else 'Log10 molar solubility',labelpad=5)
            clean(ax)
    SO=export(fig,'Figure_S_O1',180)

    fig=plt.figure(figsize=(7,77/25.4),dpi=600)
    text(fig,.025,.925,'Dictionary features pass both untrained-network calibrations',size=8,weight='bold')
    for ti,test in enumerate(('T1','T2')):
        left=.16 if ti==0 else .65
        ax=fig.add_axes([left,.23,.325,.46])
        for yi,model,color,marker in ((1,'chemberta_mtr',MTR,'o'),(0,'chemberta_mlm',MLM,'s')):
            val,(lo,hi)=calibrations[model,test]
            ax.errorbar(val,yi,xerr=[[val-lo],[hi-val]],fmt=marker,color=color,
                markersize=4.5,capsize=2.5,elinewidth=.9)
            ax.text(7.85 if ti==0 else 3.95,yi+.29,f'{val:.2f}x [{lo:.2f}, {hi:.2f}]',
                color=color,fontsize=7,ha='right',va='bottom')
        ax.axvline(1,color=MUTED,linestyle='--',linewidth=.7)
        ax.set_yticks([1,0],['MTR','MLM']);ax.set_ylim(-.45,1.75)
        ax.set_xlim(0,8 if ti==0 else 4)
        ax.set_xticks([0,2,4,6,8] if ti==0 else [0,1,2,3,4])
        ax.set_xlabel('Calibrated fold (95% interval)',labelpad=5)
        ax.set_title('Dictionary vs untrained dictionaries' if ti==0 else 'Calibrated dictionary/neuron advantage',
            fontsize=7.5,loc='left',pad=8)
        clean(ax)
    SP=export(fig,'Figure_S_P1',77)
