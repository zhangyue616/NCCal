from pathlib import Path
import csv,json,html,sys
D=Path(__file__).resolve().parents[2]/'results/artwork/supplement';F=Path(sys.argv[1]);F.mkdir(parents=True,exist_ok=True)
rows={x['metric']:x for x in csv.DictReader((D/'notation_metrics.csv').open(encoding='utf8'))}
data=[];parts=['<svg xmlns="http://www.w3.org/2000/svg" width="7in" height="3.2in" viewBox="0 0 504 230.4"><rect width="504" height="230.4" fill="white"/>']
def text(x,y,s,size=8.5,anchor='start',weight='normal',fill='#222222'):
    parts.append(f'<text x="{x}" y="{y}" font-family="Arial,Helvetica,sans-serif" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">{html.escape(str(s))}</text>')
def line(x1,y1,x2,y2,stroke='#333333',width=.6,dash=''):
    parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>')
text(252,15,'Frozen bonding-environment estimates under SMILES rewriting',9.5,'middle','bold')
for x,name,color,marker in [(135,'Mean replacement','#425C7A','circle'),(300,'Shuffled displacement','#CD7A4D','triangle')]:
    if marker=='circle':parts.append(f'<circle cx="{x-9}" cy="32" r="2.6" fill="{color}"/>')
    else:parts.append(f'<polygon points="{x-12},35 {x-6},35 {x-9},29" fill="{color}"/>')
    text(x,35,name,8.5)
for panel,(j,q,title,x0,x1,lo,hi,ticks) in enumerate([(293,115,'Hydroxyl · neuron 293 · q115',67,238,-24,2,[-24,-18,-12,-6,0]),(454,124,'Carbonyl · neuron 454 · q124',319,490,-3,5,[-3,-2,0,2,4])]):
    scale=lambda v:x0+(v-lo)/(hi-lo)*(x1-x0)
    text(x0-52,57,chr(97+panel),10,'start','bold');text((x0+x1)/2,57,title,9,'middle','bold')
    line(scale(0),70,scale(0),178,'#999999',.6,'2 2');line(x0,184,x1,184)
    for v in ticks:line(scale(v),184,scale(v),188);text(scale(v),200,str(v),8,'middle')
    for i,(encoding,label) in enumerate([('canonical','Original'),('reverse','Reverse'),('hash_permutation','Permuted')]):
        y=88+i*37;text(x0-7,y+3,label,8.5,'end')
        for k,(control,color,marker) in enumerate([('mean','#425C7A','circle'),('shuffle','#CD7A4D','triangle')]):
            row=rows[f'j{j}/q{q}/{encoding}/M2_minus_{control}_MAE'];est=float(row['estimate'])*1000;l=float(row['CI_low'])*1000;u=float(row['CI_high'])*1000;yy=y+(-5 if k==0 else 5)
            assert lo<l<=est<=u<hi
            line(scale(l),yy,scale(u),yy,color,1.15);line(scale(l),yy-2,scale(l),yy+2,color,.8);line(scale(u),yy-2,scale(u),yy+2,color,.8)
            xx=scale(est)
            if marker=='circle':parts.append(f'<circle cx="{xx}" cy="{yy}" r="2.8" fill="{color}"/>')
            else:parts.append(f'<polygon points="{xx-3.2},{yy+2.7} {xx+3.2},{yy+2.7} {xx},{yy-3.1}" fill="{color}"/>')
            data.append({'neuron':j,'output':q,'encoding':encoding,'reference':control,'estimate':float(row['estimate']),'CI_low':float(row['CI_low']),'CI_high':float(row['CI_high']),'n_molecules':int(row['n_molecules'])})
    text((x0+x1)/2,216,'Bonding environment − reference MAE (×0.001)',8,'middle')
text(252,229,'483 matched molecules · pointwise 95% molecule-bootstrap intervals · horizontal ranges differ',8,'middle')
parts.append('</svg>');(F/'Figure_S_J2.svg').write_text(''.join(parts),'utf8');(F/'Figure_S_J2_source.json').write_text(json.dumps(data,indent=2),'utf8')
print('12 paired differences drawn from frozen summary metrics.')
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Polygon,Rectangle
plt.rcParams.update({'font.family':'Arial','svg.fonttype':'none','pdf.fonttype':42})
fig=plt.figure(figsize=(7,3.2));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,504),ylim=(230.4,0));ax.axis('off')
for node in ET.fromstring(''.join(parts)):
    a=node.attrib;tag=node.tag.split('}')[-1]
    if tag=='rect':ax.add_patch(Rectangle((float(a.get('x',0)),float(a.get('y',0))),float(a['width']),float(a['height']),facecolor=a.get('fill','white'),edgecolor='none'))
    elif tag=='text':ax.text(float(a['x']),float(a['y']),node.text or'',fontsize=float(a['font-size']),ha={'start':'left','middle':'center','end':'right'}[a['text-anchor']],va='baseline',fontweight=a.get('font-weight','normal'),color=a.get('fill','#222222'))
    elif tag=='line':
        artist=ax.plot([float(a['x1']),float(a['x2'])],[float(a['y1']),float(a['y2'])],color=a['stroke'],lw=float(a['stroke-width']))[0]
        if 'stroke-dasharray'in a:artist.set_dashes([float(v)for v in a['stroke-dasharray'].split()])
    elif tag=='circle':ax.add_patch(Circle((float(a['cx']),float(a['cy'])),float(a['r']),facecolor=a['fill'],edgecolor='none'))
    elif tag=='polygon':ax.add_patch(Polygon([[float(v)for v in pair.split(',')]for pair in a['points'].split()],facecolor=a['fill'],edgecolor='none'))
for ext in ('png','pdf'):fig.savefig(F/('Figure_S_J2.'+ext),dpi=400,facecolor='white')
plt.close(fig)
