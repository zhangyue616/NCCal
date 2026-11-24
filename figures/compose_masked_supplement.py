"""Compose S-G1/G3 from newly rendered numerical protein components."""
import sys
from pathlib import Path
from xml.etree import ElementTree as ET
from PIL import Image,ImageDraw,ImageFont
from matplotlib.font_manager import FontProperties,findfont
OUT=Path(sys.argv[1]);S='http://www.w3.org/2000/svg';source=OUT/'components/Masked_protein.svg'
parent=Image.open(OUT/'components/Masked_protein.png').convert('RGBA')
for name,width,height,panels in [
 ('Figure_S_G1',355,357,[((174,0,166,180),(0,0),'b','a'),((0,184,172,162),(183,0),'d','b'),((174,184,330,162),(12.5,195),'e','c')]),
 ('Figure_S_G3',156,180,[((348,0,156,180),(0,0),'c','a')])]:
 root=ET.Element('{'+S+'}svg',{'width':str(width)+'pt','height':str(height)+'pt','viewBox':'0 0 %g %g'%(width,height)})
 ET.SubElement(root,'{'+S+'}rect',{'width':str(width),'height':str(height),'fill':'white'})
 pix=600/72.;canvas=Image.new('RGBA',(round(width*pix),round(height*pix)),'white')
 for index,(box,dest,old,new)in enumerate(panels):
  node=ET.parse(source).getroot();ids={e.get('id'):'p%d_'%index+e.get('id')for e in node.iter()if e.get('id')}
  for e in node.iter():
   for key,val in list(e.attrib.items()):
    if key=='id':e.set(key,ids[val]);continue
    for a,b in ids.items():val=val.replace('url(#'+a+')','url(#'+b+')');val='#'+b if val=='#'+a else val
    e.set(key,val)
   if e.tag=='{'+S+'}text'and e.text in (old,old.upper()):e.text=new
  x,y,w,h=box
  node.attrib.update(x=str(dest[0]),y=str(dest[1]),width=str(w),height=str(h),viewBox=' '.join(map(str,box)),overflow='hidden');root.append(node)
  scale=parent.width/504.;tile=parent.crop(tuple(round(t*scale)for t in [x,y,x+w,y+h]))
  draw=ImageDraw.Draw(tile);sx=2 if old in ('b','e')else 0;sy=11 if y==0 else 195-y
  draw.rectangle((0,0,round(13*pix),round(16*pix)),fill='white')
  font=ImageFont.truetype(findfont(FontProperties(family='Arial',weight='bold')),round(11*pix))
  draw.text((round(sx*pix),round(sy*pix)),new,font=font,fill='#20252A',anchor='ls')
  canvas.alpha_composite(tile,tuple(round(t*pix)for t in dest))
 ET.ElementTree(root).write(OUT/(name+'.svg'),encoding='utf-8',xml_declaration=True)
 canvas.save(OUT/(name+'.png'),dpi=(600,600));canvas.convert('RGB').save(OUT/(name+'.pdf'),resolution=600)
