"""Compose the retained protein panels freshly generated from numerical results."""
import sys,copy
from pathlib import Path
from xml.etree import ElementTree as ET
from PIL import Image,ImageDraw,ImageFont
from matplotlib.font_manager import FontProperties,findfont
OUT=Path(sys.argv[1]);S='http://www.w3.org/2000/svg'
root=ET.Element('{'+S+'}svg',{'width':'468pt','height':'516pt','viewBox':'0 0 468 516'})
ET.SubElement(root,'{'+S+'}rect',{'width':'468','height':'516','fill':'white'})
source=OUT/'components/Protein_panels.svg'
parent=Image.open(str(OUT/'components/Protein_panels.png')).convert('RGBA')
canvas=Image.new('RGBA',(3900,4300),'white')
panels=[((172,0,164,214),(4,2),(205,267.5),'b','a'),((0,219,168,185),(247,2),(210,231.25),'d','b'),((344,219,160,185),(120,280),(200,231.25),'f','c')]
for index,(box,dest,size,old,new)in enumerate(panels):
 node=ET.parse(source).getroot();ids={e.get('id'):'p%d_'%index+e.get('id')for e in node.iter()if e.get('id')}
 for e in node.iter():
  for key,val in list(e.attrib.items()):
   if key=='id':e.set(key,ids[val]);continue
   for a,b in ids.items():val=val.replace('url(#'+a+')','url(#'+b+')');val='#'+b if val=='#'+a else val
   e.set(key,val)
  if e.tag=='{'+S+'}text'and e.text==old:e.text=new
 node.set('x',str(dest[0]));node.set('y',str(dest[1]));node.set('width',str(size[0]));node.set('height',str(size[1]));node.set('viewBox',' '.join(map(str,box)));node.set('overflow','hidden');root.append(node)
 scale=parent.width/504.;x,y,w,h=box
 tile=parent.crop(tuple(round(t*scale)for t in [x,y,x+w,y+h]));factor=size[0]/w;pix=600/72.
 tile=tile.resize(tuple(round(t*pix)for t in size),Image.LANCZOS)
 # Replace the panel letter only, using its known source point coordinates.
 source_x={old:1.0 if old!='b'else 5.0}[old];source_y=12.0 if old=='b'else 12.0
 # Lower-row source letters lie at y=231; the crop starts at 219.
 draw=ImageDraw.Draw(tile);draw.rectangle((0,0,round(20*factor*pix),round(19*factor*pix)),fill='white')
 font=ImageFont.truetype(findfont(FontProperties(family='Arial',weight='bold')),round(11*factor*pix))
 draw.text((round(source_x*factor*pix),round(source_y*factor*pix)),new,font=font,fill='#20252A',anchor='ls')
 canvas.alpha_composite(tile,tuple(round(t*pix)for t in dest))
ET.ElementTree(root).write(OUT/'Figure_S_G5.svg',encoding='utf-8',xml_declaration=True)
canvas.save(str(OUT/'Figure_S_G5.png'),dpi=(600,600));canvas.convert('RGB').save(str(OUT/'Figure_S_G5.pdf'),resolution=600)
