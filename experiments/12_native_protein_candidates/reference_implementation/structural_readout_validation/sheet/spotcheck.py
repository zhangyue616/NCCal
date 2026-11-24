"""Independent parser and DSSP-energy neighbour recomputation, no model reads."""
from pathlib import Path
import sys,gzip,json,hashlib,itertools,argparse
from collections import defaultdict,Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'structure/runtime'))
import gemmi,numpy as np
H=Path(__file__).resolve().parent
def read(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]
def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def table(b,c):
    z=b.get_mmcif_category(c);return [dict(zip(z,r)) for r in zip(*z.values())] if z else []
def recompute(path):
    b=gemmi.cif.read_string(gzip.decompress(Path(path).read_bytes()).decode()).sole_block();raw=defaultdict(list);res={};names={}
    for x in table(b,'_atom_site.'):
        if not str(x.get('label_seq_id','')).isdigit() or x.get('label_atom_id') not in {'N','CA','C','O'}:continue
        raw[(x['label_asym_id'],int(x['label_seq_id']))].append(x)
    for k,xx in raw.items():
        if Counter(x['label_atom_id'] for x in xx)!={'N':1,'CA':1,'C':1,'O':1}:continue
        if any(x.get('label_alt_id') not in {False,None,'.','?',''} or float(x['occupancy'])<.99 for x in xx):continue
        res[k]={x['label_atom_id']:np.array([float(x[t]) for t in ('Cartn_x','Cartn_y','Cartn_z')]) for x in xx};names[k]=xx[0]['label_comp_id']
    for k,z in res.items():
        prev=(k[0],k[1]-1)
        if prev in res and names[k]!='PRO' and np.linalg.norm(z['N']-res[prev]['C'])<1.8:
            delta=res[prev]['C']-res[prev]['O'];z['H']=z['N']+delta/np.linalg.norm(delta)
    nodes={}
    for x in table(b,'_struct_sheet_range.'):
        try:ids=[(x['beg_label_asym_id'],p) for p in range(int(x['beg_label_seq_id']),int(x['end_label_seq_id'])+1)]
        except:continue
        nodes[f"{x['sheet_id']}:{x['id']}"]=ids
    def hb(d,a):
        if 'H' not in d:return False
        ds=[np.linalg.norm(d['N']-a['O']),np.linalg.norm(d['H']-a['C']),np.linalg.norm(d['H']-a['O']),np.linalg.norm(d['N']-a['C'])]
        return min(ds)>.5 and 27.888*(1/ds[0]+1/ds[1]-1/ds[2]-1/ds[3])<-.5
    edges=[];details=[]
    for (n,aa),(m,bb) in itertools.combinations(nodes.items(),2):
        pairs=[]
        for ka in aa:
            for kb in bb:
                if ka not in res or kb not in res or (ka[0]==kb[0] and abs(ka[1]-kb[1])<=2):continue
                a,z=res[ka],res[kb]
                if np.linalg.norm(a['CA']-z['CA'])<9 and (hb(a,z) or hb(z,a)):pairs.append((ka,kb))
        x=len({p[0] for p in pairs});y=len({p[1] for p in pairs})
        if x>=2 and y>=2 and min(x,y)>=.3*min(len(aa),len(bb)):
            edges.append(tuple(sorted((n,m))));details.append({'edge':sorted((n,m)),'distinct_residues_1':x,'distinct_residues_2':y,'contacts':[[list(a),list(z)] for a,z in pairs]})
    return set(edges),details
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fresh',action='store_true');ap.add_argument('--tag',default='');ap.add_argument('--input');args=ap.parse_args();s='_'+args.tag if args.tag else ('_F' if args.fresh else '')
    chains=read(H/f'chains{s}.jsonl');summary=json.loads((H/f'BUILD_SUMMARY{s}.json').read_text());path=args.input or summary.get('input_path') or (str(H.parent/'fresh/chains.jsonl') if args.fresh else str(H.parents[1]/'native_channels/cohort/chains.jsonl'));allch=read(path);by={r['record_id']:r for r in allch};rows=read(H/f'sheet_graph_audit{s}.jsonl')
    sample=[]
    for eligible in (True,False):
        rr=sorted((r for r in rows if r['eligible']==eligible),key=lambda r:sha('blind-spot-v1|'+r['sheet_id']))
        seen=set()
        for r in rr:
            if r['pdb_id'] in seen:continue
            sample.append(r);seen.add(r['pdb_id'])
            if len(seen)>=8:break
    cache={};out=[]
    for r in sample:
        path=by[r['record_id']]['source_cif']
        if path not in cache:cache[path]=recompute(path)
        ee,details=cache[path];ids={n['node_id'] for n in r['nodes']};incident={e for e in ee if any(z in ids for z in e)};archive={tuple(e) for e in r['archive_edges']}
        out.append({'sheet_id':r['sheet_id'],'record_id':r['record_id'],'eligible':r['eligible'],'failure_reasons':r['failure_reasons'],'archive_edges':r['archive_edges'],'independent_energy_edges':[list(x) for x in sorted(incident)],'archive_energy_graph_exact':archive==incident,'missing_archive_edges':[list(x) for x in sorted(archive-incident)],'extra_energy_edges':[list(x) for x in sorted(incident-archive)],'method':'independent mmCIF parser; virtual H=N+unit(previous C-O); Kabsch-Sander E<-0.5 kcal/mol, CA<9A; >=2 distinct positions each and >=30% short range; not full mkdssp','energy_contact_details':[x for x in details if any(z in ids for z in x['edge'])]})
    (H/f'blind_spot_check{s}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in out),encoding='utf-8');print(json.dumps({'sheets_checked':len(out),'eligible_checked':sum(r['eligible'] for r in out),'eligible_exact':sum(r['eligible'] and r['archive_energy_graph_exact'] for r in out),'failed_checked':sum(not r['eligible'] for r in out),'failed_exact':sum(not r['eligible'] and r['archive_energy_graph_exact'] for r in out)},indent=2))
if __name__=='__main__':main()
