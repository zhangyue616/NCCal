"""Freeze independent new natural-sequence groups before any readout."""
from prepare import *
sys.path.insert(0,str(ROOT/'structure/runtime'))
from inventory import jl,table
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import numpy as np
def init(prior):
    global PRIOR,MAT,ProteinSequence,align_optimal
    from biotite.sequence import ProteinSequence
    from biotite.sequence.align import align_optimal,SubstitutionMatrix
    MAT=SubstitutionMatrix.std_protein_matrix();PRIOR=[(h,s,ProteinSequence(s)) for h,s in prior]
def screen(r):
    s=r['sequence'];ps=ProteinSequence(s);hits=[];maxid=0.;maxcov=0.
    for h,t,pt in PRIOR:
        a=align_optimal(ps,pt,MAT,gap_penalty=(-10,-1),local=True,max_number=1)[0];tr=a.trace;both=tr[(tr[:,0]>=0)&(tr[:,1]>=0)]
        ident=float(np.mean(ps.code[both[:,0]]==pt.code[both[:,1]])) if len(both) else 0.
        cov=min(len(set(tr[tr[:,0]>=0,0])),len(set(tr[tr[:,1]>=0,1])))/min(len(s),len(t))
        if ident>=.30 and cov>=.80:hits.append(h)
        maxid=max(maxid,ident);maxcov=max(maxcov,cov)
    return r,hits,maxid,maxcov
def main():
    candidates=js(OUT/'new_source/EXACT_NATIVE_SAMPLE.json')['entries'];F=ROOT/'structural_readout_validation/fresh'
    prior={r['sequence_sha256']:r['sequence'] for r in jl(F/'prior_native_sequences.jsonl')}
    for r in jl(F/'chains.jsonl'):prior[hashlib.sha256(r['sequence'].encode()).hexdigest()]=r['sequence']
    # The reserved194 are explicitly excluded from brand-new recruited groups,
    # but remain separately eligible as the original archive reserve layer.
    for r in table(OUT/'RESERVE_194_INVENTORY.tsv'):prior[hashlib.sha256(r['sequence'].encode()).hexdigest()]=r['sequence']
    old_pfams={x['id'] for p in (OUT/'annotations').glob('uniprot_*.json') for r in js(p)['response']['results'] for x in r.get('uniProtKBCrossReferences',[]) if x['database']=='Pfam'}
    oldp=set();olde=set()
    files=[Path(x['path']) for x in js(F/'PRIOR_EXCLUSION_SCOPE.json')['files']]+[F/'chains.jsonl',F/'native_candidate_chains.jsonl']
    for p in files:
        for r in jl(p):
            if r.get('pdb_id'):
                if r.get('entity_id'):olde.add(r['pdb_id'].upper()+'_'+str(r['entity_id']))
                else:oldp.add(r['pdb_id'].upper())
    needed={e for r in candidates for e in r['entities']};clusters={};blocked=set()
    for line in (F/'sources/clusters-by-entity-30.txt').read_text().splitlines():
        toks=line.split();found=needed.intersection(toks)
        if not found:continue
        group='rcsb30-'+hashlib.sha256(line.strip().encode()).hexdigest()[:24]
        for e in found:clusters[e]=group
        if any(e in olde or e.split('_')[0] in oldp for e in toks):blocked.add(group)
    eligible=[];excluded=[]
    for r in sorted(candidates,key=lambda r:hashlib.sha256(('label-audit-native-20261123|'+r['sequence']).encode()).hexdigest()):
        h=hashlib.sha256(r['sequence'].encode()).hexdigest();r=dict(r,sequence_sha256=h)
        reasons=[]
        if h in prior:reasons.append('exact_prior_or_reserved_sequence')
        if set(r['pfam'])&old_pfams:reasons.append('shared_Pfam_legacy_component')
        cs={clusters[e] for e in r['entities'] if e in clusters}
        if cs&blocked:reasons.append('RCSB30_prior_component')
        r['rcsb30_groups']=sorted(cs)
        if reasons:excluded.append(dict(r,reasons=reasons))
        else:eligible.append(r)
    print('alignment candidates',len(eligible),'prior',len(prior),flush=True)
    kept=[];audit=[];start=time.time()
    with ProcessPoolExecutor(max_workers=6,initializer=init,initargs=(list(prior.items()),)) as pool:
        for r,hits,mi,mc in pool.map(screen,eligible,chunksize=1):
            audit.append(dict(accession=r['accession'],prior_hits=hits,max_identity_any=mi,max_coverage_any=mc))
            if hits:excluded.append(dict(r,reasons=['identity30_coverage80_to_prior']))
            else:kept.append(r)
            print('screen',r['accession'],'retained',len(kept),'seconds',time.time()-start,flush=True)
    # New-new connections use all inherited identities, shared Pfam/RCSB30,
    # then the same direct sequence threshold; a component yields one rep.
    init([]);parent=list(range(len(kept)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,r in enumerate(kept):
        for j,t in enumerate(kept[:i]):
            related=bool(set(r['pfam'])&set(t['pfam']) or set(r['rcsb30_groups'])&set(t['rcsb30_groups']))
            if not related:
                init([(t['sequence_sha256'],t['sequence'])]);related=bool(screen(r)[1])
            if related:parent[root(i)]=root(j)
    groups=defaultdict(list)
    for i,r in enumerate(kept):groups[root(i)].append(r)
    final=[]
    for rr in groups.values():
        rep=rr[0];rep['group']='label-audit-fresh-'+rep['sequence_sha256'][:24];rep['component_accessions']=[r['accession'] for r in rr];final.append(rep)
    final=final[:100]
    write('new_source/HOMOLOGY.json',{'utc':datetime.now(timezone.utc).isoformat(),'prior_sequences':len(prior),'input_native_candidates':len(candidates),'new_groups':len(final),'new_residues':sum(r['length'] for r in final),'criteria':'exact sequence, current shared Pfam legacy components, RCSB30 and local BLOSUM62 identity>=30% / shorter coverage>=80%; same criteria new-new','prior_alignment_audit':audit,'excluded':excluded,'entries':final,'structure_and_SIFTS_validation':'PENDING; not confirmation ready','no_activations':True})
    print('final',len(final),flush=True)
if __name__=='__main__':main()
