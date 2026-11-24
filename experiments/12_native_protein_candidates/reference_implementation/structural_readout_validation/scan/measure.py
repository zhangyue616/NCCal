"""All-native-channel search with locks strictly preceding heldout forward/scoring."""
from pathlib import Path
import argparse, csv, json, time, gc, math
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import runtime as rt
ROOT=rt.ROOT
TASK='sheet_edge_vs_central'

def dump(path,obj,exclusive=False):
    with open(path,'x' if exclusive else 'w',encoding='utf-8') as f:
        json.dump(obj,f,indent=2);f.flush()

def read_data(fresh=False,cohort_dir=None):
    base=Path(cohort_dir) if cohort_dir else ROOT.parent/'sheet'
    pf=base/('pairs_F.tsv' if fresh else 'pairs.tsv')
    cf=base/('chains_F.jsonl' if fresh else 'chains.jsonl')
    rows=pd.read_csv(pf,sep='\t',dtype={'group':str,'record_id':str,'pair_id':str})
    chains={r['record_id']:r for r in map(json.loads,cf.read_text(encoding='utf-8-sig').splitlines())}
    assert len(rows)>0 and rows.pair_id.is_unique and set(rows.task)=={TASK}
    assert set(rows.split)<={'F'} if fresh else set(rows.split)<={'D','T','E','H'}
    assert rows.groupby('group').split.nunique().max()==1
    for r in rows.itertuples():
        seq=chains[r.record_id]['sequence']
        assert seq[r.index_positive]==seq[r.index_negative]==r.aa
    return rows,chains,dict(pairs_path=str(pf),pairs_sha256=rt.sha(pf),chains_path=str(cf),chains_sha256=rt.sha(cf))

@torch.inference_mode()
def extract(model,mods,forward,prefix,limit,rows,chains,out,split,keep_all=True,selection=None):
    depth,width=len(mods),mods[0].weight.shape[1]
    target=out/f'{split}_delta.npy'
    if keep_all:
        buf=np.lib.format.open_memmap(target,mode='w+',dtype=np.float32,shape=(len(rows),depth,width))
    else:buf=np.full(len(rows),np.nan,dtype=np.float32)
    context={};coverage=[]
    def capture(l,args):
        a=args[0][0];delta=(a[context['pos']]-a[context['neg']]).float().cpu().numpy()
        assert delta.shape==(len(context['ix']),width) and np.isfinite(delta).all()
        if keep_all:buf[context['ix'],l,:]=delta
        elif l==selection['block_1']-1:buf[context['ix']]=delta[:,selection['channel_0']]
        context['layers'].add(l)
    handles=[m.register_forward_pre_hook(lambda m,a,l=l:capture(l,a)) for l,m in enumerate(mods) if keep_all or l==selection['block_1']-1]
    for rid,part in rows.groupby('record_id',sort=False):
        seq=chains[rid]['sequence'];assert len(seq)+2*prefix<=limit,'No truncation'
        context.update(ix=part.index.to_numpy(),pos=torch.tensor(part.index_positive.to_numpy()+prefix,device='cuda'),neg=torch.tensor(part.index_negative.to_numpy()+prefix,device='cuda'),layers=set())
        start=time.time(); result=forward(seq)
        assert torch.isfinite(result).all()
        assert len(context['layers'])==(depth if keep_all else 1)
        coverage.append(dict(record_id=rid,split=split,length=len(seq),pairs=len(part),seconds=time.time()-start,formal_forwards=1))
        del result
    for h in handles:h.remove()
    if keep_all:buf.flush()
    else:assert np.isfinite(buf).all()
    rows.to_csv(out/f'{split}_pairs.tsv',sep='\t',index=False)
    pd.DataFrame(coverage).to_csv(out/f'{split}_coverage.tsv',sep='\t',index=False)
    return buf

def select(data,rows):
    members=[part.index.to_numpy() for _,part in rows.groupby('group',sort=True)]
    common=math.lcm(*(len(ix) for ix in members));denom=2*common*len(members)
    dtype=np.int64 if denom<np.iinfo(np.int64).max else object
    best=(-1,None,None,None);ranks=[];top_ties=0
    for l in range(data.shape[1]):
        nums=np.zeros(data.shape[2],dtype=dtype)
        for ix in members:
            vals=data[ix,l,:]
            points=2*(vals>0).sum(axis=0)+(vals==0).sum(axis=0)
            nums+=(common//len(ix))*points.astype(dtype)
        signed=np.stack((nums,denom-nums),axis=-1)
        u,s=np.unravel_index(np.argmax(signed),signed.shape)
        value=int(signed[u,s])
        if value>best[0]:
            best=(value,l,int(u),1 if s==0 else -1);top_ties=int((signed==value).sum())
        elif value==best[0]:top_ties+=int((signed==value).sum())
        ranks.append((l,float(value/denom)))
    val,l,u,s=best
    return dict(task=TASK,block_1=l+1,block_0=l,channel_0=u,channel_1=u+1,sign=s,development_score=val/denom,numerator=str(val),denominator=str(denom),development_groups=len(members),development_pairs=len(rows),signed_candidates=2*data.shape[1]*data.shape[2],n_top_tied_signed_candidates=top_ties,tie_rule='block0,channel0,+ first')

def run(name,network, fresh=False,cohort_dir=None):
    rows,chains,binding=read_data(fresh,cohort_dir)
    seed=None if network=='trained' else int(network)
    out=ROOT/name/network;out.mkdir(parents=True,exist_ok=True)
    done=out/('F_done.json' if fresh else 'done.json')
    if done.exists():print('REUSE',name,network,done.name,flush=True);return
    lockpath=out/'selection_lock.json'
    if fresh:assert lockpath.exists(),'D lock required before fresh F'
    model,mods,forward,prefix,limit,meta=rt.load(name,seed=seed)
    dump(out/'model_metadata.json',meta)
    values=[]
    for split in (['F'] if fresh else ['D','T','E','H']):
        part=rows[rows.split==split].reset_index(drop=True)
        if not len(part):continue
        if split=='D' and lockpath.exists():
            selection=json.loads(lockpath.read_text())['selection']
            assert rt.sha(Path(json.loads(lockpath.read_text())['inputs']['pairs_path']))==binding['pairs_sha256']
            data=np.load(out/'D_delta.npy',mmap_mode='r')
        else:
            selection=json.loads(lockpath.read_text())['selection'] if lockpath.exists() else None
            data=extract(model,mods,forward,prefix,limit,part,chains,out,split,keep_all=network=='trained' or split=='D',selection=selection)
            if split=='D':
                selection=select(data,part)
                dump(lockpath,dict(created_unix=time.time(),model=name,network=network,selection=selection,inputs=binding,protocol_sha256=rt.sha(ROOT/'PROTOCOL.md')),exclusive=True)
                print('LOCKED',name,network,json.dumps(selection),flush=True)
        col=data[:,selection['block_1']-1,selection['channel_0']] if data.ndim==3 else data
        scored=part.copy();scored['raw_delta']=np.asarray(col);scored['delta']=selection['sign']*scored.raw_delta
        scored['correct']=(scored.delta>0).astype(float)+.5*(scored.delta==0)
        scored['model']=name;scored['network']=network;scored['block_1']=selection['block_1'];scored['channel_0']=selection['channel_0'];scored['sign']=selection['sign']
        scored['selection_lock_sha256']=rt.sha(lockpath);values.append(scored)
        print('FORWARDED',name,network,split,len(part),'pairs',flush=True)
        del data
    pd.concat(values,ignore_index=True).to_csv(out/('F_signed_pairs.tsv' if fresh else 'signed_pairs.tsv'),sep='\t',index=False)
    dump(done,dict(model=name,network=network,inputs=binding,selection_lock_sha256=rt.sha(lockpath),completed_unix=time.time()))
    del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('model',choices=['E1','ESMC']);ap.add_argument('--seeds',nargs='+',default=['trained']+[str(x) for x in range(91001,91010)]);ap.add_argument('--fresh',action='store_true');ap.add_argument('--cohort-dir');a=ap.parse_args()
    for seed in a.seeds:run(a.model,seed,a.fresh,a.cohort_dir)
