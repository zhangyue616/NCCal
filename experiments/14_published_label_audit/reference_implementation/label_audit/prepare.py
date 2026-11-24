"""Read-only archive replay and public label metadata; no model evaluation."""
from pathlib import Path
import argparse,csv,hashlib,json,math,time,sys
from datetime import datetime,timezone
sys.dont_write_bytecode=True
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,obj):
    p=OUT/p;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def replay():
    refs=[('814',ROOT/'exposure/results/35m/buried_exposed_group_errors.tsv',js(ROOT/'exposure/results/35m/results.json')['buried_exposed_pairs']),
          ('4566',ROOT/'scale/trained_650m/b16/native/buried_exposed_group_errors.tsv',js(ROOT/'scale/trained_650m/TRAINED_RESULTS.json')['blocks']['16']['native_result'])]
    results={}
    for unit,path,ref in refs:
        fullref=js(ROOT/'exposure/results/35m/results.json') if unit=='814' else ref
        with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f,delimiter='\t'))
        assert len(rows)==967==len({r['group'] for r in rows})
        a=[];b=[];row_errors=[]
        for r in rows:
            y=int(float(r.get('later_residue_exposed',r.get('target'))));pa=float(r['sequence_probability']);pb=float(r['augmented_probability'])
            ea=(pa-y)**2;eb=(pb-y)**2
            row_errors.extend([abs(ea-float(r['sequence_brier'])),abs(eb-float(r['augmented_brier'])),abs((ea-eb)-float(r.get('improvement',r.get('gain'))))]);a.append(ea);b.append(eb)
        av=math.fsum(a)/len(a);bv=math.fsum(b)/len(b);gain=math.fsum(x-y for x,y in zip(a,b))/len(a)
        if 'buried_exposed_pairs' in ref:ref=ref['buried_exposed_pairs']
        if 'brier_gain' in ref:ref=dict(ref,improvement=ref['brier_gain'])
        checks={k:{'replayed':v,'archived':ref[k],'abs_error':abs(v-ref[k])} for k,v in [('sequence_brier',av),('augmented_brier',bv),('improvement',gain)]}
        err=max(row_errors+[r['abs_error'] for r in checks.values()])
        results[unit]={'groups':len(rows),'source':str(path),'source_sha256':sha(path),'max_abs_error':err,'checks':checks,'PASS':err<=1e-12}
        continuous=path.parent/'group_errors.tsv'
        with continuous.open(encoding='utf-8-sig',newline='') as f:cr=list(csv.DictReader(f,delimiter='\t'))
        assert len(cr)==1099==len({r['group'] for r in cr})
        ca=[float(r['sequence_mae']) for r in cr];cb=[float(r['augmented_mae']) for r in cr]
        diffs=[a-b for a,b in zip(ca,cb)]
        rowerr=max(abs(d-float(r.get('gain',r.get('sequence_minus_augmented')))) for d,r in zip(diffs,cr))
        gainkey='group_mae_difference' if unit=='814' else 'mae_gain'
        cc={k:{'replayed':v,'archived':fullref[k],'abs_error':abs(v-fullref[k])} for k,v in [('sequence_mae',math.fsum(ca)/len(ca)),('augmented_mae',math.fsum(cb)/len(cb)),(gainkey,math.fsum(diffs)/len(diffs))]}
        ce=max([rowerr]+[r['abs_error'] for r in cc.values()]);results[unit]['continuous_RSA']={'groups':1099,'source':str(continuous),'source_sha256':sha(continuous),'checks':cc,'max_abs_error':ce,'coverage':'every archived group MAE and mean; no new residue prediction recomputed'}
        results[unit]['max_abs_error']=max(err,ce);results[unit]['PASS']=max(err,ce)<=1e-12
    write('ARCHIVE_REPLAY.json',{'status':'PASS' if all(r['PASS'] for r in results.values()) else 'STOP_SECTION6_2','utc':datetime.now(timezone.utc).isoformat(),'coverage':'Independent recalculation from all 967 group targets/probabilities: every squared error and mean Brier; no new fit/forward/bootstrap. Archived full-selector replay retained as separate historical evidence.','units':results})
    print(json.dumps(results,indent=2))
def discover():
    import requests
    endpoints={'hallee_tree':'https://huggingface.co/api/datasets/GleghornLab/esmc_native_units_results/tree/main?recursive=true&limit=1000',
       'banerjee_tree':'https://api.github.com/repos/arjun-banerjee/PLMNeuron/git/trees/main?recursive=1',
       'hallee_info':'https://huggingface.co/api/datasets/GleghornLab/esmc_native_units_results'}
    for name,url in endpoints.items():
        st=time.time();r=requests.get(url,timeout=45);r.raise_for_status();data=r.json();write('sources/'+name+'.json',{'url':url,'retrieved_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.time()-st,'response':data})
        paths=[x.get('path') for x in (data if isinstance(data,list) else data.get('tree',data.get('siblings',[]))) if isinstance(x,dict)]
        print(name,len(paths));print('\n'.join(p for p in paths if p and any(t in p.lower() for t in ['per_concept','label','explan','annot','csv','parquet'])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['replay','discover']);a=p.parse_args();globals()[a.mode]()
