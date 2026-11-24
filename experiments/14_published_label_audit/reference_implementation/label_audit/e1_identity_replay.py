"""Replay the four original same-layer E1 identities from archived F arrays."""
from prepare import *
from inventory import table
import numpy as np
def main():
    paths={2192:F/'fresh/unit2192_per_native_sequence.npz',1900:F/'fresh/unit1900_per_native_sequence.npz',735:old/'F_unit735.npz',2062:old/'F_unit2062.npz'}
    stored=table(old/'e1_F_groups.tsv');pp=table(F/'fixed_axes/F_pairs.tsv');annotations={p['pair_id']:p for p in table(F/'fresh/pairs.tsv')}
    for p in pp:
        a=annotations[p['pair_id']];assert p['index_positive']==a['index_positive'] and p['index_negative']==a['index_negative']
        if p['question']=='interior_vs_start':
            assert a['task']=='helix_start_vs_internal' and a['same_helix_segment']=='True' and int(a['positive_helix_offset0'])<2 and int(a['negative_helix_offset0'])>=3 and int(a['negative_helix_distance_to_end'])>=3
        else:assert a['task']=='helix_vs_strand' and a['positive_secondary']=='H' and a['negative_secondary']=='E'
    checks=[]
    for unit,path in paths.items():
        with np.load(path,allow_pickle=False) as z:
            assert len(z.files)==20
            for row in stored:
                if int(row['unit0'])!=unit:continue
                ti=tasks.index(row['question']);pairs=[p for p in pp if p['group']==row['group'] and p['question']==row['question']]
                wins=[]
                for p in pairs:
                    v=z[p['sequence_sha256']];d=signs[unit][ti]*(float(v[int(p['index_positive'])])-float(v[int(p['index_negative'])]));wins.append(float(d>0)+.5*(d==0))
                result=float(np.mean(wins));checks.append(dict(unit0=unit,group=row['group'],question=row['question'],replayed=result,archived=float(row['accuracy']),abs_error=abs(result-float(row['accuracy']))))
    error=max(r['abs_error'] for r in checks);assert error<=1e-12
    write('E1_F_IDENTITY_REPLAY.json',{'status':'PASS','utc':datetime.now(timezone.utc).isoformat(),'max_abs_error':error,'group_rows_replayed':len(checks),'source_paths_sha256':{str(p):sha(p) for p in list(paths.values())+[old/'e1_same_layer_F.py',old/'Scientific narrative is documented in PROTOCOL.md.',old/'e1_F_groups.tsv',F/'fixed_axes/F_pairs.tsv',F/'fresh/pairs.tsv']},'identities':{'2192':{'block0':14,'sign_start':-1,'sign_helix_vs_beta':1},'735':{'block0':14,'sign_start':-1,'sign_helix_vs_beta':-1},'2062':{'block0':19,'sign_start':-1,'sign_helix_vs_beta':1},'1900':{'block0':19,'sign_start':-1,'sign_helix_vs_beta':1}},'literal_task_name_issue':'interior_vs_start is a historical string only: its index_positive is the helix START, proven against positive_helix_offset0<2 and same_helix rows; preserve negative signs for START-positive new definition.','prior_F20_already_used_by_all_four':True,'checks':checks})
    print('E1 replay PASS',len(checks),error,flush=True)
if __name__=='__main__':main()
