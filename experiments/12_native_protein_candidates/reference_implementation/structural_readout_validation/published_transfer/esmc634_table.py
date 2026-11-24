"""Read the archived official E13 table; distinguish physical FFN units and poles."""
from pathlib import Path
import argparse,json,hashlib
import pandas as pd
P=Path(__file__).parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--new-axis-lock');ap.add_argument('--author-code-dir',type=Path,required=True,help='External Protify unit modules containing capture.py, bases.py and poles.py');ap.add_argument('--retest-source',type=Path,required=True,help='External author retest.py');args=ap.parse_args()
 source=P/'hallee_ESMC300_e13_archive.parquet';d=pd.read_parquet(source).reset_index().rename(columns={'index':'author_row0'})
 freeze=json.loads((P/'ESMC_STRAND_FREEZE.json').read_text(encoding='utf8'))
 assert sha(source)==freeze['author_table_sha256'].lower()
 assert sha(P/'ESMC_STRAND_PROTOCOL.md')==freeze['protocol_sha256'].lower()
 candidates=d[d.model.eq('esmc_300')&d.concept.eq('Beta strand')&d.arm.eq('mlp_positive')].sort_values(['best_test_f1','layer','best_feature'],ascending=[False,True,True])
 winner=candidates.iloc[0].to_dict()
 assert int(winner['layer'])==29 and int(winner['best_feature'])==634
 candidates.to_csv(P/'esmc634_author_beta_candidates.tsv',sep='\t',index=False)
 direct=d[d.arm.isin(['mlp_positive','mlp_poles','mlp_native'])].copy();width=2560
 direct['physical_unit0']=direct.best_feature.astype(int)%width
 direct['pole']=['positive'if f<width else'negative'for f in direct.best_feature]
 direct['direct_readout']=['ReLU(z)'if a=='mlp_positive'else 'ReLU(z),ReLU(-z)'if a=='mlp_poles'else'gate-open sign-tagged SwiGLU poles'for a in direct.arm]
 direct['physical_block0']=direct.layer.astype(int)
 direct.to_csv(P/'esmc634_author_direct_reads.tsv',sep='\t',index=False)
 code=[args.author_code_dir/'capture.py',args.author_code_dir/'bases.py',args.author_code_dir/'poles.py',args.retest_source]
 provenance={'selection_rule':'model esmc_300, concept exactly Beta strand, arm mlp_positive; maximum author best_test_f1; tie lower layer then lower unit','winner':winner,'candidate_rows':candidates.to_dict(orient='records'),'rows_total':len(d),'concepts':int(d.concept.nunique()),'author_layers0':sorted(int(x)for x in d.layer.unique()),'arms':sorted(d.arm.unique().tolist()),'direct_readout_rows':len(direct),'direct_readout_concepts':int(direct.concept.nunique()),'width':width,'readout':'z=SiLU(gate)*up; mlp_positive=ReLU(z), not the gate-tagged mlp_native','block_index_evidence':'hallee_retest.stream_esmc_arms explicitly documents RetestBatch.layer as zero-based block; author_capture.BlockReader indexes transformer_blocks[block].','normalization':'Positive common source maximum scaling omitted: within-chain order and ties are unchanged.','source_table':str(source),'source_sha256':sha(source),'source_code_sha256':{str(p):sha(p)for p in code},'protocol_sha256':sha(P/'ESMC_STRAND_PROTOCOL.md')}
 dump(P/'esmc634_author_selection.json',provenance)
 if not args.new_axis_lock:
  dump(P/'esmc634_new_axis_status.json',{'status':'WAITING_FOR_ENLARGED_D_NATIVE_AXIS_LOCK','claim':'No zero-hit novelty claim. E13 only directly covers blocks0=7,14,22,29 and636 concept labels.'})
  return
 lockpath=Path(args.new_axis_lock);lock=json.loads(lockpath.read_text(encoding='utf8'));s=lock.get('selection',lock)
 b=int(s.get('block_0',s.get('block_1',0)-1));u=int(s.get('channel_0',s.get('unit0',-1)));sign=int(s['sign'])
 assert 0<=b<30 and 0<=u<width and sign in(-1,1)
 samephysical=direct[direct.physical_block0.eq(b)&direct.physical_unit0.eq(u)].copy()
 samephysical['new_raw_sign']=sign;samephysical['matches_selected_sign_pole']=(samephysical.pole==('positive'if sign==1 else'negative'))
 samephysical['readout_relation']='Same physical FFN coordinate; rectified/gate-tagged pole differs from raw signed SwiGLU product'
 samephysical.to_csv(P/'esmc634_new_axis_same_physical_annotations.tsv',sep='\t',index=False)
 # Across the four surveyed blocks, matching unit numbers are not identical physical neurons.
 crossblocks=direct[direct.physical_unit0.eq(u)].copy();crossblocks['same_physical_block']=crossblocks.physical_block0.eq(b)
 crossblocks.to_csv(P/'esmc634_new_axis_number_across_surveyed_blocks.tsv',sep='\t',index=False)
 summary={'status':'AUDITED_OFFICIAL_TABLE_DIRECT_READOUTS','new_axis':{'block0':b,'unit0':u,'raw_sign':sign},'native_lock_path':str(lockpath),'native_lock_sha256':sha(lockpath),'table_layers0':provenance['author_layers0'],'exact_block_surveyed':b in provenance['author_layers0'],'exact_physical_coordinate_annotations':len(samephysical),'same_number_other_blocks_annotations':int((~crossblocks.same_physical_block).sum()),'same_physical_annotations':samephysical.to_dict(orient='records'),'scope':'Only winner rows for636 concepts at four blocks; not an exhaustive catalogue of all units. mlp_positive, mlp_poles and mlp_native are joined by decoded physical unit and pole. Residual dimensions, SAE latents, rotated coordinates and shuffled-checkpoint units do not directly identify the same physical FFN coordinate.'}
 dump(P/'esmc634_new_axis_audit.json',summary)
 dump(P/'esmc634_new_axis_status.json',{'status':summary['status'],'see':'esmc634_new_axis_audit.json','exact_block_surveyed':summary['exact_block_surveyed'],'new_axis':summary['new_axis']})
if __name__=='__main__':main()
