"""Rebuild the full frozen-cohort chemical pair inputs from external public structures.

No model weights or model forward are used. Required tokenizer directories are
obtained separately from the public providers. Outputs are explicit external files.
"""
from pathlib import Path
import argparse,csv,gzip,hashlib,json,sys,collections
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from nccal.molecular.pair_preparation import prepare_one
from nccal.molecular.semantic_mapper import SemanticMapper
from nccal.molecular.mapping import AtomMapper
from nccal.molecular.standardization import standardize

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--fixed-cohort',type=Path,default=ROOT/'data/molecular/evaluation_ids.tsv');p.add_argument('--chemberta-tokenizer-dir',type=Path,required=True);p.add_argument('--molformer-tokenizer-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--limit',type=int);p.add_argument('--standardize-source',action='store_true');args=p.parse_args()
    with args.fixed_cohort.open(encoding='utf-8-sig') as f:cohort=list(csv.DictReader(f,delimiter='\t'))
    if args.limit is not None:
        if args.limit<1:raise ValueError('The engineering subset size must be positive.')
        cohort=cohort[:args.limit]
    with args.input.open(encoding='utf-8-sig') as f:source=list(csv.DictReader(f,delimiter='\t'))
    by_cid={int(r['cid']):r for r in source}
    if len(by_cid)!=len(source):raise ValueError('Duplicate source CIDs are not accepted.')
    cb=SemanticMapper(args.chemberta_tokenizer_dir);mf=AtomMapper('molformer',asset_dir=args.molformer_tokenizer_dir)
    manifest=json.loads((ROOT/'data/molecular/pair_tokenizer_identity.json').read_text())
    for kind,folder in [('chemberta',args.chemberta_tokenizer_dir),('molformer',args.molformer_tokenizer_dir)]:
        actual=hashlib.sha256((folder/'tokenizer.json').read_bytes()).hexdigest();expected=manifest[kind]['sha256']
        if actual!=expected:raise ValueError(f'{kind} tokenizer identity mismatch: expected {expected}; observed {actual}')
    with gzip.open(ROOT/'data/molecular/excluded_identities.jsonl.gz','rt',encoding='utf8') as f:old_structure_ids={r['canonical_smiles_sha256'] for r in map(json.loads,f) if 'canonical_smiles_sha256' in r}
    args.output_dir.mkdir(parents=True,exist_ok=True);counts=collections.Counter();records=[]
    with gzip.open(args.output_dir/'pair_inputs.jsonl.gz','wt',encoding='utf8') as output:
        for fixed in cohort:
            cid=int(fixed['cid'])
            if cid not in by_cid:raise ValueError(f'Missing public input CID {cid}.')
            row=by_cid[cid];text=row.get('canonical_smiles') or row.get('smiles')
            if not text:raise ValueError(f'No source structure for CID {cid}.')
            if args.standardize_source:
                text,reason=standardize(text)
                if reason:raise ValueError(f'CID {cid} source qualification changed: {reason}')
            observed=hashlib.sha256(text.encode()).hexdigest();expected=fixed['canonical_smiles_sha256']
            if observed!=expected:raise ValueError(f'CID {cid} canonical input mismatch: expected {expected}; observed {observed}')
            record,status=prepare_one(cid,text,cb,mf);records.append(status);counts['parent_molecules']+=1
            if record is None:counts['excluded:'+status['reason'].split(':')[0]]+=1
            else:
                record['product_in_old_structure_pool']=hashlib.sha256(record['canonical_product_smiles'].encode()).hexdigest() in old_structure_ids
                output.write(json.dumps(record,separators=(',',':'))+'\n');counts['qualified_pairs']+=1;counts['edit_class_'+str(record['edit_class'])]+=1;counts['controlled_string_primary']+=int(record['main_string_control_pass']);counts['canonical_only_secondary']+=int(not record['main_string_control_pass'])
    with gzip.open(args.output_dir/'pair_eligibility.jsonl.gz','wt',encoding='utf8') as output:
        for record in records:output.write(json.dumps(record,separators=(',',':'))+'\n')
    summary=dict(status='INPUT_REPLAY_COMPLETE',partial_engineering_subset=args.limit is not None,counts=dict(counts),model_forwards=0,public_source_identity_checked=True,cohort_manifest_sha256=hashlib.sha256(args.fixed_cohort.read_bytes()).hexdigest(),product_history_overlap='Recalculated against the frozen canonical-identity exclusion manifest.')
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
