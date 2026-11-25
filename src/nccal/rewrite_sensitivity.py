"""Recount frozen replacement and rewritten-fit criteria by coordinate basis."""
from pathlib import Path
import argparse
import collections
import csv
import json
import math

ROOT=Path(__file__).resolve().parents[2]
INPUT='results/artwork/use/basis_rewrite_direction_inputs.tsv'


def finite(value):
    try:return math.isfinite(float(value))
    except (ValueError,TypeError):return False


def recount(root=ROOT):
    with (Path(root)/INPUT).open(encoding='utf-8',newline='') as stream:
        rows=list(csv.DictReader(stream,delimiter='\t'))
    identities=[(r['basis'],r['unit']) for r in rows]
    if len(identities)!=len(set(identities)):
        raise ValueError('Repeated basis/unit record')
    groups=collections.defaultdict(list)
    for row in rows:groups[row['basis']].append(row)
    output=[]
    for basis,observations in groups.items():
        subsets=[('full',observations)]
        if basis.startswith('sae_'):
            top=[r for r in observations if r['top464_by_validation_variance_contribution']=='True']
            if len(top)!=464:raise ValueError('Frozen variance subset must contain 464 features')
            subsets.append(('top464_by_validation_variance_contribution',top))
        for subset,data in subsets:
            usable=[r for r in data if r['canonical_usable']=='True']
            retained=[r for r in usable if r['retained']=='True']
            dual=[r for r in retained if finite(r['rewritten_R2']) and float(r['rewritten_R2'])>=.5]
            output.append(dict(basis=basis,subset=subset,physical_n=len(data),finite_canonical_R2=sum(finite(r['canonical_R2']) for r in data),finite_rewritten_R2=sum(finite(r['rewritten_R2']) for r in data),canonical_usable_n=len(usable),retained=len(retained),retained_and_rewritten_R2_ge_05=len(dual),retained_missing_rewritten_R2=sum(not finite(r['rewritten_R2']) for r in retained),canonical_usable_fraction=len(usable)/len(data),retained_fraction_all_directions=len(retained)/len(data),dual_fraction_all_directions=len(dual)/len(data)))
    return output


def validate_archived_main_counts(root=ROOT):
    root=Path(root)
    with (root/'results/artwork/use/5c_and_s_s1_basis_summary.tsv').open(encoding='utf-8-sig',newline='') as stream:
        archived=list(csv.DictReader(stream,delimiter='\t'))
    calculated=recount(root)
    for row in calculated:
        subset='full' if row['subset']=='full' else 'top464'
        candidates=[r for r in archived if r['basis']==row['basis'] and (r['subset']=='full' if subset=='full' else r['subset']!='full')]
        if len(candidates)!=1:raise ValueError('Nonunique archived basis/subset')
        for key in ['physical_n','canonical_usable_n','retained']:
            if int(candidates[0][key])!=row[key]:raise ValueError('Archived main count differs: '+repr((row['basis'],subset,key)))
    return calculated


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    values=validate_archived_main_counts()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(values,ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(bases=len({r['basis'] for r in values}),basis_subsets=len(values),archived_main_counts_match=True)))
