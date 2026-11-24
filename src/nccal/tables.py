"""Rebuild the ten manuscript tables from the study design and stored analytical results."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import json
import math

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / 'results' / 'tables'

def read(name):
    return json.loads((SOURCES / (name + '.json')).read_text(encoding='utf-8'))

def decimal(x, n):
    return f'{float(x):.{n}f}'.replace('-', '−')

def limits(values, digits=None):
    lo, hi = min(values), max(values)
    fmt = str if digits is None else lambda v: decimal(v, digits)
    return fmt(lo) if lo == hi else f'{fmt(lo)}–{fmt(hi)}'

def calibration():
    rows = dict(read('calibration_statistics'))
    order = ['ChemBERTa-77M-MLM', 'ChemBERTa-77M-MTR', 'CheMeleon', 'MolCLR GIN',
             'Hu masking', 'Hu contextpred', 'Hu edgepred', 'Hu infomax', 'MoLFormer']
    out = []
    for name in order:
        a, b = rows[name]
        if name == 'MoLFormer':
            status = 'Layer-calibrated joint (maximum-over-layers axis preference test p = 0.005)'
        elif a[2] and b[2]:
            status = 'Family-corrected joint'
        elif max(a[1],b[1]) <= .05:
            status = 'Nominal joint only'
        elif a[2]:
            status = 'Neuron readability test only'
        elif b[2]:
            status = 'Axis preference test only'
        else:
            status = 'Neither criterion'
        label = 'MoLFormer (layer 10)' if name=='MoLFormer' else name
        out.append([label,format(math.exp(a[0]),'.3g'),decimal(a[1],3),
                    format(math.exp(b[0]),'.3g'),decimal(b[1],3),status])
    return out

def exposure():
    out = []
    for r in read('protein_losses'):
        for p in (['','b17_'] if r['model']=='650m' else ['']):
            diff = lambda k: float(r[p+'sequence_'+k])-float(r[p+'augmented_'+k])
            out.append([r['model'].upper()+(' sensitivity' if p else ''),
                        f"{17 if p else r['primary_block']} / {r[p+'axis']}",
                        decimal(diff('mae'),6),decimal(diff('brier'),6),r[p+'pair_correct']])
    return out

def focal():
    r={(a['kind'], a['coordinate'], a['condition']):a for a in read('two_neuron_metrics')}
    out=[]
    for id,name in [('293','Hydroxyl 293 / 115'),('454','Carbonyl 454 / 124')]:
        b=r['activation',id,'M1']; x=r['activation',id,'M2']
        out.append([name,decimal(b['estimate'],3),
                    f"{decimal(x['estimate'],3)} [{decimal(x['ci95_lower'],3)}, {decimal(x['ci95_upper'],3)}]",
                    decimal(r['output',id,'D_mean']['estimate'],5),decimal(r['output',id,'M2']['estimate'],5)])
    return out

def layer():
    out=[]
    for model,label in [('chemberta_mtr','ChemBERTa-MTR'),('molformer','MoLFormer')]:
        for trained in [True,False]:
            rows=[r for r in read('layer_statistics') if r['model']==model and (r['network']=='trained')==trained]
            out.append([label+(' trained' if trained else ' untrained')]+
                       [limits([float(r[k]) for r in rows],4) for k in ['VE_radius0','VE_local_graph','VE_local_minus_radius0']])
    return out

def rewrites():
    out=[]
    for name,rows in zip(['ChemBERTa','MoLFormer BACE','MoLFormer ESOL'],read('rewrite_status')):
        counts=Counter(r['rewritten_status'] for r in rows if r['canonical_usable']=='True')
        out.append([name,str(len(rows)),str(sum(counts.values()))]+[str(counts[k]) for k in ['retained','failed','undetermined']])
    return out

def edits():
    source={(r['unit'],r['edit_class']):r for r in read('edit_intervals')}
    out=[]
    for unit in ['293','454']:
        values=[]
        for edit in range(1,5):
            r=source[unit,str(edit)]
            passes=float(r['mean_advantage_lo'])>0 and float(r['swap_advantage_lo'])>0
            if passes!=(r['better_than_both']=='True'):
                raise ValueError('Inconsistent interval decision')
            values.append('Yes' if passes else 'No')
        out.append([unit]+values)
    return out

def bases():
    buckets={}
    for r in read('basis_statistics'):
        group=r['basis'].split('_')[0]
        if r['subset']!='full': group='top464'
        buckets.setdefault(group,[]).append(r)
    out=[]
    for group,name in [('native','Native'),('rot','Rotations'),('pca','Principal'),('sae','Dictionaries'),('top464','Top-464 dictionaries')]:
        rows=buckets[group]
        out.append([name,limits([int(r['physical_n']) for r in rows]),limits([int(r['retained']) for r in rows])])
    return out

def labels():
    out=[]
    for frame, name in [('T','Confirmation'),('NEW','Reserved/new')]:
        eligible=[r for r in read('public_label_flags') if r['frame']==frame and r['formal_beta_gamma_eligible']=='True']
        out.append([name,str(len(eligible))]+[str(sum(r[k]=='True' for r in eligible)) for k in ['same_AA_flag','strong_increment_flag']])
    return out

def build():
    transfer=[[name,decimal(r[1],3),f'[{decimal(r[2],3)}, {decimal(r[3],3)}]']
              for r,name in zip(read('transfer_statistics'),['Axis preference (Θ)','Readability ( G init )'])]
    return {'MS1':read('model_study_design'),'MS2':calibration(),'SI1':transfer,'SI2':exposure(),
            'SI3':focal(),'SI4':layer(),'SI5':rewrites(),'SI6':edits(),'SI7':bases(),'SI8':labels()}

def check(output=None):
    expected=json.loads((ROOT/'verify/table_display.json').read_text(encoding='utf-8'))
    errors=[]
    for key,rows in build().items():
        if output:
            output.mkdir(exist_ok=True,parents=True)
            with (output/f'{key}.tsv').open('w',encoding='utf-8',newline='') as s:
                csv.writer(s,delimiter='\t').writerows([expected[key][0]]+rows)
        target=expected[key][1:]
        if len(rows)!=len(target):
            errors.append((key,'row count',len(target),len(rows)))
        for i,(rx,ry) in enumerate(zip(rows,target),1):
            if len(rx)!=len(ry): errors.append((key,i,'column count'))
            for j,(got,want) in enumerate(zip(rx,ry),1):
                if str(got)!=want: errors.append((key,i,j,want,got))
    return errors

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    issues=check(args.out)
    print(json.dumps({'tables':10,'model_inventory':'documentary','data_tables':9,'differences':issues},ensure_ascii=True))
    raise SystemExit(bool(issues))

