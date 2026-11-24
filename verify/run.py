"""Recalculate tabulated results and report coverage of numbers in the article."""
from pathlib import Path
import argparse,collections,csv,json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/nccal'))
import tables
from metrics import compute
from article_claims import sources
def save(path,data):
    path.write_text(json.dumps(data,ensure_ascii=True,indent=2),encoding='utf-8')
def verify(out,with_experiments):
    out.mkdir(exist_ok=True,parents=True)
    metrics=compute()
    claim_values=sources(ROOT)
    grids=tables.build();terr=tables.check(out/'tables')
    summary={'tables':{'n':len(grids),'documentary':1,'calculated_or_assembled':9,'differences':terr}}
    prior=json.loads((ROOT/'verify/trace_expected.json').read_text())
    raw={};trace=[]
    for r in prior:
        k=r['input']
        if k not in raw:raw[k]=json.loads((ROOT/'results/tables/trace_inputs'/(k+'.json')).read_text())
        v=raw[k][r['selector']]
        ok=math.isclose(v,r['expected'],rel_tol=1e-6,abs_tol=1e-11)
        trace.append([r['id'],k,r['selector'],r['expected'],v,'MATCH' if ok else 'DIFFERENT'])
    with (out/'keyed-trace.tsv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream,delimiter='\t');writer.writerow(['item','source','key','expected','computed','result']);writer.writerows(trace)
    summary['keyed_archived_intermediates']={'compared':len(trace),'differences':sum(r[-1]!='MATCH' for r in trace)}
    words={k:v for v,k in enumerate('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split())}
    words.update({'twenty':20,'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,'eighty':80,'ninety':90,'hundred':100,'half':.5,'twice':2})
    notation_roles = {
        'affiliation or model metadata', 'symbol, model, or record identifier',
        'identifier or cross-reference', 'equation or test identifier', 'citation identifier',
        'indexing convention', 'chemical or statistical nomenclature', 'mathematical notation',
        'availability or authorship metadata', 'bibliographic metadata',
        'zero-based block and coordinate identifier', 'section identifier'
    }
    outcomes=collections.Counter();roles=collections.Counter()
    with (ROOT/'verify/paper_numbers.tsv').open(encoding='utf-8',newline='') as f,(out/'number-coverage.tsv').open('w',encoding='utf-8',newline='') as stream:
        src=csv.DictReader(f,delimiter='\t'); dest=csv.writer(stream,delimiter='\t')
        dest.writerow([*src.fieldnames,'actual','status'])
        for row in src:
            roles[row['role']]+=1; result='';v=''
            if row['role'] in ('bound numerical result','study evidence selection'):
                label=row['display'].lower().replace('−','-').replace(',','')
                truth=words[label] if label in words else sum(words[k] for k in label.split('-')) if label[:1].isalpha() else float(label)
                decimals=len(label.split('.')[1]) if '.' in label else 1 if label=='half' else 0
                if row['role']=='study evidence selection':
                    record=claim_values[row['id']]
                    source=record['value']
                    if record['source']!=row['source'] or record['operation']!=row['field_or_operation']:
                        raise ValueError('Changed evidence definition for '+row['id'])
                else:
                    source=metrics[row['field_or_operation']]
                v=source*float(row['scale'])
                result='MATCH' if abs(v-truth)<=.5*10**(-decimals)+1e-12 else 'DIFFERENT'
            elif row['role']=='checked table cell':
                k=row['field_or_operation']; r=int(row['table_row']);c=int(row['table_column'])
                v=grids[k][r-1][c]
                wrong=any(e[0]==k and (len(e)<4 or e[1]==r and e[2]==c+1) for e in terr)
                result='DIFFERENT' if wrong else 'MATCH_TABLE_CELL'
            elif row['role']=='documentary table cell':
                result='DOCUMENTARY'
            elif row['role'] in ('editorial count','explanatory example'):
                result='EXPOSITION'
            elif row['role'] in notation_roles:
                result='IDENTIFIER_OR_NOTATION'
            else:
                result='UNVERIFIED'
            dest.writerow([*row.values(),v,result]);outcomes[result]+=1
    summary['numeric_positions']=dict(outcomes);summary['position_roles']=dict(roles)
    errors=bool(terr) or summary['keyed_archived_intermediates']['differences']>0 or outcomes['DIFFERENT']>0
    if with_experiments:
        checks={}
        for script in sorted((ROOT/'experiments').glob('*/reproduce.py')):
            name=script.parent.name; start=time.perf_counter()
            dest=out/(name+'.json')
            if dest.exists():dest.unlink()
            cmd=[sys.executable,'-B',str(script),'--mode','verify','--output',str(dest)]
            proc=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace',env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONUTF8':'1'})
            (out/(name+'.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
            checks[name]={'exit':proc.returncode,'receipt':dest.is_file(),'seconds':round(time.perf_counter()-start,3)}
            errors|=proc.returncode!=0 or not dest.is_file()
        summary['branches']=checks;errors|=len(checks)!=14
    summary['numerical_differences_or_failed_runs']=bool(errors)
    summary['all_display_positions_accounted_for']=not errors and not outcomes['UNVERIFIED']
    summary['validation_scope']='Stored-result calculations and evidence fields, table cells, documentary quantities and notation are distinguished in number-coverage.tsv. This replay does not repeat all model fits from public inputs.'
    save(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=True))
    return 1 if errors else 2 if outcomes['UNVERIFIED'] else 0
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--experiments',action='store_true')
    args=parser.parse_args();sys.exit(verify(args.out.resolve(),args.experiments))

