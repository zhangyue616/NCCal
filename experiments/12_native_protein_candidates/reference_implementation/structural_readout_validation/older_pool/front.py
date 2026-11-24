from pathlib import Path
import json,subprocess,hashlib,sys
P=Path(__file__).resolve().parent;P.mkdir(exist_ok=True)
old=P.parent/'fresh/search_query.json';q=json.loads(old.read_text());q['query']['nodes'][0]['parameters']['value']={'from':'2010-01-01','to':'2024-12-31'};q['query']['nodes'][2]['parameters']['value']=2.5;q['return_type']='polymer_entity';q['request_options'].update(group_by={'aggregation_method':'sequence_identity','similarity_cutoff':30,'ranking_criteria_type':{'sort_by':'rcsb_entry_info.resolution_combined','direction':'asc'}},group_by_return_type='representatives',results_verbosity='minimal')
q['query']['nodes'].extend([{'type':'terminal','service':'text','parameters':{'attribute':'entity_poly.rcsb_entity_polymer_type','operator':'exact_match','value':'Protein'}},{'type':'terminal','service':'text','parameters':{'attribute':'entity_poly.rcsb_sample_sequence_length','operator':'range','value':{'from':60,'to':1000}}}])
(P/'search_query.json').write_text(json.dumps(q,indent=2)+'\n')
cmd=['curl.exe','--silent','--show-error','--fail','--compressed','--location','--max-time','180','--retry','0','https://search.rcsb.org/rcsbsearch/v2/query','-H','Content-Type: application/json','--data-binary','@'+str(P/'search_query.json'),'-o',str(P/'search_response.json')]
r=subprocess.run(cmd,capture_output=True);print('SEARCH',r.returncode,r.stderr.decode(errors='replace')[-200:],flush=True)
if r.returncode:sys.exit(1)
d=json.loads((P/'search_response.json').read_text());ids={r['identifier'].upper() for r in d['result_set']};print('ENTITIES',d['total_count'],flush=True)
cluster=P.parents[1]/'cross_architecture/external/sources/clusters-by-entity-30.txt'
if not cluster.exists():cluster=P.parents[1]/'nccal_evidence_extension/cohort/public_sources/clusters-by-entity-30.txt'
groups=[];mapped=set()
def h(s):return hashlib.sha256(s.encode()).hexdigest()
for line in cluster.read_text().splitlines():
    hit=ids.intersection(line.split())
    if not hit:continue
    gid='rcsb30-'+h(line.strip())[:24];rr=sorted(hit,key=lambda e:h('older-sheet-entity-20261123|'+e));seen=set();reps=[]
    for e in rr:
        pid=e.split('_')[0]
        if pid in seen:continue
        seen.add(pid);reps.append(e)
        if len(reps)==3:break
    groups.append({'cluster':gid,'entities':reps});mapped.update(hit)
groups.sort(key=lambda r:h('older-sheet-pool-20261123|'+r['cluster']));groups=groups[:1800]
freeze={'query_sha256':h((P/'search_query.json').read_text()),'response_sha256':hashlib.sha256((P/'search_response.json').read_bytes()).hexdigest(),'cluster_file':str(cluster),'cluster_sha256':hashlib.sha256(cluster.read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256((P/'PLAN.md').read_bytes()).hexdigest(),'total_entities':len(ids),'mapped_entities':len(mapped),'unmapped_excluded':len(ids-mapped),'cluster_cap':1800,'batches':[600,1200,1800],'groups':groups,'model_scores_read':False}
(P/'ACQUISITION_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n');print('LOCKED_CLUSTERS',len(groups),flush=True)
initial=P/'ACQUISITION_FREEZE_initial.json'
if initial.exists():
    prev=json.loads(initial.read_text());a={x['cluster']for x in prev['groups']};z={x['cluster']for x in groups};note={'reason':'Supervisor authorized same-entity Protein and length60..1000 correction; intended biological scope and truth gates unchanged','initial_selected_clusters':len(a),'revised_selected_clusters':len(z),'retained_clusters':len(a&z),'entered_clusters':len(z-a),'exited_clusters':len(a-z),'initial_groups':prev.get('search_total'),'revised_total_entities':len(ids),'revised_response_group_count':d.get('group_by_count'),'revised_query_sha256':freeze['query_sha256'],'initial_query_sha256':prev['query_sha256'],'source_cache_policy':'reuse downloaded public CIFs; reuse physical label cache only when original chosen entity remains in corrected fixed candidate list','model_scores_read':False};(P/'FRAME_REVISION.json').write_text(json.dumps(note,indent=2)+'\n');print('FRAME_REVISION',note,flush=True)
