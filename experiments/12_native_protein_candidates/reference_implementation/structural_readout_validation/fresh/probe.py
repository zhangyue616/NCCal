import json, urllib.request, sys, ssl
ctx=ssl.create_default_context()
for store in ('ROOT','CA'):
 for cert,encoding,trust in ssl.enum_certificates(store):
  if encoding=='x509_asn':
   try:ctx.load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(cert))
   except ssl.SSLError:pass
q={'query':{'type':'group','logical_operator':'and','nodes':[
 {'type':'terminal','service':'text','parameters':{'attribute':'rcsb_accession_info.initial_release_date','operator':'range','value':{'from':'2025-11-14','to':'2026-11-23'}}},
 {'type':'terminal','service':'text','parameters':{'attribute':'exptl.method','operator':'exact_match','value':'X-RAY DIFFRACTION'}},
 {'type':'terminal','service':'text','parameters':{'attribute':'rcsb_entry_info.resolution_combined','operator':'less_or_equal','value':3}},
 {'type':'terminal','service':'text','parameters':{'attribute':'rcsb_entity_source_organism.taxonomy_lineage.id','operator':'in','value':['9606','10090','4932','559292','3702']}}
]},'return_type':'entry','request_options':{'return_all_hits':True}}
from pathlib import Path
P=Path(__file__).parent
(P/'search_query.json').write_text(json.dumps(q,indent=2))
import subprocess
r=subprocess.run(['curl.exe','--silent','--show-error','--fail','--max-time','50','-H','Content-Type: application/json','--data-binary','@'+str(P/'search_query.json'),'https://search.rcsb.org/rcsbsearch/v2/query'],capture_output=True,check=True)
a=json.loads(r.stdout); print('hits',a.get('total_count')); print([x['identifier'] for x in a.get('result_set',[])][:20])
(P/'search_response.json').write_text(json.dumps(a))
