"""Record bounded public metadata searches after the D-only coordinates are locked."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"analysis"

def get_json(url, params):
    for i in range(3):
        try:
            r=requests.get(url,params=params,headers={"User-Agent":"NCCal literature lookup"},timeout=25)
            if r.status_code in (403,429):
                return {"status":r.status_code,"error":"endpoint unavailable"}
            r.raise_for_status()
            return r.json()
        except (requests.RequestException,ValueError) as error:
            if i==2:
                return {"error":str(error).split("?")[0]}
            time.sleep(1)

def run():
    a=json.loads((ROOT/"amplify/results/trained/locked.json").read_text())
    e=json.loads((ROOT/"e1/trained/selection_lock.json").read_text())["selections"]
    candidate=[("AMPLIFY",k["block_1"],k["channel_0"],k["task"]) for k in a]
    candidate += [("E1",k["layer0"]+1,k["unit0"],k["task"]) for k in e]
    searches=[]
    scope={"AMPLIFY":'("AMPLIFY-120M" OR "AMPLIFY_120M" OR "AMPLIFY 120M")',
           "E1":'("Profluent-E1" OR "Profluent E1" OR "E1-150m")'}
    for model in scope:
        topic='(neuron OR neurons OR "feed-forward channel" OR "native unit" OR "sparse autoencoder")'
        queries=[(f"{scope[model]} AND {topic} AND (helix OR helices OR strand OR structure)","class")]
        for name,block,unit,task in candidate:
            if name==model:
                queries.append((f'{scope[model]} AND ("{unit}" OR "{unit+1}") AND (helix OR helical OR strand OR neuron)',
                                f"{task}; block {block} one-based; channel {unit} zero-based"))
        for query,context in queries:
            j=get_json("https://www.ebi.ac.uk/europepmc/webservices/rest/search",{"query":query,"format":"json","pageSize":8})
            hits=[(p.get("title"),p.get("doi")) for p in j.get("resultList",{}).get("result",[])]
            searches.append(dict(service="Europe PMC full-text search where available",query=query,context=context,
                                 count=j.get("hitCount"),error=j.get("error"),hits=hits))
        # Repository issues/PRs are searchable without using an account. This is not a code search.
        slug="Profluent-AI/E1" if model=="E1" else "chandar-lab/AMPLIFY"
        query=f"repo:{slug} helix"
        j=get_json("https://api.github.com/search/issues",{"q":query,"per_page":10})
        searches.append(dict(service="official repo issues and PRs",query=query,context="model release",
                             count=j.get("total_count"),error=j.get("error"),hits=[p["title"] for p in j.get("items",[])]))
    audit=dict(utc=datetime.now(timezone.utc).isoformat(),candidates=candidate,searches=searches)
    (OUT/"priority_queries.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    for q in searches:
        print(q["service"],q["context"],q["count"],q["error"],flush=True)

if __name__=="__main__":
    run()
