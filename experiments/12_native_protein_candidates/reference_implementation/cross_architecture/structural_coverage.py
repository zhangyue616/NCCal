"""Coverage and overlap counts from experimental data only; no model predictions."""
import json
from pathlib import Path
from cohorts import ROOT,OUT,read_tsv
D=ROOT/"external"
h=json.loads((ROOT/"validation"/"HOMOLOGY.json").read_text())
kept=set(h["retained_groups"])
coverage={}
for path in (D/"groups").glob("*.json"):
    r=json.loads(path.read_text(encoding="utf-8-sig"))
    sites=r.get("residues",[])
    if sites: coverage[r["group"]]=int(r["mapped_exact_observed"])/len(r["sequence"])
rows=list(read_tsv(OUT/"E_contact.tsv"))
groups_contact={r["group"] for r in rows}
g_exposure={r["group"] for r in read_tsv(OUT/"E_exposure.tsv")}
high={g for g,p in coverage.items() if p>=.8}
result=dict(cap_fraction=.8,threshold_record="Pre-external-prediction structural coverage sensitivity",covered_contact=len(groups_contact & high),screened_contact=len(groups_contact & kept),screened_exposure=len(g_exposure & kept),covered_and_screened=len(groups_contact&kept&high),high_groups=sorted(high))
(ROOT/"validation"/"COVERAGE.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
print({k:v for k,v in result.items() if k!="high_groups"})
