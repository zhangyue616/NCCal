"""One consolidated acceptance and descriptive same-helix report; no model rerun."""
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parent
C = P.parent / "cohort"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
records = [json.loads(s) for s in (C / "chains.jsonl").read_text(encoding="utf-8-sig").splitlines() if s.strip()]
cohort_pairs = pd.read_csv(C / "pairs.tsv", sep="\t")
ready = json.loads((C / "ACCEPTANCE.json").read_text())
assert ready["status"] == "PASS"
assert sha(P / "LOCKED_PROTOCOL.md") == json.loads((P / "PRE_FORWARD_LOCK_RECEIPT.json").read_text())["protocol_sha256"]
assert sha(P / "HALLEE786_PROTOCOL.md") == json.loads((P / "HALLEE_HELIX_LOCK.json").read_text())["protocol_sha256"]
assert sha(C / "chains.jsonl") == "d26c72c7c5f29a3c03f13a20d2f4adfa203c5f0d02dd83a55440a4ae86109dcb"
assert sha(C / "pairs.tsv") == "bc1054844aaef86d04b2482a50fc1dbddcdfd4f879b20e17012bcffe93356929"
model_sha = next(x["sha256"] for x in json.loads((P / "HALLEE786_INPUT_MANIFEST.json").read_text()) if x["path"].endswith("model.safetensors"))
assert model_sha == "24c5fa474c48f3b754b86efe752d5f189d2bcd88190fa2270fc92b2ef3034189"
result_names = [("UniRep1900 hidden341", "RESULTS.json", "pair_results.tsv", "hidden341_per_chain.npz"),
                ("Hallee ESM2 layer5 positive786", "HALLEE786_RESULTS.json", "hallee786_pair_results.tsv", "hallee786_positive_per_chain.npz")]
summary_rows = []
sensitivity_groups, sensitivity_rows = [], []
checks = []
for name, result_file, pair_file, act_file in result_names:
    result = json.loads((P / result_file).read_text())
    pairs = pd.read_csv(P / pair_file, sep="\t")
    assert list(pairs.pair_id) == list(cohort_pairs.pair_id)
    with np.load(P / act_file) as acts:
        assert set(acts.files) == {r["record_id"] for r in records}
        for r in records:
            assert len(acts[r["record_id"]]) == len(r["sequence"])
            assert np.isfinite(acts[r["record_id"]]).all()
    assert len(pairs) == 2593
    checks.append(dict(model=name, n_chains=len(records), n_pairs=len(pairs), result_validation=result["validation"]))
    for row in result["summaries"]:
        summary_rows.append(dict(model=name, **row))
    subset = pairs[(pairs.task == "helix_start_vs_internal") & pairs.same_helix_segment.astype(str).eq("True")].copy()
    gs = subset.groupby(["split", "group"]).agg(n_pairs=("delta", "size"),mean_start_minus_interior=("delta", "mean"),fraction_start_higher_half_ties=("correct_positive_direction", "mean")).reset_index()
    gs.insert(0, "model", name)
    sensitivity_groups.append(gs)
    for split, ss in subset.groupby("split"):
        gg = gs[gs.split == split]
        sensitivity_rows.append(dict(model=name,split=split,n_groups=len(gg),n_pairs=len(ss),
            n_start_higher=int((ss.delta > 0).sum()),n_start_lower=int((ss.delta < 0).sum()),n_ties=int((ss.delta == 0).sum()),
            pooled_mean_start_minus_interior=float(ss.delta.mean()),macro_mean_start_minus_interior=float(gg.mean_start_minus_interior.mean()),
            fraction_start_higher_half_ties=float(ss.correct_positive_direction.mean()),
            interpretation="DESCRIPTIVE_ONLY_NO_ORIGINAL_NCAP_DIRECTION"))
pd.concat(sensitivity_groups).to_csv(P / "same_helix_group_results.tsv",sep="\t",index=False)
pd.DataFrame(sensitivity_rows).to_csv(P / "same_helix_summary.tsv",sep="\t",index=False)
pd.DataFrame(summary_rows).to_csv(P / "COMBINED_SUMMARY.tsv",sep="\t",index=False)
acceptance = dict(status="PASS_ENGINEERING_FIXED_UNIT_REPLICATION_COMPLETED",utc=datetime.now(timezone.utc).isoformat(),
    n_chains=len(records), n_distinct_native_sequences=len({r["sequence"] for r in records}),n_pairs=len(cohort_pairs),
    no_new_unit_search=True, models=checks,esm2_weight_matches_author_pinned_revision=True,
    cohort_bindings_match_READY=True,pre_forward_protocol_locks_match=True,
    original_UniRep_TF_or_historical_SCOP_arrays_executed=False,
    source_to_current_sample_overlap="UNKNOWN",scientific_scope="Fixed source unit and direction survive same-AA H/E transfer; no Ncap-original-claim confirmation")
(P / "ACCEPTANCE.json").write_text(json.dumps(acceptance,indent=2),encoding="utf8")

lines = ['Scientific narrative is documented in PROTOCOL.md.', "", 'Scientific narrative is documented in PROTOCOL.md.', "",
'Scientific narrative is documented in PROTOCOL.md.', "",
'Scientific narrative is documented in PROTOCOL.md.', "",
'Scientific narrative is documented in PROTOCOL.md.', 
'Scientific narrative is documented in PROTOCOL.md.', "",
'Scientific narrative is documented in PROTOCOL.md.', "",
'Scientific narrative is documented in PROTOCOL.md.', "",
'Scientific narrative is documented in PROTOCOL.md.', "",
'Scientific narrative is documented in PROTOCOL.md.', "|---|---|---|---|---|---|---|"]
for row in summary_rows:
    if row["task"] != "helix_vs_strand": continue
    lo,hi=row["macro_fraction_positive_group_bootstrap_95ci"]
    lines.append(f"|{row['model']}|{row['split']}|{row['n_groups']}/{row['n_chains']}/{row['n_pairs']}|{row['n_positive']}/{row['n_negative']}/{row['n_ties']}|{100*row['pooled_fraction_positive']:.2f}%|{100*row['macro_fraction_positive']:.2f}% ({100*lo:.2f}–{100*hi:.2f})|{row['n_groups_mean_delta_positive']}/{row['n_groups']}|")
lines += ["", 'Scientific narrative is documented in PROTOCOL.md.', "",
'Scientific narrative is documented in PROTOCOL.md.', "",
'Scientific narrative is documented in PROTOCOL.md.', "",
'Scientific narrative is documented in PROTOCOL.md.', "|---|---|---|---|---|"]
for row in summary_rows:
    if row["task"] != "helix_start_vs_internal": continue
    lines.append(f"|{row['model']}|{row['split']}|{row['n_groups']}/{row['n_pairs']}|{row['n_positive']}/{row['n_negative']}/{row['n_ties']}|{row['macro_mean_delta']:.6f}|")
lines += ["", 'Scientific narrative is documented in PROTOCOL.md.', "", 'Scientific narrative is documented in PROTOCOL.md.', "|---|---|---|---|---|"]
for r in sensitivity_rows:
    lines.append(f"|{r['model']}|{r['split']}|{r['n_groups']}/{r['n_pairs']}|{r['n_start_higher']}/{r['n_start_lower']}/{r['n_ties']}|{r['macro_mean_start_minus_interior']:.6f}|")
lines += ["", 'Scientific narrative is documented in PROTOCOL.md.', "",
'Scientific narrative is documented in PROTOCOL.md.', "",
'Scientific narrative is documented in PROTOCOL.md.', 
'Scientific narrative is documented in PROTOCOL.md.', 
'Scientific narrative is documented in PROTOCOL.md.', 
'Scientific narrative is documented in PROTOCOL.md.', 
'Scientific narrative is documented in PROTOCOL.md.', "",
'Scientific narrative is documented in PROTOCOL.md.', "",
'Scientific narrative is documented in PROTOCOL.md.', 
'Scientific narrative is documented in PROTOCOL.md.', 
'Scientific narrative is documented in PROTOCOL.md.', 
'Scientific narrative is documented in PROTOCOL.md.', 
'Scientific narrative is documented in PROTOCOL.md.', 
'Scientific narrative is documented in PROTOCOL.md.', "",
'Scientific narrative is documented in PROTOCOL.md.']
(P / "REPORT_zh.md").write_text("\n".join(lines)+"\n",encoding="utf8")
print(json.dumps(acceptance,indent=2))
print(pd.DataFrame(sensitivity_rows).to_string(index=False))
