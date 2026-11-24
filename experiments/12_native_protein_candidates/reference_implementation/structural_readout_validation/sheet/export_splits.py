"""Export final D-add and H views of the result-blind structural cohort."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    marker = HERE/"SPLIT_EXPORT_READY.json"
    if marker.exists():
        return
    source = HERE/"READY_D_extra.json"
    # run_stages appends the final authorisation and freezes the updated hash.
    while True:
        if source.is_file():
            ready = json.loads(source.read_text(encoding="utf-8-sig"))
            if ready.get("blind_stopping") and ready.get("split_roles"):
                break
        time.sleep(10)
    assert ready["status"] == "READY_FOR_SCORING"
    assert sha(HERE/"COHORT_FREEZE_D_extra.json") == ready["freeze_sha256"]
    pf, cf = HERE/"pairs_D_extra.tsv", HERE/"chains_D_extra.jsonl"
    assert sha(pf) == ready["pairs_sha256"] and sha(cf) == ready["chains_sha256"]
    with pf.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields = reader.fieldnames
        rows = list(reader)
    chains = [json.loads(line) for line in cf.read_text(encoding="utf-8-sig").splitlines()]
    assert {r["split"] for r in rows} == {"D", "H"}
    saved_pairs, saved_chains = {}, {}
    outputs = {}
    for label, dirname in (("D", "D_add"), ("H", "H")):
        dest = HERE/dirname
        dest.mkdir(exist_ok=True)
        subset = [r for r in rows if r["split"] == label]
        seqs = [r for r in chains if r["split"] == label]
        assert {r["record_id"] for r in subset} == {r["record_id"] for r in seqs}
        pair_out, chain_out = dest/"pairs.tsv", dest/"chains.jsonl"
        with pair_out.open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(subset)
        chain_out.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in seqs),encoding="utf-8")
        with pair_out.open(encoding="utf-8", newline="") as inp:
            for r in csv.DictReader(inp, delimiter="\t"):
                assert r["pair_id"] not in saved_pairs
                saved_pairs[r["pair_id"]] = r
        for line in chain_out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            assert r["record_id"] not in saved_chains
            saved_chains[r["record_id"]] = r
        outputs[label] = dict(directory=dirname, pairs=len(subset), chains=len(seqs),
                              groups=len({r["group"] for r in seqs}), pairs_sha256=sha(pair_out),
                              chains_sha256=sha(chain_out))
    assert saved_pairs == {r["pair_id"]:r for r in rows}
    assert saved_chains == {r["record_id"]:r for r in chains}
    record = dict(utc=datetime.now(timezone.utc).isoformat(),model_scores_read=False,
                  export_matches_frozen_union=True,outputs=outputs,
                  source_freeze_sha256=ready["freeze_sha256"],blind_stopping=ready["blind_stopping"])
    tmp = marker.with_suffix(".partial")
    tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
    tmp.replace(marker)
    print(outputs, flush=True)


if __name__ == "__main__":
    main()
