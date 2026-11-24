"""Resume the frozen two-model cohort study at its first unfinished stage."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

SCAN = Path(__file__).resolve().parent
STUDY = SCAN.parent
SHEET = STUDY / "sheet"
MERGED = SCAN / "merged_cohorts"


def run(script, *arguments):
    command = [sys.executable, "-B", "-u", str(script), *map(str, arguments)]
    print(time.strftime("%Y-%m-%d %H:%M:%S"), script.name, *arguments, flush=True)
    subprocess.run(command, check=True)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    flags = [SHEET / "READY_D_extra.json", SHEET / "SPLIT_EXPORT_READY.json"]
    while not all(path.exists() for path in flags):
        time.sleep(15)
    ready = read(flags[0])
    assert ready["freeze_sha256"].lower() == digest(SHEET / "COHORT_FREEZE_D_extra.json")
    exports = read(flags[1])
    assert exports, "D/H split export must be verified before selection"
    if not (MERGED / "READY.json").exists():
        run(SCAN / "merge_cohorts.py", "--base", SHEET, "--D-add", SHEET/"D_add",
            "--H", SHEET/"H", "--out", MERGED)
    binding = read(MERGED/"READY.json")
    assert binding["pairs_sha256"] == digest(MERGED/"pairs.tsv")
    assert binding["chains_sha256"] == digest(MERGED/"chains.jsonl")
    print("Final structural denominators:", binding["counts_by_split"], flush=True)

    # Each native and reference selects only on D. Run the two trained models
    # first so the published-channel lookup can start while references run.
    for model in ("E1", "ESMC"):
        run(SCAN/"measure.py", model, "--seeds", "trained", "--cohort-dir", MERGED)
        run(SCAN/"measure.py", model, "--seeds", "trained", "--fresh", "--cohort-dir", SHEET)
    for model in ("E1", "ESMC"):
        seeds = [str(n) for n in range(91001, 91010)]
        run(SCAN/"measure.py", model, "--seeds", *seeds, "--cohort-dir", MERGED)
        run(SCAN/"measure.py", model, "--seeds", *seeds, "--fresh", "--cohort-dir", SHEET)
    run(SCAN/"analyse.py")
    run(STUDY/"strand_reference"/"run_reference.py", "legacy",
        "--development-chains", MERGED/"chains.jsonl")
    run(STUDY/"strand_reference"/"run_reference.py", "fresh",
        "--fresh-chains", STUDY/"fresh"/"chains.jsonl")
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
