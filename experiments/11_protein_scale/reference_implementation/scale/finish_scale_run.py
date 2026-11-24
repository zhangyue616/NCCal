"""Run the final scientific stage when every network-level job has published its outputs."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
deadline=datetime.fromisoformat(json.loads((root/"MAIN_GO.json").read_text(encoding="utf8"))["active_deadline_utc"])


def last_receipts():
    new=root/"sealed"/"main_650m"/"init_199"/"COMPLETE.json"
    return new.is_file() and (root/"trained_650m"/"COMPLETE.json").is_file() and (root/"directions"/"COMPLETE.json").is_file()


def main():
    print("Waiting for all three scientific tasks to complete.",flush=True)
    while not last_receipts():
        if datetime.now(timezone.utc)>=deadline:
            print("Active time limit reached with incomplete cohort. Inference withheld.",flush=True)
            return 2
        time.sleep(30)
    for filename in ("summarize_scale.py","verify_scale_results.py","write_scale_report.py"):
        subprocess.run([sys.executable,"-B","-u",str(root/filename)],check=True,cwd=root)
    print("Completed",root/"scale_results"/"RESULT_zh.md",flush=True)
    return 0


if __name__=="__main__":
    sys.exit(main())
