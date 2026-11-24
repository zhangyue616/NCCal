import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

root=Path(__file__).resolve().parent
target=root/"FROZEN.json"
if target.exists():
    raise SystemExit("Freeze exists. Verify it without replacing it.")
files=("PROTOCOL.md","external/PROTOCOL.md","cohorts/D_contact.tsv","cohorts/D_exposure.tsv","cohorts/D_grid.tsv")
items={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
with target.open("x",encoding="utf8") as f:
    json.dump({"utc":datetime.now(timezone.utc).isoformat(),"decision":"before new model-structure association scoring","files":items},f,indent=2)
    f.write("\n")
print(target.read_text())
