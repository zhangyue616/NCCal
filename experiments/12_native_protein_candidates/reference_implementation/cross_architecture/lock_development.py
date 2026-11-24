"""Seal every development fit before any scoring on the external study."""
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from cohorts import ROOT
from experiment import FITS,guard

def main():
    guard()
    expected={"source.npz"}
    for arch in ("esm2_35m","carp_38m"):
        expected.add(arch+"_trained.npz")
        for stage,n in (("random",199),("rotation",9)):
            expected.update(f"{arch}_{stage}_{j:03d}.npz" for j in range(n))
    actual={p.name for p in FITS.glob("*.npz")}
    if actual!=expected:
        raise ValueError(f"Missing fits {len(expected-actual)}, unexpected {len(actual-expected)}")
    files={}
    for name in sorted(expected):
        p=FITS/name
        with np.load(p,allow_pickle=False) as fit:
            for key in fit.files:
                v=fit[key]
                if v.dtype.kind in "if" and not np.isfinite(v).all(): raise ValueError(f"{name}/{key} contains nonfinite values")
        files[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    path=ROOT/"DEVELOPMENT_LOCK.json"
    if path.exists():
        assert json.loads(path.read_text())["fits"]==files
        return
    code_files=("analysis.py","experiment.py","cohorts.py","models/native_extractors.py")
    code={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in code_files}
    with path.open("x",encoding="utf8") as out:
        json.dump({"utc":datetime.now(timezone.utc).isoformat(),"code":code,"fits":files},out,indent=2)
        out.write("\n")
    print(f"{len(files)} fitted source/native/reference objects locked")
if __name__=="__main__": main()
