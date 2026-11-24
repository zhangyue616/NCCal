from pathlib import Path
import hashlib
import json
import time

import requests

ROOT = Path(__file__).resolve().parent / "official"
REPO = "chandar-lab/AMPLIFY_120M"
COMMIT = "d918a9e8c64dc43f6ec4996022c3deba14d66470"
WANTED = ["config.json", "tokenizer.json", "amplify.py", "rmsnorm.py", "rotary.py",
          "README.md", "model.safetensors"]

def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def main():
    ROOT.mkdir(exist_ok=True)
    listing = requests.get(f"https://huggingface.co/api/models/{REPO}/revision/{COMMIT}",
                           params={"blobs": "true"}, timeout=40)
    listing.raise_for_status()
    meta = listing.json()
    assert meta["sha"] == COMMIT
    files = {f["rfilename"]: f for f in meta["siblings"]}
    receipt = {}
    for name in WANTED:
        target = ROOT / name
        expected = files[name]
        for attempt in range(4):
            try:
                if not target.exists():
                    tmp = target.with_suffix(target.suffix + ".part")
                    url = f"https://huggingface.co/{REPO}/resolve/{COMMIT}/{name}"
                    with requests.get(url, stream=True, timeout=(30, 120)) as r:
                        r.raise_for_status()
                        with tmp.open("wb") as f:
                            for chunk in r.iter_content(1024 * 1024):
                                f.write(chunk)
                    if tmp.stat().st_size != expected["size"]:
                        raise ValueError(f"Incomplete download of {name}: {tmp.stat().st_size}")
                    tmp.replace(target)
                digest = sha256(target)
                if expected.get("lfs") and digest != expected["lfs"]["sha256"]:
                    raise ValueError(f"Upstream hash mismatch for {name}")
                receipt[name] = [target.stat().st_size, digest]
                print(name, target.stat().st_size, flush=True)
                break
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(3)
    (ROOT / "upstream.json").write_text(json.dumps({"model": REPO, "commit": COMMIT,
        "files": receipt}, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
