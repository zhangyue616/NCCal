from pathlib import Path
import requests, json, hashlib, time
ROOT=Path(__file__).resolve().parent
repo='EvolutionaryScale/esmc-300m-2024-12'
j=requests.get('https://huggingface.co/api/models/'+repo,timeout=45).json()
revision=j['sha']
out=ROOT/'esmc_weights';out.mkdir(exist_ok=True)
name='esmc_300m_2024_12_v0.pth'
target=out/name
if not target.exists():
    with requests.get(f'https://huggingface.co/{repo}/resolve/{revision}/data/weights/{name}',stream=True,timeout=(30,180)) as r:
        r.raise_for_status()
        with open(target.with_suffix('.partial'),'wb') as f:
            for chunk in r.iter_content(8*1024*1024):
                f.write(chunk)
        target.with_suffix('.partial').replace(target)
for name2 in ['README.md','config.json']:
    r=requests.get(f'https://huggingface.co/{repo}/resolve/{revision}/{name2}',timeout=45);r.raise_for_status();(out/name2).write_bytes(r.content)
with open(target,'rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
(ROOT/'esmc_acquisition.json').write_text(json.dumps(dict(repo=repo,revision=revision,gated=j.get('gated'),weight_path=str(target),bytes=target.stat().st_size,sha256=digest,acquired_unix=time.time(),authentication='anonymous public official download; no bypass',package='esm==3.2.1'),indent=2))
print('ACQUIRED',target.stat().st_size,digest,flush=True)
