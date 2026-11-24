"""Retrieve a selected upstream asset or validate a reader-supplied file."""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as s:
        for b in iter(lambda:s.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('id',nargs='?')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--check',type=Path)
    group.add_argument('--dest',type=Path)
    args=parser.parse_args()
    index={r['id']:r for r in json.loads(Path(__file__).with_name('asset_index.json').read_text(encoding='utf-8'))}
    if args.id is None:
        print('\n'.join(index));return
    if args.id not in index:parser.error('unknown asset')
    spec=index[args.id]
    if args.check:
        if digest(args.check)!=spec['sha256']:sys.exit('checksum mismatch')
        print('SHA-256 matches');return
    if args.dest is None:parser.error('specify --check FILE or --dest DIRECTORY')
    target=args.dest/args.id
    target.parent.mkdir(exist_ok=True,parents=True)
    if target.exists():
        if digest(target)!=spec['sha256']:sys.exit('existing file differs')
        return
    with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as tmp:
        temp=Path(tmp.name)
    try:
        with urllib.request.urlopen(spec['url'],timeout=90) as response, temp.open('wb') as stream:
            while True:
                block=response.read(1024*1024)
                if not block:break
                stream.write(block)
        if digest(temp)!=spec['sha256']:sys.exit('upstream content differs from the study snapshot')
        os.replace(temp,target)
    finally:
        if temp.exists():temp.unlink()
if __name__=='__main__':main()

