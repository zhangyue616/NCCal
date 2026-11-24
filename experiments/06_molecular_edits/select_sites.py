"""Select edit classes/sites from an external table of source structures.

The table columns are cid and canonical_smiles. It stays outside the repository.
This command does not accept/reject token mappings or run a model.
"""
from pathlib import Path
import argparse,csv,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from nccal.molecular.edit_selection import choose_edit
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    with args.input.open(encoding='utf-8-sig') as f:records=list(csv.DictReader(f,delimiter='\t'))
    rows=[choose_edit(int(r['cid']),r['canonical_smiles']) for r in records];args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
