"""Bindings for the narrowly authorised cTP-net text and drawing quantities."""
from pathlib import Path
import csv
import json


def bindings(root):
    root = Path(root)
    with (root/'data/ctp_case/intermediates/basis_naming_and_retrieval/basis/naming_opportunities.tsv').open(encoding='utf-8', newline='') as f:
        names = list(csv.DictReader(f, delimiter='\t'))
    widths = {int(r['coordinates']) for r in names}
    assert len(widths) == 1
    with (root/'results/ctp_case/Table_S-U5.tsv').open(encoding='utf-8-sig', newline='') as f:
        table = list(csv.DictReader(f, delimiter='\t'))
    key = list(table[0])[0]
    paired = [r for r in table if r[key].startswith(('C3/', 'E2/'))]
    cohort_keys = {r[key].split('/')[1] for r in paired}
    unused = {r[key].split('/')[1] for r in paired if r[key].startswith('E2/')}
    rotation = [r for r in paired if r[key].endswith('/1')]
    pfield = 'Holm-adjusted p'
    assert len(rotation) == len(cohort_keys) == 3
    effect = 'Mean paired difference, native minus opponent'
    assert not any(float(r[effect]) > 0 and float(r[pfield]) < .05 for r in rotation)
    values = {'hidden_units': next(iter(widths)), 'external_cohorts': len(cohort_keys), 'previously_unused_cohorts': len(unused)}
    specs = json.loads((root/'verify/ctp_claim_specs.json').read_text(encoding='utf-8'))
    return {r['id']:dict(value=values[r['selector']], source=r['source'], operation=r['operation']) for r in specs}
