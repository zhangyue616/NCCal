"""Load scientific selectors for manuscript positions from the packaged evidence."""
from importlib import import_module
from pathlib import Path
import json
ROOT = Path(__file__).resolve().parents[1]

def sources(root=ROOT):
    configuration = root / 'verify' / 'claim_sources.json'
    if not configuration.exists():
        return {}
    owners = json.loads(configuration.read_text(encoding='utf-8'))
    values = {}
    for name in sorted(set(owners.values())):
        if name not in ('molecular', 'protein', 'protein_extension', 'figure', 'shared', 'effects', 'current_figure'):
            raise ValueError('Unknown evidence module: ' + name)
        records = import_module(name + '_claim_bindings').bindings(root)
        for identifier, owner in owners.items():
            if owner == name:
                values[identifier] = records[identifier]
    return values

