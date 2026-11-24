from __future__ import annotations
import csv
import json
import os
import sys
from pathlib import Path

OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(OUT / 'runtime_cache' / 'matplotlib')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

LAYER=DICT=Path(__file__).resolve().parents[2]/'results/artwork/supplement'
TRACE = []


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return [dict(row, _row=i) for i, row in enumerate(csv.DictReader(stream,delimiter="\t"if path.suffix==".tsv"else","), 2)]


def select(rows, **keys):
    found = [r for r in rows if all(r[k] == v for k, v in keys.items())]
    assert len(found) == 1, keys
    return found[0]


def trace(path, row, field, item=None):
    value = row[field]
    TRACE.append(dict(item=item or field, value=value, source=str(path),
                      locator=f"CSV row {row['_row']}; {field}"))
    return value


def number(path, row, field, item=None):
    return float(trace(path, row, field, item))


def export_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0])
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


layer_rows = read_csv(LAYER / 'o_layer_reconstruction_8_networks.tsv')
curve_rows = read_csv(LAYER / 'o_curve_by_order.csv')
primary_rows = read_csv(LAYER / 'o_primary_12_family.csv')
fixed_rows = read_csv(LAYER / 'o_fixed_head_decomposition.csv')
refit_rows = read_csv(LAYER / 'o_refit_head_decomposition.csv')
dict_rows = read_csv(DICT / 'p_primary_four_tests.csv')
raw_rows = read_csv(DICT / 'p_raw_ratios_and_wins.csv')
training_rows = read_csv(DICT / 'p_training_log_summary.csv')
sensitivity_rows = read_csv(DICT / 'p_all_recipe_estimates.csv')
concept_rows = read_csv(DICT / 'p_per_concept_per_network_scores.csv')

layer = {}
for row in layer_rows:
    tag = f"layer.{row['model']}.{row['network']}"
    layer[row['model'], row['network']] = {
        f: number(LAYER / 'o_layer_reconstruction_8_networks.tsv', row, f, f'{tag}.{f}')
        for f in ('VE_radius0', 'VE_local_minus_radius0', 'VE_local_graph')}
    for field in ('width', 'n_molecules', 'n_atoms', 'reference_design'):
        if field in row:trace(LAYER / 'o_layer_reconstruction_8_networks.tsv', row, field, f'{tag}.{field}')
    for field in ('VE_radius0_CI95_lower', 'VE_radius0_CI95_upper',
                  'VE_local_graph_CI95_lower', 'VE_local_graph_CI95_upper',
                  'VE_local_minus_radius0_CI95_lower', 'VE_local_minus_radius0_CI95_upper'):
        if row.get(field):
            trace(LAYER / 'o_layer_reconstruction_8_networks.tsv', row, field, f'{tag}.{field}')

models = ('chemberta_mtr', 'molformer')
tasks = ('bace', 'esol')
methods = ('development_mean', 'radius0', 'local_graph', 'residual_permutation')
errors = {}
error_table = []
for model in models:
    for task in tasks:
        for method in methods:
            row = select(curve_rows, model=model, task=task, order='O1',
                         k='464' if model == 'chemberta_mtr' else '768', method=method)
            val = number(LAYER / 'o_curve_by_order.csv', row, 'mean_absolute_output_deviation',
                         f'error.{model}.{task}.{method}')
            errors[model, task, method] = val
            error_table.append(dict(model=model, task=task, condition=method,
                                    mean_absolute_output_deviation=val,
                                    unit='class-1 probability' if task == 'bace' else 'log10 mol/L'))

test_counts = {'positive_significant': 0, 'negative_significant': 0, 'not_significant': 0}
for row in primary_rows:
    tag = f"primary.{row['model']}.{row['task']}.{row['reference']}"
    effect = number(LAYER / 'o_primary_12_family.csv', row,
                    'effect_reference_error_minus_local_error', tag + '.effect')
    for field in ('CI95_percentile', 'Holm_adjusted_p_12', 'reject_at_0_05',
                  'n_molecules', 'contributing_scaffold_clusters'):
        trace(LAYER / 'o_primary_12_family.csv', row, field, f'{tag}.{field}')
    significant = row['reject_at_0_05'] == 'True'
    key = ('positive_significant' if effect > 0 else 'negative_significant') if significant else 'not_significant'
    test_counts[key] += 1
assert test_counts == {'positive_significant': 8, 'negative_significant': 3, 'not_significant': 1}

for name, rows in (('o_fixed_head_decomposition.csv', fixed_rows),
                   ('o_refit_head_decomposition.csv', refit_rows)):
    for row in rows:
        for field in ('metric_value', 'change_condition_minus_0', 'change_CI95_percentile'):
            trace(LAYER / name, row, field,
                  f"task.{row['head_mode']}.{row['model']}.{row['task']}.{row['condition']}.{field}")

calibrations = {}
for row in dict_rows:
    key = row['checkpoint'], row['test']
    tag = f'dictionary.{key[0]}.{key[1]}'
    fold = number(DICT / 'p_primary_four_tests.csv', row, 'calibrated_fold', tag + '.fold')
    ci = json.loads(trace(DICT / 'p_primary_four_tests.csv', row, 'CI95_concept_bootstrap', tag + '.CI95'))
    calibrations[key] = (fold, ci)
    for field in ('n_concepts', 'n_initializations', 'bootstrap_draws', 'p', 'Holm_adjusted_p_4',
                  'interval_conditioning'):
        trace(DICT / 'p_primary_four_tests.csv', row, field, f'{tag}.{field}')

dictionary_summary = {}
for model in ('chemberta_mtr', 'chemberta_mlm'):
    row = select(raw_rows, checkpoint=model, recipe='primary')
    summary = {}
    for field in ('trained_median_dictionary_over_native', 'trained_strict_win_concepts',
                  'trained_tied_concepts', 'initialization_ratio_median',
                  'initialization_ratio_min', 'initialization_ratio_max',
                  'initialization_win_count_median', 'initialization_win_count_min',
                  'initialization_win_count_max'):
        summary[field] = number(DICT / 'p_raw_ratios_and_wins.csv', row, field,
                                f'dictionary.{model}.raw.{field}')
    train = select(training_rows, network=model, recipe='primary')
    for field in ('validation_fvu', 'width', 'k', 'validation_rows', 'dead_features', 'alive_features'):
        summary[field] = number(DICT / 'p_training_log_summary.csv', train, field,
                                f'dictionary.{model}.training.{field}')
    dictionary_summary[model] = summary
    rows = [r for r in concept_rows if r['network'] == model and r['recipe'] == 'primary']
    assert len(rows) == len({r['concept_id'] for r in rows}) == 173
    assert sum(r['dictionary_strictly_wins'] == 'True' for r in rows) == summary['trained_strict_win_concepts']

for row in sensitivity_rows:
    if row['checkpoint'] == 'chemberta_mtr' and row['test'] == 'T2':
        for field in ('calibrated_fold', 'CI95_concept_bootstrap', 'n_initializations'):
            trace(DICT / 'p_all_recipe_estimates.csv', row, field,
                  f"sensitivity.MTR.{row['recipe']}.{field}")

