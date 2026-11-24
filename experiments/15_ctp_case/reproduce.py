"""Rebuild the five cTP-net tables from archived derived observations."""
from pathlib import Path
from functools import lru_cache
import argparse
import csv
import hashlib
import json
import math
import sys
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def close(left, right):
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-11)


@lru_cache(None)
def read(source):
    return pd.read_csv(ROOT/source, sep='\t')


def select(recipe):
    data = read(recipe['source'])
    for field, value in recipe['where'].items():
        data = data[data[field].eq(value)]
    return data.copy()


def evaluate(recipe):
    operation = recipe['operation']
    field = recipe['field']
    if operation == 'json_value':
        value = json.loads((ROOT/recipe['source']).read_text(encoding='utf-8'))
        for key in field.split('.'):
            value = value[key]
        return float(value)
    data = select(recipe)
    if operation in ['scalar', 'ratio', 'possible_pairs', 'same_name_fraction']:
        if len(data) != 1:
            raise ValueError('Nonunique scalar source selector: ' + repr(recipe))
        row = data.iloc[0]
        if operation == 'scalar':
            value = row[field]
        elif operation == 'ratio':
            value = row[field] / row[recipe['denominator_field']]
        elif operation == 'possible_pairs':
            value = math.comb(int(row[field]), 2)
        else:
            value = row.same_name_pairs / math.comb(int(row.variable_units), 2)
    elif operation == 'count':
        value = len(data)
    elif operation in ['median', 'mean', 'min', 'max', 'sum', 'nunique']:
        vector = data[field]
        if field == 'gain_both':
            vector = data.linear3_both - data.linear2_both
            assert all(close(a, b) for a, b in zip(vector, data.gain_both))
        value = getattr(vector, operation)()
    elif operation == 'median_difference':
        value = (data[field] - data[recipe['other_field']]).median()
    elif operation.startswith('donor_'):
        group = data.groupby('donor')
        if operation == 'donor_count':
            value = data.donor.nunique()
        elif operation.startswith('donor_pair_count_'):
            value = getattr(group.size(), operation.rsplit('_', 1)[1])()
        elif operation == 'donor_gain_sum_median':
            value = group.gain.sum().median()
        elif operation == 'donor_gain_conditional_median':
            value = group.gain.mean().median()
        elif operation == 'donor_gain_fixed_median':
            names = read('data/ctp_case/intermediates/basis_naming_and_retrieval/basis/naming_opportunities.tsv')
            widths = set(names.coordinates)
            assert len(widths) == 1
            value = (group.gain.sum() / math.comb(int(next(iter(widths))), 2)).median()
        elif operation.startswith('donor_fraction_'):
            method = recipe.get('method', 'three')
            win = data[f'{method}_0_margin'].gt(1e-12) & data[f'{method}_1_margin'].gt(1e-12)
            top5 = data[f'{method}_0_rank'].le(5) & data[f'{method}_1_rank'].le(5)
            top1 = data[f'{method}_0_rank'].le(1) & data[f'{method}_1_rank'].le(1)
            vectors = {'win': win, 'top5': top5, 'win_top5': win & top5, 'top1': top1, 'win_top1': win & top1}
            kind = operation[len('donor_fraction_'):-len('_median')]
            value = vectors[kind].groupby(data.donor).mean().median()
        else:
            raise ValueError('Unknown donor operation: ' + operation)
    else:
        raise ValueError('Unknown archive operation: ' + operation)
    return float(value) * recipe.get('scale', 1)


def compare_table(actual, expected):
    with actual.open(encoding='utf-8-sig', newline='') as f:
        computed = list(csv.reader(f, delimiter='\t'))
    with expected.open(encoding='utf-8-sig', newline='') as f:
        frozen = list(csv.reader(f, delimiter='\t'))
    errors = []
    if computed[0] != frozen[0] or len(computed) != len(frozen):
        return [{'type': 'shape_or_header_difference'}]
    for row, (left, right) in enumerate(zip(computed[1:], frozen[1:]), 2):
        for col, (a, b) in enumerate(zip(left, right), 1):
            if a == b:
                continue
            try:
                same = close(a, b)
            except (ValueError, TypeError):
                same = False
            if not same:
                errors.append(dict(row=row, column=col, computed=a, archived=b))
    return errors


def check_figure():
    with (ROOT/'results/artwork/ctp_case/figure_data.tsv').open(encoding='utf-8', newline='') as f:
        points = list(csv.DictReader(f, delimiter='\t'))
    tables = {}
    for i in [1, 2]:
        with (ROOT/f'results/ctp_case/Table_S-U{i}.tsv').open(encoding='utf-8-sig', newline='') as f:
            tables[i] = list(csv.DictReader(f, delimiter='\t'))
    one = {r['Record ID']: r for r in tables[1]}
    two = {r['basis']: r for r in tables[2]}
    mismatches = []
    for point in points:
        if point['panel'] == 'A':
            expected = two[point['basis']]['Same-name pairs']
        elif point['panel'] == 'B':
            expected = 100 * float(one[point['basis']][point['cohort']+'_Both-win and both-top-five fraction'])
        else:
            continue
        if not close(point['value'], expected):
            mismatches.append(point)
    counts = {panel:sum(p['panel'] == panel for p in points) for panel in ['A', 'B', 'C']}
    assert counts == {'A':16, 'B':32, 'C':63}
    return dict(panel_counts=counts, table_bound_point_differences=mismatches, C_scope='Only the 63 supplied plotting points are used. Individual donor points cannot be recovered from Table 5 cohort summaries.')


def check_limitation_reference():
    records = json.loads((HERE/'limited_reference/scvi.json').read_text(encoding='utf-8'))
    choices = read('data/ctp_case/intermediates/donor_transfer_sorting_and_marker_controls/second_model/choices_frozen.tsv')
    summary = read('data/ctp_case/intermediates/donor_transfer_sorting_and_marker_controls/second_model/summary.tsv')
    for record in records:
        parts = record['record_id'].split('_')
        description = 'KLRG1_' + parts[2]
        selector = '_'.join(parts[3:])
        choice = choices[choices.description.eq(description) & choices.selector.eq(selector)]
        assert len(choice) == 1
        choice = choice.iloc[0]
        for key, field in [('target_unit','unit'), ('development_score','development_score'), ('ranking_candidates','search_budget'), ('exact_same_name_group_size','same_name_size')]:
            assert close(record[key], choice[field])
        for prefix, cohort in [('JDM','JDM'), ('Wells','Wells_blood')]:
            for antibodies in [2, 3]:
                frame = summary[summary.source.eq(cohort) & summary.description.eq(description) & summary.selector.eq(selector) & summary.antibodies.eq(antibodies) & summary.target_kind.eq('locked')]
                assert len(frame) == 1
                row = frame.iloc[0]
                assert close(record[prefix+f'_{"two" if antibodies == 2 else "three"}_antibody_median_rank'], row.median_rank)
                if antibodies == 3:
                    for key, field in [('roster_donors','people'), ('evaluable_donors','valid'), ('three_antibody_top5_donors','top5'), ('three_antibody_top1_donors','top1')]:
                        assert close(record[prefix+'_'+key], row[field])
                    assert close(record[prefix+'_margin_over_other_description_percentage_points'], row.other_description_margin * 100)
                    assert close(record[prefix+'_margin_over_best_other_coordinate_percentage_points'], row.full_margin * 100)
    return dict(records_checked=len(records), purpose='Source check for the scVI limitation sentence only; not a sixth result table.')


def verify(output):
    output = output.resolve()
    result_dir = output.parent / (output.stem + '_tables')
    result_dir.mkdir(parents=True, exist_ok=True)
    specs = json.loads((HERE/'table_recipes.json').read_text(encoding='utf-8'))
    comparisons = {}
    recipes_checked = 0
    for specification in specs:
        data = [dict(r) for r in specification['metadata']]
        # Metadata contain descriptions and empty numeric slots, never result numbers.
        for recipe in specification['recipes']:
            assert data[recipe['row']][recipe['column']] is None
            data[recipe['row']][recipe['column']] = evaluate(recipe)
            recipes_checked += 1
        path = result_dir/specification['table']
        with path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=specification['columns'], delimiter='\t', lineterminator='\n')
            writer.writeheader()
            writer.writerows(data)
        comparisons[path.name] = compare_table(path, ROOT/'results/ctp_case'/path.name)
    manifest = json.loads((ROOT/'data/ctp_case/source_archive.json').read_text(encoding='utf-8'))
    archive_errors = [r['archive_record'] for r in manifest if hashlib.sha256((ROOT/r['archive_record']).read_bytes()).hexdigest() != r['archive_sha256']]
    locks = []
    for name in ['LOCK1', 'LOCK2']:
        actual = hashlib.sha256((HERE/'locks'/f'{name}.json').read_bytes()).hexdigest()
        expected = (HERE/'locks'/f'{name}.sha256').read_text().split()[0]
        locks.append(dict(lock=name, expected=expected, actual=actual, matches=actual == expected))
    figure = check_figure()
    limitation = check_limitation_reference()
    differences = sum(len(v) for v in comparisons.values())
    receipt = dict(experiment='15_ctp_case', mode='stored_result_reaggregation', numeric_recipes_checked=recipes_checked, table_comparisons=comparisons, differences=differences, archived_input_differences=archive_errors, lock_file_integrity=locks, figure=figure, limitation_reference=limitation, coverage='experiments/15_ctp_case/README.md', scope='The five tables are rebuilt from donor-by-pair, donor-purity, donor-by-basis, pool, basis and frozen summary records as documented. Frozen tests, intervals and winner totals are retrieved, not re-estimated. No raw matrices or public-input model rerun are included.', upstream_locked_artifact_coverage='Only selected derived inputs are included. Additional artifacts referenced by the original locks are not checked here.')
    receipt['success'] = differences == 0 and not archive_errors and all(r['matches'] for r in locks) and not figure['table_bound_point_differences']
    write_json(output, receipt)
    print(json.dumps({'experiment':'15_ctp_case', 'success':receipt['success'], 'numeric_recipes_checked':recipes_checked, 'table_differences':differences, 'archived_input_differences':len(archive_errors)}, ensure_ascii=True))
    return 0 if receipt['success'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['verify'], default='verify')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(verify(args.output))
