"""Validate and profile PaySim without loading the entire dataset into memory."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.features import FEATURES, PAYMENT_TYPES

NUMERIC = ('step', 'amount', 'oldbalanceOrg', 'oldbalanceDest', 'isFraud')
BALANCES_AND_AMOUNT = ('amount', 'oldbalanceOrg', 'oldbalanceDest')
REQUIRED = (*NUMERIC, 'type', 'nameOrig', 'nameDest')
DECISION_FIELDS = ('type', 'amount')


def step_cutoffs(steps):
    steps = sorted(set(steps))
    if len(steps) < 5:
        raise ValueError('At least five distinct time steps are required')
    return steps[int(len(steps) * .6) - 1], steps[int(len(steps) * .8) - 1]


def validate_rows(frame):
    missing_columns = set(REQUIRED) - set(frame.columns)
    if missing_columns:
        raise ValueError(f'Missing columns: {sorted(missing_columns)}')
    frame = frame.loc[:, list(REQUIRED)].copy()
    missing = frame.isna()
    for column in ('type', 'nameOrig', 'nameDest'):
        missing[column] |= frame[column].astype('string').str.strip().eq('').fillna(False)
    invalid = pd.DataFrame(False, index=frame.index, columns=REQUIRED)
    for column in NUMERIC:
        values = pd.to_numeric(frame[column], errors='coerce')
        valid = np.isfinite(values) & (values >= 0)
        if column == 'step':
            valid &= values.between(0, 2_147_483_647) & (values % 1 == 0)
        if column == 'isFraud':
            valid &= values.isin([0, 1])
        invalid[column] = ~valid & ~missing[column]
        frame[column] = values
    invalid['type'] = ~frame['type'].isin(PAYMENT_TYPES) & ~missing['type']
    clean = frame.loc[~(missing | invalid).any(axis=1)].astype({'step': 'int32', 'isFraud': 'int8'})
    return clean, missing.sum(), invalid.sum()


def profile(data, chunksize=250_000):
    missing = pd.Series(0, index=REQUIRED)
    invalid = missing.copy()
    by_step, by_type = {}, {}
    numeric = {field: {'sum': 0., 'min': None, 'max': None, 'zero_count': 0} for field in BALANCES_AND_AMOUNT}
    rows = 0
    for raw in pd.read_csv(data, chunksize=chunksize):
        rows += len(raw)
        clean, chunk_missing, chunk_invalid = validate_rows(raw)
        missing += chunk_missing
        invalid += chunk_invalid
        if clean.empty:
            continue
        for field, target in (('step', by_step), ('type', by_type)):
            for key, counts in clean.groupby(field)['isFraud'].agg(['size', 'sum']).iterrows():
                totals = target.setdefault(key, [0, 0])
                totals[0] += int(counts['size'])
                totals[1] += int(counts['sum'])
        for field, stats in numeric.items():
            values = clean[field]
            low, high = float(values.min()), float(values.max())
            stats['sum'] += float(values.sum())
            stats['min'] = low if stats['min'] is None else min(stats['min'], low)
            stats['max'] = high if stats['max'] is None else max(stats['max'], high)
            stats['zero_count'] += int(values.eq(0).sum())

    train_end, validation_end = step_cutoffs(by_step)
    splits = {'train': [], 'validation': [], 'test': []}
    for step in sorted(by_step):
        split = 'train' if step <= train_end else 'validation' if step <= validation_end else 'test'
        splits[split].append(step)

    def counts(total, fraud):
        return {'rows': total, 'fraud': fraud, 'fraud_rate': fraud / total}

    valid_rows, fraud_rows = map(sum, zip(*by_step.values()))
    return {
        'source': Path(data).name,
        'input_rows': rows,
        'rejected_rows': rows - valid_rows,
        'accepted': counts(valid_rows, fraud_rows),
        'missing_by_column': missing.to_dict(),
        'invalid_by_column': invalid.to_dict(),
        'by_type': {kind: counts(*by_type[kind]) for kind in sorted(by_type)},
        'by_step': {int(step): counts(*by_step[step]) for step in sorted(by_step)},
        'numeric': {field: {**stats, 'mean': stats['sum'] / valid_rows} for field, stats in numeric.items()},
        'splits': {name: {**counts(*map(sum, zip(*(by_step[step] for step in steps)))),
                          'first_step': int(steps[0]), 'last_step': int(steps[-1])} for name, steps in splits.items()},
        'decision_inputs': DECISION_FIELDS,
        'model_features': FEATURES,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', default='artifacts/data_profile.json')
    args = parser.parse_args()
    report = profile(args.data)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('accepted', 'rejected_rows', 'splits')}, indent=2))
