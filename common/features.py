"""Features computed only from fields available at authorization time."""
import numpy as np
import pandas as pd

PAYMENT_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")
FEATURES = ["amount_log", *(f"type_{kind}" for kind in PAYMENT_TYPES)]


def feature_row(item):
    return feature_frame([item]).iloc[0].to_dict()


def feature_frame(rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=FEATURES, index=frame.index)
    amounts = frame['amount'].astype(float)
    kinds = frame['type']
    if not (np.isfinite(amounts) & (amounts >= 0) & kinds.isin(PAYMENT_TYPES)).all():
        raise ValueError('Invalid transaction type or amount')
    return pd.DataFrame({'amount_log': np.log1p(amounts),
                         **{f'type_{kind}': kinds.eq(kind).astype('int8') for kind in PAYMENT_TYPES}})
