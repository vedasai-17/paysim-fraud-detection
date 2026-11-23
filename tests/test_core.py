import math
import pandas as pd
import pytest
from common.features import FEATURES, feature_frame
from common.risk import action_for
from training.train import recall_at_fpr, temporal_split, thresholds
from training.prepare import profile, validate_rows


SAMPLE = {"step": 1, "type": "TRANSFER", "amount": 100, "oldbalanceOrg": 100,
          "oldbalanceDest": 0, "nameOrig": "C1", "nameDest": "C2"}


def test_features_use_only_pretransaction_fields():
    row = feature_frame([{**SAMPLE, "newbalanceOrig": 0, "newbalanceDest": 100,
                          "isFraud": 1, "isFlaggedFraud": 1}])
    assert list(row.columns) == FEATURES
    assert row.iloc[0]["type_TRANSFER"] == 1
    assert row.iloc[0]["amount_log"] == pytest.approx(math.log1p(100))
    changed = {**SAMPLE, "step": 21, "oldbalanceOrg": 0, "oldbalanceDest": 600, "nameOrig": "C3"}
    pd.testing.assert_frame_equal(row, feature_frame([changed]))


def test_threshold_boundaries():
    assert [action_for(p, .2, .8) for p in (.199, .2, .799, .8)] == ["approve", "review", "review", "decline"]
    assert action_for(1, .2, None) == "review"


def test_decline_policy_requires_precision_and_support():
    labels = [0] * 1000 + [1] * 20
    scores = [.1] * 1000 + [.9] * 20
    assert thresholds(labels, scores) == (.1, .9)
    assert thresholds(labels[:-1], scores[:-1]) == (.1, None)
    assert thresholds([0] + labels[:-1], scores) == (.1, .9)
    assert thresholds([0] * 5 + labels[:-5], scores) == (.1, None)
    assert recall_at_fpr([0, 0, 1, 1], [.8, .3, .9, .8]) == .5


def test_temporal_split():
    df = pd.read_csv("data/fixture/paysim_sample.csv")
    train, val, test = temporal_split(df)
    assert train.step.max() < val.step.min() < val.step.max() < test.step.min()
    assert all(part.isFraud.nunique() == 2 for part in (train, val, test))


def test_preparation_rejects_invalid_fields_and_preserves_temporal_boundaries():
    path = "data/fixture/paysim_sample.csv"
    rows = pd.read_csv(path).head(4).astype({"amount": "object"})
    rows.loc[0, "amount"] = "broken"
    rows.loc[1, "isFraud"] = 2
    rows.loc[2, "nameOrig"] = " "
    clean, missing, invalid = validate_rows(rows)
    assert clean.index.tolist() == [3]
    assert invalid["amount"] == invalid["isFraud"] == missing["nameOrig"] == 1

    report = profile(path, chunksize=4)
    assert report["accepted"]["rows"] == 30
    assert report["accepted"]["fraud"] == 10
    assert report["rejected_rows"] == 0
    assert [split["rows"] for split in report["splits"].values()] == [18, 6, 6]
    assert report["splits"]["test"]["first_step"] == 9
