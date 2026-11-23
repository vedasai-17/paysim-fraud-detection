"""Train, calibrate and evaluate on strictly increasing PaySim time steps."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score, confusion_matrix, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common.features import FEATURES, feature_frame
from common.risk import action_for
from common.model import SigmoidCalibratedModel
from training.prepare import DECISION_FIELDS, REQUIRED, step_cutoffs, validate_rows


def temporal_split(df):
    train_end, validation_end = step_cutoffs(df.step.unique())
    train = df[df.step <= train_end]
    validation = df[(df.step > train_end) & (df.step <= validation_end)]
    test = df[df.step > validation_end]
    if any(part.isFraud.nunique() < 2 for part in (train, validation, test)):
        raise ValueError("Each temporal split needs both fraud and legitimate examples")
    return train, validation, test


def recall_at_fpr(y, probability, target=.01):
    fpr, tpr, _ = roc_curve(y, probability, drop_intermediate=False)
    return tpr[fpr <= target].max()


def metrics(y, probability):
    return {"pr_auc": float(average_precision_score(y, probability)),
            "roc_auc": float(roc_auc_score(y, probability)),
            "recall_at_1pct_fpr": float(recall_at_fpr(y, probability)),
            "brier_score": float(brier_score_loss(y, probability)),
            "log_loss": float(log_loss(y, probability))}


def thresholds(y, probability):
    scores = pd.DataFrame({'score': probability, 'fraud': np.asarray(y)})
    groups = scores.groupby('score')['fraud'].agg(['size', 'sum']).sort_index(ascending=False).cumsum()
    # All tied scores are assigned to the same band.
    review = float(np.quantile(probability, .95))
    eligible = groups[(groups.index > review) & (groups['size'] >= 20) & (groups['sum'] / groups['size'] >= .8)]
    return review, None if eligible.empty else float(eligible.index.min())


def run(data, output):
    chunks = []
    missing = pd.Series(0, index=REQUIRED)
    invalid = missing.copy()
    for raw in pd.read_csv(data, chunksize=250_000):
        clean, chunk_missing, chunk_invalid = validate_rows(raw)
        missing += chunk_missing
        invalid += chunk_invalid
        chunks.append(clean.loc[:, ['step', 'isFraud', *DECISION_FIELDS]].astype({'type': 'category'}))
    df = pd.concat(chunks, ignore_index=True)
    del chunks, raw, clean
    train, val, test = temporal_split(df)
    validation_steps = sorted(val.step.unique())
    cutoff = validation_steps[(len(validation_steps) - 1) // 2]
    early = val.step <= cutoff
    if min(val.loc[early, 'isFraud'].nunique(), val.loc[~early, 'isFraud'].nunique()) < 2:
        raise ValueError('Both validation periods require fraud and legitimate examples')
    x_train, x_val, x_test = (feature_frame(part) for part in (train, val, test))
    print(f'Training on {len(train):,} transactions', flush=True)
    baseline = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    baseline.fit(x_train, train.isFraud)
    base_prob = baseline.predict_proba(x_val)[:, 1]
    candidates = {}
    for leaves in (15, 31):
        model = LGBMClassifier(n_estimators=150, num_leaves=leaves, min_child_samples=2, learning_rate=.05,
                               class_weight="balanced", random_state=42, verbosity=-1, n_jobs=4,
                               force_col_wise=True)
        model.fit(x_train, train.isFraud)
        candidates[leaves] = (model, metrics(val.loc[early, 'isFraud'], model.predict_proba(x_val.loc[early])[:, 1]))
        print(f'LightGBM num_leaves={leaves} validation AP: {candidates[leaves][1]["pr_auc"]:.5f}', flush=True)
    selected_leaves = max(candidates, key=lambda leaves: candidates[leaves][1]["pr_auc"])
    model, candidate_metrics = candidates[selected_leaves]
    raw_val = np.clip(model.predict_proba(x_val)[:, 1], 1e-8, 1 - 1e-8)
    logits = np.log(raw_val / (1 - raw_val)).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=1000).fit(logits[early.to_numpy()], val.loc[early, 'isFraud'])
    calibrated = SigmoidCalibratedModel(model, calibrator)
    val_prob = calibrated.predict_proba(x_val)[:, 1]
    review, decline = thresholds(val.loc[~early, 'isFraud'], val_prob[~early.to_numpy()])
    test_prob = calibrated.predict_proba(x_test)[:, 1]
    actions = [action_for(p, review, decline) for p in test_prob]
    test_baseline = metrics(test.isFraud, baseline.predict_proba(x_test)[:, 1])
    report = {"rows": {"train": len(train), "validation": len(val), "test": len(test)},
              "step_ranges": {k: [int(v.step.min()), int(v.step.max())] for k, v in
                              (("train", train), ("validation", val), ("test", test))},
              "baseline_validation": metrics(val.isFraud, base_prob),
              "baseline_test": test_baseline,
              "lightgbm_validation": candidate_metrics,
              "candidate_validation": {leaves: result for leaves, (_, result) in candidates.items()},
              "selected_num_leaves": selected_leaves,
              "validation_periods": {"selection_and_calibration": [int(val.step.min()), int(cutoff)],
                                     "threshold_selection": [int(val.loc[~early, 'step'].min()), int(val.step.max())]},
              "calibrated_validation": metrics(val.loc[~early, 'isFraud'], val_prob[~early.to_numpy()]),
              "uncalibrated_validation_policy_period": metrics(val.loc[~early, 'isFraud'], raw_val[~early.to_numpy()]),
              "test": metrics(test.isFraud, test_prob),
              "thresholds": {"review": review, "decline": decline},
              "test_confusion_decline": confusion_matrix(test.isFraud, np.array(actions) == "decline", labels=[0, 1]).tolist(),
              "test_band_counts": {a: actions.count(a) for a in ("approve", "review", "decline")},
              "test_band_metrics": {a: {"count": int(sum(np.array(actions) == a)),
                                        "precision": float(np.mean(test.isFraud.to_numpy()[np.array(actions) == a])) if a in actions else 0.0,
                                        "fraud_recall": float(np.sum((test.isFraud.to_numpy() == 1) & (np.array(actions) == a)) / sum(test.isFraud))}
                                    for a in ("approve", "review", "decline")},
              "estimated_blocked_fraud_amount": float(test.loc[(np.array(actions) == "decline") & (test.isFraud.to_numpy() == 1), "amount"].sum()),
              "fraud_prevalence": float(df.isFraud.mean()),
              "missing_values": missing.to_dict(),
              "invalid_values": invalid.to_dict(),
              "transaction_types": {k: int(v) for k, v in df.type.value_counts().items()},
              "amount_summary": {k: float(v) for k, v in df.amount.describe().items()}}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": calibrated, "explainer_model": model, "features": FEATURES,
                 "review_threshold": review, "decline_threshold": decline, "version": "2.0.0"}, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    (output / "model_card.md").write_text(f"""# PaySim fraud model card

Version 2.0.0. Source: {Path(data).name}. Inputs are log amount and transaction type. Scores are intended for a synthetic analyst demonstration.

Training, validation and test steps: {report['step_ranges']}. Selected LightGBM leaf count: {selected_leaves}. Early validation selects the candidate and fits the sigmoid. Late validation chooses policy thresholds: {report['validation_periods']}.

Held-out LightGBM metrics: {report['test']}. Held-out logistic baseline: {test_baseline}. The average precision score summarizes the precision-recall curve. Recall at 1% FPR is a ranking benchmark measured on test and does not use the deployed thresholds.

Review starts at {review:.6f}. Decline threshold: {decline} (`None` disables declines when validation fails to support at least 80% precision on at least 20 transactions). Tied scores enter the same band. Test policy counts: {report['test_band_counts']}. Labeled fraud amount in the decline band: {report['estimated_blocked_fraud_amount']:.2f} simulated currency units.

PaySim is synthetic and lacks demographic labels for group fairness assessment. Fraud prevalence can shift substantially over time. Compare score distributions and alert rates to validation, review false positives and precision as labels arrive, and reassess thresholds when the review queue exceeds capacity. Transaction amounts in the decline band estimate exposure and do not account for recoveries or operational costs.
""", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    run(args.data, args.output)
