"""Build the public dashboard snapshot from the bundled model and sample."""
import csv
import json
import os
from pathlib import Path


def main():
    demo = Path(__file__).resolve().parent / "demo"
    os.environ["DATABASE_URL"] = "sqlite://"
    os.environ["MODEL_PATH"] = str(demo / "model.joblib")
    from service.app import Base, Payment, SessionLocal, engine, score

    Base.metadata.create_all(engine)
    with (demo / "transactions.csv").open(newline="", encoding="utf-8") as source, SessionLocal() as session:
        rows = []
        for row in csv.DictReader(source):
            result = score(Payment.model_validate(row), session)
            rows.append({key: result[key] for key in (
                "transaction_id", "step", "type", "amount", "isFraud", "fraud_probability",
                "recommended_action", "model_version", "explanations",
            )})
    (demo / "scores.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
