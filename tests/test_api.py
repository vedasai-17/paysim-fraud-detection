import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from service import app as module


class FakeModel:
    def predict_proba(self, frame):
        return np.array([[.1, .9]])


class FakeExplainer:
    def __init__(self, model):
        pass

    def shap_values(self, frame):
        return np.ones((1, len(frame.columns)))


def test_artifact_rejects_missing_file_and_stale_feature_schema(tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    monkeypatch.setenv("MODEL_PATH", str(path))
    module.artifact.cache_clear()
    try:
        with pytest.raises(FileNotFoundError, match="Model artifact unavailable"):
            module.artifact()
        joblib.dump({"features": ["outdated_feature"]}, path)
        with pytest.raises(ValueError, match="incompatible or unreadable"):
            module.artifact()
    finally:
        module.artifact.cache_clear()


def test_scoring_persistence_and_validation(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'api.db'}")
    module.Base.metadata.create_all(engine)
    monkeypatch.setattr(module, "engine", engine)

    def session():
        with Session(engine) as db:
            yield db

    module.app.dependency_overrides[module.db_session] = session
    monkeypatch.setattr(module, "artifact", lambda: {"model": FakeModel(), "explainer_model": object(),
                                                     "review_threshold": .2, "decline_threshold": .8,
                                                     "version": "test"})
    monkeypatch.setattr(module.shap, "TreeExplainer", FakeExplainer)
    payment = {"transaction_id": "T1", "step": 9, "type": "TRANSFER", "amount": 900,
               "nameOrig": "C1", "oldbalanceOrg": 900, "nameDest": "C2", "oldbalanceDest": 0}
    try:
        with TestClient(module.app) as client:
            assert client.get("/health").json()["model_version"] == "test"
            assert client.post("/transactions/score", json={**payment, "amount": -1}).status_code == 422
            assert client.get("/metrics").json()["processed"] == 0
            scored = client.post("/transactions/score", json=payment)
            assert scored.status_code == 201
            assert client.post("/transactions/score", json=payment).status_code == 409
            assert scored.json()["recommended_action"] == "decline"
            assert len(scored.json()["explanations"]) == 5
            assert client.get("/transactions/T1").json()["transaction_id"] == "T1"
            assert len(client.get("/transactions").json()) == 1
            assert client.get("/transactions", params={"max_amount": 800}).json() == []
            assert client.get("/transactions", params={"min_amount": 1000, "max_amount": 800}).status_code == 422
            assert client.patch("/transactions/T1/disposition", json={"disposition": "review"}).json()["disposition"] == "review"
            assert client.get("/transactions/T1").json()["recommended_action"] == "decline"
            assert client.get("/metrics").json()["decline_rate"] == 1
    finally:
        module.app.dependency_overrides.clear()
