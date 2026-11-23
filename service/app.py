import csv
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import shap
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from common.features import FEATURES, PAYMENT_TYPES, feature_frame
from common.risk import action_for

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fraud.db")
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(engine)


class Base(DeclarativeBase):
    pass


class Transaction(Base):
    __tablename__ = "transactions"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    step: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(20))
    amount: Mapped[float] = mapped_column(Float)
    name_orig: Mapped[str] = mapped_column(String(100))
    oldbalance_org: Mapped[float] = mapped_column(Float)
    name_dest: Mapped[str] = mapped_column(String(100))
    oldbalance_dest: Mapped[float] = mapped_column(Float)
    is_fraud: Mapped[int | None] = mapped_column(Integer, nullable=True)
    probability: Mapped[float] = mapped_column(Float)
    action: Mapped[str] = mapped_column(String(20))
    disposition: Mapped[str | None] = mapped_column(String(20), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    analyst_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    explanations: Mapped[str] = mapped_column(Text)
    model_version: Mapped[str] = mapped_column(String(30))


class Payment(BaseModel):
    transaction_id: str = Field(min_length=1, max_length=100)
    step: int = Field(ge=0, le=2_147_483_647)
    type: str
    amount: float = Field(ge=0, allow_inf_nan=False)
    nameOrig: str = Field(min_length=1, max_length=100)
    oldbalanceOrg: float = Field(ge=0, allow_inf_nan=False)
    nameDest: str = Field(min_length=1, max_length=100)
    oldbalanceDest: float = Field(ge=0, allow_inf_nan=False)
    isFraud: int | None = Field(default=None, ge=0, le=1)


class Disposition(BaseModel):
    disposition: str
    note: str | None = Field(default=None, max_length=2000)


@asynccontextmanager
async def lifespan(app):
    Base.metadata.create_all(engine)
    seed_path = os.getenv("DEMO_SEED_PATH")
    if seed_path:
        with Path(seed_path).open(newline="", encoding="utf-8") as source, SessionLocal() as db:
            for row in csv.DictReader(source):
                payment = Payment.model_validate(row)
                try:
                    score(payment, db)
                except HTTPException as exc:
                    if exc.status_code != 409:
                        raise
    yield


app = FastAPI(title="Payment Fraud Risk API", version="1.0.0", lifespan=lifespan)


def db_session():
    with SessionLocal() as session:
        yield session


@lru_cache
def artifact():
    full_model = Path("artifacts/phase3/model.joblib")
    configured = os.getenv("MODEL_PATH")
    path = Path(configured) if configured else full_model if full_model.is_file() else Path("artifacts/model.joblib")
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact unavailable: {path}. Run training first.")
    try:
        bundle = joblib.load(path)
        if list(bundle["features"]) != FEATURES:
            raise ValueError("Feature schema mismatch")
        if not callable(bundle["model"].predict_proba) or not bundle["version"]:
            raise ValueError("Missing model or version")
        action_for(0, bundle["review_threshold"], bundle["decline_threshold"])
        explainer(bundle["explainer_model"])
    except Exception as exc:
        raise ValueError(f"Model artifact is incompatible or unreadable: {path}. Check the model path or retrain.") from exc
    return bundle


@lru_cache(maxsize=1)
def explainer(model):
    return shap.TreeExplainer(model)


def serialize(tx):
    return {"transaction_id": tx.id, "step": tx.step, "type": tx.type, "amount": tx.amount,
            "nameOrig": tx.name_orig, "nameDest": tx.name_dest, "oldbalanceOrg": tx.oldbalance_org,
            "oldbalanceDest": tx.oldbalance_dest, "isFraud": tx.is_fraud,
            "fraud_probability": tx.probability, "risk_score": round(tx.probability * 100, 2),
            "risk_band": tx.action, "recommended_action": tx.action,
            "model_version": tx.model_version, "explanations": json.loads(tx.explanations),
            "disposition": tx.disposition, "note": tx.note,
            "analyst_at": tx.analyst_at, "scored_at": tx.scored_at}


@app.get("/health")
def health(db: Session = Depends(db_session)):
    try:
        db.execute(select(1))
        bundle = artifact()
    except SQLAlchemyError as exc:
        raise HTTPException(503, "Database unavailable") from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"status": "ok", "model_ready": True, "model_version": bundle["version"]}


@app.post("/transactions/score", status_code=201)
def score(payment: Payment, db: Session = Depends(db_session)):
    if payment.type not in PAYMENT_TYPES:
        raise HTTPException(422, f"Unsupported payment type. Use one of {PAYMENT_TYPES}")
    if db.get(Transaction, payment.transaction_id):
        raise HTTPException(409, "Transaction ID already exists")
    try:
        bundle = artifact()
    except (OSError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc
    x = feature_frame([payment.model_dump()])
    probability = float(bundle["model"].predict_proba(x)[0, 1])
    action = action_for(probability, bundle["review_threshold"], bundle["decline_threshold"])
    raw = explainer(bundle["explainer_model"]).shap_values(x)
    values = raw[1][0] if isinstance(raw, list) else raw[0]
    if getattr(values, "ndim", 1) > 1:
        values = values[:, 1]
    explanations = sorted(({"feature": name, "contribution": float(value)}
                           for name, value in zip(x.columns, values)),
                          key=lambda item: abs(item["contribution"]), reverse=True)[:5]
    tx = Transaction(id=payment.transaction_id, step=payment.step, type=payment.type,
                     amount=payment.amount, name_orig=payment.nameOrig, oldbalance_org=payment.oldbalanceOrg,
                     name_dest=payment.nameDest, oldbalance_dest=payment.oldbalanceDest,
                     is_fraud=payment.isFraud, probability=probability, action=action,
                     explanations=json.dumps(explanations), model_version=bundle["version"],
                     scored_at=datetime.now(timezone.utc))
    db.add(tx)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        if db.get(Transaction, payment.transaction_id):
            raise HTTPException(409, "Transaction ID already exists") from exc
        raise
    db.refresh(tx)
    return serialize(tx)


@app.get("/transactions")
def transactions(risk_band: str | None = None, payment_type: str | None = None,
                 disposition: str | None = None, min_amount: float = Query(default=0, ge=0),
                 max_amount: float | None = Query(default=None, ge=0),
                 limit: int = Query(default=200, ge=1, le=1000), db: Session = Depends(db_session)):
    query = select(Transaction).where(Transaction.amount >= min_amount)
    if max_amount is not None:
        if max_amount < min_amount:
            raise HTTPException(422, "Maximum amount must be at least the minimum amount")
        query = query.where(Transaction.amount <= max_amount)
    if risk_band:
        query = query.where(Transaction.action == risk_band)
    if payment_type:
        query = query.where(Transaction.type == payment_type)
    if disposition == "pending":
        query = query.where(Transaction.disposition.is_(None))
    elif disposition:
        query = query.where(Transaction.disposition == disposition)
    return [serialize(tx) for tx in db.scalars(query.order_by(Transaction.scored_at.desc()).limit(limit))]


@app.get("/transactions/{transaction_id}")
def transaction(transaction_id: str, db: Session = Depends(db_session)):
    tx = db.get(Transaction, transaction_id)
    if not tx:
        raise HTTPException(404, "Transaction not found")
    return serialize(tx)


@app.patch("/transactions/{transaction_id}/disposition")
def set_disposition(transaction_id: str, update: Disposition, db: Session = Depends(db_session)):
    if update.disposition not in ("approve", "review", "decline"):
        raise HTTPException(422, "Disposition must be approve, review or decline")
    tx = db.get(Transaction, transaction_id)
    if not tx:
        raise HTTPException(404, "Transaction not found")
    tx.disposition, tx.note, tx.analyst_at = update.disposition, update.note, datetime.now(timezone.utc)
    db.commit()
    db.refresh(tx)
    return serialize(tx)


@app.get("/metrics")
def metrics(db: Session = Depends(db_session)):
    declined = Transaction.action == "decline"
    total, alerts, declines, backlog, exposure = db.execute(select(
        func.count(Transaction.id),
        func.count(Transaction.id).filter(Transaction.action != "approve"),
        func.count(Transaction.id).filter(declined),
        func.count(Transaction.id).filter(Transaction.action == "review", Transaction.disposition.is_(None)),
        func.coalesce(func.sum(Transaction.amount).filter(declined, Transaction.is_fraud == 1), 0),
    )).one()
    return {"processed": total, "fraud_alerts": alerts, "decline_rate": declines / total if total else 0,
            "review_backlog": backlog, "estimated_blocked_loss": exposure}
