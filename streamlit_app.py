"""Read-only portfolio demo for Streamlit Community Cloud."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="PaySim fraud risk", layout="wide")
st.title("Payment fraud risk dashboard")
st.caption("Held-out PaySim examples · LightGBM 2.0.0 · Precomputed scores and SHAP explanations")

snapshot = Path(__file__).resolve().parent / "deploy" / "demo" / "scores.json"
frame = pd.DataFrame(json.loads(snapshot.read_text(encoding="utf-8")))
alerts = frame.recommended_action.ne("approve")
columns = st.columns(4)
columns[0].metric("Sample payments", len(frame))
columns[1].metric("Flagged payments", int(alerts.sum()))
columns[2].metric("Recommended review", int(frame.recommended_action.eq("review").sum()))
columns[3].metric("Recommended decline", int(frame.recommended_action.eq("decline").sum()))
st.caption("The curated sample includes extra fraud examples. Its alert rate reflects the sample selection.")

with st.sidebar:
    st.header("Transaction filters")
    action = st.selectbox("Recommended action", ["All", "approve", "review", "decline"])
    kind = st.selectbox("Payment type", ["All", *sorted(frame.type.unique())])
    minimum = st.number_input("Minimum amount", min_value=0., value=0.)

matching = frame[frame.amount >= minimum]
if action != "All":
    matching = matching[matching.recommended_action.eq(action)]
if kind != "All":
    matching = matching[matching.type.eq(kind)]

st.subheader("Transaction queue")
if matching.empty:
    st.info("No transactions match these filters.")
else:
    matching = matching.sort_values("fraud_probability", ascending=False)
    display = matching[["transaction_id", "step", "type", "amount", "fraud_probability", "recommended_action"]]
    st.dataframe(display, width="stretch", hide_index=True,
                 column_config={"fraud_probability": st.column_config.NumberColumn("Fraud probability", format="percent")})
    selected = st.selectbox("Inspect transaction", matching.transaction_id.tolist())
    row = matching.set_index("transaction_id").loc[selected]
    details = st.columns(3)
    details[0].metric("Fraud probability", f"{row.fraud_probability:.2%}")
    details[1].metric("Model recommendation", row.recommended_action.title())
    details[2].metric("PaySim label", "Fraud" if row.isFraud else "Legitimate")
    st.subheader("Top SHAP contributions")
    st.bar_chart(pd.DataFrame(row.explanations).set_index("feature")["contribution"])
    st.caption("SHAP explains the LightGBM output before probability calibration.")

st.subheader("Held-out model evaluation")
st.write("On 123,580 held-out payments, LightGBM achieved 0.341 average precision and 33.6% fraud recall at a 1% false-positive rate. Logistic regression achieved 0.114 and 15.3%, respectively.")
st.markdown("[Source code and local replay setup](https://github.com/vedasai-17/paysim-fraud-detection) · [PaySim dataset](https://www.kaggle.com/datasets/ealaxi/paysim1/metadata)")
