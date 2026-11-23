import os
from pathlib import Path
from urllib.parse import quote

import httpx
import pandas as pd
import streamlit as st

from common.features import PAYMENT_TYPES
from simulator.control import ReplayController

api_host = os.getenv("API_HOSTPORT")
API = os.getenv("API_URL", f"http://{api_host}" if api_host else "http://localhost:8000").rstrip("/")
READ_ONLY = os.getenv("DEMO_READ_ONLY") == "1"
TEST_EVENTS = Path("artifacts/phase3/test_events.csv")
ACTIONS = ["approve", "review", "decline"]

st.set_page_config(page_title="Fraud risk console", layout="wide")
st.title("Payment fraud risk console")
if READ_ONLY:
    st.caption("Read-only portfolio demo · curated synthetic PaySim examples")


@st.cache_resource
def replay_controller():
    return ReplayController()


def request(method, path, **kwargs):
    response = httpx.request(method, f"{API}{path}", timeout=10, **kwargs)
    response.raise_for_status()
    return response.json()


with st.sidebar:
    st.header("Alert filters")
    band = st.selectbox("Risk band", ["All", *ACTIONS])
    payment_type = st.selectbox("Payment type", ["All", *PAYMENT_TYPES])
    disposition = st.selectbox("Disposition", ["All", "pending", *ACTIONS])
    minimum = st.number_input("Minimum amount", min_value=0., value=0.)
    maximum = st.number_input("Maximum amount (0 for no limit)", min_value=0., value=0.)
    live = st.toggle("Live refresh", value=not READ_ONLY)
    st.button("Refresh now")


def replay_controls():
    controller = replay_controller()
    status, sent, error, run_id = controller.snapshot()
    running = status in ("Running", "Stopping")
    st.subheader("Transaction replay")
    use_full_data = TEST_EVENTS.is_file()
    if use_full_data:
        st.caption("Held-out PaySim transactions · steps 595–743")
    else:
        st.caption("Fixture transactions · steps 9–10")
    rate_col, limit_col, step_col = st.columns(3)
    rate = rate_col.number_input("Transactions per second", 0.2, 50.0, 5.0, step=0.2, disabled=running)
    limit = limit_col.number_input("Maximum transactions", 1, 123580, 200, disabled=running)
    start_step = step_col.number_input("Start step", 595, 743, 595, disabled=running) if use_full_data else 9
    start, stop, progress = st.columns([1, 1, 4])
    if start.button("Start replay", disabled=running):
        data = TEST_EVENTS if use_full_data else Path("data/fixture/paysim_sample.csv")
        controller.start(data, API, rate, limit, start_step)
        st.rerun()
    if stop.button("Stop replay", disabled=status != "Running"):
        controller.stop()
        st.rerun()
    progress.write(f"**{status}** · {sent:,} sent" + (f" · Run {run_id}" if run_id else ""))
    if error:
        st.error(f"Replay stopped with an error: {error}")


def show_details(row):
    st.subheader("Transaction details")
    a, b, c, d = st.columns(4)
    a.metric("Amount", f"{row['amount']:,.2f}")
    b.metric("Fraud score", f"{row['fraud_probability']:.2%}")
    c.metric("Recommendation", row["recommended_action"].title())
    d.metric("Analyst disposition", (row["disposition"] or "pending").title())
    st.caption(f"{row['type']} · PaySim step {row['step']} · Model {row['model_version']}")
    st.subheader("Leading SHAP contributions")
    st.bar_chart(pd.DataFrame(row["explanations"]).set_index("feature")["contribution"])
    st.caption("Contributions explain the model output before probability calibration.")
    if READ_ONLY:
        return
    transaction_id = row["transaction_id"]
    with st.form(f"disposition-{transaction_id}"):
        choice = st.selectbox("Analyst disposition", ACTIONS,
                              index=ACTIONS.index(row["disposition"] or row["recommended_action"]))
        note = st.text_area("Note", value=row["note"] or "", max_chars=2000)
        if st.form_submit_button("Save disposition"):
            request("PATCH", f"/transactions/{quote(transaction_id, safe='')}/disposition",
                    json={"disposition": choice, "note": note})
            st.session_state["saved_disposition"] = transaction_id
            st.rerun()


def show_trends(frame):
    st.subheader("Trends for matching transactions")
    frame = frame.assign(minute=pd.to_datetime(frame.scored_at, utc=True, format="mixed").dt.floor("min"),
                         alert=frame.recommended_action.ne("approve").astype(int),
                         isFraud=pd.to_numeric(frame.isFraud, errors="coerce"))
    left, right = st.columns(2)
    left.caption("Alert volume by minute (UTC)")
    left.line_chart(frame.groupby("minute")["alert"].sum())
    right.caption("Known fraud rate by minute (labeled payments only)")
    fraud_rate = frame.groupby("minute")["isFraud"].mean().dropna()
    if fraud_rate.empty:
        right.info("No replay labels in this selection.")
    else:
        right.line_chart(fraud_rate)
    left.caption("Risk band distribution")
    left.bar_chart(frame.recommended_action.value_counts().reindex(ACTIONS, fill_value=0))
    right.caption("Transaction amount by minute (UTC)")
    right.line_chart(frame.groupby("minute")["amount"].sum())


@st.fragment(run_every=3 if live else None)
def console():
    if not READ_ONLY:
        replay_controls()
    try:
        metrics = request("GET", "/metrics")
        labels = ["Processed", "Fraud alerts", "Decline rate", "Review backlog", "Known fraud amount declined"]
        values = [f"{metrics['processed']:,}", f"{metrics['fraud_alerts']:,}", f"{metrics['decline_rate']:.1%}",
                  f"{metrics['review_backlog']:,}", f"{metrics['estimated_blocked_loss']:,.0f}"]
        for column, label, value in zip(st.columns(5), labels, values):
            column.metric(label, value)

        if "saved_disposition" in st.session_state:
            st.success(f"Saved disposition for {st.session_state.pop('saved_disposition')}")
        params = {key: value for key, value in (("risk_band", band), ("payment_type", payment_type),
                                                 ("disposition", disposition)) if value != "All"}
        if maximum and maximum < minimum:
            st.warning("Maximum amount must be at least the minimum amount.")
            return
        params.update(min_amount=minimum, limit=1000)
        if maximum:
            params["max_amount"] = maximum
        rows = request("GET", "/transactions", params=params)
        st.subheader("Alert queue")
        if not rows:
            st.info("No transactions match these filters.")
            return
        frame = pd.DataFrame(rows)
        st.caption(f"{len(rows):,} matching records · up to 1,000 most recent")
        st.dataframe(frame[["transaction_id", "scored_at", "type", "amount", "fraud_probability",
                            "recommended_action", "disposition", "model_version"]], width="stretch", hide_index=True)
        selected = st.selectbox("Inspect transaction", frame.transaction_id.tolist(), key="selected_transaction")
        show_details(request("GET", f"/transactions/{quote(selected, safe='')}"))
        show_trends(frame)
    except (httpx.HTTPError, KeyError) as exc:
        st.error(f"Could not load or update the analyst console: {exc}")


console()
