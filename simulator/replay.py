import argparse
import csv
import json
from pathlib import Path
import time
from uuid import uuid4
import httpx
import pandas as pd

from training.prepare import validate_rows

PAYMENT_FIELDS = ("step", "type", "amount", "nameOrig", "oldbalanceOrg", "nameDest", "oldbalanceDest")


def export_test_events(data, profile, output):
    bounds = profile["splits"]["test"]
    chunks = []
    for raw in pd.read_csv(data, chunksize=250_000):
        clean, _, _ = validate_rows(raw)
        held_out = clean[clean.step.between(bounds["first_step"], bounds["last_step"])]
        if not held_out.empty:
            chunks.append(held_out.loc[:, [*PAYMENT_FIELDS, "isFraud"]])
    if not chunks:
        raise ValueError("No held-out events found")
    events = pd.concat(chunks).sort_values("step", kind="stable")
    if len(events) != bounds["rows"] or int(events.isFraud.sum()) != bounds["fraud"]:
        raise ValueError("Test event totals do not match the data profile")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(output, index=False)
    return len(events)


def replay(data, url, rate, limit=None, start_step=0, *, run_id=None, stop=None, on_event=print):
    if rate <= 0:
        raise ValueError("rate must be positive")
    with open(data, newline="", encoding="utf-8") as handle, httpx.Client(timeout=30) as client:
        sent = 0
        for index, row in enumerate(csv.DictReader(handle)):
            if stop is not None and stop.is_set():
                break
            if int(row["step"]) < start_step:
                continue
            if limit is not None and sent >= limit:
                break
            started = time.monotonic()
            payload = {key: row[key] for key in PAYMENT_FIELDS}
            payment_id = row.get("transaction_id") or f"replay-{index}-{row['step']}"
            payload["transaction_id"] = f"{run_id}-{payment_id}" if run_id else payment_id
            payload["isFraud"] = int(row["isFraud"]) if row.get("isFraud") else None
            response = client.post(f"{url.rstrip('/')}/transactions/score", json=payload)
            response.raise_for_status()
            on_event(payload["transaction_id"], response.status_code)
            sent += 1
            if limit is not None and sent >= limit:
                break
            delay = max(0, 1 / rate - (time.monotonic() - started))
            if stop is None:
                time.sleep(delay)
            elif stop.wait(delay):
                break
        return sent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fixture/paysim_sample.csv")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--rate", type=float, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    if args.export:
        if not args.profile:
            parser.error("--profile is required for --export")
        count = export_test_events(args.data, json.loads(args.profile.read_text(encoding="utf-8")), args.export)
        print(f"Exported {count:,} held-out events")
        parser.exit()
    try:
        replay(args.data, args.url, args.rate, args.limit, args.start_step,
               run_id=uuid4().hex[:12], on_event=lambda *event: print(*event, flush=True))
    except KeyboardInterrupt:
        print("Replay stopped")
