import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from pipeline.clean import clean_transactions
from pipeline.segment import segment_customers, segments_by_store, summarise_segments

# where things are read from and written to
RAW = Path("data/transactions.csv")
PROCESSED = Path("data/processed")
REPORTS = Path("reports")

# answer keys from the generator, only used to check our work
TRUTH = Path("data/customers_truth.csv")
PLANTED = Path("data/injected_errors.csv")


def stage(name):
    # print a header for each stage and time how long it takes
    def wrap(fn):
        def run(*args, **kwargs):
            print(f"\n[{name}]")
            start = time.time()
            result = fn(*args, **kwargs)
            print(f"  done in {time.time() - start:.1f}s")
            return result
        return run
    return wrap


@stage("generate")
def generate(rows):
    # rebuild the raw till data from scratch
    subprocess.run([sys.executable, "generate_data.py", "--rows", str(rows)], check=True)


@stage("load")
def load():
    if not RAW.exists():
        sys.exit(f"  {RAW} not found, run with --generate first")
    # read everything as text so ids like 000161 keep their leading zeros
    raw = pd.read_csv(RAW, dtype=str, keep_default_na=False, na_values=[""])
    print(f"  {len(raw):,} rows from {RAW}")
    return raw


@stage("clean")
def clean(raw):
    # fix and flag the till errors, print what was found
    df, report = clean_transactions(raw)
    for k, v in report.items():
        print(f"  {k:34} {v:>9,}")
    return df, report


@stage("segment")
def segment(df):
    # score every customer and summarise the segments
    rfm, snapshot = segment_customers(df)
    summary = summarise_segments(rfm)
    stores = segments_by_store(rfm)
    print(f"  {len(rfm):,} customers scored, snapshot {snapshot.date()}")
    print(summary[["segment", "customers", "share_of_customers", "share_of_revenue"]].to_string(index=False))
    return rfm, summary, stores, snapshot


@stage("check")
def check(report, rfm):
    # skipped automatically on real data, which has no answer keys
    if not (TRUTH.exists() and PLANTED.exists()):
        print("  no answer keys, skipping")
        return None, None

    # what we planted vs what cleaning found
    planted = pd.read_csv(PLANTED)
    found = {
        "customer_id_format": report["customer_ids_reformatted"],
        "store_mismatch": report["store_names_corrected"],
        "missing_price": report["missing_prices_filled"],
        "clock_reset": report["clock_reset_rows_removed"],
        "invoice_clash": report["clashing_receipts_split"],
        "duplicate_upload": report["receipts_sent_twice"],
        "test_transaction": report["test_receipts_removed"],
        "price_as_text": report["prices_with_pound_sign"],
        "out_of_hours": report["out_of_hours_rows_flagged"],
        "void_pair": report["void_lines_netted"],
    }
    # some errors are logged per receipt, others per row
    per_receipt = {"invoice_clash", "duplicate_upload", "test_transaction"}
    expected = {
        k: planted.loc[planted["error_type"] == k, "reference"].nunique() if k in per_receipt
        else int((planted["error_type"] == k).sum())
        for k in found
    }
    cleaning = pd.DataFrame({"planted": expected, "found": found})
    cleaning.index.name = "error_type"
    print(cleaning.to_string())

    # did the hidden customer types land in sensible segments?
    truth = pd.read_csv(TRUTH)
    merged = rfm.merge(truth[["customer_id", "true_persona"]], on="customer_id")
    crosstab = pd.crosstab(merged["segment"], merged["true_persona"])
    print()
    print(crosstab.to_string())
    return cleaning.reset_index(), crosstab.reset_index()


@stage("save")
def save(df, rfm, summary, stores, report, snapshot, cleaning, crosstab):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    # big row-level outputs go in data/processed (not in git)
    df.to_csv(PROCESSED / "transactions_clean.csv", index=False)
    rfm.to_csv(PROCESSED / "customer_segments.csv", index=False)

    # small summary outputs go in reports/ (kept in git)
    summary.to_csv(REPORTS / "segment_summary.csv", index=False)
    stores.to_csv(REPORTS / "segments_by_store.csv", index=False)
    if cleaning is not None:
        cleaning.to_csv(REPORTS / "cleaning_check.csv", index=False)
        crosstab.to_csv(REPORTS / "segment_vs_truth.csv", index=False)

    # the full cleaning tally, for the README and for comparing runs
    run_info = {"snapshot_date": str(snapshot.date()), "cleaning": report}
    (REPORTS / "run_report.json").write_text(json.dumps(run_info, indent=2))

    # list everything that was written
    for p in sorted(PROCESSED.glob("*.csv")) + sorted(REPORTS.glob("*")):
        print(f"  {p}")


def main():
    # options: --generate rebuilds the raw data first, --rows sets its size
    ap = argparse.ArgumentParser(description="till data -> clean -> customer segments")
    ap.add_argument("--generate", action="store_true", help="regenerate the raw data first")
    ap.add_argument("--rows", type=int, default=287_413, help="rows to generate (with --generate)")
    args = ap.parse_args()

    # run each stage in order, passing results along
    start = time.time()
    if args.generate:
        generate(args.rows)
    raw = load()
    df, report = clean(raw)
    rfm, summary, stores, snapshot = segment(df)
    cleaning, crosstab = check(report, rfm)
    save(df, rfm, summary, stores, report, snapshot, cleaning, crosstab)
    print(f"\npipeline finished in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
