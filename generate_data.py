import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2025-12-31")

# orders per year, mean extra lines per order, share of customers
PERSONAS = {
    "loyal":      {"orders_per_year": 30, "extra_lines": 3.0, "share": 0.10},
    "regular":    {"orders_per_year": 12, "extra_lines": 2.0, "share": 0.20},
    "occasional": {"orders_per_year": 4,  "extra_lines": 1.5, "share": 0.30},
    "one_off":    {"orders_per_year": 0,  "extra_lines": 1.0, "share": 0.25},
    "lapsed":     {"orders_per_year": 10, "extra_lines": 2.0, "share": 0.15},
}

CATEGORIES = {
    "Candles":     (["Vanilla", "Lavender", "Fig", "Cedar", "Sea Salt", "Amber"], ["Jar Candle", "Pillar Candle", "Tealight Set", "Reed Diffuser"], 9),
    "Kitchen":     (["Enamel", "Stoneware", "Bamboo", "Retro", "Copper"], ["Mug", "Teapot", "Utensil Set", "Storage Tin", "Chopping Board"], 12),
    "Home Decor":  (["Rustic", "Nordic", "Botanical", "Vintage", "Coastal"], ["Cushion", "Photo Frame", "Wall Clock", "Vase", "Throw"], 18),
    "Stationery":  (["Floral", "Kraft", "Pastel", "Monochrome"], ["Notebook", "Card Pack", "Wrapping Paper", "Planner"], 5),
    "Garden":      (["Galvanised", "Terracotta", "Solar", "Willow"], ["Planter", "Bird Feeder", "String Lights", "Watering Can"], 15),
    "Seasonal":    (["Christmas", "Easter", "Halloween", "Valentine"], ["Bauble Set", "Garland", "Gift Box", "Decoration Pack"], 8),
}

# one store per Greater Manchester borough, weighted by footfall
STORES = [
    ("S01", "Manchester Arndale", 0.18),
    ("S02", "Stockport", 0.11),
    ("S03", "Bolton", 0.11),
    ("S04", "Wigan", 0.10),
    ("S05", "Trafford", 0.10),
    ("S06", "Salford", 0.09),
    ("S07", "Oldham", 0.08),
    ("S08", "Bury", 0.08),
    ("S09", "Rochdale", 0.08),
    ("S10", "Ashton-under-Lyne", 0.07),
]


def build_products(rng):
    rows = []
    for cat, (styles, items, base) in CATEGORIES.items():
        for s in styles:
            for it in items:
                price = round(base * rng.lognormal(0, 0.35), 2)
                rows.append({"product": f"{s} {it}", "category": cat, "price": max(price, 1.5)})
    return pd.DataFrame(rows)


def day_weights(days):
    month_boost = {11: 1.4, 12: 1.9, 1: 0.75, 2: 0.85}
    w = np.array([month_boost.get(d.month, 1.0) for d in days])
    w *= np.where(days.dayofweek >= 5, 1.3, 1.0)
    return w


def estimate_lines_per_customer():
    total = 0.0
    for p in PERSONAS.values():
        if p["orders_per_year"] == 0:
            orders = 1
        else:
            orders = max(1, p["orders_per_year"] * 1.3)
        total += p["share"] * orders * (1 + p["extra_lines"])
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=287_413)
    ap.add_argument("--out", default="data/transactions.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    days = pd.date_range(START, END, freq="D")
    weights = day_weights(days)
    n_days = len(days)

    products = build_products(rng)
    product_p = rng.dirichlet(np.ones(len(products)) * 0.8)

    n_customers = max(200, int(args.rows / estimate_lines_per_customer()))
    names = list(PERSONAS)
    shares = [PERSONAS[n]["share"] for n in names]
    persona = rng.choice(names, size=n_customers, p=shares)
    customer_ids = [f"C{i:06d}" for i in range(1, n_customers + 1)]
    store_ids = np.array([s[0] for s in STORES])
    store_names = np.array([s[1] for s in STORES])
    store_p = np.array([s[2] for s in STORES])
    home_store = rng.choice(len(STORES), size=n_customers, p=store_p)

    order_cust, order_day = [], []
    for i, p in enumerate(persona):
        if p == "lapsed":
            start = int(rng.uniform(0, 0.3) * n_days)
            end = int(rng.uniform(0.35, 0.7) * n_days)
        elif p == "loyal":
            start = int(rng.uniform(0, 0.25) * n_days)
            end = n_days
        else:
            start = int(rng.uniform(0, 0.85) * n_days)
            end = n_days

        if p == "one_off":
            k = 1
        else:
            years = (end - start) / 365
            k = max(1, rng.poisson(PERSONAS[p]["orders_per_year"] * years))

        w = weights[start:end]
        picked = rng.choice(np.arange(start, end), size=k, p=w / w.sum())
        order_cust.extend([i] * k)
        order_day.extend(picked)

    order_cust = np.array(order_cust)
    order_day = np.array(order_day)
    n_orders = len(order_cust)

    # stores open 9am to 8pm, lunchtime and after-work peaks
    hours = rng.choice(np.arange(9, 20), size=n_orders, p=np.array([4, 6, 8, 11, 11, 9, 8, 9, 11, 13, 10]) / 100)
    minutes = rng.integers(0, 60, size=n_orders)
    order_ts = days[order_day] + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")

    sort_idx = np.argsort(order_ts.values)
    order_cust = order_cust[sort_idx]
    order_ts = order_ts[sort_idx]

    # most visits are to the customer's home store, some to others
    order_store = home_store[order_cust]
    away = rng.random(n_orders) < 0.15
    order_store[away] = rng.choice(len(STORES), size=away.sum(), p=store_p)
    invoice_ids = np.array([f"INV{i:07d}" for i in range(1, n_orders + 1)])

    extra = np.array([PERSONAS[persona[c]]["extra_lines"] for c in order_cust])
    lines = 1 + rng.poisson(extra)

    line_order = np.repeat(np.arange(n_orders), lines)
    n_lines = len(line_order)
    prod_idx = rng.choice(len(products), size=n_lines, p=product_p)

    qty = rng.choice([1, 2, 3, 4, 6, 12], size=n_lines, p=[0.55, 0.22, 0.1, 0.06, 0.05, 0.02])
    price = products["price"].to_numpy()[prod_idx]
    on_sale = rng.random(n_lines) < 0.08
    price = np.where(on_sale, np.round(price * 0.8, 2), price)

    cust_for_line = order_cust[line_order]
    df = pd.DataFrame({
        "invoice_id": invoice_ids[line_order],
        "customer_id": np.array(customer_ids)[cust_for_line],
        "invoice_date": order_ts[line_order],
        "product": products["product"].to_numpy()[prod_idx],
        "category": products["category"].to_numpy()[prod_idx],
        "quantity": qty,
        "unit_price": price,
        "store_id": store_ids[order_store[line_order]],
        "store_name": store_names[order_store[line_order]],
    })

    # refunds: some lines returned a few days later, sometimes at another store
    refund_mask = rng.random(len(df)) < 0.02
    refunds = df[refund_mask].copy()
    refunds["invoice_id"] = "RF" + refunds["invoice_id"].str[3:]
    refunds["quantity"] = -refunds["quantity"]
    refunds["invoice_date"] = refunds["invoice_date"] + pd.to_timedelta(rng.integers(2, 21, len(refunds)), unit="D")
    elsewhere = rng.random(len(refunds)) < 0.3
    other = rng.choice(len(STORES), size=elsewhere.sum(), p=store_p)
    refunds.loc[elsewhere, "store_id"] = store_ids[other]
    refunds.loc[elsewhere, "store_name"] = store_names[other]
    refunds = refunds[refunds["invoice_date"] <= END + pd.Timedelta(days=1)]

    # shoppers who didn't scan a loyalty card
    no_card = df["invoice_id"].isin(rng.choice(invoice_ids, size=int(n_orders * 0.04), replace=False))
    df.loc[no_card, "customer_id"] = None

    df = pd.concat([df, refunds]).reset_index(drop=True)
    df, errors = add_till_errors(df, rng, products, customer_ids, store_ids, store_names)

    df = df.sort_values("invoice_date", kind="stable").reset_index(drop=True)
    fmt = np.where(df["store_id"] == "S09", "uk", "iso")
    df["invoice_date"] = np.where(
        fmt == "uk",
        df["invoice_date"].dt.strftime("%d/%m/%Y %H:%M"),
        df["invoice_date"].dt.strftime("%Y-%m-%d %H:%M"),
    )
    errors.append(("uk_date_format", "S09", "Rochdale till exports dates as dd/mm/yyyy"))

    err_share = df["_err"].mean()
    any_share = (df["_err"] | (df["store_id"] == "S09")).mean()
    df = df.drop(columns="_err")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    truth = pd.DataFrame({
        "customer_id": customer_ids,
        "home_store": store_names[home_store],
        "true_persona": persona,
    })
    truth.to_csv(out.with_name("customers_truth.csv"), index=False)

    log = pd.DataFrame(errors, columns=["error_type", "reference", "detail"])
    log.to_csv(out.with_name("injected_errors.csv"), index=False)

    print(f"rows: {len(df):,}")
    print(f"customers: {n_customers:,}")
    print(f"invoices: {n_orders:,}")
    print(f"saved: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print()
    print(truth["true_persona"].value_counts().to_string())
    print()
    print("injected errors:")
    print(log["error_type"].value_counts().to_string())
    print()
    print(f"rows with an error: {err_share:.1%}")
    print(f"including Rochdale date format: {any_share:.1%}")


def mark(df, mask_or_idx, errors, kind, detail):
    if isinstance(mask_or_idx, pd.Series):
        idx = df.index[mask_or_idx]
    else:
        idx = pd.Index(mask_or_idx)
    df.loc[idx, "_err"] = True
    errors += [(kind, df.at[i, "invoice_id"], detail) for i in idx]


def typo(word, rng):
    if len(word) < 4:
        return word
    i = int(rng.integers(1, len(word) - 2))
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def add_till_errors(df, rng, products, customer_ids, store_ids, store_names):
    errors = []
    df["_err"] = False
    sales = df[df["quantity"] > 0]
    n = len(sales)

    def pick(share):
        return rng.choice(sales.index, size=int(n * share), replace=False)

    # keyed the wrong quantity, then voided the extra on the same receipt
    voids = []
    for i in pick(0.015):
        right = df.at[i, "quantity"]
        wrong = int(str(right) + str(rng.integers(0, 10)))
        df.at[i, "quantity"] = wrong
        df.at[i, "_err"] = True
        v = df.loc[i].copy()
        v["quantity"] = right - wrong
        v["invoice_date"] = v["invoice_date"] + pd.Timedelta(minutes=1)
        voids.append(v)
        errors.append(("void_pair", df.at[i, "invoice_id"], f"keyed {wrong}, voided {wrong - right}"))
    df = pd.concat([df, pd.DataFrame(voids)], ignore_index=True)

    # barcode wouldn't scan, cashier used the misc button
    idx = pick(0.006)
    df.loc[idx, "product"] = "MISC ITEM"
    df.loc[idx, "category"] = None
    df.loc[idx, "unit_price"] = np.round(rng.uniform(1, 30, len(idx)), 2)
    mark(df, idx, errors, "misc_item", "unscannable item keyed manually")

    # manager price override for damaged stock
    idx = pick(0.008)
    df.loc[idx, "unit_price"] = np.round(df.loc[idx, "unit_price"] * rng.uniform(0.3, 0.7, len(idx)), 2)
    mark(df, idx, errors, "price_override", "damaged stock discount")

    # wrong price in the product system for two weeks
    bad_product = products["product"].iloc[int(rng.integers(len(products)))]
    start = pd.Timestamp("2025-06-02")
    window = (df["product"] == bad_product) & df["invoice_date"].between(start, start + pd.Timedelta(days=14))
    df.loc[window, "unit_price"] = np.round(df.loc[window, "unit_price"] / 10, 2)
    df.loc[window, "_err"] = True
    errors.append(("master_price_error", bad_product, f"priced at a tenth from {start.date()} for 14 days, {int(window.sum())} lines"))

    # sync failure left the price blank
    idx = pick(0.004)
    df.loc[idx, "unit_price"] = np.nan
    mark(df, idx, errors, "missing_price", "price lost in sync")

    # a few customers share one family card
    ids = np.array(customer_ids)
    n_shared = int(len(ids) * 0.03)
    movers = rng.choice(ids, size=n_shared, replace=False)
    hosts = rng.choice(np.setdiff1d(ids, movers), size=n_shared)
    mapping = dict(zip(movers, hosts))
    moved = df["customer_id"].isin(movers)
    df.loc[moved, "customer_id"] = df.loc[moved, "customer_id"].map(mapping)
    df.loc[moved, "_err"] = True
    errors += [("shared_card", m, f"shopping recorded under {h}") for m, h in mapping.items()]

    invs = df.loc[df["invoice_id"].str.startswith("INV"), "invoice_id"].unique()

    def pick_invoices(share):
        return rng.choice(invs, size=int(len(invs) * share), replace=False)

    # bad till clock: middle of the night
    odd = pick_invoices(0.003)
    rows = df["invoice_id"].isin(odd)
    hrs = df.loc[rows, "invoice_id"].map(dict(zip(odd, rng.integers(1, 6, len(odd)))))
    df.loc[rows, "invoice_date"] = df.loc[rows, "invoice_date"].dt.normalize() + pd.to_timedelta(hrs, unit="h") \
        + pd.to_timedelta(df.loc[rows, "invoice_date"].dt.minute, unit="m")
    mark(df, rows, errors, "out_of_hours", "till clock wrong")

    # till battery died and the clock reset to a default date
    reset = pick_invoices(0.002)
    rows = df["invoice_id"].isin(reset)
    base = df.loc[rows, "invoice_id"].map(dict(zip(reset, rng.choice(["1970-01-01", "2000-01-01"], len(reset)))))
    df.loc[rows, "invoice_date"] = pd.to_datetime(base) + (df.loc[rows, "invoice_date"] - df.loc[rows, "invoice_date"].dt.normalize())
    mark(df, rows, errors, "clock_reset", "date reset to till default")

    # till moved between stores but kept its old store name
    moved_till = pick_invoices(0.005)
    rows = df["invoice_id"].isin(moved_till)
    wrong_store = df.loc[rows, "invoice_id"].map(dict(zip(moved_till, rng.integers(0, len(store_ids), len(moved_till)))))
    df.loc[rows, "store_name"] = store_names[wrong_store.to_numpy()]
    mismatch = rows & (df["store_name"] != pd.Series(store_names, index=store_ids).reindex(df["store_id"]).to_numpy())
    mark(df, mismatch, errors, "store_mismatch", "store name doesn't match store id")

    # two tills issued the same receipt number
    clash = pick_invoices(0.006)
    targets = rng.choice(np.setdiff1d(invs, clash), size=len(clash), replace=False)
    rows = df["invoice_id"].isin(clash)
    df.loc[rows, "invoice_id"] = df.loc[rows, "invoice_id"].map(dict(zip(clash, targets)))
    mark(df, rows, errors, "invoice_clash", "receipt number reused by another till")

    # staff test transactions at £0.00
    tests = []
    for k in range(max(5, len(invs) // 800)):
        day = START + pd.Timedelta(days=int(rng.integers(0, 730)))
        store = int(rng.integers(len(store_ids)))
        inv = f"TST{k + 1:05d}"
        for _ in range(int(rng.integers(1, 4))):
            tests.append({
                "invoice_id": inv, "customer_id": "C999999",
                "invoice_date": day + pd.Timedelta(hours=7, minutes=int(rng.integers(0, 60))),
                "product": "TEST ITEM", "category": None, "quantity": 1, "unit_price": 0.0,
                "store_id": store_ids[store], "store_name": store_names[store], "_err": True,
            })
        errors.append(("test_transaction", inv, "till test before opening"))
    df = pd.concat([df, pd.DataFrame(tests)], ignore_index=True)

    # typos from manual edits to the product list
    named = df.index[df["product"].ne("MISC ITEM") & df["product"].ne("TEST ITEM")]
    idx = rng.choice(named, size=int(len(df) * 0.012), replace=False)
    df.loc[idx, "product"] = [" ".join(typo(w, rng) if j == 0 else w for j, w in enumerate(p.split())) for p in df.loc[idx, "product"]]
    mark(df, idx, errors, "product_typo", "misspelt product name")

    # older till software: odd case and stray spaces
    idx = rng.choice(df.index, size=int(len(df) * 0.03), replace=False)
    style = rng.integers(0, 3, len(idx))
    col = np.where(rng.random(len(idx)) < 0.5, "product", "store_name")
    for i, s, c in zip(idx, style, col):
        v = df.at[i, c]
        df.at[i, c] = [v.upper(), v.lower(), f" {v}  "][s]
    mark(df, idx, errors, "text_format", "inconsistent case or spacing")

    # card reader quirks in customer ids
    has_id = df.index[df["customer_id"].notna() & df["customer_id"].ne("C999999")]
    idx = rng.choice(has_id, size=int(len(df) * 0.02), replace=False)
    style = rng.integers(0, 3, len(idx))
    for i, s in zip(idx, style):
        v = df.at[i, "customer_id"]
        df.at[i, "customer_id"] = [v.lower(), v[1:], f"{v} "][s]
    mark(df, idx, errors, "customer_id_format", "customer id in a different format")

    # some tills export prices as text with a pound sign
    df["unit_price"] = df["unit_price"].astype(object)
    priced = df.index[df["unit_price"].notna()]
    idx = rng.choice(priced, size=int(len(df) * 0.015), replace=False)
    df.loc[idx, "unit_price"] = [f"\u00a3{p:.2f}" for p in df.loc[idx, "unit_price"]]
    mark(df, idx, errors, "price_as_text", "price exported with a pound sign")

    # offline till re-sent whole receipts when it reconnected
    invs = df.loc[df["invoice_id"].str.startswith("INV"), "invoice_id"].unique()
    resent = rng.choice(invs, size=int(len(invs) * 0.008), replace=False)
    copies = df[df["invoice_id"].isin(resent)].copy()
    copies["_err"] = True
    df = pd.concat([df, copies], ignore_index=True)
    errors += [("duplicate_upload", inv, "whole receipt uploaded twice") for inv in resent]

    return df, errors


if __name__ == "__main__":
    main()
