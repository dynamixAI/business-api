import difflib

import numpy as np
import pandas as pd

# columns every upload must have
REQUIRED = ["invoice_id", "customer_id", "invoice_date", "product", "quantity", "unit_price", "store_id", "store_name"]

# anything dated before this is a till clock fault, not a real sale
EARLIEST_REAL_DATE = pd.Timestamp("2010-01-01")

# store opening hours (24h clock), used to flag odd timestamps
OPEN_HOUR, CLOSE_HOUR = 9, 20


# raised when an upload can't be cleaned at all, e.g. missing columns
class DataError(ValueError):
    pass


def tidy_text(s):
    # strip outer spaces, then squash repeated inner spaces (only where there are any, as regex is slow)
    s = s.str.strip()
    doubled = s.str.contains("  ", regex=False, na=False)
    if doubled.any():
        s[doubled] = s[doubled].str.replace(r"\s+", " ", regex=True)
    return s


def most_common_spelling(s):
    # group names by lowercase key, then pick the spelling used most often
    key = s.str.lower()
    best = s.groupby(key).agg(lambda x: x.value_counts().idxmax())
    return key.map(best)


def one_word_typo(a, b):
    # true if the names differ by a small spelling slip in a single word
    wa, wb = a.lower().split(), b.lower().split()
    if len(wa) != len(wb):
        return False
    diff = [(x, y) for x, y in zip(wa, wb) if x != y]
    if len(diff) != 1:
        return False
    x, y = diff[0]
    # "Floarl" vs "Floral" passes, "Pillar" vs "Jar" doesn't
    return difflib.SequenceMatcher(None, x, y).ratio() >= 0.7


def fix_typos(names):
    # a name is a typo if a much more common name is one small slip away
    counts = names.value_counts()
    fixes = {}
    for name, n in counts.items():
        if name in ("MISC ITEM", "TEST ITEM"):
            continue
        # only compare against names used at least 10x as often
        common = counts[counts >= n * 10].index
        match = [c for c in common if one_word_typo(name, c)]
        if match:
            fixes[name] = match[0]
    # map is far quicker than replace when there are hundreds of fixes
    return names.map(fixes).fillna(names), fixes


def drop_resent_receipts(df):
    # count how often each exact row appears within its receipt
    cols = list(df.columns)
    counts = df.value_counts(dropna=False).rename("n").reset_index()

    # if every row in a receipt appears 2x (or 3x...), the whole receipt was re-sent
    # i.e. each row's count divides evenly by the receipt's smallest count
    smallest = counts.groupby("invoice_id")["n"].transform("min")
    even = (counts["n"] % smallest == 0).groupby(counts["invoice_id"]).transform("all")
    counts["times_sent"] = smallest.where(even, 1)
    times_sent = counts.groupby("invoice_id")["times_sent"].first()
    counts["keep"] = counts["n"] // counts["times_sent"]

    # rebuild the table with the extra copies taken out
    out = counts.loc[counts.index.repeat(counts["keep"]), cols].reset_index(drop=True)
    removed = len(df) - len(out)
    receipts = int((times_sent > 1).sum())
    return out, removed, receipts


def clean_transactions(raw):
    # stop early if the file isn't till data at all
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise DataError(f"missing columns: {', '.join(missing)}")

    df = raw.copy()
    # running tally of everything we find and fix
    report = {"rows_in": len(df)}

    # 1. duplicate uploads: an offline till re-sends a whole receipt
    df, removed, receipts = drop_resent_receipts(df)
    report["duplicate_rows_removed"] = removed
    report["receipts_sent_twice"] = receipts

    # keep a copy of the raw text so we can count what changed later
    raw_cid = df["customer_id"].copy()
    raw_store = df["store_name"].copy()
    raw_product = df["product"].copy()

    # 2. stray spaces in every text column
    for col in ["invoice_id", "customer_id", "invoice_date", "product", "store_id", "store_name", "unit_price"]:
        df[col] = tidy_text(df[col].astype("string"))

    # 3. till test transactions: TST receipts, the test card, or the test item
    test = (
        df["invoice_id"].str.startswith("TST", na=False)
        | df["customer_id"].str.upper().eq("C999999").fillna(False)
        | df["product"].str.upper().eq("TEST ITEM").fillna(False)
    )
    report["test_rows_removed"] = int(test.sum())
    report["test_receipts_removed"] = int(df.loc[test, "invoice_id"].nunique())
    df = df[~test].copy()
    raw_cid, raw_store, raw_product = raw_cid[~test], raw_store[~test], raw_product[~test]

    # 4. customer ids: upper case, add the missing C, anything else is invalid
    cid = df["customer_id"].str.upper()
    cid = cid.where(~cid.str.fullmatch(r"\d{6}", na=False), "C" + cid)
    valid = cid.str.fullmatch(r"C\d{6}", na=False)
    df["customer_id"] = cid.where(valid)
    report["customer_ids_reformatted"] = int((df["customer_id"].ne(raw_cid) & valid).sum())

    # 5. store names: tidy the text, then trust the store id over the name
    df["store_name"] = most_common_spelling(df["store_name"])
    report["store_name_text_fixed"] = int(df["store_name"].ne(raw_store).sum())
    true_name = df.groupby("store_id")["store_name"].agg(lambda x: x.value_counts().idxmax())
    expected = df["store_id"].map(true_name)
    report["store_names_corrected"] = int(df["store_name"].ne(expected).sum())
    df["store_name"] = expected

    # 6. product names: fix case and spacing first, then typos
    df["product"] = most_common_spelling(df["product"])
    report["product_text_fixed"] = int(df["product"].ne(raw_product).sum())
    before = df["product"].copy()
    df["product"], typo_map = fix_typos(df["product"])
    report["product_typos_fixed"] = int(df["product"].ne(before).sum())
    report["product_typo_variants"] = len(typo_map)

    # 7. prices: drop the pound sign, then convert to numbers
    has_pound = df["unit_price"].str.contains("£", na=False)
    report["prices_with_pound_sign"] = int(has_pound.sum())
    df["unit_price"] = pd.to_numeric(df["unit_price"].str.replace("£", "", regex=False), errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    # missing prices: use that product's usual (median) price
    no_price = df["unit_price"].isna()
    usual = df.groupby("product")["unit_price"].transform("median")
    df["unit_price"] = df["unit_price"].fillna(usual)
    report["missing_prices_filled"] = int(no_price.sum())
    df["price_was_filled"] = no_price

    # 8. dates: most tills use 2024-01-31, Rochdale uses 31/01/2024
    iso = pd.to_datetime(df["invoice_date"], format="%Y-%m-%d %H:%M", errors="coerce")
    uk = pd.to_datetime(df["invoice_date"], format="%d/%m/%Y %H:%M", errors="coerce")
    report["uk_format_dates_parsed"] = int((iso.isna() & uk.notna()).sum())
    df["invoice_date"] = iso.fillna(uk)

    # rows we still can't read are dropped
    unreadable = df[["invoice_date", "quantity", "unit_price"]].isna().any(axis=1)
    report["unreadable_rows_removed"] = int(unreadable.sum())
    df = df[~unreadable].copy()

    # clock reset to 1970/2000: the real sale date is unknown, so drop the receipt
    reset = df["invoice_date"] < EARLIEST_REAL_DATE
    report["clock_reset_rows_removed"] = int(reset.sum())
    df = df[~reset].copy()

    # out of hours: the day is right but the time isn't, so keep and flag
    hour = df["invoice_date"].dt.hour
    df["out_of_hours"] = (hour < OPEN_HOUR) | (hour >= CLOSE_HOUR)
    report["out_of_hours_rows_flagged"] = int(df["out_of_hours"].sum())

    # 9. receipt number clashes: same id used by different stores or days
    # a real receipt happens in one store on one day
    df["visit"] = df["store_id"] + "|" + df["invoice_date"].dt.strftime("%Y%m%d")
    firsts = df[["invoice_id", "visit"]].drop_duplicates()
    firsts["part"] = firsts.groupby("invoice_id").cumcount()
    df = df.merge(firsts, on=["invoice_id", "visit"], how="left")
    report["clashing_receipts_split"] = int(firsts.loc[firsts["part"] > 0, "invoice_id"].nunique())

    # give the extra visits their own id, e.g. INV0001234-1
    df["invoice_id"] = np.where(df["part"] > 0, df["invoice_id"] + "-" + df["part"].astype(str), df["invoice_id"])
    df = df.drop(columns=["visit", "part"])

    # 10. voids: negative lines on a normal sale receipt cancel a keying mistake
    is_sale_receipt = ~df["invoice_id"].str.startswith("RF")
    voids = is_sale_receipt & (df["quantity"] < 0)
    report["void_lines_netted"] = int(voids.sum())

    # sum each product per receipt so 18 then -17 becomes 1
    sales = df[is_sale_receipt]
    pos = sales[sales["quantity"] > 0]
    price = (pos["quantity"] * pos["unit_price"]).groupby([pos["invoice_id"], pos["product"]]).sum() \
        / pos.groupby(["invoice_id", "product"])["quantity"].sum()
    keep_cols = ["customer_id", "invoice_date", "category", "store_id", "store_name"]
    netted = sales.groupby(["invoice_id", "product"], as_index=False).agg(
        quantity=("quantity", "sum"),
        price_was_filled=("price_was_filled", "any"),
        out_of_hours=("out_of_hours", "any"),
        **{c: (c, "first") for c in keep_cols},
    )
    netted["unit_price"] = netted.set_index(["invoice_id", "product"]).index.map(price)

    # a void with nothing left to cancel can't be trusted, so drop it
    leftover = (netted["quantity"] <= 0) | netted["unit_price"].isna()
    report["lines_fully_voided_or_unmatched"] = int(leftover.sum())
    netted = netted[~leftover]

    # put the netted sales back together with the refund receipts
    df = pd.concat([netted, df[~is_sale_receipt]], ignore_index=True)
    df["unit_price"] = df["unit_price"].round(2)

    # 11. odd prices: well below the usual price for that product
    usual = df.groupby("product")["unit_price"].transform("median")
    ratio = df["unit_price"] / usual
    df["price_flag"] = np.select(
        [ratio < 0.25, ratio < 0.75],
        ["suspected pricing error", "discount or override"],
        default="",
    )
    report["prices_flagged_suspect"] = int((df["price_flag"] == "suspected pricing error").sum())
    report["prices_flagged_discount"] = int((df["price_flag"] == "discount or override").sum())

    # 12. what's left: make sure numbers are stored as numbers
    df["quantity"] = df["quantity"].astype(int)
    df["unit_price"] = df["unit_price"].astype(float)
    df["revenue"] = (df["quantity"] * df["unit_price"]).round(2)
    df["is_refund"] = df["quantity"] < 0
    report["refund_rows"] = int(df["is_refund"].sum())
    report["misc_item_rows"] = int(df["product"].eq("MISC ITEM").sum())
    report["rows_without_customer"] = int(df["customer_id"].isna().sum())
    report["rows_out"] = len(df)

    # final column order, sorted by time
    cols = ["invoice_id", "customer_id", "invoice_date", "product", "category", "quantity", "unit_price",
            "revenue", "store_id", "store_name", "is_refund", "price_was_filled", "out_of_hours", "price_flag"]
    df = df[cols].sort_values(["invoice_date", "invoice_id"]).reset_index(drop=True)
    return df, report
