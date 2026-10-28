import pandas as pd

# recency score + frequency score -> segment name (standard RFM grid)
SEGMENT_MAP = {
    r"[1-2][1-2]": "Hibernating",
    r"[1-2][3-4]": "At risk",
    r"[1-2]5": "Can't lose them",
    r"3[1-2]": "About to sleep",
    r"33": "Need attention",
    r"[3-4][4-5]": "Loyal",
    r"41": "Promising",
    r"51": "New customers",
    r"[4-5][2-3]": "Potential loyalists",
    r"5[4-5]": "Champions",
}


def score(series, ascending=True):
    # split customers into 5 equal groups, 5 = best
    # rank first so lots of tied values don't break the split
    ranks = series.rank(method="first", ascending=ascending)
    return pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def segment_customers(clean, snapshot=None):
    # only loyalty card holders can be segmented
    df = clean[clean["customer_id"].notna()]
    if df["customer_id"].nunique() < 5:
        raise ValueError("need at least 5 customers to segment")

    # measure recency from the day after the last sale in the data
    if snapshot is None:
        snapshot = df["invoice_date"].max().normalize() + pd.Timedelta(days=1)

    # recency and frequency come from purchases only, not refunds
    sales = df[~df["is_refund"]]
    by_cust = sales.groupby("customer_id")
    rfm = pd.DataFrame({
        "recency_days": (snapshot - by_cust["invoice_date"].max()).dt.days,
        "frequency": by_cust["invoice_id"].nunique(),
    })

    # monetary is net spend, so refunds are taken off
    rfm["monetary"] = df.groupby("customer_id")["revenue"].sum().round(2)

    # each customer's main store = where they made most visits
    visits = sales.drop_duplicates("invoice_id")
    rfm["home_store"] = visits.groupby("customer_id")["store_name"].agg(lambda x: x.value_counts().idxmax())

    # people who only ever refunded have no purchases to score
    rfm = rfm.dropna(subset=["recency_days", "frequency"])

    # score each measure 1-5, then name the segment from recency + frequency
    rfm["r_score"] = score(rfm["recency_days"], ascending=False)
    rfm["f_score"] = score(rfm["frequency"])
    rfm["m_score"] = score(rfm["monetary"])
    rf = rfm["r_score"].astype(str) + rfm["f_score"].astype(str)
    rfm["segment"] = rf.replace(SEGMENT_MAP, regex=True)

    rfm["recency_days"] = rfm["recency_days"].astype(int)
    rfm["frequency"] = rfm["frequency"].astype(int)
    return rfm.reset_index(), snapshot


def summarise_segments(rfm):
    # one row per segment: size, value and behaviour
    s = rfm.groupby("segment").agg(
        customers=("customer_id", "count"),
        revenue=("monetary", "sum"),
        avg_recency_days=("recency_days", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_spend=("monetary", "mean"),
    )
    s["share_of_customers"] = s["customers"] / s["customers"].sum()
    s["share_of_revenue"] = s["revenue"] / s["revenue"].sum()
    return s.sort_values("revenue", ascending=False).round(2).reset_index()


def segments_by_store(rfm):
    # how many customers of each segment call each store home
    table = pd.crosstab(rfm["home_store"], rfm["segment"])
    table["total"] = table.sum(axis=1)
    return table.sort_values("total", ascending=False).reset_index()
