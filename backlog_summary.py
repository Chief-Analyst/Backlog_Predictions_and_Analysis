import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px


# =========================
# SETTINGS
# =========================
PIPELINE_PATH = "backlog_pipeline.pkl"


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_pipeline(path: str):
    return joblib.load(path)


# =========================
# SHARED HELPERS
# =========================
def _ensure_date(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.floor("D")
    return out


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom2 = denom.replace(0, np.nan)
    return (numer / denom2).fillna(0)


def safe_snapshot_subset(df: pd.DataFrame, mode: str, custom_date: pd.Timestamp | None) -> tuple[pd.DataFrame, str]:
    """
    Returns (subset_df, label_for_ui)
    mode:
      - 'Latest date'
      - 'Last 7 days'
      - 'Last 14 days'
      - 'Pick a date'
    """
    if df.empty:
        return df.copy(), "No data"

    latest = df["date"].max()

    if mode == "Latest date":
        snap = df[df["date"] == latest].copy()
        return snap, f"Latest date only: {latest.date()}"

    if mode == "Last 7 days":
        start = latest - pd.Timedelta(days=6)  # inclusive window (7 days)
        snap = df[(df["date"] >= start) & (df["date"] <= latest)].copy()
        return snap, f"Rolling window: {start.date()} → {latest.date()} (7 days)"

    if mode == "Last 14 days":
        start = latest - pd.Timedelta(days=13)  # inclusive window (14 days)
        snap = df[(df["date"] >= start) & (df["date"] <= latest)].copy()
        return snap, f"Rolling window: {start.date()} → {latest.date()} (14 days)"

    if mode == "Pick a date":
        if custom_date is None or pd.isna(custom_date):
            snap = df[df["date"] == latest].copy()
            return snap, f"Latest date only (fallback): {latest.date()}"
        snap = df[df["date"] == custom_date].copy()
        if snap.empty:
            snap = df[df["date"] == latest].copy()
            return snap, f"Selected date missing; using latest: {latest.date()}"
        return snap, f"Selected date: {custom_date.date()}"

    snap = df[df["date"] == latest].copy()
    return snap, f"Latest date only (fallback): {latest.date()}"


# =========================
# FEATURE ENGINEERING (PRED)
# =========================
def build_features_like_training(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    required = [
        "date", "customer_code", "region", "product",
        "orders_created", "orders_dispatched", "backlog_qty",
        "available_funds", "daily_target", "depot_status", "plant_status",
        "truck_availability_index", "dispatch_delay_days"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")

    num_cols = [
        "orders_created", "orders_dispatched", "backlog_qty", "available_funds",
        "daily_target", "truck_availability_index", "dispatch_delay_days"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values(["customer_code", "product", "date"]).reset_index(drop=True)

    df["orders_created"] = df["orders_created"].fillna(0)
    df["orders_dispatched"] = df["orders_dispatched"].fillna(0)
    df["backlog_qty"] = df["backlog_qty"].fillna(0)

    # daily_target safe fill
    if df["daily_target"].notna().any():
        df["daily_target"] = df["daily_target"].fillna(df["daily_target"].median())
    else:
        df["daily_target"] = df["daily_target"].fillna(0)

    # ML features
    df["backlog_cover_days"] = df["backlog_qty"] / (df["daily_target"] + 1)
    df["dispatch_rate"] = df["orders_dispatched"] / (df["backlog_qty"] + 1)
    df["backlog_change"] = df.groupby(["customer_code", "product"])["backlog_qty"].diff().fillna(0)
    df["order_gap"] = df["orders_created"] - df["orders_dispatched"]

    # Lags
    df["backlog_lag_1"] = df.groupby(["customer_code", "product"])["backlog_qty"].shift(1)
    df["backlog_lag_7"] = df.groupby(["customer_code", "product"])["backlog_qty"].shift(7)
    df["dispatch_lag_1"] = df.groupby(["customer_code", "product"])["orders_dispatched"].shift(1)
    df["dispatch_lag_7"] = df.groupby(["customer_code", "product"])["orders_dispatched"].shift(7)

    # Rolling means (past-only)
    df["backlog_roll_mean_7"] = (
        df.groupby(["customer_code", "product"])["backlog_qty"]
          .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    )
    df["dispatch_roll_mean_7"] = (
        df.groupby(["customer_code", "product"])["orders_dispatched"]
          .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    )
    df["created_roll_mean_7"] = (
        df.groupby(["customer_code", "product"])["orders_created"]
          .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    )

    return df


# =========================
# TAB 1: EXEC SUMMARY (HISTORICAL)
# =========================
def render_historical_summary_tab():
    st.subheader("📊 Historical Executive Summary (Raw Data)")

    uploaded = st.file_uploader("Upload Historical CSV (raw)", type=["csv"], key="hist_upload")
    if uploaded is None:
        st.info("Upload a historical CSV to see the executive summary.")
        return

    try:
        raw = pd.read_csv(uploaded)
        raw = _ensure_date(raw, "date")
    except Exception as e:
        st.exception(e)
        st.error("Could not read the uploaded CSV.")
        return

    required = [
        "date", "customer_code", "region", "product",
        "orders_created", "orders_dispatched", "backlog_qty"
    ]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        st.error(f"Missing required columns for historical summary: {missing}")
        return

    raw = raw.dropna(subset=["date"]).copy()
    for c in ["orders_created", "orders_dispatched", "backlog_qty"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0)

    raw["dispatch_rate"] = _safe_div(raw["orders_dispatched"], raw["backlog_qty"] + 1)
    raw["order_gap"] = raw["orders_created"] - raw["orders_dispatched"]

    # Sidebar: executive filters (light)
    with st.sidebar:
        st.header("Tab 1: Historical filters")
        date_min = raw["date"].min()
        date_max = raw["date"].max()
        d0, d1 = st.date_input(
            "Date range",
            value=(date_min.date(), date_max.date()),
            key="hist_date_range"
        )
        d0, d1 = pd.to_datetime(d0), pd.to_datetime(d1)

        region_sel = st.multiselect("Region", sorted(raw["region"].dropna().unique().tolist()), key="hist_region")
        product_sel = st.multiselect("Product", sorted(raw["product"].dropna().unique().tolist()), key="hist_product")
        top_n = st.slider("Top N customers", 5, 30, 10, key="hist_topn")

    df = raw[(raw["date"] >= d0) & (raw["date"] <= d1)].copy()
    if region_sel:
        df = df[df["region"].isin(region_sel)]
    if product_sel:
        df = df[df["product"].isin(product_sel)]

    if df.empty:
        st.warning("No data after filters. Adjust filters to see summaries.")
        return

    # KPIs
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total backlog (sum)", f"{df['backlog_qty'].sum():,.0f}")
    k2.metric("Orders created (sum)", f"{df['orders_created'].sum():,.0f}")
    k3.metric("Orders dispatched (sum)", f"{df['orders_dispatched'].sum():,.0f}")
    k4.metric("Order gap (sum)", f"{df['order_gap'].sum():,.0f}")
    k5.metric("Avg dispatch rate", f"{df['dispatch_rate'].mean():.3f}")
    k6.metric("Customers", f"{df['customer_code'].nunique():,}")

    # Charts: backlog trend + regional backlog + top customers
    st.markdown("## At-a-glance charts")

    c1, c2 = st.columns(2)
    with c1:
        daily = (
            df.groupby("date", as_index=False)
              .agg(backlog=("backlog_qty", "sum"),
                   created=("orders_created", "sum"),
                   dispatched=("orders_dispatched", "sum"))
        )
        melt = daily.melt("date", ["backlog", "created", "dispatched"], var_name="metric", value_name="value")
        fig = px.line(melt, x="date", y="value", color="metric", markers=True, title="Backlog vs Created vs Dispatched")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        by_region = (
            df.groupby("region", as_index=False)
              .agg(backlog=("backlog_qty", "sum"), gap=("order_gap", "sum"))
              .sort_values("backlog", ascending=False)
        )
        fig2 = px.bar(by_region, x="region", y="backlog", title="Total backlog by region")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("## Top customers (by backlog)")
    top_customers = (
        df.groupby(["customer_code"], as_index=False)
          .agg(backlog=("backlog_qty", "sum"),
               gap=("order_gap", "sum"),
               avg_dispatch_rate=("dispatch_rate", "mean"))
          .sort_values("backlog", ascending=False)
          .head(top_n)
    )
    st.dataframe(top_customers, use_container_width=True, height=320)

    with st.expander("Download filtered historical summary"):
        st.download_button(
            "Download (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="historical_exec_summary_filtered.csv",
            mime="text/csv",
            key="hist_dl"
        )


# =========================
# TAB 2: EXEC SUMMARY (PREDICTIONS)
# =========================
def render_prediction_summary_tab(pipe):
    st.subheader("🔮 Prediction Executive Summary")

    with st.sidebar:
        st.header("Tab 2: Prediction input")
        uploaded = st.file_uploader("Upload CSV (for predictions)", type=["csv"], key="pred_upload")

        history_mode = st.radio(
            "Insufficient history rows (lags/rolling)",
            ["Drop (recommended)", "Fill with 0"],
            index=0,
            key="pred_hist_mode_exec"
        )

        st.header("Snapshot window")
        snapshot_mode = st.selectbox(
            "Summary window",
            ["Latest date", "Last 7 days", "Last 14 days", "Pick a date"],
            index=0,
            key="pred_snap_mode"
        )

        st.header("Executive options")
        drill_customer = st.checkbox("Enable customer drilldown", value=False, key="pred_drill_exec")

        top_n = st.slider("Top N customers", 5, 30, 10, key="pred_topn_exec")

    if uploaded is None:
        st.info("Upload a CSV to view the prediction executive summary.")
        return

    # Build features + predict
    try:
        raw = pd.read_csv(uploaded)
        feat = build_features_like_training(raw)

        hist_cols = [
            "backlog_lag_1", "backlog_lag_7",
            "dispatch_lag_1", "dispatch_lag_7",
            "backlog_roll_mean_7", "dispatch_roll_mean_7",
        ]
        feat["insufficient_history"] = feat[hist_cols].isna().any(axis=1)

        dropped = 0
        if history_mode.startswith("Drop"):
            dropped = int(feat["insufficient_history"].sum())
            feat_model = feat.loc[~feat["insufficient_history"]].copy()
        else:
            feat_model = feat.copy()
            feat_model[hist_cols] = feat_model[hist_cols].fillna(0)

        expected = None
        if hasattr(pipe, "named_steps") and "preprocess" in pipe.named_steps:
            if hasattr(pipe.named_steps["preprocess"], "feature_names_in_"):
                expected = list(pipe.named_steps["preprocess"].feature_names_in_)

        drop_cols = [c for c in ["date", "customer_code", "future_backlog_qty"] if c in feat_model.columns]
        if expected is None:
            X_model = feat_model.drop(columns=drop_cols, errors="ignore")
        else:
            for c in expected:
                if c not in feat_model.columns:
                    feat_model[c] = 0
            X_model = feat_model[expected]

        preds = pd.Series(pipe.predict(X_model)).clip(lower=0)

        results = feat_model[[
            "date", "customer_code", "region", "product",
            "orders_created", "orders_dispatched",
            "dispatch_roll_mean_7", "created_roll_mean_7"
        ]].copy()

        results["predicted_backlog"] = preds.values

        results["net_reduction_per_day"] = (
            results["dispatch_roll_mean_7"] - results["created_roll_mean_7"]
        ).fillna(0)

        if dropped > 0:
            st.warning(f"Dropped {dropped:,} row(s) due to insufficient history.")

    except Exception as e:
        st.exception(e)
        st.error("Processing failed. Check your file format/columns.")
        return

    # Snapshot selection
    latest_date = results["date"].max()
    custom_date = None

    if snapshot_mode == "Pick a date":
        with st.sidebar:
            st.caption("Pick a date from your dataset")
            picked = st.date_input("Snapshot date", value=latest_date.date(), key="pred_pick_date_exec")
        custom_date = pd.to_datetime(picked).floor("D")

    snap_df, snap_label = safe_snapshot_subset(results, snapshot_mode, custom_date)
    st.caption(f"Snapshot used: {snap_label}")

    if snap_df.empty:
        st.warning("No data in snapshot window. Try another window.")
        return

    # Regional summary
    region_summary = (
        snap_df.groupby("region", dropna=False)
        .agg(
            predicted_backlog_total=("predicted_backlog", "sum"),
            total_net_reduction_per_day=("net_reduction_per_day", "sum"),
            avg_net_reduction_per_day=("net_reduction_per_day", "mean"),
            customers=("customer_code", "nunique"),
            rows=("region", "size"),
        )
        .reset_index()
        .sort_values("predicted_backlog_total", ascending=False)
    )

    eps = 1e-6
    region_summary["depletion_status"] = np.where(
        region_summary["total_net_reduction_per_day"] > eps,
        "Depleting",
        "Not depleting"
    )

    region_summary["est_days_to_deplete"] = np.where(
        region_summary["total_net_reduction_per_day"] > eps,
        region_summary["predicted_backlog_total"] / region_summary["total_net_reduction_per_day"],
        np.nan
    )

    CAP_DAYS = 365
    region_summary["est_days_to_deplete_capped"] = region_summary["est_days_to_deplete"].clip(upper=CAP_DAYS)

    def pretty_days(x):
        if pd.isna(x):
            return "N/A"
        if x > CAP_DAYS:
            return f">{CAP_DAYS} days"
        return f"{x:,.1f} days"

    region_summary["est_days_to_deplete_label"] = region_summary["est_days_to_deplete"].apply(pretty_days)

    # KPIs
    total_pred = float(region_summary["predicted_backlog_total"].sum())
    depleting_regions = int((region_summary["depletion_status"] == "Depleting").sum())
    not_depleting_regions = int((region_summary["depletion_status"] == "Not depleting").sum())
    customers_total = int(snap_df["customer_code"].nunique())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total predicted backlog", f"{total_pred:,.0f}")
    k2.metric("Regions", f"{region_summary['region'].nunique():,}")
    k3.metric("Customers", f"{customers_total:,}")
    k4.metric("Depleting regions", f"{depleting_regions}")
    k5.metric("Not depleting regions", f"{not_depleting_regions}")

    # Two visuals (executive glance)
    left, right = st.columns(2)
    with left:
        fig1 = px.bar(
            region_summary,
            x="region",
            y="predicted_backlog_total",
            title="Predicted backlog by region",
        )
        st.plotly_chart(fig1, use_container_width=True)

    with right:
        ddf = region_summary[region_summary["depletion_status"] == "Depleting"].copy()
        if ddf.empty:
            st.info("No regions are currently depleting (dispatch ≤ orders created on net).")
        else:
            fig2 = px.bar(
                ddf,
                x="region",
                y="est_days_to_deplete_capped",
                title=f"Estimated days to deplete (capped at {CAP_DAYS} days)",
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Regional Summary")
    show_cols = [
        "region",
        "predicted_backlog_total",
        "total_net_reduction_per_day",
        "avg_net_reduction_per_day",
        "est_days_to_deplete_label",
        "depletion_status",
        "customers"
    ]
    st.dataframe(region_summary[show_cols], use_container_width=True, height=240)

    # Top customers table (executive must-have)
    st.subheader(f"Top {top_n} Customers (by predicted backlog)")
    top_customers = (
        snap_df.groupby(["customer_code", "region"], as_index=False)
              .agg(predicted_backlog=("predicted_backlog", "sum"),
                   net_reduction_per_day=("net_reduction_per_day", "sum"))
              .sort_values("predicted_backlog", ascending=False)
              .head(top_n)
    )
    st.dataframe(top_customers, use_container_width=True, height=320)

    # Optional drilldown
    if drill_customer:
        st.subheader("Customer / Product contributors (within snapshot)")

        c1, c2, c3 = st.columns(3)
        with c1:
            region_pick = st.selectbox("Region", sorted(snap_df["region"].dropna().unique().tolist()), key="drill_region")
        with c2:
            customer_pick = st.selectbox(
                "Customer", ["(All)"] + sorted(snap_df["customer_code"].dropna().unique().tolist()),
                key="drill_customer"
            )
        with c3:
            drill_top_n = st.slider("Top N (drill)", 5, 30, 10, key="drill_topn")

        f = snap_df[snap_df["region"] == region_pick].copy()
        if customer_pick != "(All)":
            f = f[f["customer_code"] == customer_pick]

        cust_top = (
            f.groupby("customer_code")["predicted_backlog"]
             .sum()
             .sort_values(ascending=False)
             .head(drill_top_n)
             .reset_index()
        )

        prod_mix = (
            f.groupby("product")["predicted_backlog"]
             .sum()
             .sort_values(ascending=False)
             .reset_index()
        )

        d1, d2 = st.columns(2)
        with d1:
            st.caption(f"Top {drill_top_n} customers in {region_pick}")
            st.dataframe(cust_top, use_container_width=True, height=260)
        with d2:
            st.caption(f"Product mix in {region_pick}")
            st.dataframe(prod_mix, use_container_width=True, height=260)

    with st.expander("Download snapshot predictions"):
        st.download_button(
            "Download snapshot (CSV)",
            data=snap_df.to_csv(index=False).encode("utf-8"),
            file_name="prediction_exec_snapshot.csv",
            mime="text/csv",
            key="pred_dl_exec"
        )


# =========================
# MAIN APP (tabs)
# =========================
def main():
    st.set_page_config(page_title="Executive Backlog Summary", layout="wide")
    st.title("Executive Backlog Summary")

    try:
        pipe = load_pipeline(PIPELINE_PATH)
    except Exception as e:
        st.exception(e)
        st.error(f"Could not load '{PIPELINE_PATH}'. Put it in the same folder as this app.")
        return

    tab1, tab2 = st.tabs(["📊 Historical Summary", "🔮 Prediction Summary"])

    with tab1:
        render_historical_summary_tab()

    with tab2:
        render_prediction_summary_tab(pipe)


if __name__ == "__main__":
    main()