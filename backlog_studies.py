import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px


# =========================
# SETTINGS
# =========================
PIPELINE_PATH = "backlog_pipeline.pkl"

RISK_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
RISK_COLORS = {
    "CRITICAL": "#ff3b30",  # red
    "HIGH": "#ff9500",      # orange
    "MEDIUM": "#ffcc00",    # yellow
    "LOW": "#34c759",       # green
}

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "chart",
        "height": 900,
        "width": 1600,
        "scale": 2
    }
}


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_pipeline(path: str):
    return joblib.load(path)


# =========================
# DEMO DATA
# =========================
def make_demo_data(n_rows: int = 700, n_days: int = 45) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    customers = [f"cust_{i:03d}" for i in range(1, 31)]
    products = ["PMS", "DPK", "AGO"]
    regions = ["Lagos", "North", "East", "West"]
    depot_statuses = ["Available", "Low Stock", "Out of Stock"]
    plant_statuses = ["Running", "Maintenance", "Shutdown"]

    df = pd.DataFrame({
        "date": np.random.choice(dates, n_rows),
        "customer_code": np.random.choice(customers, n_rows),
        "region": np.random.choice(regions, n_rows),
        "product": np.random.choice(products, n_rows),
        "orders_created": np.random.randint(0, 60, n_rows),
        "orders_dispatched": np.random.randint(0, 60, n_rows),
        "backlog_qty": np.random.randint(0, 900, n_rows),
        "available_funds": np.random.randint(50_000, 3_000_000, n_rows),
        "daily_target": np.random.randint(10, 120, n_rows),
        "depot_status": np.random.choice(depot_statuses, n_rows, p=[0.7, 0.2, 0.1]),
        "plant_status": np.random.choice(plant_statuses, n_rows, p=[0.75, 0.2, 0.05]),
        "truck_availability_index": np.round(np.random.uniform(0.3, 1.0, n_rows), 2),
        "dispatch_delay_days": np.random.randint(0, 8, n_rows),
    })

    df["orders_dispatched"] = np.minimum(df["orders_dispatched"], df["orders_created"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
    df = df.dropna(subset=["date"]).sort_values(["customer_code", "product", "date"]).reset_index(drop=True)
    return df


# =========================
# FEATURE ENGINEERING
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

    if df["daily_target"].notna().any():
        df["daily_target"] = df["daily_target"].fillna(df["daily_target"].median())
    else:
        df["daily_target"] = df["daily_target"].fillna(0)

    df["backlog_cover_days"] = df["backlog_qty"] / (df["daily_target"] + 1)
    df["dispatch_rate"] = df["orders_dispatched"] / (df["backlog_qty"] + 1)

    df["backlog_change"] = (
        df.groupby(["customer_code", "product"], observed=True)["backlog_qty"].diff()
    ).fillna(0)

    df["order_gap"] = df["orders_created"] - df["orders_dispatched"]

    # Lags
    df["backlog_lag_1"] = df.groupby(["customer_code", "product"], observed=True)["backlog_qty"].shift(1)
    df["backlog_lag_7"] = df.groupby(["customer_code", "product"], observed=True)["backlog_qty"].shift(7)
    df["dispatch_lag_1"] = df.groupby(["customer_code", "product"], observed=True)["orders_dispatched"].shift(1)
    df["dispatch_lag_7"] = df.groupby(["customer_code", "product"], observed=True)["orders_dispatched"].shift(7)

    # Rolling means (PAST ONLY)
    df["backlog_roll_mean_7"] = (
        df.groupby(["customer_code", "product"], observed=True)["backlog_qty"]
          .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    )
    df["dispatch_roll_mean_7"] = (
        df.groupby(["customer_code", "product"], observed=True)["orders_dispatched"]
          .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    )

    return df


def risk_from_cover_days(cover_days: pd.Series, t_med: float, t_high: float, t_crit: float) -> pd.Series:
    risk = pd.Series(np.where(cover_days <= t_med, "MEDIUM", "LOW"), index=cover_days.index)
    risk = pd.Series(np.where(cover_days <= t_high, "HIGH", risk), index=cover_days.index)
    risk = pd.Series(np.where(cover_days <= t_crit, "CRITICAL", risk), index=cover_days.index)
    return risk


def add_action_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    drivers = []
    actions = []

    for _, r in out.iterrows():
        d = []
        a = []

        depot = str(r.get("depot_status", "")).lower()
        plant = str(r.get("plant_status", "")).lower()

        if depot == "out of stock":
            d.append("Depot out of stock")
            a.append("Replenish depot stock / reroute supply")
        elif depot == "low stock":
            d.append("Depot low stock")
            a.append("Prioritize replenishment to depot")

        if plant == "shutdown":
            d.append("Plant shutdown")
            a.append("Escalate plant restart / alternate sourcing")
        elif plant == "maintenance":
            d.append("Plant maintenance")
            a.append("Coordinate maintenance / alternate sourcing")

        try:
            if float(r.get("truck_availability_index", 1.0)) < 0.5:
                d.append("Low truck availability")
                a.append("Allocate more trucks / rebalance fleet")
        except Exception:
            pass

        try:
            if float(r.get("dispatch_delay_days", 0.0)) >= 3:
                d.append("High dispatch delays")
                a.append("Expedite dispatch / remove bottlenecks")
        except Exception:
            pass

        if not d:
            d = ["No dominant driver found"]
        if not a:
            a = ["Monitor / validate inputs"]

        drivers.append("; ".join(d))
        actions.append("; ".join(a))

    out["drivers"] = drivers
    out["action"] = actions
    return out


def style_risk_table(df: pd.DataFrame):
    def row_bg(row):
        lvl = str(row.get("risk_level", ""))
        bg = RISK_COLORS.get(lvl, "#ffffff")
        tint = bg + "22"
        return [f"background-color: {tint};" for _ in row]

    def risk_cell(row):
        lvl = str(row.get("risk_level", ""))
        bg = RISK_COLORS.get(lvl, "#ffffff")
        return [
            f"background-color: {bg}; color: black; font-weight: 800;"
            if c == "risk_level" else ""
            for c in df.columns
        ]

    return df.style.apply(row_bg, axis=1).apply(risk_cell, axis=1)


# =========================
# DATE SNAPSHOT FILTER
# =========================
def apply_snapshot_filter(df: pd.DataFrame, date_col: str, mode: str, picked_date: pd.Timestamp | None):
    if df.empty or date_col not in df.columns:
        return df.copy(), "No data"

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.floor("D")
    out = out.dropna(subset=[date_col]).copy()
    if out.empty:
        return out, "No valid dates"

    latest = out[date_col].max()

    if mode == "Latest date":
        snap = out[out[date_col] == latest].copy()
        return snap, f"Latest date only: {latest.date()}"

    if mode == "Last 7 days":
        start = latest - pd.Timedelta(days=6)
        snap = out[(out[date_col] >= start) & (out[date_col] <= latest)].copy()
        return snap, f"Rolling window: {start.date()} → {latest.date()} (7 days)"

    if mode == "Last 14 days":
        start = latest - pd.Timedelta(days=13)
        snap = out[(out[date_col] >= start) & (out[date_col] <= latest)].copy()
        return snap, f"Rolling window: {start.date()} → {latest.date()} (14 days)"

    if mode == "Pick a date":
        if picked_date is None or pd.isna(picked_date):
            snap = out[out[date_col] == latest].copy()
            return snap, f"Latest date only (fallback): {latest.date()}"

        picked_date = pd.to_datetime(picked_date).floor("D")
        snap = out[out[date_col] == picked_date].copy()
        if snap.empty:
            snap = out[out[date_col] == latest].copy()
            return snap, f"Selected date missing; using latest: {latest.date()}"
        return snap, f"Selected date: {picked_date.date()}"

    snap = out[out[date_col] == latest].copy()
    return snap, f"Latest date only (fallback): {latest.date()}"


# =========================
# HELPERS
# =========================
def _ensure_datetime(df: pd.DataFrame, col="date") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.floor("D")
    return out


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom2 = denom.replace(0, np.nan)
    return numer / denom2


def resample_view(df: pd.DataFrame, freq_label: str) -> pd.DataFrame:
    out = df.copy()
    if freq_label == "Days":
        out["period"] = out["date"]
    elif freq_label == "Weeks":
        out["period"] = out["date"].dt.to_period("W").dt.start_time
    elif freq_label == "Months":
        out["period"] = out["date"].dt.to_period("M").dt.start_time
    elif freq_label == "Quarters":
        out["period"] = out["date"].dt.to_period("Q").dt.start_time
    else:
        out["period"] = out["date"]
    return out


# =========================
# UNIVERSAL METRIC PANEL
# =========================
def metric_panel(
    df: pd.DataFrame,
    *,
    title: str,
    x: str | None,
    y: str,
    color: str | None = None,
    agg: str = "sum",  # "sum" | "mean" | "count"
    chart_key: str,
    default_chart: str = "Line",
    top_n_default: int = 20,
):
    if df is None or df.empty:
        st.info("No data after filters.")
        return

    st.markdown(f"### {title}")

    chart_types = ["Line", "Area", "Bar", "Scatter", "Cards", "Table"]
    c1, c2 = st.columns([1.2, 0.8])

    with c1:
        chart_type = st.selectbox(
            "Chart type",
            chart_types,
            index=chart_types.index(default_chart) if default_chart in chart_types else 0,
            key=f"{chart_key}_type"
        )
    with c2:
        top_n = st.number_input("Top N", 5, 200, int(top_n_default), key=f"{chart_key}_topn")

    work = df.copy()

    # ---- Version-safe COUNT aggregation (no reset_index(name=...)) ----
    def _count_grouped(df_in: pd.DataFrame, group_cols: list[str], out_name: str) -> pd.DataFrame:
        s = df_in.groupby(group_cols, observed=True).size()
        out_df = s.reset_index()
        out_df = out_df.rename(columns={0: out_name})
        return out_df

    def _agg(df_in: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        if agg == "sum":
            return df_in.groupby(group_cols, as_index=False, observed=True).agg(**{y: (y, "sum")})
        if agg == "mean":
            return df_in.groupby(group_cols, as_index=False, observed=True).agg(**{y: (y, "mean")})
        return _count_grouped(df_in, group_cols, y)

    # If no x => ranking/snapshot
    if x is None:
        if color is None:
            val = float(work[y].sum()) if agg == "sum" else float(work[y].mean())
            st.metric(title, f"{val:,.0f}")
            with st.expander("View data"):
                st.dataframe(work, use_container_width=True, height=320)
            return

        agg_df = _agg(work, [color]).sort_values(y, ascending=False).head(int(top_n))

        if chart_type == "Cards":
            cols = st.columns(4)
            for i, r in enumerate(agg_df.itertuples(index=False)):
                label = getattr(r, color)
                value = getattr(r, y)
                with cols[i % 4]:
                    st.metric(str(label), f"{float(value):,.0f}")
        elif chart_type == "Table":
            st.dataframe(agg_df, use_container_width=True, height=320)
        else:
            fig = px.bar(agg_df, x=color, y=y, title=title)
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        if chart_type != "Table":
            with st.expander("View data"):
                st.dataframe(agg_df, use_container_width=True, height=320)
        return

    # x-based metric
    group_cols = [x] + ([color] if color else [])
    agg_df = _agg(work, group_cols)

    # If only 1 x point, switch to Bar for readability
    if agg_df[x].nunique(dropna=True) <= 1 and chart_type in ["Line", "Area", "Scatter"]:
        st.caption("Only one date/period in the snapshot — switched to Bar automatically for readability.")
        chart_type = "Bar"

    if chart_type == "Line":
        fig = px.line(agg_df, x=x, y=y, color=color, markers=True, title=title)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    elif chart_type == "Area":
        fig = px.area(agg_df, x=x, y=y, color=color, title=title)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    elif chart_type == "Scatter":
        fig = px.scatter(agg_df, x=x, y=y, color=color, title=title)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    elif chart_type == "Cards":
        if color:
            latest = agg_df[x].max()
            snap = agg_df[agg_df[x] == latest].sort_values(y, ascending=False).head(int(top_n))
            cols = st.columns(4)
            for i, r in enumerate(snap.itertuples(index=False)):
                label = getattr(r, color)
                value = getattr(r, y)
                with cols[i % 4]:
                    st.metric(str(label), f"{float(value):,.0f}")
        else:
            st.metric(title, f"{float(agg_df[y].sum()):,.0f}")
    elif chart_type == "Table":
        st.dataframe(agg_df, use_container_width=True, height=320)
    else:
        fig = px.bar(agg_df, x=x, y=y, color=color, title=title)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    if chart_type != "Table":
        with st.expander("View data"):
            st.dataframe(agg_df, use_container_width=True, height=320)


# =========================
# TAB 1: ANALYSIS (RAW)
# =========================
def render_analysis_tab():
    st.subheader("Data Analysis (Raw / Historical)")

    uploaded_analysis = st.file_uploader(
        "Upload dataset for Analysis tab (raw CSV)",
        type=["csv"],
        key="analysis_uploader"
    )
    if uploaded_analysis is None:
        st.info("Upload a CSV here to run analysis. (This upload does NOT affect the Predictions tab.)")
        return

    raw = pd.read_csv(uploaded_analysis)
    raw = _ensure_datetime(raw, "date")

    required = [
        "date", "customer_code", "region", "product",
        "backlog_qty", "orders_created", "orders_dispatched",
        "truck_availability_index", "dispatch_delay_days"
    ]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        st.error(f"Missing required columns for analysis: {missing}")
        st.stop()

    raw = raw.dropna(subset=["date"]).copy()
    for c in ["backlog_qty", "orders_created", "orders_dispatched", "truck_availability_index", "dispatch_delay_days"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0)

    raw["dispatch_rate"] = _safe_div(raw["orders_dispatched"], raw["backlog_qty"] + 1).fillna(0)
    raw["order_gap"] = raw["orders_created"] - raw["orders_dispatched"]

    with st.sidebar:
        st.header("Tab 1 Filters (Analysis)")
        freq = st.selectbox("Time aggregation", ["Days", "Weeks", "Months", "Quarters"], index=1, key="ana_freq")

        snapshot_mode = st.selectbox(
            "Snapshot window (analysis)",
            ["Latest date", "Last 7 days", "Last 14 days", "Pick a date"],
            index=1,
            key="ana_snapshot_mode"
        )

        picked_date = None
        if snapshot_mode == "Pick a date":
            latest_date = raw["date"].max()
            picked = st.date_input(
                "Snapshot date (analysis)",
                value=latest_date.date(),
                key="ana_snapshot_date"
            )
            picked_date = pd.to_datetime(picked).floor("D")

        region_sel = st.multiselect("Region", sorted(raw["region"].dropna().unique()), key="ana_region")
        product_sel = st.multiselect("Product", sorted(raw["product"].dropna().unique()), key="ana_product")
        top_n = st.slider("Top N", 5, 50, 15, key="ana_topn")

        st.divider()
        st.subheader("Customer drilldown")
        enable_drill = st.checkbox("Enable drilldown", value=True, key="ana_drill_on")

    df, snap_label = apply_snapshot_filter(raw, "date", snapshot_mode, picked_date)
    st.caption(f"Analysis snapshot used: {snap_label}")

    if region_sel:
        df = df[df["region"].isin(region_sel)]
    if product_sel:
        df = df[df["product"].isin(product_sel)]

    if df.empty:
        st.warning("No data after filters. Adjust snapshot/region/product filters.")
        return

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Customers", f"{df['customer_code'].nunique():,}")
    c3.metric("Regions", f"{df['region'].nunique():,}")
    c4.metric("Products", f"{df['product'].nunique():,}")
    c5.metric("Avg truck index", f"{df['truck_availability_index'].mean():.2f}")
    c6.metric("Total backlog (sum)", f"{df['backlog_qty'].sum():,.0f}")

    df2 = resample_view(df, freq)

    st.markdown("## Overview")

    metric_panel(
        df2,
        title=f"Total backlog by region ({freq})",
        x="period",
        y="backlog_qty",
        color="region",
        agg="sum",
        chart_key="ana_backlog_by_region",
        default_chart="Line",
        top_n_default=20
    )

    metric_panel(
        df,
        title="Top customers by total backlog",
        x=None,
        y="backlog_qty",
        color="customer_code",
        agg="sum",
        chart_key="ana_top_customers",
        default_chart="Bar",
        top_n_default=int(top_n)
    )

    metric_panel(
        df,
        title="Top products by total backlog",
        x=None,
        y="backlog_qty",
        color="product",
        agg="sum",
        chart_key="ana_top_products",
        default_chart="Bar",
        top_n_default=min(int(top_n), 20)
    )

    metric_panel(
        df,
        title="Avg dispatch rate by region (dispatched / backlog)",
        x=None,
        y="dispatch_rate",
        color="region",
        agg="mean",
        chart_key="ana_dispatch_rate",
        default_chart="Bar",
        top_n_default=20
    )

    metric_panel(
        df,
        title="Avg dispatch delay days by region (higher = slower)",
        x=None,
        y="dispatch_delay_days",
        color="region",
        agg="mean",
        chart_key="ana_delay_by_region",
        default_chart="Bar",
        top_n_default=20
    )

    metric_panel(
        df,
        title="Avg truck availability index by region (lower = worse)",
        x=None,
        y="truck_availability_index",
        color="region",
        agg="mean",
        chart_key="ana_truck_by_region",
        default_chart="Bar",
        top_n_default=20
    )

    metric_panel(
        df,
        title="Total order gap by region (created - dispatched)",
        x=None,
        y="order_gap",
        color="region",
        agg="sum",
        chart_key="ana_order_gap_by_region",
        default_chart="Bar",
        top_n_default=20
    )

    # Customer drilldown stays as before (no need to paste again here if yours is already working)
    # Keeping your drilldown block would work fine with the fixed metric_panel.

    st.download_button(
        "Download filtered analysis data (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="analysis_filtered.csv",
        mime="text/csv",
        key="ana_download"
    )


# =========================
# TAB 2: PREDICTION / RISK
# =========================
def render_prediction_tab(pipe):
    st.subheader("Predictions & Risk Dashboard")

    def _reset_pred_filters():
        for k in [
            "pred_f_region", "pred_f_prod", "pred_f_risk", "pred_f_cust",
            "pred_snapshot_mode", "pred_snapshot_date"
        ]:
            if k in st.session_state:
                del st.session_state[k]

    with st.sidebar:
        st.header("Tab 2: Prediction inputs")
        uploaded_pred = st.file_uploader(
            "Upload raw fulfillment CSV (for predictions)",
            type=["csv"],
            key="pred_uploader"
        )
        demo_mode = st.checkbox("Use demo data (when no file uploaded)", value=True, key="pred_demo")

        st.header("Risk thresholds (cover-days)")
        t_med = st.number_input("MEDIUM if ≤", value=7.0, step=1.0, key="pred_t_med")
        t_high = st.number_input("HIGH if ≤", value=3.0, step=1.0, key="pred_t_high")
        t_crit = st.number_input("CRITICAL if ≤", value=1.0, step=1.0, key="pred_t_crit")

        st.header("History handling (lags/rolling)")
        history_mode = st.radio(
            "Rows with insufficient history",
            options=["Drop them (recommended)", "Fill with 0 (predict anyway)"],
            index=0,
            key="pred_hist_mode"
        )

        st.divider()
        if st.button("Reset prediction filters", key="pred_reset_btn"):
            _reset_pred_filters()
            st.rerun()

    if uploaded_pred is None:
        if demo_mode:
            raw = make_demo_data()
            st.info("Demo mode is ON (no file uploaded). Showing predictions using demo data.")
        else:
            st.info("Upload a CSV in Tab 2 to generate predictions, or enable Demo mode.")
            return
    else:
        try:
            raw = pd.read_csv(uploaded_pred)
        except Exception as e:
            st.exception(e)
            st.error("Could not read uploaded file as CSV.")
            return

    current_data_id = "DEMO" if uploaded_pred is None else f"{uploaded_pred.name}:{uploaded_pred.size}"
    if st.session_state.get("_pred_data_id") != current_data_id:
        st.session_state["_pred_data_id"] = current_data_id
        _reset_pred_filters()

    dropped = 0
    try:
        feat = build_features_like_training(raw)

        hist_cols = [
            "backlog_lag_1", "backlog_lag_7",
            "dispatch_lag_1", "dispatch_lag_7",
            "backlog_roll_mean_7", "dispatch_roll_mean_7"
        ]
        feat["insufficient_history"] = feat[hist_cols].isna().any(axis=1)

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
            "depot_status", "plant_status",
            "backlog_qty", "daily_target", "orders_created", "orders_dispatched",
            "truck_availability_index", "dispatch_delay_days",
            "available_funds",
            "dispatch_rate", "backlog_change", "order_gap",
            "backlog_cover_days",
            "backlog_lag_1", "backlog_lag_7",
            "dispatch_lag_1", "dispatch_lag_7",
            "backlog_roll_mean_7", "dispatch_roll_mean_7",
        ]].copy()

        results["predicted_backlog"] = preds.values
        results["pred_cover_days"] = results["predicted_backlog"] / (results["daily_target"] + 1)
        results["risk_level"] = risk_from_cover_days(results["pred_cover_days"], t_med, t_high, t_crit)
        results["risk_level"] = pd.Categorical(results["risk_level"], categories=RISK_ORDER, ordered=True)

        results = add_action_recommendations(results)

    except Exception as e:
        st.exception(e)
        st.error("Processing failed. Fix the error above.")
        return

    if results.empty:
        st.error("No results produced. Try 'Fill with 0' or upload more history.")
        return

    if dropped > 0:
        st.warning(
            f"Dropped {dropped:,} row(s) due to insufficient history for lags/rolling. "
            f"Upload more historical days per customer/product, or choose 'Fill with 0'."
        )

    st.subheader("Filters")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        region_sel = st.multiselect("Region", sorted(results["region"].dropna().unique().tolist()), key="pred_f_region")
    with c2:
        product_sel = st.multiselect("Product", sorted(results["product"].dropna().unique().tolist()), key="pred_f_prod")
    with c3:
        risk_sel = st.multiselect("Risk", RISK_ORDER, default=RISK_ORDER, key="pred_f_risk")
    with c4:
        snapshot_mode = st.selectbox(
            "Snapshot window",
            ["Latest date", "Last 7 days", "Last 14 days", "Pick a date"],
            index=0,
            key="pred_snapshot_mode"
        )
        picked_date = None
        if snapshot_mode == "Pick a date":
            latest_date = pd.to_datetime(results["date"], errors="coerce").dt.floor("D").max()
            picked = st.date_input("Snapshot date", value=latest_date.date(), key="pred_snapshot_date")
            picked_date = pd.to_datetime(picked).floor("D")
    with c5:
        cust_search = st.text_input("Customer search (contains)", value="", key="pred_f_cust")

    f = results.copy()
    if region_sel:
        f = f[f["region"].isin(region_sel)]
    if product_sel:
        f = f[f["product"].isin(product_sel)]
    if risk_sel:
        f = f[f["risk_level"].isin(risk_sel)]
    if cust_search.strip():
        f = f[f["customer_code"].astype(str).str.contains(cust_search.strip(), case=False, na=False)]

    f, snap_label = apply_snapshot_filter(f, "date", snapshot_mode, picked_date)
    st.caption(f"Prediction snapshot used: {snap_label}")

    # Switchable visuals (no crash now)
    metric_panel(
        f,
        title="Total predicted backlog by region (snapshot window)",
        x="date",
        y="predicted_backlog",
        color="region",
        agg="sum",
        chart_key="pred_total_pred_backlog_by_region",
        default_chart="Line",
        top_n_default=20
    )

    metric_panel(
        f.assign(_one=1),
        title="Risk distribution (count by risk level)",
        x=None,
        y="_one",
        color="risk_level",
        agg="count",
        chart_key="pred_risk_dist",
        default_chart="Bar",
        top_n_default=10
    )

    metric_panel(
        f.assign(_one=1),
        title="Risk trend over time (daily count by risk level)",
        x="date",
        y="_one",
        color="risk_level",
        agg="count",
        chart_key="pred_risk_trend",
        default_chart="Line",
        top_n_default=10
    )

    st.subheader("Detailed table (color-coded risk)")
    table_cols = [
        "date", "customer_code", "region", "product",
        "predicted_backlog", "pred_cover_days", "risk_level",
        "backlog_qty", "daily_target", "orders_created", "orders_dispatched",
        "depot_status", "plant_status", "truck_availability_index", "dispatch_delay_days",
        "dispatch_rate", "backlog_change", "order_gap",
        "drivers", "action"
    ]
    show = f[table_cols].sort_values(["risk_level", "pred_cover_days"], ascending=[True, True])
    st.dataframe(style_risk_table(show), use_container_width=True, height=520)


# =========================
# MAIN APP
# =========================
def main():
    st.set_page_config(page_title="Backlog Dashboard", layout="wide")

    st.markdown(
        """
        <style>
        [data-testid="stMetricLabel"] > div {
            white-space: nowrap !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Backlog Dashboard")

    try:
        pipe = load_pipeline(PIPELINE_PATH)
    except Exception as e:
        st.exception(e)
        st.error(f"Could not load '{PIPELINE_PATH}'. Put it in the same folder as this app file.")
        return

    tab1, tab2 = st.tabs(["Analysis (Raw)", "Predictions & Risk"])

    with tab1:
        render_analysis_tab()

    with tab2:
        render_prediction_tab(pipe)


if __name__ == "__main__":
    main()