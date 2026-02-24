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


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_pipeline(path: str):
    return joblib.load(path)


# =========================
# DEMO DATA (so app never crashes without upload)
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
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["customer_code", "product", "date"]).reset_index(drop=True)
    return df


# =========================
# FEATURE ENGINEERING (your logic + lags + rolling)
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

    # Types
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")

    num_cols = [
        "orders_created", "orders_dispatched", "backlog_qty", "available_funds",
        "daily_target", "truck_availability_index", "dispatch_delay_days"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sort for time features
    df = df.sort_values(["customer_code", "product", "date"]).reset_index(drop=True)

    # Minimal fills
    df["orders_created"] = df["orders_created"].fillna(0)
    df["orders_dispatched"] = df["orders_dispatched"].fillna(0)
    df["backlog_qty"] = df["backlog_qty"].fillna(0)
    df["daily_target"] = df["daily_target"].fillna(df["daily_target"].median())

    # Engineered features you listed
    df["backlog_cover_days"] = df["backlog_qty"] / (df["daily_target"] + 1)
    df["dispatch_rate"] = df["orders_dispatched"] / (df["backlog_qty"] + 1)

    df["backlog_change"] = (
        df.groupby(["customer_code", "product"])["backlog_qty"].diff()
    ).fillna(0)

    df["order_gap"] = df["orders_created"] - df["orders_dispatched"]

    # Lags
    df["backlog_lag_1"] = df.groupby(["customer_code", "product"])["backlog_qty"].shift(1)
    df["backlog_lag_7"] = df.groupby(["customer_code", "product"])["backlog_qty"].shift(7)

    df["dispatch_lag_1"] = df.groupby(["customer_code", "product"])["orders_dispatched"].shift(1)
    df["dispatch_lag_7"] = df.groupby(["customer_code", "product"])["orders_dispatched"].shift(7)

    # Rolling means (PAST ONLY)
    df["backlog_roll_mean_7"] = (
        df.groupby(["customer_code", "product"])["backlog_qty"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    )

    df["dispatch_roll_mean_7"] = (
        df.groupby(["customer_code", "product"])["orders_dispatched"]
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
# MAIN APP
# =========================
def main():
    st.set_page_config(page_title="Backlog Risk Dashboard", layout="wide")
    st.title("📦 Backlog Risk Dashboard")

    # Load model
    try:
        pipe = load_pipeline(PIPELINE_PATH)
    except Exception as e:
        st.exception(e)
        st.error(f"Could not load '{PIPELINE_PATH}'. Put it in the same folder as backlog.py")
        return

    # Sidebar inputs
    with st.sidebar:
        st.header("Data input")
        uploaded = st.file_uploader("Upload raw fulfillment CSV", type=["csv"])
        demo_mode = st.checkbox("Use demo data (when no file uploaded)", value=True)

        st.header("Risk thresholds (cover-days)")
        t_med = st.number_input("MEDIUM if ≤", value=7.0, step=1.0)
        t_high = st.number_input("HIGH if ≤", value=3.0, step=1.0)
        t_crit = st.number_input("CRITICAL if ≤", value=1.0, step=1.0)

        st.header("History handling (lags/rolling)")
        history_mode = st.radio(
            "Rows with insufficient history",
            options=["Drop them (recommended)", "Fill with 0 (predict anyway)"],
            index=0
        )

    # Read data safely
    if uploaded is None:
        if demo_mode:
            raw = make_demo_data()
            st.info("Demo mode is ON (no file uploaded). Showing dashboard using demo data.")
            st.download_button(
                "Download demo CSV",
                data=raw.to_csv(index=False).encode("utf-8"),
                file_name="demo_fulfillment_data.csv",
                mime="text/csv"
            )
        else:
            st.info("Upload a CSV to begin, or enable Demo mode in the sidebar.")
            return
    else:
        try:
            raw = pd.read_csv(uploaded)
        except Exception as e:
            st.exception(e)
            st.error("Could not read uploaded file as CSV.")
            return

    # Process + predict safely
    dropped = 0
    results = None

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

        # Determine expected feature columns from pipeline if available
        expected = None
        if hasattr(pipe, "named_steps") and "preprocess" in pipe.named_steps:
            if hasattr(pipe.named_steps["preprocess"], "feature_names_in_"):
                expected = list(pipe.named_steps["preprocess"].feature_names_in_)

        drop_cols = [c for c in ["date", "customer_code", "future_backlog_qty"] if c in feat_model.columns]

        if expected is None:
            X_model = feat_model.drop(columns=drop_cols)
        else:
            for c in expected:
                if c not in feat_model.columns:
                    feat_model[c] = 0
            X_model = feat_model[expected]

        preds = pipe.predict(X_model)
        preds = pd.Series(preds).clip(lower=0)

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
        st.exception(e)  # show real traceback in the UI
        st.error("Processing failed. Fix the error above.")
        return

    # Guard
    if results is None or results.empty:
        st.error("No results produced (empty dataset after processing). Try 'Fill with 0' or upload more history.")
        return

    if dropped > 0:
        st.warning(
            f"Dropped {dropped} row(s) due to insufficient history for lags/rolling. "
            f"Upload more historical days per customer/product, or choose 'Fill with 0'."
        )

    # Filters
    st.subheader("Filters")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        region_sel = st.multiselect("Region", sorted(results["region"].dropna().unique().tolist()))
    with c2:
        product_sel = st.multiselect("Product", sorted(results["product"].dropna().unique().tolist()))
    with c3:
        risk_sel = st.multiselect("Risk", RISK_ORDER, default=RISK_ORDER)
    with c4:
        date_min = results["date"].min()
        date_max = results["date"].max()
        date_range = st.date_input("Date range", value=(date_min.date(), date_max.date()))
    with c5:
        cust_search = st.text_input("Customer search (contains)", value="")

    f = results.copy()
    if region_sel:
        f = f[f["region"].isin(region_sel)]
    if product_sel:
        f = f[f["product"].isin(product_sel)]
    if risk_sel:
        f = f[f["risk_level"].isin(risk_sel)]
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        d0, d1 = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        f = f[(f["date"] >= d0) & (f["date"] <= d1)]
    if cust_search.strip():
        f = f[f["customer_code"].astype(str).str.contains(cust_search.strip(), case=False, na=False)]

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Rows", f"{len(f):,}")
    k2.metric("Customers", f"{f['customer_code'].nunique():,}")
    k3.metric("Avg predicted backlog", f"{f['predicted_backlog'].mean():.2f}")
    k4.metric("Avg cover-days", f"{f['pred_cover_days'].mean():.2f}")
    k5.metric("Critical rows", f"{int((f['risk_level'] == 'CRITICAL').sum()):,}")

    # Critical cards (latest date)
    st.subheader("🚨 Critical customers (latest date) + recommended actions")
    latest_date = f["date"].max()
    today_df = f[f["date"] == latest_date].copy()
    crit_today = today_df[today_df["risk_level"] == "CRITICAL"].sort_values("pred_cover_days").head(12)

    if crit_today.empty:
        st.success("No CRITICAL customers on the latest date in the filtered data.")
    else:
        cols = st.columns(3)
        for i, (_, r) in enumerate(crit_today.iterrows()):
            col = cols[i % 3]
            with col:
                st.markdown(
                    f"""
                    <div style="border-radius:16px;padding:14px;border:1px solid #ff3b30;background:#ff3b3018;">
                        <div style="font-size:16px;font-weight:800;">{r['customer_code']} • {r['product']} • {r['region']}</div>
                        <div style="margin-top:6px;">
                            <span style="padding:4px 10px;border-radius:999px;background:#ff3b30;color:black;font-weight:800;">
                                CRITICAL
                            </span>
                        </div>
                        <div style="margin-top:10px;">
                            <b>Pred backlog:</b> {r['predicted_backlog']:.0f}<br/>
                            <b>Cover-days:</b> {r['pred_cover_days']:.2f}<br/>
                            <b>Depot:</b> {r['depot_status']} • <b>Plant:</b> {r['plant_status']}<br/>
                            <b>Delay days:</b> {r['dispatch_delay_days']} • <b>Truck idx:</b> {r['truck_availability_index']}
                        </div>
                        <div style="margin-top:10px;">
                            <b>Drivers:</b> {r['drivers']}<br/>
                            <b>Action:</b> {r['action']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # Charts
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Risk distribution")
        risk_counts = f["risk_level"].value_counts().reindex(RISK_ORDER).fillna(0).reset_index()
        risk_counts.columns = ["risk_level", "count"]
        fig = px.bar(
            risk_counts,
            x="risk_level",
            y="count",
            color="risk_level",
            color_discrete_map=RISK_COLORS,
            title="Count by risk level"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Top risky customers (severity score)")
        sev = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        tmp = f.copy()
        tmp["_sev"] = tmp["risk_level"].astype(str).map(sev).fillna(0)

        cust_risk = (
            tmp.groupby("customer_code")
               .agg(
                    rows=("customer_code", "size"),
                    max_severity=("_sev", "max"),
                    severity_sum=("_sev", "sum"),
                    avg_predicted_backlog=("predicted_backlog", "mean"),
                    max_predicted_backlog=("predicted_backlog", "max"),
                    avg_cover_days=("pred_cover_days", "mean"),
               )
               .sort_values(["max_severity", "severity_sum", "max_predicted_backlog"], ascending=False)
               .head(20)
               .reset_index()
        )
        st.dataframe(cust_risk, use_container_width=True, hide_index=True)

    st.subheader("Risk trend over time")
    daily = f.groupby(["date", "risk_level"], observed=True).size().reset_index(name="count")
    fig2 = px.line(
        daily,
        x="date",
        y="count",
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        markers=True,
        title="Daily count by risk level"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Customer × Product severity heatmap")
    sev = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    tmp = f.copy()
    tmp["_sev"] = tmp["risk_level"].astype(str).map(sev).fillna(0)
    pivot = tmp.pivot_table(index="customer_code", columns="product", values="_sev", aggfunc="max", fill_value=0)
    top_customers = tmp.groupby("customer_code")["_sev"].max().sort_values(ascending=False).head(30).index
    pivot = pivot.loc[top_customers]

    fig3 = px.imshow(pivot, aspect="auto", title="Severity heatmap (0=LOW, 3=CRITICAL)")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Driver breakdown (CRITICAL/HIGH)")
    focus = f[f["risk_level"].isin(["CRITICAL", "HIGH"])].copy()
    if focus.empty:
        st.info("No CRITICAL/HIGH rows in the filtered data.")
    else:
        driver_rows = []
        for _, r in focus.iterrows():
            for d in str(r["drivers"]).split("; "):
                driver_rows.append({"risk_level": str(r["risk_level"]), "driver": d})
        drv = pd.DataFrame(driver_rows)
        drv_counts = drv.groupby(["risk_level", "driver"]).size().reset_index(name="count")
        fig_drv = px.bar(
            drv_counts,
            x="driver",
            y="count",
            color="risk_level",
            color_discrete_map=RISK_COLORS,
            barmode="group",
            title="Drivers for CRITICAL/HIGH"
        )
        fig_drv.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(fig_drv, use_container_width=True)

    # Table + download
    st.subheader("Detailed table (color-coded risk)")
    table_cols = [
        "date","customer_code","region","product",
        "predicted_backlog","pred_cover_days","risk_level",
        "backlog_qty","daily_target","orders_created","orders_dispatched",
        "depot_status","plant_status","truck_availability_index","dispatch_delay_days",
        "dispatch_rate","backlog_change","order_gap",
        "drivers","action"
    ]
    show = f[table_cols].sort_values(["risk_level", "pred_cover_days"], ascending=[True, True])
    st.dataframe(style_risk_table(show), use_container_width=True, height=520)

    st.download_button(
        "Download filtered results (CSV)",
        data=show.to_csv(index=False).encode("utf-8"),
        file_name="backlog_risk_predictions.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
