import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import hashlib
import logging
import datetime
import io
import os
import re

# =========================
# SETTINGS
# =========================
PIPELINE_PATH = "backlog_pipeline.pkl"

# ── Paste the SHA-256 of your .pkl here (run generate_model_hash.py once) ──
EXPECTED_MODEL_HASH = os.environ.get("MODEL_HASH", "")   # or hardcode after first run

# Allowed company email domain(s)  e.g. "@mycompany.com"
ALLOWED_DOMAINS = [d.strip() for d in os.environ.get("ALLOWED_DOMAINS", "@gmail.com").split(",")]

MAX_UPLOAD_MB   = 20
MAX_ROWS        = 100_000

ALLOWED_COLUMNS = {
    "date", "customer_code", "region", "product",
    "orders_created", "orders_dispatched", "backlog_qty",
    "available_funds", "daily_target", "depot_status", "plant_status",
    "truck_availability_index", "dispatch_delay_days",
}

RISK_ORDER  = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
RISK_COLORS = {
    "CRITICAL": "#ff3b30",
    "HIGH":     "#ff9500",
    "MEDIUM":   "#ffcc00",
    "LOW":      "#34c759",
}

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive":  True,
    "scrollZoom":  True,
    "toImageButtonOptions": {
        "format": "png", "filename": "chart",
        "height": 900,   "width": 1600, "scale": 2
    },
}

# =========================
# LOGGING  (audit trail)
# =========================
logging.basicConfig(
    filename="audit.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
audit = logging.getLogger("audit")

def _user() -> str:
    """Return current authenticated user email (or 'unknown')."""
    return st.session_state.get("user_email", "unknown")

def log(action: str, detail: str = ""):
    audit.info(f"user={_user()} | action={action} | {detail}")


# =========================
# GOOGLE OAUTH  (built into Streamlit Community Cloud)
# =========================
def check_auth():
    """
    Streamlit Community Cloud exposes st.experimental_user (or st.user in newer versions).
    We validate the email domain and store it in session_state.
    """
    # ── Streamlit >= 1.35 uses st.user; older uses st.experimental_user ──
    user_obj = getattr(st, "user", None) or getattr(st, "experimental_user", None)

    if user_obj is None:
        # Running locally without auth configured — warn developer
        st.warning(
            "⚠️ Auth not configured. "
            "Deploy to Streamlit Community Cloud and enable Google OAuth in your app settings."
        )
        st.info("Running in LOCAL DEV mode — authentication bypassed.")
        st.session_state["user_email"]      = "dev@local"
        st.session_state["user_name"]       = "Local Developer"
        st.session_state["authenticated"]   = True
        return

    email = getattr(user_obj, "email", None)

    if not email:
        _show_login_wall("Could not retrieve your email from Google. Please try again.")
        st.stop()

    # Domain check
    domain_ok = any(email.lower().endswith(d.lower()) for d in ALLOWED_DOMAINS)
    if not domain_ok:
        log("AUTH_DENIED", f"email={email}")
        _show_login_wall(
            f"**Access denied.** Your account (`{email}`) is not authorised.\n\n"
            f"Contact your administrator to get access."
        )
        st.stop()

    # All good
    st.session_state["user_email"]    = email
    st.session_state["user_name"]     = getattr(user_obj, "name", email)
    st.session_state["authenticated"] = True
    log("AUTH_SUCCESS")


def _show_login_wall(message: str):
    st.set_page_config(page_title="Backlog Dashboard — Login", layout="centered")
    st.markdown(
        """
        <style>
        .login-box {
            max-width: 480px; margin: 80px auto; padding: 2.5rem 3rem;
            border: 1px solid #e0e0e0; border-radius: 12px;
            box-shadow: 0 4px 24px rgba(0,0,0,.08);
            text-align: center; font-family: sans-serif;
        }
        .login-box h2 { margin-bottom: .25rem; }
        .login-box p  { color: #666; font-size: .95rem; }
        </style>
        <div class="login-box">
            <h2>🔒 Backlog Dashboard</h2>
            <p>Sign in with your company Google account to continue.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.error(message)


# =========================
# SECURE MODEL LOADING
# =========================
@st.cache_resource
def load_pipeline(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    actual_hash = hashlib.sha256(open(path, "rb").read()).hexdigest()

    if EXPECTED_MODEL_HASH and actual_hash != EXPECTED_MODEL_HASH:
        log("MODEL_HASH_MISMATCH", f"expected={EXPECTED_MODEL_HASH} actual={actual_hash}")
        raise ValueError(
            "Model file integrity check failed.\n"
            f"Expected: {EXPECTED_MODEL_HASH}\nGot:      {actual_hash}\n"
            "The file may have been tampered with."
        )

    if not EXPECTED_MODEL_HASH:
        st.sidebar.warning(
            f"⚠️ Model hash not set. Set `EXPECTED_MODEL_HASH` in secrets.\n\n"
            f"Current hash: `{actual_hash}`"
        )

    log("MODEL_LOADED", f"path={path} hash={actual_hash}")
    return joblib.load(path)


# =========================
# SAFE CSV READER
# =========================
def safe_read_csv(uploaded_file) -> pd.DataFrame:
    # 1. Size guard
    if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"File too large. Maximum allowed size is {MAX_UPLOAD_MB} MB.")
        log("UPLOAD_REJECTED", f"reason=size_exceeded file={uploaded_file.name} size={uploaded_file.size}")
        st.stop()

    # 2. Filename sanitisation
    safe_name = re.sub(r"[^\w.\-]", "_", uploaded_file.name)

    # 3. Parse
    try:
        content = uploaded_file.read()
        df = pd.read_csv(
            io.BytesIO(content),
            nrows=MAX_ROWS,
            on_bad_lines="skip",
        )
    except Exception as e:
        log("UPLOAD_PARSE_ERROR", f"file={safe_name} error={e}")
        st.error(f"Could not parse CSV: {e}")
        st.stop()

    if len(df) >= MAX_ROWS:
        st.warning(f"File has been capped at {MAX_ROWS:,} rows for safety.")

    # 4. Column allow-list  (drop anything unexpected)
    unexpected = set(df.columns) - ALLOWED_COLUMNS
    if unexpected:
        st.warning(f"Unexpected columns were ignored: {unexpected}")
        df = df[[c for c in df.columns if c in ALLOWED_COLUMNS]]

    log("UPLOAD_OK", f"file={safe_name} rows={len(df)} cols={list(df.columns)}")
    return df


# =========================
# DEMO DATA
# =========================
def make_demo_data(n_rows: int = 700, n_days: int = 45) -> pd.DataFrame:
    np.random.seed(42)
    dates          = pd.date_range("2024-01-01", periods=n_days, freq="D")
    customers      = [f"cust_{i:03d}" for i in range(1, 31)]
    products       = ["PMS", "DPK", "AGO"]
    regions        = ["Lagos", "North", "East", "West"]
    depot_statuses = ["Available", "Low Stock", "Out of Stock"]
    plant_statuses = ["Running", "Maintenance", "Shutdown"]

    df = pd.DataFrame({
        "date":                    np.random.choice(dates, n_rows),
        "customer_code":           np.random.choice(customers, n_rows),
        "region":                  np.random.choice(regions, n_rows),
        "product":                 np.random.choice(products, n_rows),
        "orders_created":          np.random.randint(0, 60, n_rows),
        "orders_dispatched":       np.random.randint(0, 60, n_rows),
        "backlog_qty":             np.random.randint(0, 900, n_rows),
        "available_funds":         np.random.randint(50_000, 3_000_000, n_rows),
        "daily_target":            np.random.randint(10, 120, n_rows),
        "depot_status":            np.random.choice(depot_statuses, n_rows, p=[0.7, 0.2, 0.1]),
        "plant_status":            np.random.choice(plant_statuses, n_rows, p=[0.75, 0.2, 0.05]),
        "truck_availability_index": np.round(np.random.uniform(0.3, 1.0, n_rows), 2),
        "dispatch_delay_days":     np.random.randint(0, 8, n_rows),
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

    df["orders_created"]    = df["orders_created"].fillna(0)
    df["orders_dispatched"] = df["orders_dispatched"].fillna(0)
    df["backlog_qty"]       = df["backlog_qty"].fillna(0)

    if df["daily_target"].notna().any():
        df["daily_target"] = df["daily_target"].fillna(df["daily_target"].median())
    else:
        df["daily_target"] = df["daily_target"].fillna(0)

    df["backlog_cover_days"] = df["backlog_qty"] / (df["daily_target"] + 1)
    df["dispatch_rate"]      = df["orders_dispatched"] / (df["backlog_qty"] + 1)
    df["backlog_change"]     = (
        df.groupby(["customer_code", "product"], observed=True)["backlog_qty"].diff()
    ).fillna(0)
    df["order_gap"] = df["orders_created"] - df["orders_dispatched"]

    df["backlog_lag_1"]  = df.groupby(["customer_code", "product"], observed=True)["backlog_qty"].shift(1)
    df["backlog_lag_7"]  = df.groupby(["customer_code", "product"], observed=True)["backlog_qty"].shift(7)
    df["dispatch_lag_1"] = df.groupby(["customer_code", "product"], observed=True)["orders_dispatched"].shift(1)
    df["dispatch_lag_7"] = df.groupby(["customer_code", "product"], observed=True)["orders_dispatched"].shift(7)

    df["backlog_roll_mean_7"]  = (
        df.groupby(["customer_code", "product"], observed=True)["backlog_qty"]
          .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    )
    df["dispatch_roll_mean_7"] = (
        df.groupby(["customer_code", "product"], observed=True)["orders_dispatched"]
          .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    )
    return df


def risk_from_cover_days(cover_days, t_med, t_high, t_crit):
    risk = pd.Series(np.where(cover_days <= t_med,  "MEDIUM", "LOW"),    index=cover_days.index)
    risk = pd.Series(np.where(cover_days <= t_high, "HIGH",   risk),     index=cover_days.index)
    risk = pd.Series(np.where(cover_days <= t_crit, "CRITICAL", risk),   index=cover_days.index)
    return risk


def add_action_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    drivers, actions = [], []
    for _, r in out.iterrows():
        d, a = [], []
        depot = str(r.get("depot_status", "")).lower()
        plant = str(r.get("plant_status", "")).lower()

        if depot == "out of stock":
            d.append("Depot out of stock");   a.append("Replenish depot stock / reroute supply")
        elif depot == "low stock":
            d.append("Depot low stock");      a.append("Prioritize replenishment to depot")

        if plant == "shutdown":
            d.append("Plant shutdown");       a.append("Escalate plant restart / alternate sourcing")
        elif plant == "maintenance":
            d.append("Plant maintenance");    a.append("Coordinate maintenance / alternate sourcing")

        try:
            if float(r.get("truck_availability_index", 1.0)) < 0.5:
                d.append("Low truck availability"); a.append("Allocate more trucks / rebalance fleet")
        except Exception:
            pass
        try:
            if float(r.get("dispatch_delay_days", 0.0)) >= 3:
                d.append("High dispatch delays");   a.append("Expedite dispatch / remove bottlenecks")
        except Exception:
            pass

        drivers.append("; ".join(d) if d else "No dominant driver found")
        actions.append("; ".join(a) if a else "Monitor / validate inputs")

    out["drivers"] = drivers
    out["action"]  = actions
    return out


def style_risk_table(df: pd.DataFrame):
    def row_bg(row):
        bg = RISK_COLORS.get(str(row.get("risk_level", "")), "#ffffff")
        return [f"background-color: {bg}22;" for _ in row]

    def risk_cell(row):
        bg = RISK_COLORS.get(str(row.get("risk_level", "")), "#ffffff")
        return [
            f"background-color: {bg}; color: black; font-weight: 800;"
            if c == "risk_level" else ""
            for c in df.columns
        ]
    return df.style.apply(row_bg, axis=1).apply(risk_cell, axis=1)


# =========================
# DATE SNAPSHOT FILTER
# =========================
def apply_snapshot_filter(df, date_col, mode, picked_date):
    if df.empty or date_col not in df.columns:
        return df.copy(), "No data"

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.floor("D")
    out = out.dropna(subset=[date_col]).copy()
    if out.empty:
        return out, "No valid dates"

    latest = out[date_col].max()

    if mode == "Latest date":
        return out[out[date_col] == latest].copy(), f"Latest date only: {latest.date()}"
    if mode == "Last 7 days":
        s = latest - pd.Timedelta(days=6)
        return out[(out[date_col] >= s) & (out[date_col] <= latest)].copy(), f"Rolling 7d: {s.date()} → {latest.date()}"
    if mode == "Last 14 days":
        s = latest - pd.Timedelta(days=13)
        return out[(out[date_col] >= s) & (out[date_col] <= latest)].copy(), f"Rolling 14d: {s.date()} → {latest.date()}"
    if mode == "Pick a date":
        if picked_date is None or pd.isna(picked_date):
            return out[out[date_col] == latest].copy(), f"Latest (fallback): {latest.date()}"
        pd_date = pd.to_datetime(picked_date).floor("D")
        snap = out[out[date_col] == pd_date].copy()
        if snap.empty:
            return out[out[date_col] == latest].copy(), f"Date missing; using latest: {latest.date()}"
        return snap, f"Selected date: {pd_date.date()}"

    return out[out[date_col] == latest].copy(), f"Latest (fallback): {latest.date()}"


# =========================
# HELPERS
# =========================
def _ensure_datetime(df, col="date"):
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.floor("D")
    return out


def _safe_div(n, d):
    return n / d.replace(0, np.nan)


def resample_view(df, freq_label):
    out = df.copy()
    m = {"Days": None, "Weeks": "W", "Months": "M", "Quarters": "Q"}
    freq = m.get(freq_label)
    out["period"] = out["date"].dt.to_period(freq).dt.start_time if freq else out["date"]
    return out


# =========================
# UNIVERSAL METRIC PANEL
# =========================
def metric_panel(df, *, title, x, y, color=None, agg="sum",
                 chart_key, default_chart="Line", top_n_default=20):
    if df is None or df.empty:
        st.info("No data after filters.")
        return

    st.markdown(f"### {title}")
    chart_types = ["Line", "Area", "Bar", "Scatter", "Cards", "Table"]
    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        chart_type = st.selectbox("Chart type", chart_types,
                                  index=chart_types.index(default_chart) if default_chart in chart_types else 0,
                                  key=f"{chart_key}_type")
    with c2:
        top_n = st.number_input("Top N", 5, 200, int(top_n_default), key=f"{chart_key}_topn")

    work = df.copy()

    def _count_grouped(df_in, group_cols, out_name):
        s = df_in.groupby(group_cols, observed=True).size().reset_index()
        return s.rename(columns={0: out_name})

    def _agg(df_in, group_cols):
        if agg == "sum":  return df_in.groupby(group_cols, as_index=False, observed=True).agg(**{y: (y, "sum")})
        if agg == "mean": return df_in.groupby(group_cols, as_index=False, observed=True).agg(**{y: (y, "mean")})
        return _count_grouped(df_in, group_cols, y)

    if x is None:
        if color is None:
            val = float(work[y].sum()) if agg == "sum" else float(work[y].mean())
            st.metric(title, f"{val:,.0f}")
            with st.expander("View data"): st.dataframe(work, use_container_width=True, height=320)
            return
        agg_df = _agg(work, [color]).sort_values(y, ascending=False).head(int(top_n))
        if chart_type == "Cards":
            cols = st.columns(4)
            for i, r in enumerate(agg_df.itertuples(index=False)):
                with cols[i % 4]: st.metric(str(getattr(r, color)), f"{float(getattr(r, y)):,.0f}")
        elif chart_type == "Table":
            st.dataframe(agg_df, use_container_width=True, height=320)
        else:
            st.plotly_chart(px.bar(agg_df, x=color, y=y, title=title), use_container_width=True, config=PLOTLY_CONFIG)
        if chart_type != "Table":
            with st.expander("View data"): st.dataframe(agg_df, use_container_width=True, height=320)
        return

    group_cols = [x] + ([color] if color else [])
    agg_df = _agg(work, group_cols)

    if agg_df[x].nunique(dropna=True) <= 1 and chart_type in ["Line", "Area", "Scatter"]:
        st.caption("One date/period — switched to Bar automatically.")
        chart_type = "Bar"

    if   chart_type == "Line":    fig = px.line(agg_df,    x=x, y=y, color=color, markers=True, title=title)
    elif chart_type == "Area":    fig = px.area(agg_df,    x=x, y=y, color=color, title=title)
    elif chart_type == "Scatter": fig = px.scatter(agg_df, x=x, y=y, color=color, title=title)
    elif chart_type == "Cards":
        if color:
            snap = agg_df[agg_df[x] == agg_df[x].max()].sort_values(y, ascending=False).head(int(top_n))
            cols  = st.columns(4)
            for i, r in enumerate(snap.itertuples(index=False)):
                with cols[i % 4]: st.metric(str(getattr(r, color)), f"{float(getattr(r, y)):,.0f}")
        else:
            st.metric(title, f"{float(agg_df[y].sum()):,.0f}")
        with st.expander("View data"): st.dataframe(agg_df, use_container_width=True, height=320)
        return
    elif chart_type == "Table":
        st.dataframe(agg_df, use_container_width=True, height=320)
        return
    else:
        fig = px.bar(agg_df, x=x, y=y, color=color, title=title)

    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    with st.expander("View data"): st.dataframe(agg_df, use_container_width=True, height=320)


# =========================
# TAB 1: ANALYSIS
# =========================
def render_analysis_tab():
    st.subheader("Data Analysis (Raw / Historical)")

    uploaded_analysis = st.file_uploader(
        "Upload dataset for Analysis tab (raw CSV)",
        type=["csv"], key="analysis_uploader"
    )
    if uploaded_analysis is None:
        st.info("Upload a CSV here to run analysis.")
        return

    raw = safe_read_csv(uploaded_analysis)
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
    raw["order_gap"]     = raw["orders_created"] - raw["orders_dispatched"]

    with st.sidebar:
        st.header("Tab 1 Filters (Analysis)")
        freq          = st.selectbox("Time aggregation", ["Days", "Weeks", "Months", "Quarters"], index=1, key="ana_freq")
        snapshot_mode = st.selectbox("Snapshot window", ["Latest date","Last 7 days","Last 14 days","Pick a date"], index=1, key="ana_snapshot_mode")
        picked_date   = None
        if snapshot_mode == "Pick a date":
            picked      = st.date_input("Snapshot date", value=raw["date"].max().date(), key="ana_snapshot_date")
            picked_date = pd.to_datetime(picked).floor("D")
        region_sel  = st.multiselect("Region",  sorted(raw["region"].dropna().unique()),  key="ana_region")
        product_sel = st.multiselect("Product", sorted(raw["product"].dropna().unique()), key="ana_product")
        top_n       = st.slider("Top N", 5, 50, 15, key="ana_topn")

    df, snap_label = apply_snapshot_filter(raw, "date", snapshot_mode, picked_date)
    st.caption(f"Analysis snapshot: {snap_label}")

    if region_sel:  df = df[df["region"].isin(region_sel)]
    if product_sel: df = df[df["product"].isin(product_sel)]

    if df.empty:
        st.warning("No data after filters.")
        return

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Rows",             f"{len(df):,}")
    c2.metric("Customers",        f"{df['customer_code'].nunique():,}")
    c3.metric("Regions",          f"{df['region'].nunique():,}")
    c4.metric("Products",         f"{df['product'].nunique():,}")
    c5.metric("Avg truck index",  f"{df['truck_availability_index'].mean():.2f}")
    c6.metric("Total backlog",    f"{df['backlog_qty'].sum():,.0f}")

    df2 = resample_view(df, freq)
    st.markdown("## Overview")

    for cfg in [
        dict(title=f"Total backlog by region ({freq})", x="period", y="backlog_qty", color="region", agg="sum", chart_key="ana_backlog_region",  default_chart="Line",  top_n_default=20),
        dict(title="Top customers by total backlog",    x=None,     y="backlog_qty", color="customer_code", agg="sum", chart_key="ana_top_customers", default_chart="Bar", top_n_default=top_n),
        dict(title="Top products by total backlog",     x=None,     y="backlog_qty", color="product",       agg="sum", chart_key="ana_top_products",  default_chart="Bar", top_n_default=20),
        dict(title="Avg dispatch rate by region",       x=None,     y="dispatch_rate",         color="region", agg="mean", chart_key="ana_dispatch_rate",   default_chart="Bar", top_n_default=20),
        dict(title="Avg dispatch delay (days) by region", x=None,   y="dispatch_delay_days",   color="region", agg="mean", chart_key="ana_delay_region",    default_chart="Bar", top_n_default=20),
        dict(title="Avg truck availability index by region", x=None, y="truck_availability_index", color="region", agg="mean", chart_key="ana_truck_region", default_chart="Bar", top_n_default=20),
        dict(title="Total order gap by region",         x=None,     y="order_gap",             color="region", agg="sum",  chart_key="ana_order_gap",       default_chart="Bar", top_n_default=20),
    ]:
        src = df2 if cfg.get("x") == "period" else df
        metric_panel(src, **cfg)

    # ── Secure download (authenticated users only) ──
    if st.session_state.get("authenticated"):
        st.download_button(
            "⬇ Download filtered analysis data (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="analysis_filtered.csv",
            mime="text/csv",
            key="ana_download"
        )
        log("DOWNLOAD", f"tab=analysis rows={len(df)}")


# =========================
# TAB 2: PREDICTIONS
# =========================
def render_prediction_tab(pipe):
    st.subheader("Predictions & Risk Dashboard")

    def _reset():
        for k in ["pred_f_region","pred_f_prod","pred_f_risk","pred_f_cust","pred_snapshot_mode","pred_snapshot_date"]:
            st.session_state.pop(k, None)

    with st.sidebar:
        st.header("Tab 2: Prediction inputs")
        uploaded_pred = st.file_uploader("Upload raw fulfillment CSV", type=["csv"], key="pred_uploader")
        demo_mode     = st.checkbox("Use demo data (when no file uploaded)", value=True, key="pred_demo")

        st.header("Risk thresholds (cover-days)")
        t_med  = st.number_input("MEDIUM if ≤", value=7.0, step=1.0, key="pred_t_med")
        t_high = st.number_input("HIGH if ≤",   value=3.0, step=1.0, key="pred_t_high")
        t_crit = st.number_input("CRITICAL if ≤", value=1.0, step=1.0, key="pred_t_crit")

        st.header("History handling")
        history_mode = st.radio(
            "Rows with insufficient history",
            ["Drop them (recommended)", "Fill with 0 (predict anyway)"],
            index=0, key="pred_hist_mode"
        )
        st.divider()
        if st.button("Reset prediction filters", key="pred_reset_btn"):
            _reset(); st.rerun()

    if uploaded_pred is None:
        if demo_mode:
            raw = make_demo_data()
            st.info("Demo mode ON — using synthetic demo data.")
        else:
            st.info("Upload a CSV or enable Demo mode.")
            return
    else:
        raw = safe_read_csv(uploaded_pred)

    current_id = "DEMO" if uploaded_pred is None else f"{uploaded_pred.name}:{uploaded_pred.size}"
    if st.session_state.get("_pred_data_id") != current_id:
        st.session_state["_pred_data_id"] = current_id
        _reset()

    dropped = 0
    try:
        feat      = build_features_like_training(raw)
        hist_cols = ["backlog_lag_1","backlog_lag_7","dispatch_lag_1","dispatch_lag_7","backlog_roll_mean_7","dispatch_roll_mean_7"]
        feat["insufficient_history"] = feat[hist_cols].isna().any(axis=1)

        if history_mode.startswith("Drop"):
            dropped    = int(feat["insufficient_history"].sum())
            feat_model = feat.loc[~feat["insufficient_history"]].copy()
        else:
            feat_model = feat.copy()
            feat_model[hist_cols] = feat_model[hist_cols].fillna(0)

        expected = None
        if hasattr(pipe, "named_steps") and "preprocess" in pipe.named_steps:
            if hasattr(pipe.named_steps["preprocess"], "feature_names_in_"):
                expected = list(pipe.named_steps["preprocess"].feature_names_in_)

        drop_cols = [c for c in ["date","customer_code","future_backlog_qty"] if c in feat_model.columns]
        X_model   = feat_model[expected] if expected else feat_model.drop(columns=drop_cols, errors="ignore")

        if expected:
            for c in expected:
                if c not in feat_model.columns:
                    feat_model[c] = 0
            X_model = feat_model[expected]

        preds   = pd.Series(pipe.predict(X_model)).clip(lower=0)
        results = feat_model[[
            "date","customer_code","region","product",
            "depot_status","plant_status","backlog_qty","daily_target",
            "orders_created","orders_dispatched","truck_availability_index","dispatch_delay_days",
            "available_funds","dispatch_rate","backlog_change","order_gap","backlog_cover_days",
            "backlog_lag_1","backlog_lag_7","dispatch_lag_1","dispatch_lag_7",
            "backlog_roll_mean_7","dispatch_roll_mean_7",
        ]].copy()

        results["predicted_backlog"] = preds.values
        results["pred_cover_days"]   = results["predicted_backlog"] / (results["daily_target"] + 1)
        results["risk_level"]        = risk_from_cover_days(results["pred_cover_days"], t_med, t_high, t_crit)
        results["risk_level"]        = pd.Categorical(results["risk_level"], categories=RISK_ORDER, ordered=True)
        results = add_action_recommendations(results)
        log("PREDICTION_RUN", f"rows={len(results)} dropped={dropped}")

    except Exception as e:
        logging.exception("Prediction failed")
        st.error("Prediction failed. Check your CSV format and try again.")
        if st.secrets.get("ENV", "prod") == "dev":
            st.exception(e)
        return

    if results.empty:
        st.error("No results. Try 'Fill with 0' or upload more historical data.")
        return

    if dropped:
        st.warning(f"Dropped {dropped:,} rows due to insufficient lag history. Upload more data or choose 'Fill with 0'.")

    st.subheader("Filters")
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: region_sel   = st.multiselect("Region",  sorted(results["region"].dropna().unique()),  key="pred_f_region")
    with c2: product_sel  = st.multiselect("Product", sorted(results["product"].dropna().unique()), key="pred_f_prod")
    with c3: risk_sel     = st.multiselect("Risk",    RISK_ORDER, default=RISK_ORDER,               key="pred_f_risk")
    with c4:
        snapshot_mode = st.selectbox("Snapshot window", ["Latest date","Last 7 days","Last 14 days","Pick a date"], index=0, key="pred_snapshot_mode")
        picked_date   = None
        if snapshot_mode == "Pick a date":
            latest  = pd.to_datetime(results["date"], errors="coerce").dt.floor("D").max()
            picked  = st.date_input("Snapshot date", value=latest.date(), key="pred_snapshot_date")
            picked_date = pd.to_datetime(picked).floor("D")
    with c5: cust_search  = st.text_input("Customer search", value="", key="pred_f_cust")

    f = results.copy()
    if region_sel:  f = f[f["region"].isin(region_sel)]
    if product_sel: f = f[f["product"].isin(product_sel)]
    if risk_sel:    f = f[f["risk_level"].isin(risk_sel)]
    if cust_search.strip():
        f = f[f["customer_code"].astype(str).str.contains(cust_search.strip(), case=False, na=False)]

    f, snap_label = apply_snapshot_filter(f, "date", snapshot_mode, picked_date)
    st.caption(f"Prediction snapshot: {snap_label}")

    for cfg in [
        dict(title="Total predicted backlog by region", x="date", y="predicted_backlog", color="region", agg="sum", chart_key="pred_total_backlog", default_chart="Line", top_n_default=20),
        dict(title="Risk distribution (count)", x=None, y="_one", color="risk_level", agg="count", chart_key="pred_risk_dist", default_chart="Bar", top_n_default=10),
        dict(title="Risk trend over time (daily count)", x="date", y="_one", color="risk_level", agg="count", chart_key="pred_risk_trend", default_chart="Line", top_n_default=10),
    ]:
        src = f.assign(_one=1) if "_one" in cfg["y"] else f
        metric_panel(src, **cfg)

    table_cols = [
        "date","customer_code","region","product",
        "predicted_backlog","pred_cover_days","risk_level",
        "backlog_qty","daily_target","orders_created","orders_dispatched",
        "depot_status","plant_status","truck_availability_index","dispatch_delay_days",
        "dispatch_rate","backlog_change","order_gap","drivers","action"
    ]
    show = f[table_cols].sort_values(["risk_level","pred_cover_days"], ascending=[True, True])
    st.subheader("Detailed table (colour-coded risk)")
    st.dataframe(style_risk_table(show), use_container_width=True, height=520)

    if st.session_state.get("authenticated"):
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇ Download predictions (CSV)",
                data=f.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                key="pred_download"
            )
        with col2:
            critical_df = f[f["risk_level"] == "CRITICAL"]
            st.download_button(
                f"⬇ Download CRITICAL only ({len(critical_df):,} rows)",
                data=critical_df.to_csv(index=False).encode("utf-8"),
                file_name="predictions_critical.csv",
                mime="text/csv",
                key="pred_download_critical"
            )
        log("DOWNLOAD", f"tab=predictions rows={len(f)}")


# =========================
# MAIN
# =========================
def main():
    st.set_page_config(page_title="Backlog Dashboard", layout="wide")

    # ── Authentication gate — must pass before anything renders ──
    check_auth()

    # ── User info badge in sidebar ──
    with st.sidebar:
        st.markdown(
            f"**👤 {st.session_state.get('user_name', '')}**  \n"
            f"`{st.session_state.get('user_email', '')}`"
        )
        st.divider()

    st.markdown(
        """
        <style>
        [data-testid="stMetricLabel"] > div {
            white-space: nowrap !important; overflow: visible !important; text-overflow: clip !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Backlog Dashboard")

    try:
        pipe = load_pipeline(PIPELINE_PATH)
    except FileNotFoundError as e:
        st.error(str(e))
        return
    except ValueError as e:
        st.error(str(e))
        return
    except Exception:
        logging.exception("Model load error")
        st.error(f"Could not load model. Place `{PIPELINE_PATH}` in the same folder as this file.")
        return

    tab1, tab2 = st.tabs(["📊 Analysis (Raw)", "🔮 Predictions & Risk"])
    with tab1: render_analysis_tab()
    with tab2: render_prediction_tab(pipe)


if __name__ == "__main__":
    main()