# app.py — FINAL (multi-file loader + 3 dual-axis panels)
import os
import re
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Stream Dashboards", layout="wide")

# ========= Config =========
# Load ALL CSVs from a folder (default: processed/) OR a single file if DASH_DATA is set
DATA_DIR = os.getenv("DATA_DIR", "processed")           # folder of CSVs
DATA_PATH = (os.getenv("DASH_DATA") or "").strip() or None  # optional single-file CSV override

FILE_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")  # yyyy-mm-dd in filename

REQUIRED_NUMERIC_COLS = [
    "oi", "mid", "ltp", "spot", "iv", "strike",
    "delta", "gamma", "vega", "theta", "rho", "opt_price"
]

# ========= Utilities =========
def _coerce_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Coerce required columns to numeric; create if missing."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    return df

def _date_from_filename(path: str) -> str | None:
    """Extract yyyy-mm-dd from filename if present."""
    m = FILE_DATE_RE.search(os.path.basename(path))
    return m.group(1) if m else None

def _read_one_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # ts → datetime
    if "ts" not in df.columns:
        raise ValueError(f"{os.path.basename(path)} missing 'ts' column")
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])

    # option_type normalize
    if "option_type" in df.columns:
        df["option_type"] = (
            df["option_type"].astype(str).str.upper().replace({"CALL": "CE", "PUT": "PE"})
        )
    else:
        df["option_type"] = "CE"

    # expiry as string
    df["expiry"] = df.get("expiry", "NA").astype(str)

    # ensure numeric cols exist
    df = _coerce_numeric(df, REQUIRED_NUMERIC_COLS)

    # date columns
    df["date"] = df["ts"].dt.date.astype(str)
    file_date = _date_from_filename(path)
    if file_date:
        df["file_date"] = file_date
        # if for some reason date is NA (unlikely), fill from filename
        df.loc[df["date"].isna(), "date"] = file_date
    else:
        df["file_date"] = np.nan

    # symbol fallback
    if "symbol" not in df.columns:
        df["symbol"] = "NIFTY"

    return df

@st.cache_data
def load_all_data(data_dir: str, single_path: str | None) -> pd.DataFrame:
    """Read either one CSV (DASH_DATA) or all CSVs in DATA_DIR/*.csv."""
    if single_path:
        if not os.path.exists(single_path):
            raise FileNotFoundError(f"No such file: {single_path}")
        out = _read_one_csv(single_path)
        return out.sort_values("ts")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"DATA_DIR '{data_dir}' not found")
    paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not paths:
        cols = ["ts","symbol","option_type","strike","oi","mid","ltp","expiry","spot","T",
                "opt_price","iv","delta","gamma","vega","theta","rho","date","file_date"]
        return pd.DataFrame(columns=cols)

    frames = []
    for p in paths:
        try:
            frames.append(_read_one_csv(p))
        except Exception as e:
            st.warning(f"[skip] {os.path.basename(p)}: {e}")
    if not frames:
        cols = ["ts","symbol","option_type","strike","oi","mid","ltp","expiry","spot","T",
                "opt_price","iv","delta","gamma","vega","theta","rho","date","file_date"]
        return pd.DataFrame(columns=cols)

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["ts"]).sort_values("ts")
    return out

def apply_filters(d: pd.DataFrame, dates, expiries, strikes, typ: str) -> pd.DataFrame:
    g = d.copy()
    if dates:
        g = g[g["date"].isin(dates)]
    if expiries:
        g = g[g["expiry"].isin(expiries)]
    if strikes:
        g = g[g["strike"].isin(strikes)]
    if typ != "Both":
        g = g[g["option_type"] == typ]
    return g.sort_values("ts")

def resample_df(d: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Downsample per (symbol, strike, option_type, expiry).
       Last for prices/greeks, sum for OI."""
    if rule == "tick" or d.empty:
        return d
    keep_last = ["spot","ltp","mid","iv","delta","gamma","vega","theta","rho","opt_price"]
    sum_cols  = ["oi"]
    grp = ["symbol","strike","option_type","expiry"]

    out = []
    for keys, g in d.groupby(grp, dropna=False):
        g = g.set_index("ts").sort_index()
        last_part = g[keep_last].resample(rule).last()
        sum_part  = g[sum_cols].resample(rule).sum()
        merged = last_part.join(sum_part, how="outer")
        for i, k in enumerate(grp):
            merged[k] = keys[i]
        merged = merged.reset_index()
        out.append(merged)

    o = pd.concat(out, ignore_index=True) if out else d
    if o.empty:
        return o
    o["date"] = o["ts"].dt.date.astype(str)
    return o.dropna(subset=["ts"]).sort_values("ts")

def add_dual_traces(fig, x, y1, name1, y2, name2, y1_title, y2_title):
    fig.add_trace(go.Scatter(x=x, y=y1, name=name1, mode="lines", yaxis="y1"))
    fig.add_trace(go.Scatter(x=x, y=y2, name=name2, mode="lines", yaxis="y2"))
    fig.update_layout(
        yaxis=dict(title=y1_title),
        yaxis2=dict(title=y2_title, overlaying="y", side="right"),
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=10, b=10),
        hovermode="x unified",
        height=330,
    )
    # x-axis ticks every 15 minutes; show spikelines
    fig.update_xaxes(dtick=15*60*1000, showspikes=True, spikemode="across")
    fig.update_yaxes(showspikes=True, spikemode="across")
    return fig

# ========= Load data =========
df = load_all_data(DATA_DIR, DATA_PATH)

if df.empty:
    st.error("No data found. Add CSVs to processed/ (e.g., ticks_YYYY-MM-DD_greeks.csv) or set DASH_DATA.")
    st.stop()

st.title("Stream Dashboards — Final")
st.caption(
    f"Loaded {df['date'].nunique()} day(s) · rows={len(df)} · "
    f"source={'DASH_DATA:'+os.path.abspath(DATA_PATH) if DATA_PATH else 'DATA_DIR:'+os.path.abspath(DATA_DIR)}"
)

# ========= Layout =========
left, right = st.columns([1, 3], gap="large")

with left:
    st.markdown("## Inputs")

    # Date selector — default to ALL available dates
    dates = sorted(df["date"].dropna().unique().tolist())
    date_sel = st.multiselect("Date", options=dates, default=dates)

    # Expiry selector — limit to selections in date range
    df_base = df[df["date"].isin(date_sel)] if date_sel else df.copy()
    expiries = sorted(df_base["expiry"].dropna().unique().tolist())
    expiry_sel = st.multiselect("Expiry", options=expiries, default=expiries)

    # Strike selector — after date+expiry filters
    df_base2 = df_base[df_base["expiry"].isin(expiry_sel)] if expiry_sel else df_base
    strikes = sorted([s for s in df_base2["strike"].dropna().unique().tolist()])
    strike_sel = st.multiselect("Strike", options=strikes, default=strikes)

    typ = st.radio("Type", options=["CE", "PE", "Both"], index=2, horizontal=True)
    ds = st.selectbox("Downsample", options=["tick", "1min", "5min", "15min"], index=1)

# Apply filters + resample
df_f = apply_filters(df, date_sel, expiry_sel, strike_sel, typ)
df_plot = resample_df(df_f, ds)

# Friendly empty state post-filter
if df_plot.empty:
    st.warning("No data after filters. Try widening Date/Expiry/Strike/Type.")
    st.stop()

with right:
    # 1) OI + Option price (dual axis)
    st.markdown("### OI + Option price")
    fig1 = go.Figure()
    for (k_strike, k_type), g in df_plot.groupby(["strike", "option_type"], dropna=False):
        price = g["mid"].where(g["mid"].notna(), g["opt_price"]).where(lambda s: s.notna(), g["ltp"])
        fig1.add_trace(go.Scatter(x=g["ts"], y=g["oi"],
                                  name=f"{int(k_strike) if pd.notna(k_strike) else 'NA'} {k_type} OI",
                                  mode="lines", yaxis="y1"))
        fig1.add_trace(go.Scatter(x=g["ts"], y=price,
                                  name=f"{int(k_strike) if pd.notna(k_strike) else 'NA'} {k_type} price",
                                  mode="lines", yaxis="y2"))
    fig1.update_layout(
        yaxis=dict(title="OI"),
        yaxis2=dict(title="Option price", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=10, b=10),
        hovermode="x unified",
        height=330,
    )
    fig1.update_xaxes(dtick=15*60*1000)
    st.plotly_chart(fig1, use_container_width=True)

    # 2) IV + Option price (dual axis)
    st.markdown("### IV + Option price")
    fig2 = go.Figure()
    for (k_strike, k_type), g in df_plot.groupby(["strike", "option_type"], dropna=False):
        price = g["mid"].where(g["mid"].notna(), g["opt_price"]).where(lambda s: s.notna(), g["ltp"])
        fig2.add_trace(go.Scatter(x=g["ts"], y=g["iv"],
                                  name=f"{int(k_strike) if pd.notna(k_strike) else 'NA'} {k_type} IV",
                                  mode="lines", yaxis="y1"))
        fig2.add_trace(go.Scatter(x=g["ts"], y=price,
                                  name=f"{int(k_strike) if pd.notna(k_strike) else 'NA'} {k_type} price",
                                  mode="lines", yaxis="y2"))
    fig2.update_layout(
        yaxis=dict(title="IV"),
        yaxis2=dict(title="Option price", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=10, b=10),
        hovermode="x unified",
        height=330,
    )
    fig2.update_xaxes(dtick=15*60*1000)
    st.plotly_chart(fig2, use_container_width=True)

    # 3) OI + Spot (dual axis)
    st.markdown("### OI + Spot")
    fig3 = go.Figure()
    # average spot across filtered series at each timestamp for a clean single spot line
    spot_series = df_plot.groupby("ts", as_index=False)["spot"].mean()
    for (k_strike, k_type), g in df_plot.groupby(["strike", "option_type"], dropna=False):
        merged = g[["ts", "oi"]].merge(spot_series, on="ts", how="left")
        fig3.add_trace(go.Scatter(x=merged["ts"], y=merged["oi"],
                                  name=f"{int(k_strike) if pd.notna(k_strike) else 'NA'} {k_type} OI",
                                  mode="lines", yaxis="y1"))
    fig3.add_trace(go.Scatter(x=spot_series["ts"], y=spot_series["spot"],
                              name="Spot (NIFTY)", mode="lines", yaxis="y2"))
    fig3.update_layout(
        yaxis=dict(title="OI"),
        yaxis2=dict(title="Spot", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=10, b=10),
        hovermode="x unified",
        height=330,
    )
    fig3.update_xaxes(dtick=15*60*1000)
    st.plotly_chart(fig3, use_container_width=True)

    # Table + download
    st.markdown("### Data (filtered)")
    st.dataframe(df_plot.sort_values("ts"))
    csv = df_plot.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", csv, file_name="filtered.csv", mime="text/csv")
