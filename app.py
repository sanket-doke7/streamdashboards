# app.py (BUILD 2)
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Stream Dashboards", layout="wide")
st.title("Stream Dashboards — BUILD 2")

DATA_PATH = os.getenv("DASH_DATA", "processed/ticks_2025-10-28_greeks.csv")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])
    df["option_type"] = df["option_type"].astype(str).str.upper().replace({"CALL":"CE","PUT":"PE"})
    df["date"] = df["ts"].dt.date.astype(str)
    df["expiry"] = df["expiry"].astype(str)
    for c in ["oi","mid","ltp","spot","iv","strike","delta","gamma","vega","theta","rho","opt_price"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df = load_data(DATA_PATH)

# Small diagnostic so you can confirm we’re reading the CSV:
st.caption(f"Loaded: {DATA_PATH} | rows={len(df)} | cols={list(df.columns)[:8]}...")

left, right = st.columns([1,3], gap="large")

with left:
    st.markdown("## Inputs")

    dates = sorted(df["date"].dropna().unique().tolist())
    date_sel = st.multiselect("Date", dates, default=dates[:1] if dates else [])

    expiries = sorted(df["expiry"].dropna().unique().tolist())
    expiry_sel = st.multiselect("Expiry", expiries, default=expiries[:1] if expiries else [])

    df_base = df.copy()
    if date_sel: df_base = df_base[df_base["date"].isin(date_sel)]
    if expiry_sel: df_base = df_base[df_base["expiry"].isin(expiry_sel)]

    strikes = sorted(df_base["strike"].dropna().unique().tolist())
    strike_sel = st.multiselect("Strike", strikes, default=strikes[:1] if strikes else [])

    typ = st.radio("Type", ["CE","PE","Both"], index=2, horizontal=True)
    ds = st.selectbox("Downsample", ["tick","1min","5min","15min"], index=1)

def apply_filters(d):
    g = d.copy()
    if date_sel: g = g[g["date"].isin(date_sel)]
    if expiry_sel: g = g[g["expiry"].isin(expiry_sel)]
    if strike_sel: g = g[g["strike"].isin(strike_sel)]
    if typ != "Both": g = g[g["option_type"] == typ]
    return g.sort_values("ts")

def resample_df(d, rule):
    if rule == "tick": return d
    keep_last = ["spot","ltp","mid","iv","delta","gamma","vega","theta","rho","opt_price"]
    sum_cols  = ["oi"]
    grp = ["symbol","strike","option_type","expiry"]
    out = []
    for keys, g in d.groupby(grp, dropna=False):
        g = g.set_index("ts").sort_index()
        last_part = g[keep_last].resample(rule).last().fillna(method="ffill", limit=1)
        sum_part  = g[sum_cols].resample(rule).sum()
        merged = last_part.join(sum_part, how="outer")
        for i,k in enumerate(grp): merged[k] = keys[i]
        merged = merged.reset_index()
        out.append(merged)
    if not out: return d
    o = pd.concat(out, ignore_index=True)
    o["date"] = o["ts"].dt.date.astype(str)
    return o.dropna(subset=["ts"]).sort_values("ts")

def dual(fig, x, y1, n1, y2, n2, y1_title, y2_title):
    fig.add_trace(go.Scatter(x=x, y=y1, name=n1, mode="lines", yaxis="y1"))
    fig.add_trace(go.Scatter(x=x, y=y2, name=n2, mode="lines", yaxis="y2"))
    fig.update_layout(
        yaxis=dict(title=y1_title),
        yaxis2=dict(title=y2_title, overlaying="y", side="right"),
        legend=dict(orientation="h"),
        margin=dict(l=20,r=20,t=10,b=10),
        hovermode="x unified",
        height=330,
    )
    fig.update_xaxes(dtick=15*60*1000, showspikes=True, spikemode="across")
    fig.update_yaxes(showspikes=True, spikemode="across")
    return fig

dff = resample_df(apply_filters(df), ds)

with right:
    # 1) OI + Option price
    st.markdown("### OI + Option price")
    price_col = dff["mid"].where(dff["mid"].notna(), dff.get("opt_price", dff["ltp"]))
    fig1 = go.Figure()
    # combine multiple strikes/types into a single chart with separate traces
    for (k_strike, k_type), g in dff.groupby(["strike","option_type"], dropna=False):
        price = g["mid"].where(g["mid"].notna(), g.get("opt_price", g["ltp"]))
        fig1.add_trace(go.Scatter(x=g["ts"], y=g["oi"], name=f"{int(k_strike) if pd.notna(k_strike) else 'NA'} {k_type} OI", mode="lines", yaxis="y1"))
        fig1.add_trace(go.Scatter(x=g["ts"], y=price, name=f"{int(k_strike) if pd.notna(k_strike) else 'NA'} {k_type} price", mode="lines", yaxis="y2"))
    fig1.update_layout(yaxis=dict(title="OI"), yaxis2=dict(title="Option price", overlaying="y", side="right"),
                       legend=dict(orientation="h"), margin=dict(l=20,r=20,t=10,b=10),
                       hovermode="x unified", height=330)
    fig1.update_xaxes(dtick=15*60*1000)
    st.plotly_chart(fig1, use_container_width=True)

    # 2) IV + Option price
    st.markdown("### IV + Option price")
    fig2 = go.Figure()
    for (k_strike, k_type), g in dff.groupby(["strike","option_type"], dropna=False):
        price = g["mid"].where(g["mid"].notna(), g.get("opt_price", g["ltp"]))
        fig2.add_trace(go.Scatter(x=g["ts"], y=g["iv"], name=f"{int(k_strike) if pd.notna(k_strike) else 'NA'} {k_type} IV", mode="lines", yaxis="y1"))
        fig2.add_trace(go.Scatter(x=g["ts"], y=price, name=f"{int(k_strike) if pd.notna(k_strike) else 'NA'} {k_type} price", mode="lines", yaxis="y2"))
    fig2.update_layout(yaxis=dict(title="IV"), yaxis2=dict(title="Option price", overlaying="y", side="right"),
                       legend=dict(orientation="h"), margin=dict(l=20,r=20,t=10,b=10),
                       hovermode="x unified", height=330)
    fig2.update_xaxes(dtick=15*60*1000)
    st.plotly_chart(fig2, use_container_width=True)

    # 3) OI + Spot
    st.markdown("### OI + Spot")
    fig3 = go.Figure()
    spot_series = dff.groupby("ts", as_index=False)["spot"].mean()
    for (k_strike, k_type), g in dff.groupby(["strike","option_type"], dropna=False):
        merged = g[["ts","oi"]].merge(spot_series, on="ts", how="left")
        fig3.add_trace(go.Scatter(x=merged["ts"], y=merged["oi"], name=f"{int(k_strike) if pd.notna(k_strike) else 'NA'} {k_type} OI", mode="lines", yaxis="y1"))
    fig3.add_trace(go.Scatter(x=spot_series["ts"], y=spot_series["spot"], name="Spot (NIFTY)", mode="lines", yaxis="y2"))
    fig3.update_layout(yaxis=dict(title="OI"), yaxis2=dict(title="Spot", overlaying="y", side="right"),
                       legend=dict(orientation="h"), margin=dict(l=20,r=20,t=10,b=10),
                       hovermode="x unified", height=330)
    fig3.update_xaxes(dtick=15*60*1000)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Data (filtered)")
    st.dataframe(dff.sort_values("ts"))
