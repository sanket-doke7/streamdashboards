import os
import glob
from datetime import datetime, timezone
from typing import Optional, Union

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from bs_math import implied_vol, bs_greeks, bs_price

load_dotenv()

R = float(os.getenv("RISK_FREE_RATE", "0.065"))
Q = float(os.getenv("DIVIDEND_YIELD", "0.00"))
INDEX_SYMBOL = os.getenv("INDEX_SYMBOL", "NIFTY")
CSV_TZ = os.getenv("CSV_TZ", "Asia/Kolkata")

IN_DIR = os.path.join(os.getcwd(), "input_tickdata")
OUT_DIR = os.path.join(os.getcwd(), "processed")
os.makedirs(OUT_DIR, exist_ok=True)

def parse_ts(val: Union[str, int, float, pd.Timestamp]) -> pd.Timestamp:
    """Parse varied timestamp formats (ns integers, ISO strings) into tz-aware values."""
    if pd.isna(val):
        return pd.NaT

    target_tz = CSV_TZ or "Asia/Kolkata"

    if isinstance(val, pd.Timestamp):
        ts = val
        if ts.tzinfo is None:
            try:
                ts = ts.tz_localize(target_tz)
            except (TypeError, ValueError):
                return pd.NaT
        else:
            ts = ts.tz_convert(target_tz)
        return ts

    ts = pd.NaT
    try:
        if isinstance(val, (np.integer, int)):
            ts = pd.to_datetime(int(val), errors="coerce", utc=True, unit="ns")
        else:
            s = str(val).strip()
            if not s or s.lower() in {"nan", "nat"}:
                return pd.NaT
            if s.lstrip("-").isdigit():
                ts = pd.to_datetime(int(s), errors="coerce", utc=True, unit="ns")
            else:
                ts = pd.to_datetime(s, errors="coerce", utc=True)
                if ts is pd.NaT or pd.isna(ts):
                    ts = pd.to_datetime(s, errors="coerce")
                    if ts is pd.NaT or pd.isna(ts):
                        return pd.NaT
                    if ts.tzinfo is None:
                        ts = ts.tz_localize(target_tz)
                    else:
                        ts = ts.tz_convert(target_tz)
                    return ts
    except Exception:
        return pd.NaT

    if ts is pd.NaT or pd.isna(ts):
        return pd.NaT

    return ts.tz_convert(target_tz)

def time_to_expiry_yrs(now_ts: pd.Timestamp, expiry: pd.Timestamp) -> float:
    # year fraction ACT/365
    dt = (expiry - now_ts).total_seconds()
    return max(dt, 0.0) / (365.0 * 24 * 3600)

def load_index_spot(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    # 1) pick the symbol column present
    sym_col = None
    for cand in ["tradingsymbol", "symbol", "underlying", "instrument", "name"]:
        if cand in df.columns:
            sym_col = cand
            break
    if sym_col is None:
        return None

    # 2) build a case-insensitive, trimmed match set
    #    If INDEX_SYMBOL contains "NIFTY", also accept plain "NIFTY"
    target = INDEX_SYMBOL.strip().upper()
    targets = {target}
    if "NIFTY" in target:
        targets.add("NIFTY")
        targets.add("NIFTY50")
        targets.add("NIFTY 50")

    sym_up = (
        df[sym_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)  # collapse multiple spaces
    )

    idx_rows = df[sym_up.isin(targets)].copy()
    if idx_rows.empty:
        # Light debug to help if it still fails
        # print("DEBUG: unique symbols (top 10):", sym_up.value_counts().head(10).to_dict())
        return None

    # 3) choose spot column
    spot_col = None
    for c in ["last_price", "ltp", "spot", "close", "price"]:
        if c in idx_rows.columns:
            spot_col = c
            break
    if spot_col is None:
        return None

    # 4) choose timestamp column
    ts_col = None
    for c in ["time", "timestamp", "date_time", "ts"]:
        if c in idx_rows.columns:
            ts_col = c
            break
    if ts_col is None:
        return None

    # 5) build output (ts, spot)
    idx_rows["ts"] = idx_rows[ts_col].map(parse_ts)
    idx_rows["spot"] = pd.to_numeric(idx_rows[spot_col], errors="coerce")
    out = (
        idx_rows[["ts", "spot"]]
        .dropna()
        .drop_duplicates(subset=["ts"])
        .sort_values("ts")
    )
    return out if not out.empty else None

def maybe_load_external_spot() -> Optional[pd.DataFrame]:
    ext = os.path.join(IN_DIR, "nifty_spot.csv")
    if not os.path.exists(ext):
        return None
    df = pd.read_csv(ext)
    # Must contain ts,spot
    if not set(["ts","spot"]).issubset(df.columns):
        return None
    df["ts"] = df["ts"].map(parse_ts)
    df["spot"] = pd.to_numeric(df["spot"], errors="coerce")
    return df.dropna().sort_values("ts")[["ts","spot"]]

def normalize_option_frame(df: pd.DataFrame, expiry_hint: Optional[pd.Timestamp]) -> pd.DataFrame:
    # Column mapping/guessing
    colmap = {}
    def first_match(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    ts_col = first_match(["time","timestamp","date_time","ts"])
    bid_col = first_match(["best_bid","bid","buy_price"])
    ask_col = first_match(["best_ask","ask","sell_price"])
    ltp_col = first_match(["ltp","last_price","trade_price","price"])
    oi_col  = first_match(["oi","open_interest"])
    sym_col = first_match(["symbol","tradingsymbol","instrument","name"])
    typ_col = first_match(["option_type","type","opt_type","cp","call_put"])
    strike_col = first_match(["strike","strike_price","k"])
    exp_col = first_match(["expiry","expiration","expiry_date","exp"])

    need = [ts_col, ltp_col, sym_col, typ_col, strike_col]
    if any(x is None for x in need):
        raise ValueError("Missing required columns (need ts, ltp, symbol, option_type, strike).")

    out = pd.DataFrame()
    out["ts"] = df[ts_col].map(parse_ts)
    out["symbol"] = df[sym_col].astype(str)
    typ = df[typ_col].astype(str).str.upper().str.replace("CALL","CE").str.replace("PUT","PE")
    typ = typ.replace({"C":"CE","P":"PE"})
    out["option_type"] = typ
    out["strike"] = pd.to_numeric(df[strike_col], errors="coerce")
    out["oi"] = pd.to_numeric(df[oi_col], errors="coerce") if oi_col else np.nan
    bid = pd.to_numeric(df[bid_col], errors="coerce") if bid_col else np.nan
    ask = pd.to_numeric(df[ask_col], errors="coerce") if ask_col else np.nan
    ltp = pd.to_numeric(df[ltp_col], errors="coerce")
    out["mid"] = np.where(np.isfinite(bid) & np.isfinite(ask), (bid + ask) / 2.0, ltp)
    out["ltp"] = ltp

    if exp_col:
        expiry = pd.to_datetime(df[exp_col].astype(str), errors="coerce")
        # Localize if naive
        if expiry.dt.tz is None:
            expiry = expiry.dt.tz_localize("Asia/Kolkata")
        out["expiry"] = expiry
    else:
        out["expiry"] = expiry_hint  # can be None; later we’ll drop rows without expiry

    return out.dropna(subset=["ts","option_type","strike","mid"])

def attach_spot(df_opt: pd.DataFrame, spot_df: pd.DataFrame) -> pd.DataFrame:
    # Merge-asof on timestamp (forward/backward tolerance small)
    df_opt = df_opt.sort_values("ts")
    spot_df = spot_df.sort_values("ts")
    merged = pd.merge_asof(df_opt, spot_df, on="ts", direction="nearest", tolerance=pd.Timedelta("1min"))
    return merged

def compute_iv_greeks(df: pd.DataFrame) -> pd.DataFrame:
    # Requires columns: ts, spot, mid, ltp, strike, option_type, expiry
    df = df.copy()
    df = df.dropna(subset=["expiry","spot","mid","strike"])
    df["T"] = (df["expiry"] - df["ts"]).dt.total_seconds().clip(lower=0) / (365.0*24*3600)
    df["opt_price"] = np.where(np.isfinite(df["mid"]), df["mid"], df["ltp"])

    ivs = []
    deltas, gammas, vegas, thetas, rhos = [], [], [], [], []
    for S, K, T, typ, p in zip(df["spot"], df["strike"], df["T"], df["option_type"], df["opt_price"]):
        if not (np.isfinite(S) and np.isfinite(K) and np.isfinite(T) and T > 0 and np.isfinite(p) and p > 0):
            ivs.append(np.nan); deltas.append(np.nan); gammas.append(np.nan); vegas.append(np.nan); thetas.append(np.nan); rhos.append(np.nan)
            continue
        iv = implied_vol(price=float(p), S=float(S), K=float(K), r=R, q=Q, T=float(T), typ=typ)
        ivs.append(iv)
        greeks = bs_greeks(S=float(S), K=float(K), r=R, q=Q, sigma=iv if np.isfinite(iv) else np.nan, T=float(T), typ=typ)
        deltas.append(greeks["delta"]); gammas.append(greeks["gamma"]); vegas.append(greeks["vega"]); thetas.append(greeks["theta"]); rhos.append(greeks["rho"])
    df["iv"] = ivs
    df["delta"] = deltas
    df["gamma"] = gammas
    df["vega"]  = vegas
    df["theta"] = thetas
    df["rho"]   = rhos
    return df

def main():
    files = sorted(glob.glob(os.path.join(IN_DIR, "*.csv")))
    if not files:
        print(f"No CSV files found in {IN_DIR}")
        return

    combined = []
    external_spot = maybe_load_external_spot()

    for f in files:
        try:
            raw = pd.read_csv(f)
        except Exception as e:
            print(f"[skip] {f}: {e}")
            continue

        # Guess a single expiry in the file if present
        expiry_hint = None
        for c in ["expiry","expiration","expiry_date","exp"]:
            if c in raw.columns:
                expiry_series = pd.to_datetime(raw[c], errors="coerce")
                expiry_hint = expiry_series.dropna().iloc[0] if not expiry_series.dropna().empty else None
                if expiry_hint is not None and expiry_hint.tz is None:
                    expiry_hint = expiry_hint.tz_localize("Asia/Kolkata")
                break

        try:
            opt = normalize_option_frame(raw, expiry_hint)
        except Exception as e:
            print(f"[skip] {f}: {e}")
            continue

        spot_df = load_index_spot(raw)
        if spot_df is None:
            spot_df = external_spot
        if spot_df is None:
            print(f"[warn] No index spot available for {os.path.basename(f)}; skipping IV/Greeks.")
            continue

        merged = attach_spot(opt, spot_df)
        out = compute_iv_greeks(merged)

        out_name = os.path.splitext(os.path.basename(f))[0] + "_greeks.csv"
        out_path = os.path.join(OUT_DIR, out_name)
        out.to_csv(out_path, index=False)
        print(f"[ok] wrote {out_path}  ({len(out)} rows)")
        combined.append(out)

    if combined:
        big = pd.concat(combined, ignore_index=True)
        big.to_parquet(os.path.join(OUT_DIR, "all_options_greeks.parquet"))
        print(f"[ok] wrote combined parquet with {len(big)} rows")

if __name__ == "__main__":
    main()
