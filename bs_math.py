import math
from math import exp, log, sqrt
from typing import Literal, Tuple

import numpy as np

OptionType = Literal["CE", "PE"]

SQRT_TWO_PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_TWO_PI


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1_d2(S: float, K: float, r: float, q: float, sigma: float, T: float) -> Tuple[float, float]:
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return float("nan"), float("nan")
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_price(S: float, K: float, r: float, q: float, sigma: float, T: float, typ: OptionType) -> float:
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    if np.isnan(d1) or np.isnan(d2):
        return float("nan")
    if typ == "CE":
        return S * exp(-q * T) * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)
    else:
        return K * exp(-r * T) * _norm_cdf(-d2) - S * exp(-q * T) * _norm_cdf(-d1)


def bs_greeks(S: float, K: float, r: float, q: float, sigma: float, T: float, typ: OptionType):
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    if np.isnan(d1) or np.isnan(d2):
        return {k: float("nan") for k in ["delta", "gamma", "vega", "theta", "rho"]}

    pdf = _norm_pdf(d1)
    sqrtT = sqrt(T)
    disc_r = exp(-r * T)
    disc_q = exp(-q * T)

    if typ == "CE":
        delta = disc_q * _norm_cdf(d1)
        theta = (
            -(S * pdf * sigma * disc_q) / (2 * sqrtT)
            - r * K * disc_r * _norm_cdf(d2)
            + q * S * disc_q * _norm_cdf(d1)
        )
        rho = K * T * disc_r * _norm_cdf(d2)
    else:
        delta = -disc_q * _norm_cdf(-d1)
        theta = (
            -(S * pdf * sigma * disc_q) / (2 * sqrtT)
            + r * K * disc_r * _norm_cdf(-d2)
            - q * S * disc_q * _norm_cdf(-d1)
        )
        rho = -K * T * disc_r * _norm_cdf(-d2)

    gamma = (disc_q * pdf) / (S * sigma * sqrtT)
    vega = S * disc_q * pdf * sqrtT
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }


def implied_vol(price: float, S: float, K: float, r: float, q: float, T: float, typ: OptionType, floor: float = 1e-4, cap: float = 5.0) -> float:
    # Guardrails
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return float("nan")

    intrinsic = max(0.0, (S * exp(-q * T) - K * exp(-r * T)) if typ == "CE" else (K * exp(-r * T) - S * exp(-q * T)))
    if price < intrinsic - 1e-8:
        return float("nan")

    def price_minus_market(sig: float) -> float:
        return bs_price(S, K, r, q, sig, T, typ) - price

    low = floor
    high = max(cap, floor * 2.0)
    f_low = price_minus_market(low)
    f_high = price_minus_market(high)

    attempts = 0
    while attempts < 20 and not (np.isnan(f_low) or np.isnan(f_high)) and f_low * f_high > 0:
        high *= 2.0
        if high > 100.0:
            break
        f_high = price_minus_market(high)
        attempts += 1

    if np.isnan(f_low) or np.isnan(f_high) or f_low * f_high > 0:
        return float("nan")

    for _ in range(100):
        mid = 0.5 * (low + high)
        f_mid = price_minus_market(mid)
        if np.isnan(f_mid):
            return float("nan")
        if abs(f_mid) < 1e-6 or (high - low) < 1e-6:
            return float(mid)
        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    return float(mid) if abs(f_mid) < 1e-4 else float("nan")
