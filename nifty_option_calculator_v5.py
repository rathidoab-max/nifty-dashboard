import streamlit as st
from streamlit_autorefresh import st_autorefresh
from kiteconnect import KiteConnect
import pandas as pd
import numpy as np
from scipy.stats import norm
import datetime
import os
import matplotlib.pyplot as plt

# Auto-refresh every 15 seconds
st_autorefresh(interval=15000, key="refresh")

# CONFIG
api_key = "bwhraj28ii33624u"
access_token = "3KZ7HAYfKjA13lToH18kkPc7os4W2FiM"
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# Black-Scholes and Greeks
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2), d1, d2
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1), d1, d2

def greeks(S, K, T, r, sigma, option_type='call'):
    _, d1, d2 = black_scholes(S, K, T, r, sigma, option_type)
    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    theta = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T))
    vega = S*norm.pdf(d1)*np.sqrt(T)
    rho = K*T*np.exp(-r*T)*norm.cdf(d2 if option_type == 'call' else -d2)
    return delta, gamma, theta, vega, rho

def implied_volatility(price, S, K, T, r, option_type='call'):
    sigma = 0.2
    for _ in range(100):
        theoretical, d1, _ = black_scholes(S, K, T, r, sigma, option_type)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        diff = price - theoretical
        if abs(diff) < 1e-5:
            return sigma
        sigma += diff / vega
    return sigma

# Get live spot
nifty_spot = kite.ltp("NSE:NIFTY 50")["NSE:NIFTY 50"]["last_price"]
atm_strike = int(round(nifty_spot / 50) * 50)

# Load live option chain
instruments = kite.instruments("NFO")
df = pd.DataFrame(instruments)
df['tradingsymbol'] = df['tradingsymbol'].astype(str).str.upper()
df['instrument_type'] = df['instrument_type'].astype(str).str.upper()
df['segment'] = df['segment'].astype(str).str.upper()
df = df[
    (df['tradingsymbol'].str.startswith("NIFTY")) &
    (df['segment'] == 'NFO-OPT') &
    (df['instrument_type'].isin(['CE', 'PE'])) &
    (df['expiry'].notnull())
]

# UI inputs
expiry_options = sorted(df['expiry'].dropna().unique())
expiry = st.selectbox("Select Expiry Date", expiry_options)
strikes = list(range(atm_strike - 500, atm_strike + 550, 50))
strike_ce = st.selectbox("Select Strike for CE", strikes, index=10)
strike_pe = st.selectbox("Select Strike for PE", strikes, index=10)
vol = st.slider("Assumed Volatility (%)", 5.0, 60.0, 15.0) / 100
num_lots = st.slider("Number of Lots", 1, 50, 1)
r = 0.06
T = (expiry - datetime.date.today()).days / 365

# Helpers
def get_option(df, expiry, strike, opt_type):
    return df[(df['expiry'] == expiry) & (df['strike'] == strike) & (df['instrument_type'] == opt_type)]
def fetch_ltp(ts):
    try:
        return kite.ltp(f"NFO:{ts}")[f"NFO:{ts}"]["last_price"]
    except:
        return None

def render_panel(label, ltp, strike, opt_type):
    if ltp is None:
        st.warning(f"{label} LTP not found.")
        return
    theo, d1, d2 = black_scholes(nifty_spot, strike, T, r, vol, opt_type)
    iv = implied_volatility(ltp, nifty_spot, strike, T, r, opt_type)
    delta, gamma, theta, vega, rho = greeks(nifty_spot, strike, T, r, vol, opt_type)
    intrinsic = max(0, nifty_spot - strike) if opt_type == 'call' else max(0, strike - nifty_spot)
    time_val = ltp - intrinsic

    if iv > 0.3:
        st.warning(f"⚠️ High IV Alert for {label}: {round(iv * 100, 2)}%")
    if abs(delta) > 0.7:
        st.warning(f"⚠️ High Delta Alert for {label}: {round(delta, 2)}")

    st.metric(f"{label} LTP", round(ltp, 2))
    st.metric("Theo Value", round(theo, 2))
    st.metric("Reverse IV", f"{round(iv * 100, 2)}%")
    st.metric("Intrinsic", round(intrinsic, 2))
    st.metric("Time Value", round(time_val, 2))
    col1, col2, col3 = st.columns(3)
    col1.metric("Delta", round(delta, 4))
    col2.metric("Gamma", round(gamma, 4))
    col3.metric("Theta", round(theta, 2))
    col4, col5 = st.columns(2)
    col4.metric("Vega", round(vega, 2))
    col5.metric("Rho", round(rho, 2))

    return {
        "ltp": ltp,
        "iv": iv,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho
    }

# CE + PE
st.subheader(f"NIFTY Spot: {nifty_spot}")
col1, col2 = st.columns(2)
ce_row = get_option(df, expiry, strike_ce, 'CE')
pe_row = get_option(df, expiry, strike_pe, 'PE')
ce_ltp = fetch_ltp(ce_row['tradingsymbol'].values[0]) if not ce_row.empty else None
pe_ltp = fetch_ltp(pe_row['tradingsymbol'].values[0]) if not pe_row.empty else None

with col1:
    st.markdown("### CALL Option")
    ce_data = render_panel("CALL", ce_ltp, strike_ce, 'call')
with col2:
    st.markdown("### PUT Option")
    pe_data = render_panel("PUT", pe_ltp, strike_pe, 'put')

# ---- P&L Graph with Zones ----
st.markdown("---")
st.markdown("### 📉 P&L Curve with Visual Zones")
lot_size = 50 * num_lots
strike_range = list(range(atm_strike - 1000, atm_strike + 1000 + 1, 50))
pl_data = []
for spot in strike_range:
    ce_pnl = (ce_ltp - max(0, spot - strike_ce)) * lot_size if ce_ltp else 0
    pe_pnl = (pe_ltp - max(0, strike_pe - spot)) * lot_size if pe_ltp else 0
    pl_data.append(ce_pnl + pe_pnl)

breakevens = [s for s, pl in zip(strike_range, pl_data) if abs(pl) < 5]  # near-zero P&L
max_pl = max(pl_data)
max_pl_spot = strike_range[pl_data.index(max_pl)]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(strike_range, pl_data, label="Net P&L", linewidth=2, color='blue')
ax.axhline(0, color='black', linestyle='--', linewidth=1)
ax.axvline(nifty_spot, color='red', linestyle='--', linewidth=1, label='Current Spot')
ax.axvline(max_pl_spot, color='green', linestyle=':', linewidth=1, label='Max Profit Spot')
for be in breakevens:
    ax.axvline(be, color='orange', linestyle=':', linewidth=0.8)
ax.fill_between(strike_range, pl_data, 0, where=np.array(pl_data) > 0, color='lightgreen', alpha=0.3)
ax.fill_between(strike_range, pl_data, 0, where=np.array(pl_data) < 0, color='salmon', alpha=0.3)
ax.set_xlabel("NIFTY Spot at Expiry")
ax.set_ylabel("Profit / Loss")
ax.set_title("P&L Range at Expiry")
ax.legend()
st.pyplot(fig)

# Remaining logging + summary code stays as-is...
