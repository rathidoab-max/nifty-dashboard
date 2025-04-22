import streamlit as st
from streamlit_autorefresh import st_autorefresh
from kiteconnect import KiteConnect
import pandas as pd
import numpy as np
from scipy.stats import norm
import datetime
import os

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

# ---- P&L Panel ----
st.markdown("---")
st.markdown("### 💰 P&L Simulator (for Sellers)")
lot_size = 50
spot_min = int(nifty_spot - 500)
spot_max = int(nifty_spot + 500)
spot_default = int(nifty_spot)
spot_input = st.slider("NIFTY at Expiry", spot_min, spot_max, spot_default, step=50)

ce_intrinsic = max(0, spot_input - strike_ce)
pe_intrinsic = max(0, strike_pe - spot_input)
ce_pl = (ce_ltp - ce_intrinsic) * lot_size
pe_pl = (pe_ltp - pe_intrinsic) * lot_size

col3, col4 = st.columns(2)
col3.metric("CALL P&L", f"₹{round(ce_pl, 2)}")
col4.metric("PUT P&L", f"₹{round(pe_pl, 2)}")

# ---- Logger ----
st.markdown("---")
st.markdown("### 📊 Auto Logger (to CSV)")
log_enabled = st.toggle("Enable Logging", value=False)
log_path = "option_log.csv"

if log_enabled and ce_data and pe_data:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "timestamp": now,
        "nifty": nifty_spot,
        "ce_strike": strike_ce,
        "ce_ltp": ce_data['ltp'],
        "ce_iv": ce_data['iv'],
        "pe_strike": strike_pe,
        "pe_ltp": pe_data['ltp'],
        "pe_iv": pe_data['iv']
    }
    df_log = pd.DataFrame([row])
    if not os.path.exists(log_path):
        df_log.to_csv(log_path, index=False)
    else:
        df_log.to_csv(log_path, mode='a', header=False, index=False)
    st.success("Logged!")
