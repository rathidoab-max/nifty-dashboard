# 📊 NIFTY Trading Intelligence Engine – v6 (Unified)
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

# Core Models
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

# 🎯 LIVE NIFTY STATUS
nifty_spot = kite.ltp("NSE:NIFTY 50")["NSE:NIFTY 50"]["last_price"]
atm_strike = int(round(nifty_spot / 50) * 50)
st.title("📈 NIFTY Options Intelligence – v6")
st.info(f"Live NIFTY Spot: {nifty_spot} | ATM: {atm_strike}")

# 📦 Load NFO Option Chain
instruments = kite.instruments("NFO")
df = pd.DataFrame(instruments)
df = df[(df['segment'] == 'NFO-OPT') & (df['tradingsymbol'].str.startswith('NIFTY')) & (df['expiry'].notnull())]

# 📌 USER SELECTORS
expiry = st.selectbox("Select Expiry", sorted(df['expiry'].unique()))
strikes = list(range(atm_strike - 500, atm_strike + 550, 50))
strike_ce = st.selectbox("Strike for CALL", strikes, index=10)
strike_pe = st.selectbox("Strike for PUT", strikes, index=10)
vol = st.slider("Assumed Volatility (%)", 5.0, 60.0, 15.0) / 100
lots = st.slider("Number of Lots", 1, 50, 1)
r = 0.06
T = (expiry - datetime.date.today()).days / 365

# ⛓️ Strike Selectors + LTP/Greeks
def get_option(df, expiry, strike, opt_type):
    return df[(df['expiry'] == expiry) & (df['strike'] == strike) & (df['instrument_type'] == opt_type)]

def fetch_ltp(ts):
    return kite.ltp(f"NFO:{ts}")[f"NFO:{ts}"]["last_price"]

st.markdown("## 🔍 CE / PE Snapshot")
ce_row = get_option(df, expiry, strike_ce, 'CE')
pe_row = get_option(df, expiry, strike_pe, 'PE')
ce_ltp = fetch_ltp(ce_row['tradingsymbol'].values[0]) if not ce_row.empty else None
pe_ltp = fetch_ltp(pe_row['tradingsymbol'].values[0]) if not pe_row.empty else None

col1, col2 = st.columns(2)
with col1:
    st.markdown("### CALL")
    if ce_ltp:
        ce_iv = implied_volatility(ce_ltp, nifty_spot, strike_ce, T, r, 'call')
        ce_delta, _, ce_theta, ce_vega, _ = greeks(nifty_spot, strike_ce, T, r, ce_iv, 'call')
        st.metric("LTP", round(ce_ltp, 2))
        st.metric("IV", round(ce_iv * 100, 2))
        st.metric("Delta", round(ce_delta, 2))
        st.metric("Theta", round(ce_theta, 2))
        st.metric("Vega", round(ce_vega, 2))
with col2:
    st.markdown("### PUT")
    if pe_ltp:
        pe_iv = implied_volatility(pe_ltp, nifty_spot, strike_pe, T, r, 'put')
        pe_delta, _, pe_theta, pe_vega, _ = greeks(nifty_spot, strike_pe, T, r, pe_iv, 'put')
        st.metric("LTP", round(pe_ltp, 2))
        st.metric("IV", round(pe_iv * 100, 2))
        st.metric("Delta", round(pe_delta, 2))
        st.metric("Theta", round(pe_theta, 2))
        st.metric("Vega", round(pe_vega, 2))

# 📉 P&L SIMULATOR
st.markdown("## 💰 P&L Simulator")
spot_input = st.slider("NIFTY at Expiry", nifty_spot - 500, nifty_spot + 500, nifty_spot, step=50)
lot_size = 50 * lots
ce_intrinsic = max(0, spot_input - strike_ce)
pe_intrinsic = max(0, strike_pe - spot_input)
ce_pl = (ce_ltp - ce_intrinsic) * lot_size if ce_ltp else 0
pe_pl = (pe_ltp - pe_intrinsic) * lot_size if pe_ltp else 0
st.success(f"CALL P&L: ₹{round(ce_pl)} | PUT P&L: ₹{round(pe_pl)} | TOTAL: ₹{round(ce_pl + pe_pl)}")

# 📈 P&L Range Curve
strike_range = list(range(atm_strike - 1000, atm_strike + 1000 + 1, 50))
pl_data = []
for spot in strike_range:
    ce_pnl = (ce_ltp - max(0, spot - strike_ce)) * lot_size if ce_ltp else 0
    pe_pnl = (pe_ltp - max(0, strike_pe - spot)) * lot_size if pe_ltp else 0
    pl_data.append(ce_pnl + pe_pnl)

fig1, ax1 = plt.subplots(figsize=(8, 3))
breakevens = [s for s, pl in zip(strike_range, pl_data) if abs(pl) < 5]
max_pl = max(pl_data)
max_spot = strike_range[pl_data.index(max_pl)]
ax1.plot(strike_range, pl_data, label="Net P&L", linewidth=2)
ax1.axhline(0, color='gray', linestyle='--')
ax1.axvline(nifty_spot, color='red', linestyle='--', label='Spot')
ax1.axvline(max_spot, color='green', linestyle=':')
for be in breakevens:
    ax1.axvline(be, color='orange', linestyle=':', linewidth=0.8)
ax1.fill_between(strike_range, pl_data, 0, where=np.array(pl_data) > 0, color='lightgreen', alpha=0.3)
ax1.fill_between(strike_range, pl_data, 0, where=np.array(pl_data) < 0, color='salmon', alpha=0.3)
ax1.set_title("P&L vs NIFTY Spot")
ax1.legend()
st.pyplot(fig1)

# 🧠 IV SHAPE TRACKER + LOGGER
st.markdown("## 🧠 IV Curve Logger")
log_toggle = st.toggle("Enable IV Logging", value=True)
option_chain = df[(df['expiry'] == expiry)].copy()
iv_data = []
for _, row in option_chain.iterrows():
    try:
        strike = row['strike']
        tsym = row['tradingsymbol']
        ltp = kite.ltp(f"NFO:{tsym}")[f"NFO:{tsym}"]["last_price"]
        opt_type = 'call' if row['instrument_type'] == 'CE' else 'put'
        iv = implied_volatility(ltp, nifty_spot, strike, T, r, opt_type)
        delta, gamma, theta, vega, _ = greeks(nifty_spot, strike, T, r, iv, opt_type)
        iv_data.append({"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "strike": strike, "type": opt_type, "ltp": ltp, "iv": iv * 100, "delta": delta, "theta": theta, "vega": vega, "gamma": gamma, "atm": atm_strike, "spot": nifty_spot})
    except: continue

iv_df = pd.DataFrame(iv_data)
fig2, ax2 = plt.subplots(figsize=(8, 3))
for label, grp in iv_df.groupby("type"):
    ax2.plot(grp["strike"] - atm_strike, grp["iv"], marker='o', label=label.upper())
ax2.axvline(0, color='gray', linestyle='--')
ax2.set_title("IV vs Strike Distance")
ax2.set_xlabel("Strike Distance from ATM")
ax2.set_ylabel("IV (%)")
ax2.legend()
st.pyplot(fig2)

# Log to CSV
if log_toggle:
    path = f"iv_curve_log_{datetime.date.today()}.csv"
    if not os.path.exists(path):
        iv_df.to_csv(path, index=False)
    else:
        iv_df.to_csv(path, mode='a', header=False, index=False)
    st.success(f"Logged {len(iv_df)} rows to {path}")
