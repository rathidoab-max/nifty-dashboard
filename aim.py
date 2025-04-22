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

# ---- IV Intelligence Chart ----
st.markdown("---")
st.markdown("### 🧠 IV Pattern Detector (vs Strike Distance)")
option_chain = df[(df['expiry'] == expiry)].copy()

iv_data = []
for _, row in option_chain.iterrows():
    strike = row['strike']
    option_type = 'call' if row['instrument_type'] == 'CE' else 'put'
    try:
        ltp = kite.ltp(f"NFO:{row['tradingsymbol']}")[f"NFO:{row['tradingsymbol']}"]["last_price"]
        iv = implied_volatility(ltp, nifty_spot, strike, T, r, option_type)
        distance = strike - atm_strike
        iv_data.append({"distance": distance, "iv": iv * 100, "type": option_type})
    except:
        continue

iv_df = pd.DataFrame(iv_data)
fig, ax = plt.subplots(figsize=(8, 4))
for label, grp in iv_df.groupby("type"):
    ax.plot(grp["distance"], grp["iv"], marker='o', label=label.upper())
ax.axvline(0, color='gray', linestyle='--', linewidth=1)
ax.set_xlabel("Strike Distance from ATM")
ax.set_ylabel("Implied Volatility (%)")
ax.set_title("IV vs Strike Distance")
ax.legend()
st.pyplot(fig)

# (rest of the app code continues...)
