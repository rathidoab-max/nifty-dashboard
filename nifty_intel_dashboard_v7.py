# 📊 Unified NIFTY Trading Intelligence Dashboard – v7.1
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
from kiteconnect import KiteConnect
from scipy.stats import norm

# ---- SIMPLE PASSWORD PROTECTION ----
password = st.text_input("Enter access password", type="password")
if password != st.secrets.get("app_password", "mysecret"):
    st.warning("🔐 This app is private. Please enter the correct password.")
    st.stop()

# ---- CONFIG ----
st.set_page_config(page_title="NIFTY Intelligence Dashboard", layout="wide")
st.title("📈 NIFTY Trading Intelligence Dashboard – v7.1")
api_key = st.secrets["api_key"]
access_token = st.secrets["access_token"]
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# ---- LOGGER TOGGLE ----
enable_logger = st.sidebar.checkbox("🟢 Enable IV Logging")

# ---- Black-Scholes / Greeks ----
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
    return delta, gamma, theta, vega

def implied_volatility(price, S, K, T, r, option_type='call'):
    sigma = 0.2
    for _ in range(100):
        try:
            theoretical, d1, _ = black_scholes(S, K, T, r, sigma, option_type)
            vega = S * norm.pdf(d1) * np.sqrt(T)
            if vega < 1e-6:
                break
            diff = price - theoretical
            if abs(diff) < 1e-5:
                return sigma
            sigma += diff / vega
            if sigma <= 0 or sigma > 5:
                break
        except:
            break
    return max(sigma, 0.01)

# ---- DATA LOADING ----
nifty_spot = kite.ltp("NSE:NIFTY 50")["NSE:NIFTY 50"]["last_price"]
atm_strike = int(round(nifty_spot / 50) * 50)
instruments = kite.instruments("NFO")
df = pd.DataFrame(instruments)
df = df[(df['segment'] == 'NFO-OPT') & (df['tradingsymbol'].str.startswith('NIFTY')) & (df['expiry'].notnull())]

# ---- Logger (activated only if toggle ON) ----
if enable_logger:
    now = datetime.datetime.now()
    today = now.strftime("%Y-%m-%d")
    log_path = f"/mount/temp/iv_curve_log_{today}.csv"
    expiry = sorted(df['expiry'].unique())[0]
    r = 0.06
    T = (expiry - datetime.date.today()).days / 365
    log_rows = []
    for option_type in ['CE', 'PE']:
        subset = df[(df['expiry'] == expiry) & (df['instrument_type'] == option_type)]
        for _, row in subset.iterrows():
            try:
                ltp = kite.ltp(f"NFO:{row['tradingsymbol']}")[f"NFO:{row['tradingsymbol']}"]["last_price"]
                if not ltp or ltp <= 0: continue
                opt_type = 'call' if option_type == 'CE' else 'put'
                iv = implied_volatility(ltp, nifty_spot, row['strike'], T, r, opt_type)
                delta, gamma, theta, vega = greeks(nifty_spot, row['strike'], T, r, iv, opt_type)
                log_rows.append({
                    "time": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "spot": nifty_spot,
                    "atm": atm_strike,
                    "strike": row['strike'],
                    "type": opt_type,
                    "ltp": ltp,
                    "iv": round(iv * 100, 2),
                    "delta": delta,
                    "theta": theta,
                    "vega": vega
                })
            except: continue
    log_df = pd.DataFrame(log_rows)
    if not log_df.empty:
        if os.path.exists(log_path):
            log_df.to_csv(log_path, mode='a', header=False, index=False)
        else:
            log_df.to_csv(log_path, index=False)

# [Place rest of your existing Streamlit tab structure and UI rendering below here...]

