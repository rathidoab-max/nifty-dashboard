import streamlit as st
from streamlit_autorefresh import st_autorefresh
from kiteconnect import KiteConnect
import pandas as pd
import numpy as np
from scipy.stats import norm
import datetime

# Auto-refresh every 15 seconds
st_autorefresh(interval=15000, key="refresh")

# ---- CONFIG ----
api_key = "bwhraj28ii33624u"
access_token = "3KZ7HAYfKjA13lToH18kkPc7os4W2FiM"

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# ---- Black-Scholes and Greeks ----
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

# ---- Get Live NIFTY Spot ----
try:
    nifty_spot = kite.ltp("NSE:NIFTY 50")["NSE:NIFTY 50"]["last_price"]
except Exception as e:
    st.error(f"❌ Failed to fetch NIFTY price: {e}")
    st.stop()

atm_strike = int(round(nifty_spot / 50) * 50)

# ---- Fetch Live Option Chain ----
try:
    instruments = kite.instruments("NFO")
except Exception as e:
    st.error(f"❌ Failed to fetch instruments: {e}")
    st.stop()

df = pd.DataFrame(instruments)

# ---- Filter NIFTY Options ----
df['tradingsymbol'] = df['tradingsymbol'].astype(str).str.upper()
df['instrument_type'] = df['instrument_type'].astype(str).str.upper()
df['segment'] = df['segment'].astype(str).str.upper()

df = df[
    (df['tradingsymbol'].str.startswith("NIFTY")) &
    (df['segment'] == 'NFO-OPT') &
    (df['instrument_type'].isin(['CE', 'PE'])) &
    (df['expiry'].notnull())
]

if df.empty:
    st.error("❌ No NIFTY options found.")
    st.dataframe(pd.DataFrame(instruments).head())
    st.stop()

# ---- UI Controls ----
expiry_options = sorted(df['expiry'].dropna().unique())
selected_expiry = st.selectbox("Select Expiry Date", expiry_options)

strikes = list(range(atm_strike - 500, atm_strike + 550, 50))
selected_strike = st.selectbox("Select Strike", strikes)

vol = st.slider("Assumed Volatility (%)", 5.0, 60.0, 15.0) / 100
r = 0.06
T = (selected_expiry - datetime.date.today()).days / 365
# ---- Get CE and PE Rows ----
def get_option(df, expiry, strike, opt_type):
    return df[
        (df['expiry'] == expiry) &
        (df['strike'] == strike) &
        (df['instrument_type'] == opt_type)
    ]

ce_row = get_option(df, selected_expiry, selected_strike, 'CE')
pe_row = get_option(df, selected_expiry, selected_strike, 'PE')

def fetch_ltp(tradingsymbol):
    try:
        return kite.ltp(f"NFO:{tradingsymbol}")[f"NFO:{tradingsymbol}"]["last_price"]
    except:
        return None

ce_ltp = fetch_ltp(ce_row['tradingsymbol'].values[0]) if not ce_row.empty else None
pe_ltp = fetch_ltp(pe_row['tradingsymbol'].values[0]) if not pe_row.empty else None

# ---- Output ----
st.subheader(f"NIFTY Spot: {nifty_spot}")
st.markdown(f"**ATM Strike:** {atm_strike}")

def render_option(label, ltp, opt_type):
    if ltp is None:
        st.warning(f"{label}: LTP not found.")
        return
    theo, d1, d2 = black_scholes(nifty_spot, selected_strike, T, r, vol, opt_type)
    iv = implied_volatility(ltp, nifty_spot, selected_strike, T, r, opt_type)
    delta, gamma, theta, vega, rho = greeks(nifty_spot, selected_strike, T, r, vol, opt_type)
    intrinsic = max(0, nifty_spot - selected_strike) if opt_type == 'call' else max(0, selected_strike - nifty_spot)
    time_val = ltp - intrinsic

    st.metric(f"🔹 {label} LTP", round(ltp, 2))
    st.metric("📐 Theo Value", round(theo, 2))
    st.metric("🧠 Reverse IV", f"{round(iv * 100, 2)}%")
    st.metric("Intrinsic Value", round(intrinsic, 2))
    st.metric("Time Value", round(time_val, 2))

    col1, col2, col3 = st.columns(3)
    col1.metric("Delta", round(delta, 4))
    col2.metric("Gamma", round(gamma, 4))
    col3.metric("Theta", round(theta, 2))

    col4, col5 = st.columns(2)
    col4.metric("Vega", round(vega, 2))
    col5.metric("Rho", round(rho, 2))

col1, col2 = st.columns(2)
with col1:
    render_option("CALL", ce_ltp, 'call')
with col2:
    render_option("PUT", pe_ltp, 'put')