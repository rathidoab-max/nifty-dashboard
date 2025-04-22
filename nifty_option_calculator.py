import streamlit as st
from streamlit_autorefresh import st_autorefresh
from kiteconnect import KiteConnect
import numpy as np
from scipy.stats import norm
import datetime

# Auto-refresh every 15 seconds
st_autorefresh(interval=15000, key="refresh")

# --- Kite API Setup ---
api_key = "bwhraj28ii33624u"
api_secret = "2926p2qcpjcb61aectwo9z2m24bb6x8y"
access_token = "3KZ7HAYfKjA13lToH18kkPc7os4W2FiM"

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# --- Black-Scholes Formula ---
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price, d1, d2

# --- Greeks ---
def greeks(S, K, T, r, sigma, option_type='call'):
    _, d1, d2 = black_scholes(S, K, T, r, sigma, option_type)
    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)
    return delta, gamma, theta, vega, rho

# --- Reverse IV Calculation ---
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

# --- Streamlit UI ---
st.title("📈 NIFTY Option True Value + Greeks")

strike = st.number_input("Strike Price", value=22500)
option_type = st.radio("Option Type", ["call", "put"])
expiry_date = st.date_input("Expiry Date", value=datetime.date.today() + datetime.timedelta(days=7))
vol = st.slider("Volatility (%)", min_value=5.0, max_value=60.0, value=15.0) / 100
r = 0.06  # Fixed risk-free rate

# Fetch live NIFTY price
try:
    ltp_data = kite.ltp("NSE:NIFTY 50")
    spot_price = ltp_data["NSE:NIFTY 50"]["last_price"]
except Exception as e:
    st.error(f"Error fetching live price: {e}")
    st.stop()

T = (expiry_date - datetime.date.today()).days / 365.0

# Compute values
theoretical, d1, d2 = black_scholes(spot_price, strike, T, r, vol, option_type)
delta, gamma, theta, vega, rho = greeks(spot_price, strike, T, r, vol, option_type)
intrinsic = max(0, spot_price - strike) if option_type == 'call' else max(0, strike - spot_price)
time_value = theoretical - intrinsic
iv_reverse = implied_volatility(theoretical, spot_price, strike, T, r, option_type)

# --- Output ---
st.metric("📊 Spot Price", round(spot_price, 2))
st.metric("💰 Theoretical Value", round(theoretical, 2))
st.metric("🔍 Implied Volatility (Reverse)", f"{round(iv_reverse * 100, 2)}%")

col1, col2, col3 = st.columns(3)
col1.metric("Delta", round(delta, 4))
col2.metric("Gamma", round(gamma, 4))
col3.metric("Theta", round(theta, 2))

col4, col5, col6 = st.columns(3)
col4.metric("Vega", round(vega, 2))
col5.metric("Rho", round(rho, 2))
col6.metric("Intrinsic", round(intrinsic, 2))

st.metric("⏱ Time Value", round(time_value, 2))