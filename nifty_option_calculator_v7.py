# 📊 Unified NIFTY Trading Intelligence Dashboard – v7
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
from kiteconnect import KiteConnect
from scipy.stats import norm

# ---- CONFIG ----
st.set_page_config(page_title="NIFTY Intelligence Dashboard", layout="wide")
st.title("📈 NIFTY Trading Intelligence Dashboard – v7")
api_key = "bwhraj28ii33624u"
access_token = "3KZ7HAYfKjA13lToH18kkPc7os4W2FiM"
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

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

# ---- Data ----
nifty_spot = kite.ltp("NSE:NIFTY 50")["NSE:NIFTY 50"]["last_price"]
atm_strike = int(round(nifty_spot / 50) * 50)
instruments = kite.instruments("NFO")
df = pd.DataFrame(instruments)
df = df[(df['segment'] == 'NFO-OPT') & (df['tradingsymbol'].str.startswith('NIFTY')) & (df['expiry'].notnull())]

# ---- UI Controls ----
expiry = st.sidebar.selectbox("Expiry", sorted(df['expiry'].unique()))
r = 0.06
T = (expiry - datetime.date.today()).days / 365

# ---- Tabs ----
tab1, tab2, tab3, tab4 = st.tabs(["🔴 Live Monitor", "📊 IV Insights", "🌡️ Heatmap", "📋 Daily Report"])

with tab1:
    st.header("🔴 CE/PE Snapshot + P&L Simulator")
    strike_ce = st.selectbox("Strike for CE", list(range(atm_strike - 500, atm_strike + 550, 50)), index=10)
    strike_pe = st.selectbox("Strike for PE", list(range(atm_strike - 500, atm_strike + 550, 50)), index=10)
    lots = st.slider("Lots", 1, 50, 1)
    vol = st.slider("Assumed Volatility (%)", 5.0, 60.0, 15.0) / 100

    def get_ltp(sym):
        try: return kite.ltp(f"NFO:{sym}")[f"NFO:{sym}"]["last_price"]
        except: return None

    def get_opt(exp, strike, typ):
        row = df[(df['expiry'] == exp) & (df['strike'] == strike) & (df['instrument_type'] == typ)]
        return row.iloc[0] if not row.empty else None

    ce_row, pe_row = get_opt(expiry, strike_ce, 'CE'), get_opt(expiry, strike_pe, 'PE')
    ce_ltp = get_ltp(ce_row['tradingsymbol']) if ce_row is not None else None
    pe_ltp = get_ltp(pe_row['tradingsymbol']) if pe_row is not None else None

    col1, col2 = st.columns(2)
    if ce_ltp:
        ce_iv = implied_volatility(ce_ltp, nifty_spot, strike_ce, T, r, 'call')
        d, g, t, v = greeks(nifty_spot, strike_ce, T, r, ce_iv, 'call')
        with col1:
            st.subheader("CALL")
            st.metric("LTP", round(ce_ltp, 2))
            st.metric("IV", round(ce_iv*100, 2))
            st.metric("Delta", round(d, 2))
    if pe_ltp:
        pe_iv = implied_volatility(pe_ltp, nifty_spot, strike_pe, T, r, 'put')
        d, g, t, v = greeks(nifty_spot, strike_pe, T, r, pe_iv, 'put')
        with col2:
            st.subheader("PUT")
            st.metric("LTP", round(pe_ltp, 2))
            st.metric("IV", round(pe_iv*100, 2))
            st.metric("Delta", round(d, 2))

    spot_input = st.slider("Simulated NIFTY Expiry", int(nifty_spot - 500), int(nifty_spot + 500), int(nifty_spot), step=50)
    lot_size = 50 * lots
    ce_pl = (ce_ltp - max(0, spot_input - strike_ce)) * lot_size if ce_ltp else 0
    pe_pl = (pe_ltp - max(0, strike_pe - spot_input)) * lot_size if pe_ltp else 0
    st.success(f"P&L: CALL ₹{round(ce_pl)} + PUT ₹{round(pe_pl)} = TOTAL ₹{round(ce_pl + pe_pl)}")

with tab2:
    st.header("📊 IV Bulge Detector")
    today = datetime.date.today()
    file = f"iv_curve_log_{today}.csv"
    if os.path.exists(file):
        df_iv = pd.read_csv(file)
        df_iv = df_iv.sort_values("time")
        latest = df_iv.groupby(['strike', 'type']).tail(1)
        latest['distance'] = latest['strike'] - latest['atm']
        bulges = []
        for t in ["call", "put"]:
            sub = latest[latest['type'] == t].sort_values("distance")
            for i in range(1, len(sub)-1):
                row = sub.iloc[i]
                iv = row['iv']
                med = np.median([sub.iloc[i-1]['iv'], sub.iloc[i+1]['iv']])
                if iv > 1.5 * med and row['delta'] < 0.3:
                    bulges.append(row)
        outliers = pd.DataFrame(bulges)
        if not outliers.empty:
            st.warning(f"{len(outliers)} bulges found")
            st.dataframe(outliers[['strike', 'type', 'iv', 'ltp', 'delta', 'distance']])

with tab3:
    st.header("🌡️ IV Heatmap")
    if os.path.exists(file):
        df_iv = pd.read_csv(file)
        df_iv['time'] = pd.to_datetime(df_iv['time'])
        df_iv['hour_min'] = df_iv['time'].dt.strftime("%H:%M")
        df_iv['bucket'] = (df_iv['strike'] - df_iv['atm']) // 50 * 50
        opt = st.radio("Type", ["call", "put"], horizontal=True)
        pivot = df_iv[df_iv['type'] == opt].pivot_table(index='bucket', columns='hour_min', values='iv', aggfunc='mean')
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(pivot, cmap="coolwarm", annot=False, fmt=".1f", linewidths=0.5, ax=ax)
        ax.set_title(f"{opt.upper()} IV Heatmap")
        st.pyplot(fig)

with tab4:
    st.header("📋 Daily Summary Report")
    if os.path.exists(file):
        df_iv = pd.read_csv(file)
        df_iv['time'] = pd.to_datetime(df_iv['time'])
        df_iv['hour'] = df_iv['time'].dt.hour
        df_iv['distance'] = df_iv['strike'] - df_iv['atm']

        summary = []
        open_spot = df_iv[df_iv['hour'] == df_iv['hour'].min()]['spot'].mean()
        close_spot = df_iv[df_iv['hour'] == df_iv['hour'].max()]['spot'].mean()
        spot_chg = round(((close_spot - open_spot) / open_spot) * 100, 2)
        summary.append(f"🔹 NIFTY moved {round(open_spot)} → {round(close_spot)} ({spot_chg}%)")

        iv_start = df_iv[df_iv['hour'] == df_iv['hour'].min()].groupby('type')['iv'].mean()
        iv_end = df_iv[df_iv['hour'] == df_iv['hour'].max()].groupby('type')['iv'].mean()
        for t in ['call', 'put']:
            chg = round(iv_end.get(t, 0) - iv_start.get(t, 0), 2)
            summary.append(f"{t.upper()} IV change: {chg}%")

        otm = df_iv[(df_iv['delta'] < 0.3) & (df_iv['iv'] > 25)]
        if not otm.empty:
            hot = otm['distance'].mode()[0]
            iv_avg = otm[otm['distance'] == hot]['iv'].mean()
            summary.append(f"⚠️ High IV seen at ±{hot} (avg {round(iv_avg, 1)}%)")

        decay = df_iv.groupby('strike')['theta'].mean().sort_values().head(1)
        best = decay.index[0]
        summary.append(f"📉 Best theta decay at {best}")

        for line in summary:
            st.write(line)
