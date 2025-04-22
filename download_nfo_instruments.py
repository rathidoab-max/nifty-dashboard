from kiteconnect import KiteConnect
import pandas as pd

# 🔐 Replace with your values
api_key = "bwhraj28ii33624u"
access_token = "2926p2qcpjcb61aectwo9z2m24bb6x8y"

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

print("🔄 Downloading full NFO instrument list...")
instruments = kite.instruments("NFO")

df = pd.DataFrame(instruments)

# ✅ Filter only NIFTY options with expiry
df = df[
    (df['name'] == 'NIFTY') &
    (df['instrument_type'] == 'OPTIDX') &
    (df['expiry'].notnull())
]

df.to_csv("nfo_instruments.csv", index=False)
print("✅ Clean NIFTY option file saved to nfo_instruments.csv")