from kiteconnect import KiteConnect
import pandas as pd

api_key = "your_api_key"
access_token = "your_access_token"

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

print("📦 Fetching instruments...")
instruments = kite.instruments("NFO")
df = pd.DataFrame(instruments)

print("🔍 Unique names in NFO instrument list:")
print(df['name'].dropna().unique())