from kiteconnect import KiteConnect
import pandas as pd

api_key = "your_api_key"
access_token = "your_access_token"

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

print("🔄 Fetching all NFO instruments...")
all_instruments = kite.instruments("NFO")
df = pd.DataFrame(all_instruments)

print(f"Total instruments: {len(df)}")

# ✅ Clean filter for NIFTY options
nifty_df = df[
    (df['instrument_type'] == 'OPTIDX') &
    (df['name'].str.strip().str.upper() == 'NIFTY') &
    (df['segment'] == 'NFO-OPT') &
    (df['expiry'].notnull())
]

print(f"NIFTY options found: {len(nifty_df)}")
print("Saving to nfo_instruments.csv")

nifty_df.to_csv("nfo_instruments.csv", index=False)
print("✅ Done!")