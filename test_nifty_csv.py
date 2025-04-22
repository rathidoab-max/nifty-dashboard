import pandas as pd

df = pd.read_csv("nfo_instruments.csv")
print(f"Total rows: {len(df)}")

# Filter NIFTY + OPTIDX
df = df[df['instrument_type'] == 'OPTIDX']
df = df[df['name'] == 'NIFTY']
df = df[df['expiry'].notnull()]

print(f"NIFTY OPTIDX rows with expiry: {len(df)}")
print("Unique expiry dates:")
print(sorted(df['expiry'].astype(str).unique()))