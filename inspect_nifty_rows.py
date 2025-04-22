import pandas as pd

df = pd.read_csv("nfo_instruments.csv")
print("Total rows:", len(df))

print("\nUnique values in instrument_type:", df['instrument_type'].unique())
print("Unique values in segment:", df['segment'].unique())
print("Unique names:", df['name'].unique())

# Try looser filter
nifty_df = df[
    (df['name'].astype(str).str.contains('NIFTY', case=False, na=False)) &
    (df['instrument_type'].astype(str).str.upper() == 'OPTIDX')
]

print(f"\nNIFTY OPTIDX rows: {len(nifty_df)}")
print(nifty_df[['tradingsymbol', 'expiry', 'strike', 'option_type']].head())