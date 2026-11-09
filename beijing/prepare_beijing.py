"""Turn one Beijing station into the LTSF-Linear 'custom' format (date + 11 channels, target PM2.5 last).
Usage: python beijing/prepare_beijing.py [station]      (default Dongsi)
Gaps are forward-filled (last known value); any leading gaps at the very start are back-filled."""
import os, sys, glob
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LTSF = os.environ.get("LTSF_DIR", os.path.join(REPO, "LTSF-Linear"))
station = sys.argv[1] if len(sys.argv) > 1 else "Dongsi"
f = glob.glob(os.path.join(REPO, "beijing", "raw", "PRSA_Data_20130301-20170228", f"PRSA_Data_{station}_*.csv"))[0]
df = pd.read_csv(f)
df["date"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
print("station", station, "| rows", len(df), "|", df["date"].min(), "->", df["date"].max())
full = pd.date_range(df["date"].min(), df["date"].max(), freq="h")
print("regular hourly grid:", len(full) == len(df) and bool((df["date"].values == full.values).all()))
num = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "PM2.5"]
print("missing % before fill:"); print((df[num].isna().mean() * 100).round(2))
out = df[["date"] + num].copy()
out[num] = out[num].ffill().bfill()
out = out.rename(columns={"PM2.5": "OT"})
print("missing after fill:", int(out.isna().sum().sum()))
dest = os.path.join(LTSF, "dataset", f"beijing_{station}.csv")
out.to_csv(dest, index=False)
print("saved", dest, out.shape)
