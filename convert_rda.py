"""
convert_rda.py - turn the denguedatahub R data file into a normal CSV.

Usage (from the project folder):
    python convert_rda.py                          # uses data/srilanka_weekly_data.rda
    python convert_rda.py path/to/srilanka_weekly_data.rda
Output:
    data/sl_dengue_weekly.csv  (columns: year, week, start.date, end.date, district, cases)
"""
import sys
from pathlib import Path

import pyreadr

DATA_DIR = Path(__file__).resolve().parent / "data"

path = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_DIR / "srilanka_weekly_data.rda"
result = pyreadr.read_r(str(path))
print("Objects found in file:", list(result.keys()))

df = result.get("srilanka_weekly_data", next(iter(result.values())))
print("Shape:", df.shape)
print(df.head())

DATA_DIR.mkdir(exist_ok=True)
out = DATA_DIR / "sl_dengue_weekly.csv"
df.to_csv(out, index=False)
print("Saved ->", out)
