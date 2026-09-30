"""
Pull daily exchange-rate series from the Bank of Canada Valet API and build
a small SQLite database for analysis.

Run:  python fetch_data.py
Out:  data/fx_daily.csv   (date, usdcad, eurusd, gbpusd, audusd, jpyusd)
      data/fx.sqlite      (table fx + view fx_returns, see sql/returns.sql)

Series are quoted by the Bank as CAD per foreign unit (e.g. FXUSDCAD = CAD per
1 USD). Control rates against the USD are built as cross rates:
    EUR/USD = FXEURCAD / FXUSDCAD, and so on.
"""
import sqlite3
from pathlib import Path

import pandas as pd
import requests

START = "2024-01-01"  # long pre-period so every estimation window is clean
SERIES = ["FXUSDCAD", "FXEURCAD", "FXGBPCAD", "FXAUDCAD", "FXJPYCAD"]
URL = "https://www.bankofcanada.ca/valet/observations/{}/json"

root = Path(__file__).parent
(root / "data").mkdir(exist_ok=True)

resp = requests.get(URL.format(",".join(SERIES)), params={"start_date": START}, timeout=30)
resp.raise_for_status()
obs = resp.json()["observations"]

rows = []
for o in obs:
    row = {"date": o["d"]}
    for s in SERIES:
        v = o.get(s, {}).get("v")
        row[s] = float(v) if v not in (None, "") else None
    rows.append(row)

raw = pd.DataFrame(rows).dropna().sort_values("date").reset_index(drop=True)
raw["date"] = pd.to_datetime(raw["date"])

out = pd.DataFrame({
    "date": raw["date"].dt.strftime("%Y-%m-%d"),
    "usdcad": raw["FXUSDCAD"],
    "eurusd": raw["FXEURCAD"] / raw["FXUSDCAD"],
    "gbpusd": raw["FXGBPCAD"] / raw["FXUSDCAD"],
    "audusd": raw["FXAUDCAD"] / raw["FXUSDCAD"],
    "jpyusd": raw["FXJPYCAD"] / raw["FXUSDCAD"],
})
out.to_csv(root / "data" / "fx_daily.csv", index=False)
print(f"Saved {len(out)} rows: {out['date'].iloc[0]} to {out['date'].iloc[-1]}")

# ---- SQLite layer: load the table, then create the returns view ----
con = sqlite3.connect(root / "data" / "fx.sqlite")
out.to_sql("fx", con, if_exists="replace", index=False)
con.executescript((root / "sql" / "returns.sql").read_text())
n = con.execute("SELECT COUNT(*) FROM fx_returns").fetchone()[0]
print(f"SQLite view fx_returns ready ({n} rows)")
con.close()
