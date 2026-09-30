"""
Out-of-sample forecast evaluation for USD/CAD.

Question: does any model beat a random walk at 1, 5 and 10 trading days ahead?

Design
  - Rolling-origin (expanding-window) evaluation: at every origin the models are
    re-fit on data up to that day and forecast h = 1..10 days ahead.
  - Models: random walk (benchmark), random walk with drift, ARIMA(1,1,1).
  - Loss: squared error and absolute error, in percent log-price points.
  - Inference: Diebold-Mariano test vs the random walk, with a Bartlett/HAC
    variance using h-1 lags (forecast errors overlap when h > 1) and the
    Harvey-Leybourne-Newbold small-sample correction (t-distribution).

Usage:  python forecast_model.py [path/to/fx_daily.csv]
Needs:  pandas, numpy, scipy, statsmodels, matplotlib
Output: output/forecast_eval.csv, output/latest_forecast.csv, output/usdcad_forecast.png
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# ---- Settings ----
MIN_TRAIN = 120           # first origin uses this many observations
HORIZONS = (1, 5, 10)     # evaluated horizons (trading days)
H_MAX = max(HORIZONS)
ARIMA_ORDER = (1, 1, 1)

csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/fx_daily.csv")
Path("output").mkdir(exist_ok=True)

df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
y = 100 * np.log(df["usdcad"].to_numpy())          # log price x 100 (percent points)
dates = df["date"]
N = len(y)
assert N >= MIN_TRAIN + H_MAX + 20, "Not enough data for this evaluation design"
print(f"Data: {dates.iloc[0].date()} to {dates.iloc[-1].date()} | {N} observations")


# ---- Forecasters: each returns an array of H_MAX point forecasts ----
def f_naive(train):
    return np.full(H_MAX, train[-1])


def f_drift(train):
    drift = np.mean(np.diff(train))                 # expanding-window mean change
    return train[-1] + drift * np.arange(1, H_MAX + 1)


def f_arima(train):
    fit = ARIMA(train, order=ARIMA_ORDER).fit()
    return np.asarray(fit.forecast(steps=H_MAX))


MODELS = {"Random walk": f_naive, "RW with drift": f_drift,
          f"ARIMA{ARIMA_ORDER}": f_arima}

# ---- Rolling-origin loop ----
# errors[model][h] = list of (actual - forecast) for every origin where the target exists
errors = {m: {h: [] for h in HORIZONS} for m in MODELS}
origins = range(MIN_TRAIN, N - 1)                    # train = y[:t], targets y[t + h - 1]
for t in origins:
    train = y[:t]
    for name, fn in MODELS.items():
        try:
            fc = fn(train)
        except Exception:
            fc = f_naive(train)                      # rare ARIMA failure: fall back
        for h in HORIZONS:
            if t + h - 1 < N:
                errors[name][h].append(y[t + h - 1] - fc[h - 1])

# Errors for a model and the benchmark are appended in the same origin order, so the
# arrays are paired for the DM test at each horizon.


def dm_test(e_bench, e_model, h, power=2):
    """Diebold-Mariano with HLN correction. Positive stat => model beats benchmark."""
    e_bench, e_model = np.asarray(e_bench), np.asarray(e_model)
    d = np.abs(e_bench) ** power - np.abs(e_model) ** power
    T = len(d)
    dbar = d.mean()
    lrv = np.mean((d - dbar) ** 2)                   # gamma_0
    for k in range(1, h):                            # Bartlett weights, h-1 lags
        cov = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        lrv += 2 * (1 - k / h) * cov
    lrv = max(lrv, 1e-18)
    dm = dbar / np.sqrt(lrv / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_c = dm * hln
    p = 2 * (1 - stats.t.cdf(abs(dm_c), df=T - 1))
    return dm_c, p


bench = "Random walk"
rows = []
for name in MODELS:
    for h in HORIZONS:
        e = np.asarray(errors[name][h])
        eb = np.asarray(errors[bench][h])
        rmse, mae = np.sqrt(np.mean(e ** 2)), np.mean(np.abs(e))
        rmse_b, mae_b = np.sqrt(np.mean(eb ** 2)), np.mean(np.abs(eb))
        if name == bench:
            dm_stat = p_val = np.nan
        else:
            dm_stat, p_val = dm_test(eb, e, h)
        rows.append({"model": name, "horizon_days": h, "n_forecasts": len(e),
                     "RMSE_pct": rmse, "MAE_pct": mae,
                     "RMSE_vs_RW": rmse / rmse_b, "MAE_vs_RW": mae / mae_b,
                     "DM_stat": dm_stat, "DM_p_value": p_val})

res = pd.DataFrame(rows)
res.to_csv("output/forecast_eval.csv", index=False)
print("\n=== Rolling-origin results (losses in percent log points; ratios < 1 beat the random walk) ===")
print(res.round(4).to_string(index=False))

sig = res[(res["model"] != bench) & (res["DM_p_value"] < 0.05) & (res["DM_stat"] > 0)]
print("\nModels that significantly beat the random walk (5% level):",
      "none" if sig.empty else sig[["model", "horizon_days"]].to_dict("records"))

# ---- Latest 10-day forecast from the full sample (for display only) ----
fit_full = ARIMA(y, order=ARIMA_ORDER).fit()
fc = fit_full.get_forecast(steps=H_MAX)
mean = np.exp(np.asarray(fc.predicted_mean) / 100)
ci = np.exp(np.asarray(fc.conf_int(alpha=0.10)) / 100)
fdates = pd.bdate_range(dates.iloc[-1], periods=H_MAX + 1)[1:]     # ignores holidays
latest = pd.DataFrame({"date": fdates, "forecast": mean, "lo90": ci[:, 0], "hi90": ci[:, 1]})
latest.to_csv("output/latest_forecast.csv", index=False)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(dates.iloc[-60:], df["usdcad"].iloc[-60:], color="#12315C", lw=1.4, label="Observed USD/CAD")
ax.plot(latest["date"], latest["forecast"], color="#B22222", lw=2, label=f"ARIMA{ARIMA_ORDER} forecast")
ax.fill_between(latest["date"], latest["lo90"], latest["hi90"], color="#B22222", alpha=0.15, label="90% interval")
ax.set_title("USD/CAD: last 60 observations and 10-day forecast (display only)")
ax.set_xlabel("Date"); ax.set_ylabel("USD/CAD"); ax.legend()
fig.tight_layout(); fig.savefig("output/usdcad_forecast.png", dpi=120)
print("\nSaved output/forecast_eval.csv, output/latest_forecast.csv, output/usdcad_forecast.png")
