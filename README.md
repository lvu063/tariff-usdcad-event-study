# Tariff Announcements and the Canadian Dollar: An Event Study

Live dashboard: https://tariff-dashboard.lovable.app

## Question
How did USD/CAD respond to U.S. tariff announcements on Canada (2024-2026), after
accounting for broad U.S. dollar moves?

## Headline results
<!-- Fill in AFTER rerunning the analysis. One table, 3-5 rows, one line of interpretation. -->
| Event | Date | Abnormal return, day 0 | CAR [0,+5] | p-value |
|-------|------|------------------------|------------|---------|
| ...   | ...  | ...                    | ...        | ...     |

## Data
- Bank of Canada Valet API, daily rates (FXUSDCAD, FXEURCAD, FXGBPCAD, FXAUDCAD, FXJPYCAD)
- Pulled by `fetch_data.py` on <date>. Re-run it to refresh.
- Event dates and sources: see `docs/events.md` (each date linked to a primary source)

## Method
1. Daily log returns of USD/CAD (levels are non-stationary; ADF test in `docs/diagnostics.md`)
2. Market-model benchmark: USD/CAD returns regressed on USD cross-rate returns
   (EUR, GBP, AUD, JPY) over a pre-event estimation window
3. Abnormal and cumulative abnormal returns around each event, Newey-West HAC inference
4. Robustness: alternative windows, alternative control currencies, placebo dates
5. Forecast benchmark: ARIMA vs random walk vs drift, rolling-origin evaluation, Diebold-Mariano test

## Reproduce
```
pip install -r requirements.txt     # Python
Rscript -e "renv::restore()"        # R
python fetch_data.py
Rscript event_study.R
python forecast_model.py
```

## Repo layout
```
data/        fx_daily.csv, fx.sqlite
sql/         returns.sql (window-function returns + rolling volatility)
output/      figures and result tables
docs/        events.md, diagnostics.md, report.pdf
```

## Limitations
<!-- Keep this honest and specific: confounding events, small samples, what a control cannot fix. -->

## Author
<Your name> | <LinkedIn> | Not affiliated with the Bank of Canada or Statistics Canada.
