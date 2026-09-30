-- Daily percentage changes and a 20-day rolling volatility, using window functions.
DROP VIEW IF EXISTS fx_returns;
CREATE VIEW fx_returns AS
WITH r AS (
  SELECT
    date,
    usdcad,
    eurusd,
    (usdcad / LAG(usdcad) OVER (ORDER BY date) - 1) * 100 AS usdcad_ret,
    (eurusd / LAG(eurusd) OVER (ORDER BY date) - 1) * 100 AS eurusd_ret
  FROM fx
)
SELECT
  date, usdcad, eurusd, usdcad_ret, eurusd_ret,
  AVG(usdcad_ret * usdcad_ret) OVER (
    ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
  ) AS usdcad_var_20d
FROM r
WHERE usdcad_ret IS NOT NULL;
