# =============================================================================
# Event study: USD/CAD response to U.S.-Canada tariff announcements
#
# Design (replaces the single-dummy levels regression):
#   1. Work in daily LOG RETURNS (levels of FX are non-stationary).
#   2. Market-model benchmark: USD/CAD returns explained by USD cross-rate
#      returns (EUR, GBP, AUD, JPY per USD) over a pre-event estimation window.
#      This nets out broad U.S. dollar moves, the main confounder.
#   3. Abnormal return (AR) and cumulative abnormal return (CAR) per event.
#   4. Inference: (a) classic t-stats, (b) placebo test on non-event days,
#      (c) full-sample regression with event dummies and Newey-West HAC SEs.
#   5. Newey-West is implemented by hand AND cross-checked against the
#      sandwich package.
#   6. Sensitivity to estimation-window length and event-window length.
#
# Units: returns are in PERCENT. Positive AR = USD/CAD rose = CAD weakened.
#
# Needs:  install.packages(c("sandwich", "lmtest"))   ("tseries" optional, for ADF)
# Input:  data/fx_daily.csv  (from fetch_data.py)
# Output: output/*.csv, output/event_windows.png, output/run_info.txt
# =============================================================================

suppressPackageStartupMessages({
  library(sandwich)
  library(lmtest)
})
set.seed(2026)
dir.create("output", showWarnings = FALSE)

# ---- Settings ---------------------------------------------------------------
EST_LEN <- 100   # trading days in the estimation window
GAP     <- 10    # trading days between estimation window and event day
H       <- 5     # post-event window: day 0 to day +H
PRE     <- 5     # pre-event days shown in the plots
CTRL    <- c("eurusd", "gbpusd", "audusd", "jpyusd")
FORM    <- as.formula(paste("usdcad ~", paste(CTRL, collapse = " + ")))

# ---- Events -----------------------------------------------------------------
# Day 0 = first trading day ON OR AFTER `date`. For announcements made after
# the close or on a weekend, enter the date the market could first react.
# !! VERIFY every date against a primary source and record the link in docs/events.md
events <- data.frame(
  label = c("Nov 2024: 25% tariff threat",
            "Feb 2025: tariff order signed",
            "Feb 2025: 30-day pause announced",
            "Mar 2025: tariffs take effect",
            "Mar 2025: USMCA-goods exemption",
            "Aug 2026: escalation to 50%",
            "Sep 2026: Canadian countermeasures"),
  date  = as.Date(c("2024-11-26",   # threat made evening of Nov 25
                    "2025-02-01",   # Saturday -> maps to Feb 3
                    "2025-02-04",   # check: pause was announced Feb 3 evening
                    "2025-03-04",
                    "2025-03-06",
                    "2026-08-22",   # Saturday -> maps to Aug 24 (from your report)
                    "2026-09-08")),
  stringsAsFactors = FALSE
)

# Bank of Canada decision dates, used only to FLAG overlapping event windows.
# !! Incomplete: add the rest from the Bank's published schedule.
boc_dates <- as.Date(c("2024-10-23", "2024-12-11", "2025-01-29", "2025-03-12",
                       "2025-04-16", "2025-06-04", "2025-07-30", "2025-09-17",
                       "2025-10-29", "2026-03-18", "2026-06-10", "2026-07-15",
                       "2026-09-02"))

# ---- Data -------------------------------------------------------------------
df <- read.csv("data/fx_daily.csv", stringsAsFactors = FALSE)
df$date <- as.Date(df$date)
df <- df[order(df$date), ]
stopifnot(!anyNA(df), all(c("usdcad", CTRL) %in% names(df)))

ret <- data.frame(date = df$date[-1])
for (v in c("usdcad", CTRL)) ret[[v]] <- 100 * diff(log(df[[v]]))
n <- nrow(ret)
cat(sprintf("Data: %s to %s | %d daily returns\n", min(df$date), max(df$date), n))

# ---- Stationarity check (optional) ------------------------------------------
if (requireNamespace("tseries", quietly = TRUE)) {
  adf <- do.call(rbind, lapply(c("usdcad", CTRL), function(v) {
    data.frame(series = v,
               p_value_log_level = suppressWarnings(tseries::adf.test(log(df[[v]]))$p.value),
               p_value_return    = suppressWarnings(tseries::adf.test(ret[[v]])$p.value))
  }))
  write.csv(adf, "output/adf_tests.csv", row.names = FALSE)
  print(adf)
} else {
  message("Package 'tseries' not installed: skipping ADF tests.")
}

# ---- Core: market model, AR and CAR for one event day -----------------------
run_event <- function(i0, est_len = EST_LEN, gap = GAP, h = H, pre = PRE) {
  est_rows <- (i0 - gap - est_len):(i0 - gap - 1)
  if (min(est_rows) < 1) return(NULL)                      # not enough history
  fit  <- lm(FORM, data = ret[est_rows, ])
  rows <- (i0 - pre):min(i0 + h, n)                        # truncated if data end
  ar   <- as.numeric(ret$usdcad[rows] - predict(fit, newdata = ret[rows, ]))
  rel  <- rows - i0
  post <- rel >= 0
  s    <- summary(fit)$sigma
  list(rel = rel, ar = ar, sigma = s, h_eff = max(rel),
       ar0 = ar[rel == 0], car = sum(ar[post]),
       t_ar0 = ar[rel == 0] / s,
       t_car = sum(ar[post]) / (s * sqrt(sum(post))),
       r2 = summary(fit)$r.squared)
}

# ---- 1. Event-by-event results ----------------------------------------------
event_idx <- rep(NA_integer_, nrow(events))
res <- list(); windows <- list()
for (k in seq_len(nrow(events))) {
  i0 <- which(ret$date >= events$date[k])[1]
  if (is.na(i0)) { message("No data on/after ", events$date[k], " - skipped"); next }
  r <- run_event(i0)
  if (is.null(r)) { message("Not enough pre-event data for: ", events$label[k]); next }
  event_idx[k] <- i0
  win_dates <- ret$date[i0:(i0 + r$h_eff)]
  res[[length(res) + 1]] <- data.frame(
    event = events$label[k], input_date = events$date[k], day0 = ret$date[i0],
    window_days = r$h_eff + 1,
    AR_day0_pct = r$ar0, t_AR_day0 = r$t_ar0,
    CAR_pct = r$car, t_CAR = r$t_car,
    mkt_model_R2 = r$r2,
    boc_decision_in_window = any(boc_dates %in% win_dates),
    truncated_window = r$h_eff < H)
  r$label <- events$label[k]
  windows[[length(windows) + 1]] <- r
}
stopifnot(length(res) > 0)
out <- do.call(rbind, res)

# ---- 2. Placebo test: same statistic on non-event days ----------------------
idx <- seq_len(n)
excl <- rep(FALSE, n)
for (i0 in event_idx[!is.na(event_idx)]) {
  excl[idx >= (i0 - GAP - PRE) & idx <= (i0 + H + GAP)] <- TRUE
}
cand <- which(!excl & idx >= (GAP + EST_LEN + 1) & idx <= (n - H))
placebo <- t(sapply(cand, function(i) { r <- run_event(i); c(r$ar0, r$car) }))
colnames(placebo) <- c("AR_day0", "CAR")
write.csv(data.frame(day0 = ret$date[cand], placebo),
          "output/placebo_distribution.csv", row.names = FALSE)

emp_p <- function(x, dist) (1 + sum(abs(dist) >= abs(x))) / (1 + length(dist))
out$placebo_p_AR  <- sapply(out$AR_day0_pct, emp_p, dist = placebo[, "AR_day0"])
out$placebo_p_CAR <- sapply(out$CAR_pct,     emp_p, dist = placebo[, "CAR"])
cat(sprintf("Placebo days: %d | sd(AR day0) = %.3f | sd(CAR) = %.3f\n",
            length(cand), sd(placebo[, 1]), sd(placebo[, 2])))

write.csv(out, "output/event_results.csv", row.names = FALSE)
print(round_df <- within(out, { AR_day0_pct <- round(AR_day0_pct, 3); t_AR_day0 <- round(t_AR_day0, 2)
                                CAR_pct <- round(CAR_pct, 3); t_CAR <- round(t_CAR, 2)
                                mkt_model_R2 <- round(mkt_model_R2, 2) }))

# ---- 3. Full-sample regression with event dummies + Newey-West HAC ----------
X <- ret
dummy_map <- character()
for (k in which(!is.na(event_idx))) {
  i0 <- event_idx[k]
  d0 <- paste0("d0_", k); dw <- paste0("dw_", k)
  X[[d0]] <- as.integer(idx == i0)
  X[[dw]] <- as.integer(idx > i0 & idx <= i0 + H)
  dummy_map[d0] <- paste(events$label[k], "(day 0)")
  dummy_map[dw] <- paste(events$label[k], sprintf("(days +1 to +%d)", H))
}
dn <- names(dummy_map)
keep <- dn[sapply(dn, function(v) sum(X[[v]]) > 0)]
m <- lm(as.formula(paste("usdcad ~", paste(c(CTRL, keep), collapse = " + "))), data = X)

nw_manual <- function(model, L) {
  Xm <- model.matrix(model); u <- residuals(model); nn <- nrow(Xm)
  bread <- solve(crossprod(Xm)); Xu <- Xm * u
  S <- crossprod(Xu)
  if (L >= 1) for (l in 1:L) {
    w <- 1 - l / (L + 1)                                   # Bartlett weight
    G <- crossprod(Xu[(l + 1):nn, , drop = FALSE], Xu[1:(nn - l), , drop = FALSE])
    S <- S + w * (G + t(G))
  }
  bread %*% S %*% bread
}
L <- floor(4 * (n / 100)^(2 / 9))
V_man <- nw_manual(m, L)
V_pkg <- NeweyWest(m, lag = L, prewhite = FALSE, adjust = FALSE)
rel_diff <- max(abs(V_man - V_pkg)) / max(abs(V_pkg))
cat(sprintf("\nNewey-West check (lag %d): max relative difference manual vs sandwich = %.2e\n",
            L, rel_diff))
if (rel_diff > 1e-6) warning("Manual and sandwich Newey-West differ - investigate before publishing.")

ct <- unclass(coeftest(m, vcov. = V_pkg))
hac <- data.frame(term = rownames(ct), estimate = ct[, 1], hac_se = ct[, 2],
                  t = ct[, 3], p_value = ct[, 4], row.names = NULL)
hac$term <- ifelse(hac$term %in% names(dummy_map), dummy_map[hac$term], hac$term)
write.csv(hac, "output/hac_regression.csv", row.names = FALSE)
writeLines(sprintf("lag=%d; max relative difference manual vs sandwich=%.3e", L, rel_diff),
           "output/nw_crosscheck.txt")
print(hac[grepl("\\(", hac$term), ], digits = 3)

# ---- 4. Sensitivity: estimation length x event window -----------------------
sens <- do.call(rbind, lapply(which(!is.na(event_idx)), function(k) {
  do.call(rbind, lapply(c(60, 100, 150), function(el) {
    do.call(rbind, lapply(c(1, 5, 10), function(hh) {
      r <- run_event(event_idx[k], est_len = el, h = hh)
      if (is.null(r)) return(NULL)
      data.frame(event = events$label[k], est_len = el, post_days = hh,
                 CAR_pct = r$car, t_CAR = r$t_car)
    }))
  }))
}))
write.csv(sens, "output/sensitivity.csv", row.names = FALSE)

# ---- 5. Plot: daily AR (bars) and CAR from day 0 (line) per event ----------
nev <- length(windows)
png("output/event_windows.png", width = 1300, height = 330 * ceiling(nev / 2), res = 120)
par(mfrow = c(ceiling(nev / 2), 2), mar = c(4, 4, 3, 1))
for (w in windows) {
  cum <- cumsum(ifelse(w$rel >= 0, w$ar, 0)); cum[w$rel < 0] <- NA
  plot(w$rel, w$ar, type = "h", lwd = 6, lend = 1, col = "#12315C",
       ylim = range(c(w$ar, cum, 0), na.rm = TRUE),
       xlab = "Trading days from event", ylab = "% (abnormal)", main = w$label, cex.main = 0.9)
  abline(h = 0, col = "grey50"); abline(v = -0.5, lty = 2, col = "#B22222")
  lines(w$rel, cum, col = "#B22222", lwd = 2, type = "b", pch = 16)
}
invisible(dev.off())

# ---- Run record -------------------------------------------------------------
writeLines(c(sprintf("Data: %s to %s (%d returns)", min(df$date), max(df$date), n),
             sprintf("EST_LEN=%d GAP=%d H=%d PRE=%d", EST_LEN, GAP, H, PRE),
             capture.output(sessionInfo())), "output/run_info.txt")
cat("\nDone. See output/ (event_results.csv is the main table).\n")
