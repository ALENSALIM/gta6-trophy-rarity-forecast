# Leonida Rarity Forecast

Forecasts every Grand Theft Auto VI PS5 trophy's Exophase earn rate at the
target time, as five quantiles per trophy, for the make.fun bounty
`gta6-trophy-rarity-forecast-01`.

## How it works

`main.ts` reads every `<id>.in` case file from `$MAKEFUN_INPUT_DIR`
(default `/input`) and writes one `<id>.out` forecast per case to
`$MAKEFUN_OUTPUT_DIR` (default `/output`).

For each trophy it fits a saturating growth curve to the history:

- Earn rates only rise, fastest just after launch and ever more slowly as
  the tracked player base saturates, so the trophy's logit rate is fitted
  as a straight line in log time since launch (log hours + 1) over the
  last day of reads, with recent reads weighted most (6-hour half-life).
  The fitted line is read at the target time.
- The slope is shrunk toward zero when the fit cannot tell it from noise
  and is never allowed negative, so a flat page forecasts persistence.
- The median is capped at four logit units above the last read and at 99%
  unless the trophy is already there.

The five quantiles come from the median plus a logit spread built from
four honest sources: a random-walk volatility (half the house's
0.5-logit-per-root-day prior, half what the fit's residuals imply, floored
at 0.30 a day), the slope's standard error times the extrapolation
distance, the fit residual, and a small horizon-scaled floor. Stable
trophies get tight bands, noisy or fast-moving ones wide ones, and longer
horizons get wider bands than short ones.

The program is deterministic (same input, same output bytes), uses no
network and no wall clock — the only times it reads are the case file's
own `at` and `target` — and runs in well under a second per case.

## Validation

Rolling-origin backtests (see the backtest script kept outside this repo)
against the house persistence baseline, scored by pinball loss on the
logit scale:

- Practice histories (flat launch-night page): **0.012226 vs 0.016952**
  for the house baseline.
- Synthetic log-time growth histories: **0.048996 vs 0.065063**.
- Final-like synthetic run (2 days fitted, 4.5 days ahead):
  **0.180243 vs 0.375752**.

## License

MIT — see `LICENSE`.
