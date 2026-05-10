# Research Status

Current status: the MACD+VWAP setup family failed 730-day walk-forward validation.

- Do not trade this family live.
- Do not paper trade this family yet.
- MACD+VWAP should be retired as the primary setup.
- Keep `MACD_VWAP_VOLUME` only as a sample-limited observation, not as a strategy.
- The research framework is now the valuable output of this repo state.

## Failed Setups Summary

- `MACD_VWAP_BASE`: average/median OOS profit factor `0.573 / 0.446` over `185` OOS trades.
- `MACD_VWAP_SHORT_ONLY`: average/median OOS profit factor `0.483 / 0.265`.
- `MACD_VWAP_NO_MEDIUM_ATR`: average/median OOS profit factor `0.733 / 0.320`.
- `MACD_VWAP_VOLUME`: `30` OOS trades; promising pocket, but too small and unstable.
- `MACD_VOLUME_ONLY`: `0` OOS trades.
- `VOLUME_BREAKOUT_CONTINUATION`: retired research setup. It was implemented as a research-only setup and failed 730-day walk-forward validation with a large sample, so the failure is not due to lack of data.
- `PULLBACK_TO_TREND_CONTINUATION`: retired research setup. It failed 730-day walk-forward validation with a large sample, so the failure is not sample-limited.
- `RANGE_MEAN_REVERSION`: broad research setup failed promotion rules. It is not paper tradeable or live tradeable, but it is not retired yet because one diagnostic pocket remains interesting.

## Retired Breakout Research

- `VOLUME_BREAKOUT_CONTINUATION` total OOS trades: `856`
- Average/median OOS profit factor: `0.640 / 0.568`
- Profitable/losing windows: `3 / 14`
- Worst window profit factor: `0.218`
- Average/median `R`: `-0.315 / -1.149`
- TP hit rate: `32.6%`
- Longs and shorts both failed.
- ATR buckets all failed.
- Volume buckets all failed, with the high-volume bucket worst.
- Conclusion: raw breakout continuation with volume expansion should be retired immediately.
- Do not patch, paper trade, or live trade this setup.

## Retired Pullback Research

- `PULLBACK_TO_TREND_CONTINUATION` failed 730-day walk-forward validation.
- It had a large sample, so the failure is not sample-limited.
- Total OOS trades: `1120`
- Average/median OOS profit factor: `0.634 / 0.559`
- Profitable/losing windows: `1 / 16`
- Worst window profit factor: `0.388`
- Average/median `R`: `-0.302 / -1.131`
- TP hit rate: `29.9%`
- Longs and shorts both failed.
- ATR, VWAP distance, and volume buckets all failed.
- Conclusion: simple pullback-to-trend continuation should be retired immediately.
- Do not patch, paper trade, or live trade this setup.

## Range Mean Reversion Research

- `RANGE_MEAN_REVERSION` broad version status: failed, not retired yet.
- It is not paper tradeable.
- It is not live tradeable.
- The broad setup failed promotion rules.
- Mean reversion to VWAP happens often, but VWAP-touch exit is not profitable by itself.
- `MR_EXIT_0` remained the least bad exit.
- `MR_EXIT_1` full VWAP exit failed badly.
- `MR_EXIT_3` range midpoint exit was mildly interesting, but still failed.
- Shorts are weak.
- High-volume fades are weak.
- The far-VWAP bucket is the only clearly interesting pocket.

Key exit results:

- `MR_EXIT_0`: `140` OOS trades, median PF `0.780`, AvgR `-0.187`
- `MR_EXIT_1`: `165` OOS trades, median PF `0.096`, AvgR `-0.500`
- `MR_EXIT_3`: `158` OOS trades, median PF `0.328`, AvgR `-0.325`

## Far-VWAP Mean Reversion Robustness

- `FAR_VWAP_MEAN_REVERSION` status: unstable / failed robustness validation.
- It is not paper tradeable.
- It is not live tradeable.
- Do not promote this family.
- The 730-day result looked promising but sample-limited.
- The 1095-day robustness run expanded the sample and the edge degraded.
- `FV_MR_0` and `FV_MR_5` OOS trades increased from `29` to `52`.
- `FV_MR_0` and `FV_MR_5` median PF fell from `1.288` to `0.340`.
- `FV_MR_0` and `FV_MR_5` AvgR fell from `+0.159` to `+0.030`.
- `FV_MR_0` and `FV_MR_5` median R fell from `+0.076` to `+0.005`.
- `FV_MR_0` and `FV_MR_5` profitable/losing windows worsened from `10 / 7` to `12 / 17`.
- High-volume far-VWAP trades still did not appear, so the high-volume exclusion remains untested.
- `FV_MR_6` short-only increased from `17` to `30` OOS trades, but it also degraded on the longer-history run.
- No far-VWAP variant passed promotion rules.
- Do not paper trade this family.
- Do not live trade this family.
- Do not promote this family.
- Conclusion: far-VWAP mean reversion produced interesting pockets over 730 days, but failed robustness validation over 1095 days.

## Safety Notes

- No setup in the current MACD+VWAP family passed the promotion rules.
- Nothing in this family should be promoted to paper trading or live trading.
- Research flags remain opt-in by default:
  - `RUN_SIGNAL_EXPERIMENTS=false`
  - `RUN_EXIT_EXPERIMENTS=false`
  - `RUN_WALK_FORWARD_RESEARCH=false`

## Next Research Direction

The next strategy should not be another patch of MACD+VWAP. Build one new setup family at a time and validate it through the research framework.

Candidate research families:

- Breakout continuation with volume expansion
- Pullback-to-trend continuation
- Range mean-reversion
- Volatility expansion after compression

Do not implement or promote any new family until it passes walk-forward validation and the repo promotion rules.
