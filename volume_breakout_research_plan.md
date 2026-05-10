# VOLUME_BREAKOUT_CONTINUATION Research Plan

Research-only document. This file does not define live trading behavior and should not be treated as a strategy implementation.

## Setup Family

`VOLUME_BREAKOUT_CONTINUATION`

## Scope

- Do not modify the current live strategy.
- Do not implement trading logic yet.
- Use this plan to guide a new research-only setup family.
- Promote nothing unless it passes walk-forward validation and repo promotion rules.

## Setup Hypothesis

The hypothesis is that a breakout can become tradable when price leaves a clear recent balance area and volume expands enough to signal real participation, not just noise. The expected edge is not from predicting reversals or patching MACD+VWAP, but from identifying situations where price discovery has already started and continuation is more likely than immediate mean reversion.

A valid setup family should show:

- Better immediate follow-through than the retired MACD+VWAP family.
- Higher frequency of trades reaching `0.5R` and `1.0R`.
- No-TP trades with meaningfully stronger MFE than the prior family.
- Performance that survives walk-forward windows instead of one small pocket.

## Entry Idea

Initial research idea only:

- Identify a recent compression, balance, or short consolidation.
- Mark a breakout above resistance for longs or below support for shorts.
- Require volume expansion relative to recent baseline activity.
- Enter only when the breakout is not excessively extended from the breakout level at signal time.
- Evaluate variants such as:
  - immediate breakout continuation
  - breakout after one-bar hold above or below the level
  - breakout plus retest hold

Do not choose numeric thresholds yet. Use quantile-driven diagnostics first.

## Invalidation Idea

The setup should be considered invalid when the breakout fails structurally rather than just fluctuating:

- Price quickly falls back inside the broken range for longs or back above it for shorts.
- The breakout candle or immediate follow-through shows strong rejection.
- Volume expands on the breakout but continuation does not appear afterward.
- MAE happens early and MFE remains weak, indicating false breakout behavior.

Research should compare invalidation ideas such as:

- return back inside range
- failure to hold breakout level after `N` candles
- initial structure stop beyond the opposite side of the range
- ATR-scaled protective stop

## Regime Tags Needed

Use existing data first and bucket regimes with quantiles where possible.

- direction: `long` / `short`
- ATR bucket: `low` / `medium` / `high`
- volume ratio bucket: `low` / `medium` / `high`
- breakout extension bucket at entry: distance from breakout level in `R`
- distance from VWAP bucket
- pre-entry 3-candle move bucket
- pre-entry 6-candle move bucket
- compression width bucket
- compression duration bucket
- breakout candle range bucket
- breakout close-location bucket if measurable
- trend context tag
- range/compression context tag
- retest vs no-retest tag

## Required Forensic Fields

Every accepted trade and every skipped candidate should be analyzable with breakout-specific context.

Core fields:

- `setup_name`
- `split`
- `window_id`
- `direction`
- `signal_time`
- `entry_time`
- `entry_price`
- `exit_time`
- `exit_price`
- `initial_stop`
- `current_stop_at_exit`
- `exit_reason`
- `exit_r`
- `pnl_net`

Breakout structure fields:

- breakout reference level
- prior range high
- prior range low
- range width
- range width in `R`
- compression duration in candles
- breakout distance from level in `R`
- breakout candle range in `R`
- close position within breakout candle if measurable
- retest occurred yes/no
- retest delay in candles if measurable

Regime and context fields:

- ATR pct at entry
- ATR bucket
- volume ratio at entry
- volume bucket
- distance from VWAP at entry
- VWAP distance bucket
- pre-entry 3-candle move in `R`
- pre-entry 6-candle move in `R`
- pre-entry move 3c bucket
- pre-entry move 6c bucket
- trend/regime tag
- compression width bucket
- compression duration bucket

Path and follow-through fields:

- `partial_tp_hit`
- `mfe_r`
- `mae_r`
- `candles_held`
- `candles_to_mfe`
- `candles_to_mae`
- `reached_0_5r`
- `reached_1_0r`
- `reached_1_5r`
- `reached_2_0r`
- `candles_to_0_5r`
- `candles_to_1_0r`
- `candles_to_1_5r`
- `hit_1_0r_before_minus_1r`
- `mae_happened_before_mfe`

## Walk-Forward Promotion Rules

This family should use conservative promotion rules. A setup variant can only become a paper-trading candidate if it passes all of the following:

- Total OOS trades across walk-forward tests `>= 50`
- Median window profit factor `> 1.05`
- Average window profit factor `> 1.10`
- Worst 25% of windows are not catastrophic
- Average `R` is positive
- Median `R` is not deeply negative
- No-TP median MFE improves meaningfully over the retired MACD+VWAP family
- A healthy share of trades reaches `0.5R` and `1.0R`
- IS and OOS do not completely disagree
- Profit is not concentrated in one tiny 3-trade or low-sample cluster
- Results are not dependent on one single direction or one single regime bucket unless the setup is explicitly designed as direction-specific

## Failure Conditions

This setup family should be considered weak if any of these patterns appear:

- Most losers never reach `0.5R`
- No-TP trades show poor MFE, similar to the retired MACD+VWAP family
- Performance depends on one tiny volume or ATR pocket
- Trade count collapses when basic quality constraints are added
- IS looks good but OOS degrades sharply across windows
- One side works only because the other side is consistently poor
- Breakouts mostly mean-revert instead of continuing
- Walk-forward profitability vanishes after fees and slippage

## Retirement Criteria

This setup family should be retired if research shows:

- No variant passes the promotion rules after a full walk-forward study
- OOS edge remains sample-limited or unstable
- The apparent edge comes from a tiny cluster of trades
- No-TP MFE stays weak, meaning breakout entries still do not generate real follow-through
- Robustness does not improve after testing multiple structurally sensible entry and invalidation variants
- The family becomes another threshold-tuning exercise instead of a repeatable setup with stable behavior

## Next Research Workflow

Recommended order:

1. Define the candidate universe for breakout-continuation events.
2. Tag regimes and skipped candidates before selecting any final rules.
3. Run trade forensics on accepted and rejected candidates.
4. Compare simple named setup variants only.
5. Use walk-forward validation before considering any paper-trading promotion.

The goal is to discover whether this family has a real edge, not to rescue it with repeated threshold patching.

## Final Result

- Status: retired
- Reason: failed 730-day walk-forward validation with a large sample
- Main lesson: simple volume expansion does not distinguish continuation from exhaustion or liquidity grabs
