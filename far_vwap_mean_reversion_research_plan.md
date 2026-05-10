# FAR_VWAP_MEAN_REVERSION Research Plan

Research-only document. This file is planning only and does not define or change live trading behavior.

## Setup Family

`FAR_VWAP_MEAN_REVERSION`

## Context

Three continuation-style families have already been retired:

- MACD+VWAP trend continuation
- `VOLUME_BREAKOUT_CONTINUATION`
- `PULLBACK_TO_TREND_CONTINUATION`

Broad `RANGE_MEAN_REVERSION` also failed promotion rules, but one diagnostic pocket remains interesting:

- far VWAP-distance trades performed better than the broad mean-reversion set
- longs looked materially stronger than shorts
- high-volume fades remained weak
- VWAP-touch exits were not enough to rescue the broad family

The next hypothesis is narrower:

- the edge, if any, may exist only when price is far from VWAP, especially on the long side and outside high-volume continuation conditions

## Scope

- Do not implement anything yet.
- Do not change live behavior.
- Do not change strategy logic in this planning step.
- Use this file only to guide future research implementation.

## 1. Setup Hypothesis

Broad mean reversion fails because too many entries are taken when price is not truly stretched or when the market is still in real expansion. A trade may only have edge when price is far from VWAP and the move looks more like exhaustion than continuation.

The core idea is:

- focus on the far VWAP-distance bucket only
- bias toward long-side exhaustion first
- avoid obvious continuation conditions such as high-volume expansion
- keep the current `MR_EXIT_0` baseline at first, because broad exit testing did not find a clearly better replacement

## 2. Entry Idea

Initial research direction only:

- use the existing `RANGE_MEAN_REVERSION` structure as the base candidate universe
- require the `far` VWAP-distance bucket
- test long-only first as the most promising direction
- treat high-volume fades as lower quality or exclude them in one diagnostic variant
- keep VWAP touch and range midpoint as diagnostic outcome fields, not as final exit commitments

The goal is not to redesign the whole setup yet, only to isolate whether the strongest observed pocket survives walk-forward on its own.

## 3. Invalidation Idea

Candidate invalidation ideas for later testing:

- stop beyond the recent extreme
- keep the current stop baseline unchanged at first for comparability
- invalid if price continues expanding away from VWAP instead of reverting
- monitor whether losers still fail immediately before reaching `0.5R`

Research should keep invalidation simple first and avoid new threshold tuning until the narrower pocket proves it has structural promise.

## 4. Diagnostic Variants

Named research variants to compare later:

- far-VWAP both directions
- far-VWAP long-only
- far-VWAP long-only excluding high-volume
- far-VWAP medium-ATR only
- far-VWAP low/medium-volume only

These are diagnostic variants only. None should be promoted unless they pass full walk-forward rules.

## 5. Required Forensic Fields

Every accepted and rejected candidate should preserve the existing range mean-reversion forensic fields, plus support variant-specific slicing.

Core fields:

- `setup_name`
- `variant_name`
- `split`
- `window_id`
- `signal_time`
- `entry_time`
- `entry_price`
- `exit_time`
- `exit_price`
- `exit_reason`
- `exit_r`
- `pnl_net`

Context fields:

- VWAP distance in `R`
- VWAP distance bucket
- direction
- ATR bucket
- volume bucket
- range high
- range low
- range midpoint
- distance from range boundary
- reclaim detected
- rejection detected
- candles outside range
- MACD agreed
- price vs VWAP

Path and outcome fields:

- `mfe_r`
- `mae_r`
- `reached_0_5r`
- `reached_1_0r`
- `reached_1_5r`
- `candles_to_0_5r`
- `candles_to_1_0r`
- `touched_vwap_after_entry`
- `candles_to_vwap_touch`
- `touched_range_mid_after_entry`
- `candles_to_range_mid_touch`
- `max_continuation_away_from_vwap_before_reversion_r`

## 6. Walk-Forward Promotion Rules

Use the existing promotion rules:

- Total OOS trades `>= 50`
- Average window PF `> 1.10`
- Median window PF `> 1.05`
- Average `R` positive
- Median `R` not deeply negative
- Profitable windows meaningfully outnumber losing windows
- No tiny pocket creates most profit

Additionally for this family:

- the far-VWAP edge should not depend entirely on one tiny long cluster
- results should remain acceptable after excluding obviously weak buckets such as high-volume continuation

## 7. Failure Conditions

Treat the family as failed if:

- OOS PF stays below `1.0` across walk-forward
- median `R` stays materially negative
- the long-only far-VWAP pocket also fails
- performance disappears when high-volume fades are removed
- trades still fail before reaching `0.5R`
- the apparent edge only exists in one tiny calendar segment

## 8. Retirement Criteria

Retire `FAR_VWAP_MEAN_REVERSION` if research shows:

- no far-VWAP variant passes promotion rules
- the long-only version does not stabilize the setup
- the low/medium-volume filter does not improve robustness
- shorts remain weak and longs still do not produce positive walk-forward expectancy
- the supposed edge collapses once evaluated across enough OOS windows

## Planning Only

This file is planning only.

- No live behavior changes
- No strategy code changes
- No implementation in this step

## Final Robustness Result

- Status: unstable / failed robustness validation.
- Reason: the promising 730-day pocket degraded on 1095-day validation.
- Main lesson: far-VWAP mean reversion may create occasional positive pockets, but the effect is not stable across longer history.
