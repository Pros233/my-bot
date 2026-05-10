# PULLBACK_TO_TREND_CONTINUATION Research Plan

Research-only document. This file is planning only and does not define or change live trading behavior.

## Setup Family

`PULLBACK_TO_TREND_CONTINUATION`

## Context

Two setup families have now been retired:

- MACD+VWAP trend setup
- `VOLUME_BREAKOUT_CONTINUATION`

Main lesson so far:

- Raw momentum and chase entries are failing.
- The next hypothesis is that waiting for an established trend, then a pullback, then a reclaim may produce better continuation entries than entering the initial impulse or breakout.

## Scope

- Do not implement anything yet.
- Do not change live behavior.
- Do not change strategy logic in this planning step.
- Use this file only to guide future research implementation.

## 1. Setup Hypothesis

A trade is only valid after all of the following:

- a clear trend context exists
- price pulls back toward a reference area
- price then reclaims in the direction of the trend with confirmation
- entry is not taken on the initial breakout or impulse candle

The core idea is to avoid raw chase entries and instead test whether continuation is stronger after price pauses, resets, and then proves the trend is resuming.

## 2. Candidate Trend Context Ideas

Use existing data where possible before introducing anything new.

Candidate context signals:

- EMA alignment or slope
- price above VWAP for longs or below VWAP for shorts
- recent swing structure
- ATR regime tag
- MACD agreement as optional context, not as the primary trigger

Research goal:

- determine whether trend context improves entry quality only when paired with a pullback and reclaim
- avoid turning MACD back into the main decision engine

## 3. Pullback Definition Ideas

Possible pullback definitions to compare in research:

- pullback toward VWAP
- pullback toward an already available moving average
- pullback after a prior impulse leg
- pullback into recent structure rather than straight continuation

Quality guard:

- avoid entries if price is still too extended after the pullback

Research should treat pullback depth and pullback duration as measurable context first, not optimized rules.

## 4. Reclaim / Confirmation Ideas

Possible reclaim or continuation confirmation ideas:

- close back in the trend direction after the pullback
- reclaim of VWAP or reclaim away from VWAP in trend direction
- reclaim of short-term structure
- volume confirmation as optional support
- MACD agreement logged as context

Avoid:

- raw high-volume breakout chase
- entering the first impulse candle without a reset

## 5. Invalidation Idea

Candidate invalidation ideas for later research:

- recent swing invalidation
- ATR-based stop
- failure to reclaim after the pullback
- time-based invalidation tracked as a research field only, not optimized initially

The main goal is to test whether structure-based invalidation and better entry timing improve follow-through compared with the retired chase-style families.

## 6. Required Forensic Fields

Every accepted and rejected candidate should support pullback-specific forensics.

Core fields:

- `setup_name`
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

Trend and pullback context:

- trend direction
- pullback depth in `R` or ATR
- distance from VWAP
- distance from recent swing
- candles since prior impulse
- candles spent in pullback
- reclaim candle size
- volume ratio on reclaim
- MACD agreement
- ATR bucket
- pre-entry move bucket

Path behavior after entry:

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

## 7. Walk-Forward Promotion Rules

Use the existing promotion rules:

- Total OOS trades `>= 50`
- Median window PF `> 1.05`
- Average window PF `> 1.10`
- Average `R` positive
- Median `R` not deeply negative
- TP hit rate stable
- no small cluster creates most profit

Nothing should be promoted unless these are met in walk-forward validation.

## 8. Failure Conditions

Retire the family if:

- OOS PF stays below `1.0` across walk-forward
- median `R` remains near full-stop loss
- both longs and shorts fail
- performance depends on tiny pockets
- pullback entries still fail early before reaching `0.5R`

This family should also be retired if it becomes another threshold-patching exercise without a clear structural edge.

## 9. Planning Only

This file is planning only.

- No live behavior changes
- No strategy code changes
- No implementation in this step

## Final Result

- Status: retired
- Reason: failed walk-forward with large sample
- Main lesson: waiting for a simple pullback/reclaim did not solve the continuation problem
