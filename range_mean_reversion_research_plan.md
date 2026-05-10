# RANGE_MEAN_REVERSION Research Plan

Research-only document. This file is planning only and does not define or change live trading behavior.

## Setup Family

`RANGE_MEAN_REVERSION`

## Context

Three continuation-style setup families have now been retired:

- MACD+VWAP trend continuation
- `VOLUME_BREAKOUT_CONTINUATION`
- `PULLBACK_TO_TREND_CONTINUATION`

Shared lesson so far:

- Continuation entries on BTC 1H are failing under the current risk and exit model.
- Median `R` remains near full-stop loss.
- TP hit rate is too low.
- Most walk-forward windows lose.

New hypothesis:

- Instead of chasing continuation, BTC 1H may offer better opportunity when price stretches away from fair value and then reverts back toward mean.

## Scope

- Do not implement anything yet.
- Do not change live behavior.
- Do not change strategy logic in this planning step.
- Use this file only to guide future research implementation.

## 1. Setup Hypothesis

A trade is only valid when all of the following are true:

- price is range-bound or otherwise non-trending
- price stretches far from VWAP or another already-available fair-value reference
- momentum or exhaustion behavior suggests the move may revert
- entry fades the extension back toward mean instead of joining the move

The core idea is to test whether BTC 1H offers more reliable edge from fading extension than from participating in continuation.

## 2. Candidate Range Context Ideas

Use existing data only where possible before adding anything new.

Candidate context signals:

- ATR regime
- VWAP distance
- recent high-low range compression
- failed breakout behavior
- slope or flatness of an existing trend reference if available
- avoid strong trend regimes

Research goal:

- determine whether mean-reversion behavior is stronger when price is rotating inside a range rather than trending cleanly
- identify whether some apparent extensions are actually just trend continuation and should be excluded

## 3. Entry Idea

Initial research ideas only:

For longs:

- price extends below VWAP or below recent range low
- then shows reclaim, rejection, or failed continuation downward

For shorts:

- price extends above VWAP or above recent range high
- then shows rejection, reclaim failure, or failed continuation upward

Quality guard:

- fade the extension only after rejection behavior appears
- avoid blind catch-the-knife entries without reclaim or rejection evidence

## 4. Invalidation Idea

Candidate invalidation ideas for later research:

- stop beyond the recent extreme
- ATR-based stop
- invalid if price continues expanding away from mean
- time-based invalidation tracked only as a research field at first

The main goal is to test whether fading extension becomes tradable when invalidation is tied to structural failure rather than arbitrary noise.

## 5. Exit Idea

Do not optimize exits yet.

- Use the existing `F2` exit baseline initially for comparison.
- Also log whether price reaches VWAP or the fair-value reference before stop.
- Treat mean-touch behavior as a diagnostic outcome first, not a tuned exit rule.

## 6. Required Forensic Fields

Every accepted and rejected candidate should support mean-reversion-specific forensics.

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

Range and extension context:

- distance from VWAP at entry
- range high
- range low
- distance from range boundary
- whether breakout failed
- candles outside range
- reclaim candle size
- volume ratio
- ATR bucket
- VWAP distance bucket
- trend or range bucket if available

Post-entry path behavior:

- `mfe_r`
- `mae_r`
- whether price touched VWAP after entry
- candles to VWAP touch
- max continuation away from VWAP before reversion
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
- Average window PF `> 1.10`
- Median window PF `> 1.05`
- Average `R` positive
- Median `R` not deeply negative
- Profitable windows meaningfully outnumber losing windows
- No tiny pocket creates most profit

Nothing should be promoted unless these are met in walk-forward validation.

## 8. Failure Conditions

Retire the family if:

- OOS PF stays below `1.0` across walk-forward
- median `R` remains near full-stop loss
- price usually continues away instead of reverting
- both longs and shorts fail
- only tiny sample pockets work
- performance depends on one regime bucket only

This family should also be retired if it becomes another threshold-tuning exercise without a stable structural edge.

## 9. Planning Only

This file is planning only.

- No live behavior changes
- No strategy code changes
- No implementation in this step
