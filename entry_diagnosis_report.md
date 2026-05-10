# Entry Diagnosis

Research-only continuation diagnosis across the current BTC dataset. No live trading logic was changed.

## Structural Summary
- Broad 1H continuation baseline remains weak. Breakout baseline was 0.6338 median PF with 1952 OOS trades.
- Across 1H baseline continuation trades (`VOLUME_BREAKOUT_CONTINUATION` + `PULLBACK_TO_TREND_CONTINUATION`), 40.6% never reached 0.5R and 55.0% never reached 1.0R.
- The original `ENTRY_ANTI_CHASE_LONG_ONLY` failure mode stayed real: 52.9% of its 1H full trades failed before 0.5R, and 81.8% of losers never reached 0.5R.

## Entry Profile Research
- 1H breakout `ENTRY_ANTI_CHASE_LONG_ONLY`: 60 OOS trades, median PF 0.0, AvgR -0.1253, profitable/losing windows 15/26.
- 1H breakout `ENTRY_ANTI_CHASE_LONG_ONLY_ADAPTIVE`: 131 OOS trades, median PF 0.8523, AvgR -0.0835, profitable/losing windows 19/22, cluster profit share 0.6056.
- Adaptive anti-chase helped by reopening some medium/far VWAP sample and cutting early outright failures, but it still failed because median PF stayed below 1.05, AvgR stayed negative, worst-window PF stayed at 0.0, and profit still leaned on small windows.

## Time Horizon Effect
- Breakout baseline 1H: trades=1952, median PF=0.6338, AvgR=-0.2883, windows=2/39.
- Breakout baseline 2H: trades=954, median PF=0.875, AvgR=-0.084, windows=16/25.
- Breakout baseline 4H: trades=467, median PF=0.9383, AvgR=0.0228, windows=18/23.
- Higher timeframes improved follow-through directionally: breakout baseline median PF rose from 0.634 on 1H to 0.875 on 2H and 0.938 on 4H.
- That improvement was not enough to create a robust edge. Even 4H baseline still failed median PF > 1.05 and stable-window requirements.
- Refined long-only overlays on 4H produced better-looking trade quality, but sample stayed too small and profitable windows were still dominated by one- or two-trade pockets.

## Reward / Exit Effect
- Lower continuation targets did not reliably fix the problem.
- On 1H breakout baseline, 1.0R and 0.75R increased win rate but reduced median PF versus the current baseline. VWAP-touch and range-mid exits were decisively bad.
- On 1H adaptive anti-chase, the baseline target still beat 1.0R and 0.75R on median PF. Smaller targets improved hit rate but did not create stable expectancy.
- Conclusion: the reward model is demanding, but it is not the only blocker. Easier targets alone do not rescue BTC continuation.

## Structural Constraints
- Breakout baseline 1H: 1128 full trades, median R -1.144, fail-before-0.5R 40.0%.
- Pullback baseline 1H: 1474 full trades, median R -1.132, fail-before-0.5R 41.0%.
- Low ATR is consistently the weakest broad bucket. Volume is not a strong separator for breakout continuation because accepted breakout trades are almost entirely high-volume already.
- Multi-candle extension filters help avoid some obvious chases, but simply tightening or relaxing them mostly trades sample size against a still-weak underlying edge.

## Recommendations
- Do not promote any continuation profile to paper trading.
- If continuation research continues, prefer higher timeframes before more 1H refinement because 2H/4H improved follow-through more than further 1H filter tweaks.
- Consider testing a different asset with cleaner directional follow-through if continuation remains the goal.
- If staying on BTC, prioritize alternative entry concepts such as range mean-reversion or range-trading, because continuation still clusters around median losses near the stop.
- Any future reward-model work should be paired with a setup that first proves stable entry quality. Smaller targets by themselves were not enough.
