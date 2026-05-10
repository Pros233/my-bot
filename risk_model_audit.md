# Risk Model Audit

Research-only audit. No live behavior, entry logic, or exit logic was changed.

## Scope

- Source files read: `research_status.md`, `backtest.py`, `config.py`, `forensics.py`, `setup_trades.csv`, and `walk_forward_summary.csv`.
- Trade evidence uses `2407` accepted research setup-trades from `setup_trades.csv` (`FULL_1095D`).
- Important: these are setup-trade observations, not unique market events. The same market move can appear in multiple research setups.

## 1. Cost Model

| Item | Value | Evidence |
| --- | ---: | --- |
| Maker fee per leg | 0.100% | `config.MAKER_FEE = 0.001` |
| Entry slippage | 0.100% | Adverse entry adjustment on every fill |
| Exit slippage modeled | No | Stops, trails, and time exits fill at the modeled exit price with no extra slip |
| Fees on entry and exit | Yes | `risk.net_pnl()` and research close helpers charge both legs |
| Partial exits pay fees correctly | Yes | Each closed leg pays its proportional entry fee and exit fee |
| Stop exits pay fees | Yes | Stop finalization uses the same fee-aware PnL helpers |
| Median fee drag | 0.222R | Round-trip fees alone |
| Median total friction | 0.333R | Fees plus entry slippage |
| Low-ATR median total friction | 0.494R | Thin stops magnify cost drag |

Interpretation: the model is mixed but mildly optimistic overall. Entry fills are conservative because they always include 0.10% adverse slippage, but all exits, including stop exits, pay maker fees and get no additional exit slippage. That makes stop execution a little too clean for live conditions.

## 2. Stop Model

- Initial stop distance is `ATR * ATR_STOP_MULTIPLIER`, with `ATR_STOP_MULTIPLIER = 1.5`, rounded to 2 decimals.
- Position size is `risk_amount / stop_distance`, so narrower ATR stops increase both leverage and cost drag in R terms.

| Stop Geometry | Value |
| --- | ---: |
| Average stop distance | 0.973% of price |
| Median stop distance | 0.900% of price |
| Median expected full-stop loss | -1.226R |
| Median actual stop-exit loss | -1.221R |
| Losing trades exiting via STOP | 93.5% |

### Stop Distance by Setup Family

| Setup family | Trades | Avg stop % | Median stop % |
| --- | ---: | ---: | ---: |
| MACD_VWAP_BASE | 31 | 0.893% | 0.871% |
| MACD_VWAP_SHORT_ONLY | 13 | 0.883% | 0.871% |
| MACD_VWAP_NO_MEDIUM_ATR | 18 | 0.889% | 0.730% |
| MACD_VWAP_SHORT_ONLY_NO_MEDIUM_ATR | 7 | 0.877% | 0.758% |
| MACD_VWAP_VOLUME | 30 | 0.872% | 0.853% |
| MACD_ONLY_REFERENCE | 31 | 0.893% | 0.871% |
| VOLUME_BREAKOUT_CONTINUATION | 881 | 0.972% | 0.907% |
| PULLBACK_TO_TREND_CONTINUATION | 1113 | 0.990% | 0.905% |
| RANGE_MEAN_REVERSION | 157 | 0.953% | 0.896% |
| FAR_VWAP_MEAN_REVERSION | 126 | 0.944% | 0.890% |

### Stop Distance by ATR Bucket

| ATR bucket | Trades | Avg stop % | Median stop % |
| --- | ---: | ---: | ---: |
| low | 835 | 0.587% | 0.607% |
| medium | 821 | 0.919% | 0.916% |
| high | 751 | 1.461% | 1.328% |

Most losers are full-stop losses: `1523` losing trades were recorded, and `93.5%` of them exited via `STOP`. The median losing trade was `-1.212R`, which is very close to the modeled median fee-loaded stop of `-1.226R`. That means the stop logic is behaving as designed; the extra loss beyond `-1R` mostly comes from fees on relatively tight ATR stops, not from runaway stop behavior.

The deeper issue is entry failure, not stop drift: `64.2%` of losers never reached `0.5R`, and `86.0%` never reached `1.0R`.

## 3. Reward Model

- Partial TP is triggered at `entry ± stop_distance * 1.5R` and closes 50% of the position.
- Stage B activates at `0.8R`; Stage C begins only after the 1.5R partial TP.

| MFE milestone | Trades reaching milestone | Rate |
| --- | ---: | ---: |
| >= 0.5R | 1425 | 59.2% |
| >= 1.0R | 1068 | 44.4% |
| >= 1.5R | 781 | 32.4% |
| >= 2.0R | 545 | 22.6% |
| Median MFE | — | 0.762R |
| Mean MFE | — | 1.430R |

### Universal Target Math (no cherry-picking)

| Exit model | Avg R | Median R | PF | Win rate |
| --- | ---: | ---: | ---: | ---: |
| Actual | -0.281 | -1.144 | 0.613 | 36.7% |
| Full exit at 0.50R | -0.207 | 0.500 | 0.589 | 59.4% |
| Full exit at 0.75R | -0.211 | 0.750 | 0.644 | 51.2% |
| Full exit at 1.00R | -0.208 | -1.097 | 0.682 | 45.6% |
| Full exit at 1.50R | -0.227 | -1.142 | 0.685 | 37.3% |
| Full exit at 2.00R | -0.200 | -1.143 | 0.724 | 37.0% |

Interpretation: `1.5R` is demanding on BTC 1H under the current research families. Only `32.4%` of setup-trades ever print `1.5R`, while `44.4%` print `1.0R`. A universal full exit at `1.0R` improves the aggregate math from `PF 0.613 / AvgR -0.281` to `PF 0.682 / AvgR -0.208`, but it still does not produce positive expectancy. Exit tuning alone does not rescue the system.

### Mean-Target Exit Evidence Already Tested

| Range MR exit experiment | OOS trades | Median PF | Avg R | Median R | Win rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| MR_EXIT_0 (Current F2 baseline) | 140 | 0.7796 | -0.187 | -0.835 | 39.3% |
| MR_EXIT_1 (Full exit at VWAP touch) | 165 | 0.0961 | -0.500 | -0.497 | 17.0% |
| MR_EXIT_3 (Full exit at range midpoint touch) | 158 | 0.3280 | -0.325 | -0.142 | 44.9% |

VWAP-touch and range-mid exits were already tested on `RANGE_MEAN_REVERSION` and both failed. That is good evidence that mean-target exits are not a general mathematical fix for the current research families.

## 4. Break-Even Logic

- Before partial TP, Stage B can trail toward breakeven once profit reaches `0.8R`.
- After partial TP, 50% is closed and the stop floor becomes `BE + 0.1R`, then Stage C trails with `STAGE_C_ATR_MULT = 1.5`.

| Partial-TP Behavior | Value |
| --- | ---: |
| Trades hitting partial TP | 765 / 2407 (31.8%) |
| Partial-TP trades later exiting via STOP | 764 / 765 (99.9%) |
| Average final R after partial TP | +1.376R |
| Median final R after partial TP | +1.046R |
| Partial-TP trades finishing `<= 0.5R` | 53 / 765 (6.9%) |
| Partial-TP trades finishing `<= 0R` | 1 / 765 |

Interpretation: break-even movement is not obviously too early. Once trades actually reach the `1.5R` partial TP, the model usually captures around `1R+` net on the final trade. The main weakness is that too few trades get there in the first place.

## 5. Time / Stall Logic

| Logic | State | Evidence |
| --- | --- | --- |
| Stall exit | Disabled | `STALL_EXIT_ENABLED = False` and observed stall exits = `0` |
| Time exit | Enabled | Triggers after `30` candles if profit is below `0.5R` |
| Observed time exits | Active | `144` trades, avg `-0.180R`, median `-0.171R` |

Interpretation: stall exits were correctly disabled where expected. Time exits still affect research setups and act as a modest loss-reducer, but they are only `144 / 2407` trades and are not the primary performance driver.

## 6. Directional Asymmetry

| Direction | Trades | Win rate | Avg R | Median R | PF | Avg stop % | Avg MFE | Avg MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LONG | 1255 | 37.5% | -0.276 | -1.140 | 0.612 | 0.961% | 1.380R | 1.050R |
| SHORT | 1152 | 35.9% | -0.287 | -1.147 | 0.614 | 0.986% | 1.486R | 1.023R |

Interpretation: one side is not catastrophically worse across all research setups. Shorts have slightly wider stops and slightly worse average `R`, but long/short performance is weak on both sides. Directional asymmetry exists inside some individual families, but it is not the main portfolio-level problem.

## 7. BTC 1H / Market Fit

- `40.8%` of setup-trades never reached `0.5R`.
- `55.6%` never reached `1.0R`.
- `67.6%` never reached `1.5R`.
- Median MFE across all setup-trades was only `0.762R`.
- Median total modeled friction was `0.333R`, and low-ATR trades paid about `0.494R` before any edge.

BTC 1H does have enough movement to print `0.5R` and sometimes `1.0R`, but the current research families do not generate consistent enough follow-through for a 1.5R partial-TP model to feel easy. That said, the dominant failure is still entry quality: most losing trades do not get far enough for any exit logic to matter.

## Verdict

- `cost model issue`: secondary. Fees are modeled on all exits, but stop/trail exits are still a bit optimistic because they use maker fees and no extra exit slippage.
- `stop model issue`: no clear evidence. Loss geometry matches the ATR stop plus fee drag.
- `risk model too demanding`: partially yes. Median modeled friction is about `0.33R`, and low-ATR trades pay almost `0.50R` of friction before any edge.
- `entry model still main issue`: yes. `64.2%` of losers never reach `0.5R`, and `86.0%` never reach `1.0R`.
- `timeframe likely unsuitable`: not conclusively for every future setup, but the current BTC 1H setup families do not provide enough reliable follow-through for the present reward model.
- **Overall verdict:** the current risk/exit model is demanding, but it is not the main cause of failure. Entry quality is still the dominant problem, with cost drag and ambitious reward geometry acting as secondary headwinds.
