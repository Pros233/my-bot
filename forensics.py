"""
forensics.py — Per-trade diagnostic analysis.

Called from backtest.run() after IS and OOS simulations complete.
Writes trade_forensics.csv and prints a structured diagnostic report.
Does NOT modify any strategy logic or live trading behavior.
"""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

import config

FORENSICS_CSV = Path(__file__).parent / "trade_forensics.csv"

_CSV_FIELDS = [
    "split",
    "trade_num",
    "direction",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "initial_stop",
    "stop_distance",
    "current_stop_at_exit",
    "exit_reason",
    "exit_r",
    "pnl_net",
    "signals_fired",
    "partial_tp_hit",
    "max_favorable_excursion_price",
    "max_adverse_excursion_price",
    "mfe_r",
    "mae_r",
    "candles_to_mfe",
    "candles_to_mae",
    "candles_held",
    "reached_0_5r",
    "reached_1_0r",
    "reached_1_5r",
    "reached_2_0r",
    "candles_to_0_5r",
    "candles_to_1_0r",
    "candles_to_1_5r",
    "hit_1_0r_before_minus_1r",
    "hit_1_5r_before_minus_1r",
    "price_move_last_3_candles_before_entry_r",
    "price_move_last_6_candles_before_entry_r",
    "distance_from_recent_swing_high_r",
    "distance_from_recent_swing_low_r",
    "distance_from_vwap_at_entry_r",
    "atr_pct_at_entry",
    "volume_ratio_at_entry",
    "macd_histogram_at_entry",
    "macd_histogram_slope",
]


def _precompute(df: pd.DataFrame) -> dict:
    macd_df = df.ta.macd(
        fast=config.MACD_FAST,
        slow=config.MACD_SLOW,
        signal=config.MACD_SIGNAL,
    )
    hist_col = f"MACDh_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"
    macd_hist = (
        macd_df[hist_col]
        if macd_df is not None and hist_col in macd_df.columns
        else None
    )

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vwap_20 = (tp * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    avg_vol_20 = df["volume"].rolling(20).mean()

    return {
        "macd_hist": macd_hist,
        "vwap_20": vwap_20,
        "avg_vol_20": avg_vol_20,
    }


def _iso(ts: Any) -> str:
    return ts.isoformat() if ts is not None else ""


def _round_or_none(value: float | None, digits: int) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def _valid(values: list[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(num):
            out.append(num)
    return out


def _avg(values: list[Any]) -> str:
    nums = _valid(values)
    return f"{statistics.mean(nums):+.3f}" if nums else "—"


def _med(values: list[Any]) -> str:
    nums = _valid(values)
    return f"{statistics.median(nums):+.3f}" if nums else "—"


def _median_value(values: list[Any]) -> float | None:
    nums = _valid(values)
    return statistics.median(nums) if nums else None


def _pct(n: int, d: int) -> str:
    return f"{n / d * 100:.1f}%" if d else "—"


def _signal_names(rec: dict) -> list[str]:
    raw = rec.get("signals_fired", "")
    return [name for name in raw.split("|") if name]


def _signal_combo_key(rec: dict) -> str:
    names = sorted(_signal_names(rec))
    return "+".join(names) if names else "(none)"


def _swing_distance(rec: dict) -> float | None:
    hi = rec.get("distance_from_recent_swing_high_r")
    lo = rec.get("distance_from_recent_swing_low_r")
    return hi if hi is not None else lo


def _is_mae_before_mfe(rec: dict) -> bool:
    mae = rec["candles_to_mae"]
    mfe = rec["candles_to_mfe"]
    return mae > 0 and (mfe <= 0 or mae < mfe)


def _is_large_pre_entry_move(rec: dict) -> bool:
    move_3 = rec["price_move_last_3_candles_before_entry_r"] or 0.0
    move_6 = rec["price_move_last_6_candles_before_entry_r"] or 0.0
    return move_3 > 1.0 or move_6 > 1.5


def _build_record(trade: Any, split: str, df: pd.DataFrame, ind: dict) -> dict | None:
    sd = float(trade.stop_distance)
    ep = float(trade.entry_price)
    if sd <= 0:
        return None

    try:
        entry_idx = df.index.get_loc(trade.entry_time)
    except KeyError:
        return None

    signal_idx = entry_idx - 1
    if signal_idx < 25:
        return None

    close = df["close"]
    high = df["high"]
    low = df["low"]

    if trade.direction == "LONG":
        mfe_r = (trade.mfe_price - ep) / sd
        mae_r = (ep - trade.mae_price) / sd
    else:
        mfe_r = (ep - trade.mfe_price) / sd
        mae_r = (trade.mae_price - ep) / sd

    c1r = trade.candles_to_1_0r
    c15r = trade.candles_to_1_5r
    cneg = trade.candle_first_neg_1r
    hit_1r_before_neg = trade.reached_1_0r and (cneg == -1 or c1r <= cneg)
    hit_15r_before_neg = trade.reached_1_5r and (cneg == -1 or c15r <= cneg)

    close_sig = float(close.iloc[signal_idx])
    close_3 = float(close.iloc[signal_idx - 3])
    close_6 = float(close.iloc[signal_idx - 6])

    lk_hi = high.iloc[max(0, signal_idx - 19): signal_idx + 1]
    lk_lo = low.iloc[max(0, signal_idx - 19): signal_idx + 1]

    if trade.direction == "LONG":
        move_3r = (close_sig - close_3) / sd
        move_6r = (close_sig - close_6) / sd
        dist_swing_high_r = (float(lk_hi.max()) - ep) / sd
        dist_swing_low_r = None
    else:
        move_3r = (close_3 - close_sig) / sd
        move_6r = (close_6 - close_sig) / sd
        dist_swing_high_r = None
        dist_swing_low_r = (ep - float(lk_lo.min())) / sd

    vwap_s = ind.get("vwap_20")
    dist_vwap_r = None
    if vwap_s is not None:
        vwap_val = float(vwap_s.iloc[signal_idx])
        if math.isfinite(vwap_val) and vwap_val > 0:
            if trade.direction == "LONG":
                dist_vwap_r = (ep - vwap_val) / sd
            else:
                dist_vwap_r = (vwap_val - ep) / sd

    atr_pct = trade.atr_at_entry / ep * 100.0 if ep > 0 else None

    avg_vs = ind.get("avg_vol_20")
    vol_ratio = None
    if avg_vs is not None:
        avg_v = float(avg_vs.iloc[signal_idx])
        sig_v = float(df["volume"].iloc[signal_idx])
        if math.isfinite(avg_v) and avg_v > 0:
            vol_ratio = sig_v / avg_v

    mh_s = ind.get("macd_hist")
    macd_h = None
    macd_h_slope = None
    if mh_s is not None and signal_idx >= 1:
        mh_now = float(mh_s.iloc[signal_idx])
        mh_prev = float(mh_s.iloc[signal_idx - 1])
        if math.isfinite(mh_now) and math.isfinite(mh_prev):
            macd_h = mh_now
            macd_h_slope = mh_now - mh_prev

    return {
        "split": split,
        "trade_num": trade.trade_num,
        "direction": trade.direction,
        "entry_time": _iso(trade.entry_time),
        "exit_time": _iso(trade.exit_time),
        "entry_price": round(ep, 2),
        "exit_price": round(trade.exit_price or 0.0, 2),
        "initial_stop": round(trade.stop_price, 2),
        "stop_distance": round(sd, 4),
        "current_stop_at_exit": round(trade.current_stop, 2),
        "exit_reason": trade.exit_reason,
        "exit_r": round(trade.exit_r, 4),
        "pnl_net": round(trade.pnl_net or 0.0, 4),
        "signals_fired": "|".join(trade.signals_fired),
        "partial_tp_hit": trade.tp_hit,
        "max_favorable_excursion_price": round(trade.mfe_price, 2),
        "max_adverse_excursion_price": round(trade.mae_price, 2),
        "mfe_r": round(mfe_r, 3),
        "mae_r": round(mae_r, 3),
        "candles_to_mfe": trade.candle_at_mfe,
        "candles_to_mae": trade.candle_at_mae,
        "candles_held": trade.candles_in_trade,
        "reached_0_5r": trade.reached_0_5r,
        "reached_1_0r": trade.reached_1_0r,
        "reached_1_5r": trade.reached_1_5r,
        "reached_2_0r": trade.reached_2_0r,
        "candles_to_0_5r": trade.candles_to_0_5r,
        "candles_to_1_0r": trade.candles_to_1_0r,
        "candles_to_1_5r": trade.candles_to_1_5r,
        "hit_1_0r_before_minus_1r": hit_1r_before_neg,
        "hit_1_5r_before_minus_1r": hit_15r_before_neg,
        "price_move_last_3_candles_before_entry_r": round(move_3r, 3),
        "price_move_last_6_candles_before_entry_r": round(move_6r, 3),
        "distance_from_recent_swing_high_r": _round_or_none(dist_swing_high_r, 3),
        "distance_from_recent_swing_low_r": _round_or_none(dist_swing_low_r, 3),
        "distance_from_vwap_at_entry_r": _round_or_none(dist_vwap_r, 3),
        "atr_pct_at_entry": _round_or_none(atr_pct, 3),
        "volume_ratio_at_entry": _round_or_none(vol_ratio, 3),
        "macd_histogram_at_entry": _round_or_none(macd_h, 6),
        "macd_histogram_slope": _round_or_none(macd_h_slope, 6),
    }


def _write_csv(records: list[dict]) -> None:
    with open(FORENSICS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"\n  [Forensics] CSV saved → {FORENSICS_CSV}  ({len(records)} trades)")


def _grp(recs: list[dict]) -> tuple[int, int, list[float], list[float]]:
    wins = [r for r in recs if r["exit_r"] > 0]
    rs = [r["exit_r"] for r in recs]
    mfes = [r["mfe_r"] for r in recs]
    return len(recs), len(wins), rs, mfes


def _row(label: str, recs: list[dict]) -> None:
    if not recs:
        return
    n, w, rs, mfes = _grp(recs)
    wr = _pct(w, n)
    ar = _avg(rs)
    mr = _med(rs)
    amfe = _avg(mfes)
    tp_n = sum(1 for r in recs if r["partial_tp_hit"])
    print(
        f"  {label:<32} {n:>4} trades  WR={wr:>6}  "
        f"Avg-R={ar}  Med-R={mr}  AvgMFE={amfe}  TP={_pct(tp_n, n)}"
    )


def _print_signal_combo_rows(recs: list[dict]) -> None:
    combos: dict[str, list[dict]] = {}
    for rec in recs:
        combos.setdefault(_signal_combo_key(rec), []).append(rec)

    if not combos:
        return

    print("\n  Signal combinations:")
    for combo, combo_recs in sorted(
        combos.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        _row(combo, combo_recs)


def _print_summary(is_recs: list[dict], oos_recs: list[dict]) -> None:
    width = 90
    print(f"\n{'━' * width}")
    print("  TRADE FORENSICS REPORT")
    print(f"{'━' * width}")
    print(f"  IS: {len(is_recs)} trades   OOS: {len(oos_recs)} trades")

    for split_label, recs in [("IN-SAMPLE", is_recs), ("OUT-OF-SAMPLE", oos_recs)]:
        if not recs:
            continue

        print(f"\n  {'─' * width}")
        print(f"  {split_label}")
        print(f"  {'─' * width}")

        wins = [r for r in recs if r["exit_r"] > 0]
        losses = [r for r in recs if r["exit_r"] <= 0]
        tp_hit = [r for r in recs if r["partial_tp_hit"]]
        no_tp = [r for r in recs if not r["partial_tp_hit"]]

        all_r = [r["exit_r"] for r in recs]
        all_mfe = [r["mfe_r"] for r in recs]
        all_mae = [r["mae_r"] for r in recs]
        win_r = [r["exit_r"] for r in wins]
        loss_r = [r["exit_r"] for r in losses]
        no_tp_mfe = [r["mfe_r"] for r in no_tp]
        no_tp_mae = [r["mae_r"] for r in no_tp]

        def milestone_count(field: str) -> int:
            return sum(1 for rec in recs if rec[field])

        n_half = milestone_count("reached_0_5r")
        n_1r = milestone_count("reached_1_0r")
        n_15r = milestone_count("reached_1_5r")
        n_2r = milestone_count("reached_2_0r")
        n = len(recs)

        stalled_1r = [r for r in recs if r["reached_1_0r"] and not r["reached_1_5r"]]
        never_half = [r for r in recs if not r["reached_0_5r"]]
        mae_first = [r for r in recs if _is_mae_before_mfe(r)]
        chased = [r for r in recs if _is_large_pre_entry_move(r)]

        print("\n  Overall R stats:")
        print(
            f"    Avg-R={_avg(all_r)}  Med-R={_med(all_r)}  "
            f"Max={max(all_r):+.3f}  Min={min(all_r):+.3f}"
        )
        print(f"    Winners avg-R={_avg(win_r)}  Losers avg-R={_avg(loss_r)}")
        print(f"    Avg MFE={_avg(all_mfe)}  Avg MAE={_avg(all_mae)}")

        print(f"\n  Milestone reach rates (out of {n} trades):")
        print(f"    Reached 0.5R : {n_half:>3} / {n}  ({_pct(n_half, n)})")
        print(f"    Reached 1.0R : {n_1r:>3} / {n}  ({_pct(n_1r, n)})")
        print(f"    Reached 1.5R : {n_15r:>3} / {n}  ({_pct(n_15r, n)})  ← partial TP target")
        print(f"    Reached 2.0R : {n_2r:>3} / {n}  ({_pct(n_2r, n)})")
        print(
            f"\n  Partial TP:  hit={len(tp_hit)}/{n}  ({_pct(len(tp_hit), n)})  "
            f"no-TP avg-MFE={_avg(no_tp_mfe)}  no-TP avg-MAE={_avg(no_tp_mae)}"
        )

        print("\n  Path shape:")
        print(f"    Reached 1R then stalled at <1.5R : {len(stalled_1r):>3} ({_pct(len(stalled_1r), n)})")
        print(f"    Never reached 0.5R               : {len(never_half):>3} ({_pct(len(never_half), n)})")
        print(f"    MAE hit before MFE               : {len(mae_first):>3} ({_pct(len(mae_first), n)})")
        print(f"    Large pre-entry move             : {len(chased):>3} ({_pct(len(chased), n)})")

        if losses:
            l_reached_half = sum(1 for r in losses if r["reached_0_5r"])
            l_reached_1r = sum(1 for r in losses if r["reached_1_0r"])
            ln = len(losses)
            print(f"\n  Losing trades ({ln} total):")
            print(f"    Reached 0.5R before loss : {l_reached_half}/{ln} ({_pct(l_reached_half, ln)})")
            print(f"    Reached 1.0R before loss : {l_reached_1r}/{ln}  ({_pct(l_reached_1r, ln)})")
            print(f"    Avg MFE of losses        : {_avg([r['mfe_r'] for r in losses])}")
            print(f"    Median MFE of losses     : {_med([r['mfe_r'] for r in losses])}")

        move3 = [r["price_move_last_3_candles_before_entry_r"] for r in recs]
        move6 = [r["price_move_last_6_candles_before_entry_r"] for r in recs]
        vr = [r["volume_ratio_at_entry"] for r in recs]
        macdh = [r["macd_histogram_at_entry"] for r in recs]
        mslope = [r["macd_histogram_slope"] for r in recs]
        dist_s = [_swing_distance(r) for r in recs]
        dist_vwap = [r["distance_from_vwap_at_entry_r"] for r in recs]
        print("\n  Entry timing stats:")
        print(f"    Avg 3c pre-move (R)    : {_avg(move3)}  (+ = in trade direction)")
        print(f"    Avg 6c pre-move (R)    : {_avg(move6)}  (+ = in trade direction)")
        print(f"    Avg dist from swing (R): {_avg(dist_s)}")
        print(f"    Avg dist from VWAP (R) : {_avg(dist_vwap)}  (+ = stretched in trade direction)")
        print(f"    Avg volume ratio       : {_avg(vr)}")
        print(f"    Avg MACD histogram     : {_avg(macdh)}")
        print(f"    Avg MACD hist slope    : {_avg(mslope)}  (+ = accelerating)")
        print(
            f"    Winners MACD hist avg  : {_avg([r['macd_histogram_at_entry'] for r in wins])}  "
            f"slope avg: {_avg([r['macd_histogram_slope'] for r in wins])}"
        )
        print(
            f"    Losers  MACD hist avg  : {_avg([r['macd_histogram_at_entry'] for r in losses])}  "
            f"slope avg: {_avg([r['macd_histogram_slope'] for r in losses])}"
        )

        print(f"\n  {'─' * 72}")
        print(f"  BREAKDOWNS  {'label':<32} {'N':>4}  WR  Avg-R  Med-R  AvgMFE  TP%")
        print(f"  {'─' * 72}")

        _row("Winners", wins)
        _row("Losers", losses)
        _row("LONG", [r for r in recs if r["direction"] == "LONG"])
        _row("SHORT", [r for r in recs if r["direction"] == "SHORT"])
        _row("TP hit", tp_hit)
        _row("No TP hit", no_tp)

        def has_signal(rec: dict, name: str) -> bool:
            return name in _signal_names(rec)

        _row("MACD+EMA", [r for r in recs if has_signal(r, "MACD") and has_signal(r, "EMA Cross")])
        _row("MACD+Volume", [r for r in recs if has_signal(r, "MACD") and has_signal(r, "Volume")])
        _row(
            "MACD+VWAP only",
            [
                r
                for r in recs
                if has_signal(r, "MACD")
                and not has_signal(r, "EMA Cross")
                and not has_signal(r, "Volume")
            ],
        )
        _row("WITH Volume confirm", [r for r in recs if has_signal(r, "Volume")])
        _row("WITHOUT Volume confirm", [r for r in recs if not has_signal(r, "Volume")])

        atr_vals = _valid([r["atr_pct_at_entry"] for r in recs])
        if atr_vals:
            atr_med = statistics.median(atr_vals)
            _row(f"HIGH ATR (>={atr_med:.2f}%)", [r for r in recs if (r["atr_pct_at_entry"] or 0.0) >= atr_med])
            _row(f"LOW  ATR (< {atr_med:.2f}%)", [r for r in recs if (r["atr_pct_at_entry"] or 0.0) < atr_med])

        vr_vals = _valid([r["volume_ratio_at_entry"] for r in recs])
        if vr_vals:
            vr_med = statistics.median(vr_vals)
            _row(
                f"HIGH volume ratio (>={vr_med:.2f}x)",
                [r for r in recs if r["volume_ratio_at_entry"] is not None and r["volume_ratio_at_entry"] >= vr_med],
            )
            _row(
                f"LOW  volume ratio (< {vr_med:.2f}x)",
                [r for r in recs if r["volume_ratio_at_entry"] is not None and r["volume_ratio_at_entry"] < vr_med],
            )

        _row("Reached 1R, stalled <1.5R", stalled_1r)
        _row("Never reached 0.5R", never_half)
        _row("MAE before MFE", mae_first)
        _row("Large pre-entry move", chased)
        _print_signal_combo_rows(recs)

    if oos_recs:
        print(f"\n{'━' * width}")
        print("  DIAGNOSTIC ANSWERS  (OOS)")
        print(f"{'━' * width}")

        oos_losses = [r for r in oos_recs if r["exit_r"] <= 0]
        oos_no_tp = [r for r in oos_recs if not r["partial_tp_hit"]]
        chased_oos = [r for r in oos_recs if _is_large_pre_entry_move(r)]
        not_chased_oos = [r for r in oos_recs if not _is_large_pre_entry_move(r)]
        neg_slope_recs = [
            r for r in oos_recs
            if r["macd_histogram_slope"] is not None and r["macd_histogram_slope"] < 0
        ]
        pos_slope_recs = [
            r for r in oos_recs
            if r["macd_histogram_slope"] is not None and r["macd_histogram_slope"] >= 0
        ]

        loss_half = sum(1 for r in oos_losses if r["reached_0_5r"])
        loss_1r = sum(1 for r in oos_losses if r["reached_1_0r"])
        loss_mae_first = sum(1 for r in oos_losses if _is_mae_before_mfe(r))
        ln = len(oos_losses)
        n = len(oos_recs)
        no_tp_mfe_list = [r["mfe_r"] for r in oos_no_tp]
        no_tp_mae_list = [r["mae_r"] for r in oos_no_tp]
        all_mae = [r["mae_r"] for r in oos_recs]
        med_mae = _median_value(all_mae)

        print("\n  Q: Are losing trades failing immediately or first going into profit?")
        print(f"     {loss_half}/{ln} losers ({_pct(loss_half, ln)}) reached 0.5R before loss")
        print(f"     {loss_1r}/{ln} losers ({_pct(loss_1r, ln)}) reached 1.0R before loss")
        print(f"     {loss_mae_first}/{ln} losers ({_pct(loss_mae_first, ln)}) saw MAE before MFE")
        print(f"     Losers avg MFE = {_avg([r['mfe_r'] for r in oos_losses])}")
        if ln:
            immediate = loss_half / ln < 0.35 and loss_mae_first / ln >= 0.5
            print(f"     → Most losers are {'failing quickly' if immediate else 'showing some profit before reversing'}")

        print("\n  Q: Is 1.5R too far, or are entries poor?")
        print(f"     No-TP trades avg MFE-R = {_avg(no_tp_mfe_list)}")
        print(f"     No-TP trades med MFE-R = {_med(no_tp_mfe_list)}")
        print(f"     No-TP trades avg MAE-R = {_avg(no_tp_mae_list)}")
        med_mfe = _median_value(no_tp_mfe_list)
        if med_mfe is not None:
            if med_mfe < 0.5:
                print(f"     → Median no-TP MFE of {med_mfe:.2f}R says entries are poor; price rarely gets moving")
            elif med_mfe < 1.0:
                print(f"     → Median no-TP MFE of {med_mfe:.2f}R says trades move somewhat, but 1.5R is usually too far")
            else:
                print(f"     → Median no-TP MFE of {med_mfe:.2f}R says trades often get traction before reversing short of 1.5R")

        print("\n  Q: Are stops too tight or too wide?")
        stops_close = sum(1 for val in all_mae if val > 0.8)
        print(f"     {stops_close}/{n} trades ({_pct(stops_close, n)}) had MAE > 0.8R")
        print(f"     Avg MAE-R = {_avg(all_mae)}  Median MAE-R = {_med(all_mae)}")
        if med_mae is not None:
            if med_mae > 0.8:
                print("     → Most trades use a large share of the stop; stops do not look obviously too wide")
            elif med_mae < 0.5 and loss_half / max(ln, 1) > 0.4:
                print("     → A chunk of losers reverse after some profit, which hints the stop may be somewhat tight")
            else:
                print("     → Stop width is not the clearest problem; entry quality looks more important")

        print("\n  Q: Are longs or shorts causing most losses?")
        long_losses = [r for r in oos_losses if r["direction"] == "LONG"]
        short_losses = [r for r in oos_losses if r["direction"] == "SHORT"]
        print(f"     Long losses: {len(long_losses)}/{ln}    Short losses: {len(short_losses)}/{ln}")

        print("\n  Q: Are entries happening after price has already moved too far?")
        chased_tp = sum(1 for r in chased_oos if r["partial_tp_hit"])
        non_chased_tp = sum(1 for r in not_chased_oos if r["partial_tp_hit"])
        print(
            f"     Large pre-entry move trades: {len(chased_oos)}/{n} ({_pct(len(chased_oos), n)})  "
            f"TP={_pct(chased_tp, len(chased_oos))}"
        )
        print(
            f"     Not stretched at entry     : {len(not_chased_oos)}/{n} ({_pct(len(not_chased_oos), n)})  "
            f"TP={_pct(non_chased_tp, len(not_chased_oos))}"
        )

        print("\n  Q: Does MACD fire too late?")
        ns_tp = sum(1 for r in neg_slope_recs if r["partial_tp_hit"])
        ps_tp = sum(1 for r in pos_slope_recs if r["partial_tp_hit"])
        print(f"     MACD slope negative (weakening): {len(neg_slope_recs)} trades  TP={_pct(ns_tp, len(neg_slope_recs))}")
        print(f"     MACD slope positive (accel.)   : {len(pos_slope_recs)} trades  TP={_pct(ps_tp, len(pos_slope_recs))}")
        if neg_slope_recs and pos_slope_recs:
            neg_rate = ns_tp / len(neg_slope_recs)
            pos_rate = ps_tp / len(pos_slope_recs)
            print(
                "     → MACD looks "
                + ("late on weakening entries" if neg_rate < pos_rate else "not obviously late from slope alone")
            )

        print("\n  Q: Which signal combination has the highest TP-hit rate?")
        combos: dict[str, list[dict]] = {}
        for rec in oos_recs:
            combos.setdefault(_signal_combo_key(rec), []).append(rec)
        combo_rows = sorted(
            combos.items(),
            key=lambda item: (
                -(sum(1 for rec in item[1] if rec["partial_tp_hit"]) / len(item[1])),
                -len(item[1]),
                item[0],
            ),
        )
        if combo_rows:
            best_key, best_recs = combo_rows[0]
            best_tp = sum(1 for rec in best_recs if rec["partial_tp_hit"])
            print(f"     Best combo: {best_key}  TP={_pct(best_tp, len(best_recs))}  ({best_tp}/{len(best_recs)})")
            for combo, combo_recs in combo_rows:
                combo_tp = sum(1 for rec in combo_recs if rec["partial_tp_hit"])
                print(f"     {combo:<35} {len(combo_recs):>3} trades  TP={_pct(combo_tp, len(combo_recs))}")

    print()


def run_forensics(
    df: pd.DataFrame,
    is_trades: list,
    oos_trades: list,
) -> None:
    """Build forensics records from IS+OOS trades, write CSV, print report."""
    print("\n  [Forensics] Pre-computing indicators…", flush=True)
    ind = _precompute(df)

    is_recs = [record for trade in is_trades if (record := _build_record(trade, "IS", df, ind))]
    oos_recs = [record for trade in oos_trades if (record := _build_record(trade, "OOS", df, ind))]

    _write_csv(is_recs + oos_recs)
    _print_summary(is_recs, oos_recs)
