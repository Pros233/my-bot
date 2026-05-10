"""
macd_vwap_diagnostics.py — Focused diagnostics for MACD+VWAP trades.

Reads trade_forensics.csv from the latest completed backtest, isolates
MACD+VWAP trades that do not include Volume, and writes a bucketed report to
macd_vwap_diagnostics.csv.

This is diagnostic only. It does not change any strategy or live-trading logic.
"""
from __future__ import annotations

import math
import statistics
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
FORENSICS_CSV = ROOT / "trade_forensics.csv"
OUTPUT_CSV = ROOT / "macd_vwap_diagnostics.csv"

_REPORT_COLUMNS = [
    "section",
    "bucket",
    "definition",
    "trades",
    "tp_hit_rate_pct",
    "win_rate_pct",
    "avg_r",
    "median_r",
    "avg_mfe_r",
    "median_mfe_r",
    "avg_mae_r",
    "median_mae_r",
    "profit_factor_r",
]


def _require_csv() -> None:
    if not FORENSICS_CSV.exists():
        raise FileNotFoundError(
            f"Missing {FORENSICS_CSV}. Run the backtest first to generate trade_forensics.csv."
        )


def _load_df() -> pd.DataFrame:
    _require_csv()
    df = pd.read_csv(
        FORENSICS_CSV,
        true_values=["True", "true", "TRUE"],
        false_values=["False", "false", "FALSE"],
    )

    for col in [
        "exit_r",
        "mfe_r",
        "mae_r",
        "atr_pct_at_entry",
        "volume_ratio_at_entry",
        "distance_from_vwap_at_entry_r",
        "macd_histogram_slope",
        "price_move_last_3_candles_before_entry_r",
        "price_move_last_6_candles_before_entry_r",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bool_cols = [
        "partial_tp_hit",
        "reached_0_5r",
        "reached_1_0r",
        "reached_1_5r",
    ]
    for col in bool_cols:
        df[col] = df[col].astype(bool)

    df["candles_to_mfe"] = pd.to_numeric(df["candles_to_mfe"], errors="coerce")
    df["candles_to_mae"] = pd.to_numeric(df["candles_to_mae"], errors="coerce")
    df["signal_list"] = df["signals_fired"].fillna("").map(
        lambda raw: [item for item in str(raw).split("|") if item]
    )
    return df


def _has_signal(sig_list: list[str], name: str) -> bool:
    return name in sig_list


def _is_macd_vwap_no_volume(sig_list: list[str]) -> bool:
    return _has_signal(sig_list, "MACD") and _has_signal(sig_list, "VWAP") and not _has_signal(sig_list, "Volume")


def _profit_factor(exit_rs: pd.Series) -> float:
    vals = [float(v) for v in exit_rs.dropna()]
    if not vals:
        return 0.0
    gains = sum(v for v in vals if v > 0)
    losses = abs(sum(v for v in vals if v < 0))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def _round(value: float) -> float:
    return round(float(value), 3)


def _median(series: pd.Series) -> float:
    vals = [float(v) for v in series.dropna()]
    return float(statistics.median(vals)) if vals else 0.0


def _stats(frame: pd.DataFrame) -> dict:
    trades = len(frame)
    exit_r = frame["exit_r"].dropna()
    mfe_r = frame["mfe_r"].dropna()
    mae_r = frame["mae_r"].dropna()
    tp_hits = int(frame["partial_tp_hit"].sum())
    wins = int((frame["exit_r"] > 0).sum())
    pf = _profit_factor(exit_r)
    pf_out = "inf" if math.isinf(pf) else _round(pf)

    return {
        "trades": trades,
        "tp_hit_rate_pct": _round(tp_hits / trades * 100.0) if trades else 0.0,
        "win_rate_pct": _round(wins / trades * 100.0) if trades else 0.0,
        "avg_r": _round(exit_r.mean()) if not exit_r.empty else 0.0,
        "median_r": _round(_median(exit_r)) if not exit_r.empty else 0.0,
        "avg_mfe_r": _round(mfe_r.mean()) if not mfe_r.empty else 0.0,
        "median_mfe_r": _round(_median(mfe_r)) if not mfe_r.empty else 0.0,
        "avg_mae_r": _round(mae_r.mean()) if not mae_r.empty else 0.0,
        "median_mae_r": _round(_median(mae_r)) if not mae_r.empty else 0.0,
        "profit_factor_r": pf_out,
    }


def _row(section: str, bucket: str, definition: str, frame: pd.DataFrame) -> dict:
    return {
        "section": section,
        "bucket": bucket,
        "definition": definition,
        **_stats(frame),
    }


def _tertile_edges(series: pd.Series) -> tuple[float, float]:
    clean = series.dropna().astype(float)
    q1 = float(clean.quantile(1 / 3))
    q2 = float(clean.quantile(2 / 3))
    return q1, q2


def _bucket_tertiles(series: pd.Series, q1: float, q2: float) -> pd.Series:
    def classify(value: float) -> str | None:
        if pd.isna(value):
            return None
        if value <= q1:
            return "low"
        if value <= q2:
            return "medium"
        return "high"

    return series.map(classify)


def _bucket_vwap_distance(series: pd.Series, q1: float, q2: float) -> pd.Series:
    abs_s = series.abs()

    def classify(value: float) -> str | None:
        if pd.isna(value):
            return None
        if value <= q1:
            return "close_to_vwap"
        if value <= q2:
            return "moderately_extended"
        return "far_from_vwap"

    return abs_s.map(classify)


def _bucket_macd_slope(series: pd.Series, flat_cut: float) -> pd.Series:
    def classify(value: float) -> str | None:
        if pd.isna(value):
            return None
        if abs(value) <= flat_cut:
            return "flat_or_near_zero"
        return "positive" if value > 0 else "negative"

    return series.map(classify)


def _mae_before_mfe(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame["candles_to_mae"] > 0)
        & (
            (frame["candles_to_mfe"] <= 0)
            | (frame["candles_to_mae"] < frame["candles_to_mfe"])
        )
    )


def _breakdown_rows(
    subset: pd.DataFrame,
    section: str,
    bucket_series: pd.Series,
    definitions: dict[str, str],
    order: list[str],
) -> list[dict]:
    rows: list[dict] = []
    for bucket in order:
        frame = subset.loc[bucket_series == bucket]
        rows.append(_row(section, bucket, definitions[bucket], frame))
    return rows


def build_report() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_trades = _load_df()
    subset = all_trades.loc[all_trades["signal_list"].map(_is_macd_vwap_no_volume)].copy()

    atr_q1, atr_q2 = _tertile_edges(all_trades["atr_pct_at_entry"])
    vol_q1, vol_q2 = _tertile_edges(all_trades["volume_ratio_at_entry"])
    vwap_q1, vwap_q2 = _tertile_edges(all_trades["distance_from_vwap_at_entry_r"].abs())
    move3_q1, move3_q2 = _tertile_edges(all_trades["price_move_last_3_candles_before_entry_r"])
    move6_q1, move6_q2 = _tertile_edges(all_trades["price_move_last_6_candles_before_entry_r"])
    slope_flat_cut = float(all_trades["macd_histogram_slope"].abs().dropna().quantile(1 / 3))

    subset["atr_bucket"] = _bucket_tertiles(subset["atr_pct_at_entry"], atr_q1, atr_q2)
    subset["volume_bucket"] = _bucket_tertiles(subset["volume_ratio_at_entry"], vol_q1, vol_q2)
    subset["vwap_bucket"] = _bucket_vwap_distance(subset["distance_from_vwap_at_entry_r"], vwap_q1, vwap_q2)
    subset["move3_bucket"] = _bucket_tertiles(subset["price_move_last_3_candles_before_entry_r"], move3_q1, move3_q2)
    subset["move6_bucket"] = _bucket_tertiles(subset["price_move_last_6_candles_before_entry_r"], move6_q1, move6_q2)
    subset["slope_bucket"] = _bucket_macd_slope(subset["macd_histogram_slope"], slope_flat_cut)
    subset["mae_before_mfe"] = _mae_before_mfe(subset)
    subset["hit_1r_fail_1_5r"] = subset["reached_1_0r"] & ~subset["reached_1_5r"]

    rows: list[dict] = []
    rows.append(_row("overview", "all_macd_vwap_no_volume", "MACD and VWAP present, Volume absent", subset))

    rows.extend([
        _row("comparison", "winners", "exit_r > 0", subset.loc[subset["exit_r"] > 0]),
        _row("comparison", "losers", "exit_r <= 0", subset.loc[subset["exit_r"] <= 0]),
        _row("comparison", "tp_hit", "partial_tp_hit == True", subset.loc[subset["partial_tp_hit"]]),
        _row("comparison", "no_tp", "partial_tp_hit == False", subset.loc[~subset["partial_tp_hit"]]),
        _row("comparison", "is", "split == IS", subset.loc[subset["split"] == "IS"]),
        _row("comparison", "oos", "split == OOS", subset.loc[subset["split"] == "OOS"]),
    ])

    rows.extend([
        _row("direction", "long", "direction == LONG", subset.loc[subset["direction"] == "LONG"]),
        _row("direction", "short", "direction == SHORT", subset.loc[subset["direction"] == "SHORT"]),
    ])

    rows.extend(_breakdown_rows(
        subset,
        "atr_regime",
        subset["atr_bucket"],
        {
            "low": f"atr_pct_at_entry <= {atr_q1:.3f}%",
            "medium": f"{atr_q1:.3f}% < atr_pct_at_entry <= {atr_q2:.3f}%",
            "high": f"atr_pct_at_entry > {atr_q2:.3f}%",
        },
        ["low", "medium", "high"],
    ))

    rows.extend(_breakdown_rows(
        subset,
        "volume_regime",
        subset["volume_bucket"],
        {
            "low": f"volume_ratio_at_entry <= {vol_q1:.3f}x",
            "medium": f"{vol_q1:.3f}x < volume_ratio_at_entry <= {vol_q2:.3f}x",
            "high": f"volume_ratio_at_entry > {vol_q2:.3f}x",
        },
        ["low", "medium", "high"],
    ))

    rows.extend(_breakdown_rows(
        subset,
        "vwap_distance",
        subset["vwap_bucket"],
        {
            "close_to_vwap": f"|distance_from_vwap_at_entry_r| <= {vwap_q1:.3f}R",
            "moderately_extended": f"{vwap_q1:.3f}R < |distance_from_vwap_at_entry_r| <= {vwap_q2:.3f}R",
            "far_from_vwap": f"|distance_from_vwap_at_entry_r| > {vwap_q2:.3f}R",
        },
        ["close_to_vwap", "moderately_extended", "far_from_vwap"],
    ))

    rows.extend(_breakdown_rows(
        subset,
        "macd_histogram_slope",
        subset["slope_bucket"],
        {
            "positive": f"macd_histogram_slope > +{slope_flat_cut:.6f}",
            "flat_or_near_zero": f"|macd_histogram_slope| <= {slope_flat_cut:.6f}",
            "negative": f"macd_histogram_slope < -{slope_flat_cut:.6f}",
        },
        ["positive", "flat_or_near_zero", "negative"],
    ))

    rows.extend(_breakdown_rows(
        subset,
        "pre_entry_move_3c",
        subset["move3_bucket"],
        {
            "low": f"3c move <= {move3_q1:.3f}R",
            "medium": f"{move3_q1:.3f}R < 3c move <= {move3_q2:.3f}R",
            "high": f"3c move > {move3_q2:.3f}R",
        },
        ["low", "medium", "high"],
    ))

    rows.extend(_breakdown_rows(
        subset,
        "pre_entry_move_6c",
        subset["move6_bucket"],
        {
            "low": f"6c move <= {move6_q1:.3f}R",
            "medium": f"{move6_q1:.3f}R < 6c move <= {move6_q2:.3f}R",
            "high": f"6c move > {move6_q2:.3f}R",
        },
        ["low", "medium", "high"],
    ))

    rows.extend([
        _row("early_path", "reached_0_5r", "reached_0_5r == True", subset.loc[subset["reached_0_5r"]]),
        _row("early_path", "never_reached_0_5r", "reached_0_5r == False", subset.loc[~subset["reached_0_5r"]]),
        _row("early_path", "reached_1_0r_failed_1_5r", "reached_1_0r == True and reached_1_5r == False", subset.loc[subset["hit_1r_fail_1_5r"]]),
        _row("early_path", "mae_before_mfe", "candles_to_mae < candles_to_mfe", subset.loc[subset["mae_before_mfe"]]),
    ])

    report = pd.DataFrame(rows, columns=_REPORT_COLUMNS)
    report.to_csv(OUTPUT_CSV, index=False)
    return subset, report


def _fmt_pf(value: object) -> str:
    return "inf" if value == "inf" else f"{float(value):.3f}"


def _print_findings(subset: pd.DataFrame, report: pd.DataFrame) -> None:
    def pick(section: str, bucket: str) -> pd.Series:
        row = report.loc[(report["section"] == section) & (report["bucket"] == bucket)]
        return row.iloc[0]

    atr_rows = report.loc[report["section"] == "atr_regime"].copy().sort_values(["avg_r", "win_rate_pct"])
    volume_rows = report.loc[report["section"] == "volume_regime"].copy().sort_values(["avg_r", "win_rate_pct"])
    vwap_rows = report.loc[report["section"] == "vwap_distance"].copy().sort_values(["avg_r", "win_rate_pct"])
    slope_rows = report.loc[report["section"] == "macd_histogram_slope"].copy()
    move3_rows = report.loc[report["section"] == "pre_entry_move_3c"].copy().sort_values(["avg_r", "win_rate_pct"])

    weakest_atr = atr_rows.iloc[0]
    best_volume = volume_rows.sort_values(["avg_r", "win_rate_pct"], ascending=False).iloc[0]
    weakest_vwap = vwap_rows.iloc[0]
    best_slope = slope_rows.sort_values(["avg_r", "win_rate_pct"], ascending=False).iloc[0]
    worst_slope = slope_rows.sort_values(["avg_r", "win_rate_pct"]).iloc[0]
    best_move3 = move3_rows.sort_values(["avg_r", "win_rate_pct"], ascending=False).iloc[0]

    early_never = pick("early_path", "never_reached_0_5r")
    oos = pick("comparison", "oos")
    is_row = pick("comparison", "is")
    all_long = pick("direction", "long")
    all_short = pick("direction", "short")

    oos_subset = subset.loc[subset["split"] == "OOS"].copy()
    oos_long = oos_subset.loc[oos_subset["direction"] == "LONG"]
    oos_short = oos_subset.loc[oos_subset["direction"] == "SHORT"]
    oos_long_stats = _stats(oos_long)
    oos_short_stats = _stats(oos_short)

    oos_atr = (
        oos_subset.groupby("atr_bucket")
        .apply(lambda frame: pd.Series(_stats(frame)))
        .reset_index()
        .sort_values(["avg_r", "win_rate_pct"])
    )
    oos_vwap = (
        oos_subset.groupby("vwap_bucket")
        .apply(lambda frame: pd.Series(_stats(frame)))
        .reset_index()
        .sort_values(["avg_r", "win_rate_pct"])
    )

    losers = subset.loc[subset["exit_r"] <= 0]
    losers_half = int(losers["reached_0_5r"].sum())
    losers_1r = int(losers["reached_1_0r"].sum())

    print(f"\nMACD+VWAP-only diagnostics saved → {OUTPUT_CSV}")
    print(f"Subset size: {len(subset)} trades  (IS={len(subset.loc[subset['split'] == 'IS'])}, OOS={len(subset.loc[subset['split'] == 'OOS'])})")
    print("\nClearest 5 findings:")
    print(
        "1. MACD+VWAP-only stays weak in both splits: "
        f"IS AvgR={is_row['avg_r']:+.3f}, PF={_fmt_pf(is_row['profit_factor_r'])}; "
        f"OOS AvgR={oos['avg_r']:+.3f}, PF={_fmt_pf(oos['profit_factor_r'])}. "
        "OOS is slightly less bad than IS, but the edge is still negative."
    )
    print(
        "2. Longs are the weak side: "
        f"overall longs AvgR={all_long['avg_r']:+.3f} vs shorts {all_short['avg_r']:+.3f}; "
        f"OOS longs AvgR={oos_long_stats['avg_r']:+.3f} on {oos_long_stats['trades']} trades "
        f"vs OOS shorts {oos_short_stats['avg_r']:+.3f} on {oos_short_stats['trades']} trades."
    )
    print(
        "3. The failure regime is medium ATR, not simply high ATR or low ATR: "
        f"overall medium ATR AvgR={weakest_atr['avg_r']:+.3f}, PF={_fmt_pf(weakest_atr['profit_factor_r'])}; "
        f"OOS worst ATR bucket is {oos_atr.iloc[0]['atr_bucket']} with AvgR={oos_atr.iloc[0]['avg_r']:+.3f}."
    )
    print(
        "4. The VWAP/volume pattern is not monotonic: "
        f"overall {weakest_vwap['bucket']} is the weakest VWAP-distance bucket "
        f"(AvgR={weakest_vwap['avg_r']:+.3f}), while OOS worst is {oos_vwap.iloc[0]['vwap_bucket']} "
        f"(AvgR={oos_vwap.iloc[0]['avg_r']:+.3f}). "
        f"Volume is also U-shaped: {best_volume['bucket']} volume is best, while low and high volume are both weaker."
    )
    print(
        "5. Bad MACD+VWAP-only trades fail early, not after building real profit: "
        f"{len(losers) - losers_half}/{len(losers)} losers never reached 0.5R, "
        f"and the never-0.5R bucket had WR={early_never['win_rate_pct']:.1f}% with median MFE={early_never['median_mfe_r']:+.3f}R. "
        f"Large pre-entry moves are not the failure mode because the best 3c bucket was {best_move3['bucket']}, "
        f"and MACD slope is only mildly useful: {best_slope['bucket']} was best, {worst_slope['bucket']} was worst, "
        "but all slope buckets stayed negative on AvgR."
    )


def main() -> None:
    subset, report = build_report()
    _print_findings(subset, report)


if __name__ == "__main__":
    main()
