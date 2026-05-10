"""
analyze_mr_exits.py — Post-run analysis for range MR dynamic exits.

Run after: python3 main.py --mode backtest (with RUN_RANGE_MEAN_REVERSION=True)

Usage:
    python3 analyze_mr_exits.py
"""
from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import monte_carlo as mc

_PF_CAP = 10.0
_MIN_TRADES_FOR_WQ = 5
_LOW_SAMPLE_THRESHOLD = 10

SUMMARY_CSV = Path("range_mr_research_summary.csv")
WINDOWS_CSV = Path("range_mr_research_windows.csv")
TRADES_CSV = Path("range_mr_research_trades.csv")

ALL_EXITS = (
    "MR_EXIT_0", "MR_EXIT_7", "MR_EXIT_8", "MR_EXIT_9",
    "MR_EXIT_10", "MR_EXIT_11", "MR_EXIT_12", "MR_EXIT_13", "MR_EXIT_14",
)
DYNAMIC_EXITS = ALL_EXITS[1:]


def cap_pf(v: object) -> float:
    try:
        f = float(str(v))
        return _PF_CAP if not math.isfinite(f) else min(f, _PF_CAP)
    except (ValueError, TypeError):
        return 0.0


def pf(rs: list[float]) -> float:
    gross = sum(r for r in rs if r > 0)
    loss = abs(sum(r for r in rs if r < 0))
    return min(gross / loss, _PF_CAP) if loss > 0 else _PF_CAP


def _passes_gate(r: dict, win_by: dict) -> tuple[bool, list[str]]:
    n = int(r["total_oos_trades"])
    avg_per_win = float(r["avg_trades_per_window"])
    low_sample = avg_per_win < _LOW_SAMPLE_THRESHOLD

    avg_pf = cap_pf(r["average_oos_pf"])
    med_pf = cap_pf(r["median_oos_pf"])
    avg_is_pf = cap_pf(r["average_is_pf"])
    tp_std = float(r["tp_hit_rate_std_pct"])
    avg_oos_r = float(r["average_oos_avg_r"])
    avg_is_r = float(r["average_is_avg_r"])
    clust = float(r["small_cluster_profit_share"])
    med_r = float(r["median_oos_median_r"])

    combo = r["combo_name"]
    exit_code = r.get("research_exit_code", "")
    setup_windows = win_by.get((combo, exit_code), [])

    wq_pfs = [
        cap_pf(w["oos_profit_factor"])
        for w in setup_windows
        if int(w["oos_trades"]) >= _MIN_TRADES_FOR_WQ
    ] or [cap_pf(w["oos_profit_factor"]) for w in setup_windows]
    wq_pfs.sort()
    wq_mean = statistics.mean(wq_pfs[: max(1, len(wq_pfs) // 4)]) if wq_pfs else 0.0

    wq_threshold = 0.60 if low_sample else 0.80
    tp_threshold = 30.0 if low_sample else 25.0

    sign_ok = avg_is_r == 0.0 or math.copysign(1, avg_is_r) == math.copysign(1, avg_oos_r)
    is_oos_ok = sign_ok and (low_sample or abs(avg_is_pf - avg_pf) <= 0.75)

    criteria = {
        "min_trades": n >= 50,
        "med_pf": med_pf > 1.05,
        "avg_pf": avg_pf > 1.10,
        f"wq_pf≥{wq_threshold}": wq_mean >= wq_threshold,
        "avg_r_pos": avg_oos_r > 0,
        "med_r": med_r > -0.25,
        f"tp_std≤{tp_threshold}%": tp_std <= tp_threshold,
        "is_oos_agree": is_oos_ok,
        "no_cluster": clust <= 0.50,
    }
    fails = [k for k, v in criteria.items() if not v]
    return len(fails) == 0, fails


def main() -> None:
    if not SUMMARY_CSV.exists():
        print(f"[error] {SUMMARY_CSV} not found. Run backtest first.")
        sys.exit(1)

    rows = list(csv.DictReader(open(SUMMARY_CSV)))
    win_rows = list(csv.DictReader(open(WINDOWS_CSV))) if WINDOWS_CSV.exists() else []
    trade_rows = list(csv.DictReader(open(TRADES_CSV))) if TRADES_CSV.exists() else []

    win_by: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for w in win_rows:
        win_by[(w["combo_name"], w.get("research_exit_code", ""))].append(w)

    # ── 1. Exit code coverage ─────────────────────────────────────────────────
    exits_present = sorted(set(r.get("research_exit_code", "N/A") for r in rows))
    print(f"\n{'='*70}")
    print("  RANGE MR DYNAMIC EXIT ANALYSIS")
    print(f"{'='*70}")
    print(f"\n  Exit codes in summary: {exits_present}")
    print(f"  Total rows: {len(rows)}")
    if "MR_EXIT_12" not in exits_present:
        print("  ⚠  MR_EXIT_12/13/14 NOT PRESENT — re-run backtest after registering specs")

    # ── 2. Per-exit aggregated stats (2H RMR setups only) ────────────────────
    rmr2h = [r for r in rows if r["setup_name"].startswith("RMR_2H")]
    by_exit: dict[str, list[dict]] = defaultdict(list)
    for r in rmr2h:
        by_exit[r.get("research_exit_code", "N/A")].append(r)

    print(f"\n{'─'*70}")
    print("  2H RMR SETUPS — AGGREGATE EXIT COMPARISON")
    print(f"{'─'*70}")
    header = f"  {'Exit':<12} {'Setups':>6} {'n_trades':>9} {'avgR':>7} {'WR%':>5} {'PF_p50':>7} {'P(PF>1)':>8} {'Passed':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    exit_summaries: dict[str, dict] = {}
    for code in ALL_EXITS:
        rlist = by_exit.get(code, [])
        if not rlist:
            continue
        all_oos_rs: list[float] = []
        for r in rlist:
            try:
                all_oos_rs.append(float(r["average_oos_avg_r"]))
            except (ValueError, KeyError):
                pass
        n_total = sum(int(r.get("total_oos_trades", 0)) for r in rlist)
        avg_wr = statistics.mean(float(r["oos_win_rate_pct"]) for r in rlist) if rlist else 0.0
        mc_p50 = statistics.median(float(r.get("mc_pf_p50", 0)) for r in rlist) if rlist else 0.0
        mc_prob = statistics.mean(float(r.get("mc_prob_pf_above_1", 0)) for r in rlist) if rlist else 0.0
        n_passed, _ = zip(*[_passes_gate(r, win_by) for r in rlist]) if rlist else ([], [])
        passed_count = sum(n_passed)
        avg_r = statistics.mean(all_oos_rs) if all_oos_rs else 0.0
        print(f"  {code:<12} {len(rlist):>6} {n_total:>9} {avg_r:>7.3f} {avg_wr:>5.0f}% {mc_p50:>7.3f} {mc_prob:>8.2%} {passed_count:>5}/{len(rlist)}")
        exit_summaries[code] = {
            "avg_r": avg_r, "n_total": n_total, "mc_prob": mc_prob, "passed": passed_count
        }

    # ── 3. Best vs baseline per setup ────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  BASELINE vs DYNAMIC EXITS (per-setup winner)")
    print(f"{'─'*70}")
    by_setup: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rmr2h:
        by_setup[r["setup_name"]][r.get("research_exit_code", "")] = r

    baseline_wins = 0
    dynamic_wins: dict[str, int] = defaultdict(int)
    for setup_name, exits in sorted(by_setup.items()):
        if "MR_EXIT_0" not in exits:
            continue
        base = exits["MR_EXIT_0"]
        base_r = float(base.get("average_oos_avg_r", 0))
        best_dyn_code = None
        best_dyn_r = base_r
        for code in DYNAMIC_EXITS:
            if code not in exits:
                continue
            r_val = float(exits[code].get("average_oos_avg_r", -99))
            if r_val > best_dyn_r:
                best_dyn_r = r_val
                best_dyn_code = code
        if best_dyn_code:
            dynamic_wins[best_dyn_code] += 1
            winner = best_dyn_code
            winner_r = best_dyn_r
        else:
            baseline_wins += 1
            winner = "MR_EXIT_0"
            winner_r = base_r
        print(f"  {setup_name[:55]:<55} base={base_r:>6.3f}  best={winner:<10} ({winner_r:>6.3f})")

    print(f"\n  Baseline (MR_EXIT_0) wins: {baseline_wins}")
    for code, n in sorted(dynamic_wins.items(), key=lambda x: -x[1]):
        print(f"  {code} wins: {n}")

    # ── 4. Promotion gate summary ─────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  PROMOTION GATE RESULTS (all RMR_2H setups)")
    print(f"{'─'*70}")
    all_passing: list[dict] = []
    for r in rmr2h:
        passed, fails = _passes_gate(r, win_by)
        if passed:
            all_passing.append(r)

    print(f"\n  PASSED: {len(all_passing)} setups")
    if all_passing:
        # Sort by avgR descending
        all_passing.sort(key=lambda r: -float(r.get("average_oos_avg_r", 0)))
        print(f"\n  {'Setup':<60} {'Exit':<12} {'n':>5} {'avgR':>6} {'WR':>4}")
        print("  " + "-" * 90)
        for r in all_passing:
            print(f"  {r['setup_name'][:60]:<60} {r.get('research_exit_code','?'):<12} "
                  f"{int(r['total_oos_trades']):>5} {float(r['average_oos_avg_r']):>6.3f} "
                  f"{float(r['oos_win_rate_pct']):>3.0f}%")

    # ── 5. MC bootstrap on best candidate ────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  MONTE CARLO ANALYSIS — TOP CANDIDATES")
    print(f"{'─'*70}")
    if all_passing:
        top = all_passing[:3]
        for r in top:
            mc_p05 = float(r.get("mc_pf_p05", 0))
            mc_p50 = float(r.get("mc_pf_p50", 0))
            mc_p95 = float(r.get("mc_pf_p95", 0))
            mc_prob = float(r.get("mc_prob_pf_above_1", 0))
            mc_r_p05 = float(r.get("mc_avg_r_p05", 0))
            mc_r_p95 = float(r.get("mc_avg_r_p95", 0))
            low_s = r.get("mc_low_sample_adaptive", "False") == "True"
            print(f"\n  {r['setup_name']}")
            print(f"    Exit: {r.get('research_exit_code','?')}  n={int(r['total_oos_trades'])}  "
                  f"avgR={float(r['average_oos_avg_r']):.3f}  WR={float(r['oos_win_rate_pct']):.0f}%")
            print(f"    MC PF:   p05={mc_p05:.3f}  p50={mc_p50:.3f}  p95={mc_p95:.3f}")
            print(f"    MC avgR: p05={mc_r_p05:.3f}  p95={mc_r_p95:.3f}")
            print(f"    P(PF>1)={mc_prob:.0%}  low_sample={low_s}")
    else:
        print("  No setups passed promotion gate.")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
