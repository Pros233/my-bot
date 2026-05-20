"""
report_comparative.py — CLI tool: live vs shadow comparative report.

Usage:
  python3 report_comparative.py              # 30-day report
  python3 report_comparative.py --days 7    # 7-day report
  python3 report_comparative.py --days 90   # 90-day report
  python3 report_comparative.py --json      # machine-readable JSON output

Prints a markdown-style table comparing live vs shadow performance
per engine and flags outperforming shadow strategies.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone


def _pct(v: float | None, suffix: str = "%") -> str:
    if v is None:
        return "—"
    return f"{v:+.3f}{suffix}"


def _num(v, fmt: str = ".0f") -> str:
    if v is None:
        return "—"
    return f"{v:{fmt}}"


def _bar(score: float, width: int = 20) -> str:
    filled = int(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


def run_report(days: int = 30, as_json: bool = False) -> None:
    try:
        import shadow_analytics as sa
        import confidence_score as cs
        import anomaly_detector as ad
    except ImportError as e:
        print(f"Error: could not import required modules — {e}", file=sys.stderr)
        sys.exit(1)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    comparison = sa.get_live_vs_shadow(days)
    outperformers = sa.get_outperforming_shadows(days)
    anomalies = ad.get_active_anomalies()
    conf_summary = cs.get_confidence_summary()

    if as_json:
        print(json.dumps({
            "generated_at":  now,
            "days":          days,
            "comparison":    comparison,
            "outperformers": outperformers,
            "anomalies":     anomalies,
            "confidence":    conf_summary,
        }, indent=2))
        return

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  BTC Bot — Live vs Shadow Comparative Report")
    print(f"  Generated: {now}  |  Period: last {days} days")
    print(f"{'='*72}\n")

    # ── Confidence score ──────────────────────────────────────────────────────
    score = conf_summary.get("score", 0.0)
    state = conf_summary.get("state", "?")
    scale = conf_summary.get("risk_scale", 1.0)
    bar   = _bar(score)
    print(f"System Confidence: {score:.0f}/100 [{state}]  risk_scale={scale:.2f}")
    print(f"  [{bar}]")
    comps = conf_summary.get("components", {})
    for k, v in comps.items():
        print(f"  {k:<24} {v:5.1f}")
    print()

    # ── Anomalies ─────────────────────────────────────────────────────────────
    if anomalies:
        print(f"Active Anomalies ({len(anomalies)}):")
        for a in anomalies:
            sev = a.get("severity", "?")
            atype = a.get("anomaly_type", "?")
            sym   = a.get("symbol", "?")
            msg   = a.get("message", "")
            cnt   = a.get("count", 1)
            print(f"  [{sev:8s}] {atype:<16} {sym:<12} {msg}  (×{cnt})")
        print()
    else:
        print("Active Anomalies: none\n")

    # ── Live vs Shadow table ──────────────────────────────────────────────────
    if not comparison:
        print("No live or shadow trade data available for this period.\n")
        return

    print(f"{'Engine':<14} {'Live T':>6} {'Live WR':>8} {'Live Exp':>10}"
          f" {'Shad T':>6} {'Shad WR':>8} {'Shad Exp%':>10}"
          f" {'Delta':>8} {'Flag':>6}")
    print("-" * 80)

    for eng, data in sorted(comparison.items()):
        live   = data.get("live",   {})
        shadow = data.get("shadow", {})
        delta  = data.get("delta_expectancy")
        flag   = "★ OUT" if data.get("outperforms") else ""

        l_t  = _num(live.get("trades"),   ".0f")
        l_wr = _pct(live.get("win_rate") * 100 if live.get("win_rate") is not None else None, "")
        l_ex = _pct(live.get("expectancy"), "")

        s_t  = _num(shadow.get("trades"),  ".0f")
        s_wr = _pct(shadow.get("win_rate") * 100 if shadow.get("win_rate") is not None else None, "")
        s_ex = _pct(shadow.get("expectancy_pct"), "")

        delta_str = _pct(delta, "") if delta is not None else "—"

        print(f"{eng:<14} {l_t:>6} {l_wr:>8} {l_ex:>10}"
              f" {s_t:>6} {s_wr:>8} {s_ex:>10}"
              f" {delta_str:>8} {flag:>6}")

    print()

    # ── Outperformer highlight ────────────────────────────────────────────────
    if outperformers:
        print(f"Outperforming Shadow Strategies ({len(outperformers)}):")
        for item in outperformers:
            eng   = item.get("engine", "?")
            delta = item.get("delta_expectancy", 0.0)
            s     = item.get("shadow", {})
            print(f"  ★  {eng:<14}  shadow_exp={s.get('expectancy_pct',0):+.4f}%  "
                  f"delta={delta:+.4f}  shadow_trades={s.get('trades',0)}")
        print("\n  → Consider promoting these to live testing.\n")
    else:
        print("No shadow strategy outperforms live in this period.\n")

    print(f"{'='*72}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live vs Shadow comparative report for BTC bot."
    )
    parser.add_argument("--days", type=int, default=30,
                        help="Lookback period in days (default: 30)")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of text table")
    args = parser.parse_args()
    run_report(days=args.days, as_json=args.json)


if __name__ == "__main__":
    main()
