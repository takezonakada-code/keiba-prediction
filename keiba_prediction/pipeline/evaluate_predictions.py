"""
週次評価パイプライン: ROI / ドローダウン / Racing Sharpe を集計してDB保存。
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from db.database import fetch_bet_results, get_conn
from evaluation.metrics import (
    compute_max_drawdown,
    compute_racing_sharpe,
    compute_roi,
    weekly_roi_series,
)


def evaluate_weekly(target_week_start: date | None = None) -> dict:
    """
    先週分のベット結果を集計して返す。

    Returns
    -------
    dict: {roi_pct, sharpe, max_drawdown_yen, hit_rate, tickets}
    """
    if target_week_start is None:
        today = date.today()
        target_week_start = today - timedelta(days=today.weekday() + 7)

    week_end = target_week_start + timedelta(days=6)
    date_start = target_week_start.isoformat()
    date_end   = week_end.isoformat()

    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM bet_results WHERE race_date BETWEEN ? AND ?",
            (date_start, date_end),
        ).fetchall()

    if not rows:
        print(f"[評価] {date_start}〜{date_end}: データなし")
        return {}

    df = pd.DataFrame([dict(r) for r in rows])
    stakes  = df["stake"].values
    payouts = df["payout"].values

    roi    = compute_roi(stakes, payouts)
    sharpe = compute_racing_sharpe(stakes, payouts)
    mdd    = compute_max_drawdown(stakes, payouts)
    hits   = int(df["is_hit"].sum())
    tickets = len(df)

    result = {
        "week":            f"{date_start}〜{date_end}",
        "roi_pct":         round(roi, 1),
        "sharpe":          round(sharpe, 3),
        "max_drawdown_yen": round(mdd, 0),
        "hit_rate":        round(100.0 * hits / tickets, 1),
        "tickets":         tickets,
        "hits":            hits,
    }

    print(f"[週次評価] {result}")

    # モデル評価ログに保存
    _save_weekly_eval(result)
    return result


def evaluate_cumulative() -> dict:
    """累計ROI・Sharpeを計算して返す。"""
    results = fetch_bet_results()
    if not results:
        return {}

    df = pd.DataFrame(results)
    stakes  = df["stake"].values
    payouts = df["payout"].values

    return {
        "roi_pct":         round(compute_roi(stakes, payouts), 1),
        "sharpe":          round(compute_racing_sharpe(stakes, payouts), 3),
        "max_drawdown_yen": round(compute_max_drawdown(stakes, payouts), 0),
        "tickets":         len(df),
        "hits":            int(df["is_hit"].sum()),
    }


def _save_weekly_eval(result: dict) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO model_eval_logs (test_month, roi_pct, sharpe, n_test)
            VALUES (?, ?, ?, ?)
            """,
            (result["week"], result["roi_pct"], result["sharpe"], result["tickets"]),
        )


if __name__ == "__main__":
    evaluate_weekly()
    print("\n--- 累計 ---")
    print(evaluate_cumulative())
