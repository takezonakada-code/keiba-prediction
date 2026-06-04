"""
前日・当日の全レース結果を取得してDBに保存する。
対象: NAR全場（15競馬場）+ JRA（scraper.py経由）

23:00の自動実行で当日結果を取得し、
翌朝5:30のメイン処理で照合・評価に使用する。
"""
from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.nar_scraper import NARScraper
from db.database import get_conn, init_db


def fetch_results_for_date(
    target_date: date,
    force: bool = False,
) -> dict:
    """
    指定日の全NAR開催レース結果を取得してDBに保存する。

    Parameters
    ----------
    target_date : 取得対象日
    force       : True の場合は取得済みでも再取得

    Returns
    -------
    dict: {total, success, failure, skipped}
    """
    date_str = target_date.isoformat()
    print(f"[fetch_results] {date_str} の結果取得開始")

    init_db()
    scraper = NARScraper(sleep_sec=2.0, max_retry=3)

    # 対象race_idを取得
    with get_conn() as conn:
        race_ids = [r["race_id"] for r in conn.execute(
            "SELECT race_id FROM nar_races WHERE race_date = ? AND race_type != 'banei'",
            (date_str,)
        ).fetchall()]

    if not race_ids:
        # netkeibaからその日のレース一覧を取得
        race_ids = scraper.fetch_race_ids(target_date)

    stats = {"total": len(race_ids), "success": 0, "failure": 0, "skipped": 0}

    for race_id in race_ids:
        # 結果取得済みかチェック
        if not force:
            with get_conn() as conn:
                exists = conn.execute(
                    "SELECT 1 FROM nar_results WHERE race_id=? AND finish_position IS NOT NULL LIMIT 1",
                    (race_id,)
                ).fetchone()
            if exists:
                stats["skipped"] += 1
                continue

        ok = scraper.scrape_race(race_id, scrape_result=True, skip_if_exists=(not force))
        if ok:
            stats["success"] += 1
        else:
            stats["failure"] += 1
        time.sleep(0.5)

    print(f"[fetch_results] 完了: 取得{stats['success']} / スキップ{stats['skipped']} / 失敗{stats['failure']}")
    return stats


def fetch_yesterday_results() -> dict:
    """前日の結果を取得する（23:00自動実行用）。"""
    yesterday = date.today() - timedelta(days=1)
    return fetch_results_for_date(yesterday)


def fetch_today_results() -> dict:
    """本日の結果を取得する（翌朝の照合に使用）。"""
    return fetch_results_for_date(date.today())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="YYYY-MM-DD（省略時=昨日）")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    target = date.fromisoformat(args.date) if args.date else date.today() - timedelta(days=1)
    fetch_results_for_date(target, force=args.force)
