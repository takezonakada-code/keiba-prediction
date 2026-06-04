"""
NAR過去データ一括取得スクレイパー（2022〜2024年）。
途中で止まっても再開できる進捗管理付き。

使い方:
  # テスト（今日の園田・名古屋のみ）
  python -m data.historical_scraper --test

  # 過去3年分（全場）
  python -m data.historical_scraper --start 2022-01-01 --end 2024-12-31

  # 特定場のみ
  python -m data.historical_scraper --start 2022-01-01 --end 2024-12-31 --tracks 46 43
"""
from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from typing import Optional

from data.nar_scraper import NARScraper, TRACK_MAP
from db.database import get_conn


def fetch_date_range(
    start_date: date,
    end_date: date,
    track_filter: Optional[list[str]] = None,
    sleep_between_days: float = 3.0,
) -> None:
    """
    指定期間の全NAR開催データを取得する。

    Parameters
    ----------
    start_date        : 開始日
    end_date          : 終了日（含む）
    track_filter      : 場コードリスト（None=全場）
    sleep_between_days: 日付をまたぐときの追加待機秒数
    """
    scraper = NARScraper(sleep_sec=2.5, max_retry=3)
    current = start_date
    total_days = (end_date - start_date).days + 1
    day_count = 0

    print(f"=== NAR歴史データ取得: {start_date} 〜 {end_date} ===")
    if track_filter:
        names = [TRACK_MAP.get(c, {}).get("name", c) for c in track_filter]
        print(f"    対象場: {', '.join(names)}")

    while current <= end_date:
        day_count += 1
        print(f"\n[{day_count}/{total_days}] {current} を処理中...")

        # 進捗DB確認（この日が完了済みかどうか）
        if _day_completed(current, track_filter):
            print(f"  → スキップ（取得済み）")
            current += timedelta(days=1)
            continue

        try:
            stats = scraper.run_today(current, track_filter=track_filter)
            _mark_day_done(current, track_filter, stats)
        except Exception as e:
            print(f"  [エラー] {current}: {e}")
            _mark_day_error(current, str(e))

        # 日付間の追加待機
        if current < end_date:
            time.sleep(sleep_between_days)

        current += timedelta(days=1)

    print(f"\n=== 完了: {total_days}日分処理 ===")
    print(f"    スクレイピング成功率: {scraper.success_rate:.1%}")


# ──────────────────────────────────────────────────
# 進捗管理
# ──────────────────────────────────────────────────
def _day_completed(target_date: date, track_filter: Optional[list[str]]) -> bool:
    """この日+場の取得が完了済みかDBで確認。"""
    date_str     = target_date.isoformat()
    track_filter = track_filter or []
    key = f"{date_str}|{'_'.join(sorted(track_filter)) if track_filter else 'ALL'}"
    with get_conn() as conn:
        row = conn.execute(
            "SELECT status FROM historical_progress WHERE day_key = ?", (key,)
        ).fetchone()
    return row is not None and row["status"] == "done"


def _mark_day_done(
    target_date: date,
    track_filter: Optional[list[str]],
    stats: dict,
) -> None:
    date_str = target_date.isoformat()
    track_filter = track_filter or []
    key = f"{date_str}|{'_'.join(sorted(track_filter)) if track_filter else 'ALL'}"
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO historical_progress
              (day_key, race_date, status, success_count, failure_count, updated_at)
            VALUES (?, ?, 'done', ?, ?, datetime('now'))
        """, (key, date_str, stats.get("success", 0), stats.get("failure", 0)))


def _mark_day_error(target_date: date, error_msg: str) -> None:
    date_str = target_date.isoformat()
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO historical_progress
              (day_key, race_date, status, error_message, updated_at)
            VALUES (?, ?, 'error', ?, datetime('now'))
        """, (date_str + "|ALL", date_str, error_msg))


def print_progress_summary() -> None:
    """取得進捗サマリーを表示。"""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT status, COUNT(*) as cnt, SUM(success_count) as ok, SUM(failure_count) as ng
            FROM historical_progress
            GROUP BY status
        """).fetchall()
        total_races = conn.execute("SELECT COUNT(*) as cnt FROM nar_races").fetchone()
        total_results = conn.execute("SELECT COUNT(*) as cnt FROM nar_results").fetchone()

    print("\n=== 進捗サマリー ===")
    for r in rows:
        print(f"  {r['status']}: {r['cnt']}日 (成功{r['ok']}件, 失敗{r['ng']}件)")
    print(f"  nar_races テーブル: {total_races['cnt']}件")
    print(f"  nar_results テーブル: {total_results['cnt']}件")


# ──────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    from db.database import init_db
    init_db()

    parser = argparse.ArgumentParser(description="NAR歴史データ取得")
    parser.add_argument("--test",  action="store_true", help="今日の園田(46)・名古屋(43)のみテスト実行")
    parser.add_argument("--start", default="2022-01-01", help="開始日 YYYY-MM-DD")
    parser.add_argument("--end",   default="2024-12-31", help="終了日 YYYY-MM-DD")
    parser.add_argument("--tracks", nargs="+", help="場コード（例: 46 43）")
    parser.add_argument("--status", action="store_true", help="進捗確認のみ")
    args = parser.parse_args()

    if args.status:
        print_progress_summary()
        raise SystemExit(0)

    if args.test:
        print("=== テスト実行: 今日の園田(50)・名古屋(48) ===")
        scraper = NARScraper(sleep_sec=2.5)
        for track_code in ["50", "48"]:
            track_name = TRACK_MAP[track_code]["name"]
            print(f"\n--- {track_name} ---")
            stats = scraper.run_today(track_filter=[track_code])
            print(f"  結果: {stats}")
        print_progress_summary()
        raise SystemExit(0)

    # 通常実行
    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)
    tracks = args.tracks or None

    fetch_date_range(start, end, track_filter=tracks)
    print_progress_summary()
