"""
NAR過去データ一括取得スクレイパー。
途中で止まっても再開できる進捗管理付き。
3並列ワーカー・1秒待機でnetkeiba BANリスクを最小化。

使い方:
  python -m data.historical_scraper --test
  python -m data.historical_scraper --start 2023-01-01 --end 2026-06-03
  python -m data.historical_scraper --status
"""
from __future__ import annotations

import argparse
import queue
import threading
import time
from datetime import date, timedelta
from typing import Optional

from data.nar_scraper import NARScraper, TRACK_MAP
from db.database import get_conn

# 並列設定（上限: 3ワーカー・1秒待機）
MAX_WORKERS  = 3
SLEEP_SEC    = 1.0   # リクエスト間隔
SLEEP_BETWEEN_DAYS = 0.5  # 日付間の追加待機（ワーカーごとに独立）


# ──────────────────────────────────────────────────
# 並列取得
# ──────────────────────────────────────────────────
def fetch_date_range(
    start_date: date,
    end_date: date,
    track_filter: Optional[list[str]] = None,
    workers: int = MAX_WORKERS,
) -> None:
    """
    指定期間の全NAR開催データを並列取得する。

    Parameters
    ----------
    start_date   : 開始日
    end_date     : 終了日（含む）
    track_filter : 場コードリスト（None=全場）
    workers      : 並列ワーカー数（最大3）
    """
    workers = min(workers, MAX_WORKERS)

    # 未完了の日付リストを構築
    all_dates = []
    current = start_date
    while current <= end_date:
        if not _day_completed(current, track_filter):
            all_dates.append(current)
        current += timedelta(days=1)

    total_days = (end_date - start_date).days + 1
    skip_count = total_days - len(all_dates)

    print(f"=== NAR歴史データ取得: {start_date} 〜 {end_date} ===")
    print(f"    対象: {len(all_dates)}日 / スキップ(取得済み): {skip_count}日")
    print(f"    ワーカー: {workers}並列 / 待機: {SLEEP_SEC}秒/リクエスト")
    if track_filter:
        names = [TRACK_MAP.get(c, {}).get("name", c) for c in track_filter]
        print(f"    場フィルタ: {', '.join(names)}")

    if not all_dates:
        print("=== 全日付取得済み ===")
        return

    # キュー + スレッドセーフカウンター
    date_queue: queue.Queue[date] = queue.Queue()
    for d in all_dates:
        date_queue.put(d)

    lock      = threading.Lock()
    counters  = {"done": 0, "fail": 0, "total": len(all_dates)}
    t_start   = time.time()

    def worker(worker_id: int) -> None:
        """1ワーカーのメインループ。"""
        scraper = NARScraper(sleep_sec=SLEEP_SEC, max_retry=3)
        while True:
            try:
                target = date_queue.get_nowait()
            except queue.Empty:
                break

            try:
                stats = scraper.run_today(target, track_filter=track_filter)
                _mark_day_done(target, track_filter, stats)
                with lock:
                    counters["done"] += 1
                    done  = counters["done"]
                    total = counters["total"]
                    elapsed = time.time() - t_start
                    rate = done / elapsed * 60  # 件/分
                    eta_min = (total - done) / (done / elapsed) / 60 if done > 0 else 0
                    print(f"  [W{worker_id}] {target} 完了 "
                          f"({done}/{total}) "
                          f"速度:{rate:.1f}日/分 "
                          f"残り約{eta_min:.0f}分")
            except Exception as e:
                with lock:
                    counters["fail"] += 1
                print(f"  [W{worker_id}] {target} エラー: {e}")
                _mark_day_error(target, str(e))

            date_queue.task_done()

    # ワーカー起動
    threads = []
    for i in range(workers):
        t = threading.Thread(target=worker, args=(i,), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.3)   # ワーカー起動ずらし（同時リクエスト集中を防ぐ）

    for t in threads:
        t.join()

    elapsed = (time.time() - t_start) / 60
    print(f"\n=== 完了: {counters['done']}日成功 / {counters['fail']}日失敗 "
          f"({elapsed:.1f}分) ===")


# ──────────────────────────────────────────────────
# 進捗管理
# ──────────────────────────────────────────────────
def _day_completed(target_date: date, track_filter: Optional[list[str]]) -> bool:
    date_str = target_date.isoformat()
    key = _day_key(date_str, track_filter)
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
    key = _day_key(date_str, track_filter)
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


def _day_key(date_str: str, track_filter: Optional[list[str]]) -> str:
    tf = track_filter or []
    return f"{date_str}|{'_'.join(sorted(tf)) if tf else 'ALL'}"


def print_progress_summary() -> None:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT status, COUNT(*) as cnt,
                   SUM(success_count) as ok, SUM(failure_count) as ng
            FROM historical_progress
            GROUP BY status
        """).fetchall()
        total_races   = conn.execute("SELECT COUNT(*) as cnt FROM nar_races").fetchone()
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

    parser = argparse.ArgumentParser(description="NAR歴史データ並列取得")
    parser.add_argument("--test",    action="store_true", help="今日の園田・名古屋テスト")
    parser.add_argument("--start",   default="2023-01-01")
    parser.add_argument("--end",     default="2026-06-03")
    parser.add_argument("--tracks",  nargs="+")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"並列ワーカー数（上限{MAX_WORKERS}）")
    parser.add_argument("--status",  action="store_true")
    args = parser.parse_args()

    if args.status:
        print_progress_summary()
        raise SystemExit(0)

    if args.test:
        print("=== テスト実行: 今日の園田(50)・名古屋(48) ===")
        scraper = NARScraper(sleep_sec=SLEEP_SEC)
        for code in ["50", "48"]:
            print(f"\n--- {TRACK_MAP[code]['name']} ---")
            print(scraper.run_today(track_filter=[code]))
        print_progress_summary()
        raise SystemExit(0)

    start   = date.fromisoformat(args.start)
    end     = date.fromisoformat(args.end)
    tracks  = args.tracks or None
    workers = min(args.workers, MAX_WORKERS)

    fetch_date_range(start, end, track_filter=tracks, workers=workers)
    print_progress_summary()
