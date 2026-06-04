"""
NAR + JRA 歴史データ一括取得スクレイパー。
途中で止まっても再開できる進捗管理付き。
50,000件達成ごとに特徴量計算・モデル再学習を自動実行。

使い方:
  python -m data.historical_scraper --status
  python -m data.historical_scraper --start 2023-01-01 --end 2026-06-03
  python -m data.historical_scraper --start 2023-01-01 --end 2026-06-03 --jra
  python -m data.historical_scraper --test
"""
from __future__ import annotations

import argparse
import queue
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from data.nar_scraper import NARScraper, TRACK_MAP
from db.database import get_conn

# 設定
MAX_WORKERS      = 3
SLEEP_SEC        = 1.5   # netkeiba負荷配慮（最低1.5秒）
MILESTONE_RACES  = 50_000  # この件数ごとに特徴量再計算


# ──────────────────────────────────────────────────
# DB初期化（拡張スキーマ）
# ──────────────────────────────────────────────────
def init_extended_schema() -> None:
    schema_path = Path(__file__).parent.parent / "db" / "schema_extended.sql"
    if not schema_path.exists():
        return
    with get_conn() as conn:
        conn.executescript(schema_path.read_text(encoding="utf-8"))


# ──────────────────────────────────────────────────
# 並列取得（NAR）
# ──────────────────────────────────────────────────
def fetch_date_range_nar(
    start_date: date,
    end_date:   date,
    track_filter: Optional[list[str]] = None,
    workers:      int = MAX_WORKERS,
) -> None:
    workers = min(workers, MAX_WORKERS)

    # 未完了の日付リスト
    all_dates = [
        d for d in _date_range(start_date, end_date)
        if not _day_completed("NAR", d, track_filter)
    ]
    skip = (end_date - start_date).days + 1 - len(all_dates)

    print(f"\n=== NAR取得: {start_date}〜{end_date} ===")
    print(f"    対象: {len(all_dates)}日 / スキップ: {skip}日")
    print(f"    ワーカー: {workers}並列 / 待機: {SLEEP_SEC}秒")
    if track_filter:
        names = [TRACK_MAP.get(c,{}).get("name",c) for c in track_filter]
        print(f"    場フィルタ: {', '.join(names)}")

    if not all_dates:
        print("=== 全日付取得済み ===")
        return

    dq: queue.Queue[date] = queue.Queue()
    for d in all_dates: dq.put(d)

    lock    = threading.Lock()
    cnt     = {"done": 0, "fail": 0, "total": len(all_dates)}
    t_start = time.time()
    prev_milestone = [_get_total_races()]

    def worker(wid: int) -> None:
        scraper = NARScraper(sleep_sec=SLEEP_SEC, max_retry=5)
        while True:
            try: target = dq.get_nowait()
            except queue.Empty: break

            t0 = time.time()
            try:
                stats = scraper.run_today(target, track_filter=track_filter)
                _mark_done("NAR", target, track_filter, stats, time.time()-t0)
                with lock:
                    cnt["done"] += 1
                    done  = cnt["done"]
                    total = cnt["total"]
                    elapsed = time.time() - t_start
                    rate  = done / elapsed * 60
                    eta   = (total - done) / (done / elapsed) / 60 if done > 0 else 0
                    print(f"  [W{wid}] {target} ({done}/{total}) {rate:.1f}日/分 残~{eta:.0f}分")

                    # マイルストーン確認
                    cur = _get_total_races()
                    if cur // MILESTONE_RACES > prev_milestone[0] // MILESTONE_RACES:
                        prev_milestone[0] = cur
                        threading.Thread(target=_run_milestone, args=(cur,), daemon=True).start()

            except Exception as e:
                with lock: cnt["fail"] += 1
                print(f"  [W{wid}] {target} エラー: {e}")
                _mark_error("NAR", target, str(e))

            dq.task_done()

    threads = []
    for i in range(workers):
        t = threading.Thread(target=worker, args=(i,), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.5)
    for t in threads: t.join()

    elapsed = (time.time() - t_start) / 60
    print(f"\n=== 完了: {cnt['done']}日成功/{cnt['fail']}日失敗 ({elapsed:.1f}分) ===")


# ──────────────────────────────────────────────────
# 並列取得（JRA）
# ──────────────────────────────────────────────────
def fetch_date_range_jra(
    start_date: date,
    end_date:   date,
    workers:    int = 2,  # JRAは2並列に抑制
) -> None:
    from data.jra_scraper import JRAScraper

    workers = min(workers, 2)

    all_dates = [
        d for d in _date_range(start_date, end_date)
        if not _day_completed("JRA", d, None)
    ]
    skip = (end_date - start_date).days + 1 - len(all_dates)

    print(f"\n=== JRA取得: {start_date}〜{end_date} ===")
    print(f"    対象: {len(all_dates)}日 / スキップ: {skip}日 / {workers}並列")

    if not all_dates: return

    dq: queue.Queue[date] = queue.Queue()
    for d in all_dates: dq.put(d)

    lock    = threading.Lock()
    cnt     = {"done": 0, "fail": 0, "total": len(all_dates)}
    t_start = time.time()

    def worker(wid: int) -> None:
        scraper = JRAScraper(sleep_sec=SLEEP_SEC * 1.5, max_retry=5)
        while True:
            try: target = dq.get_nowait()
            except queue.Empty: break

            t0 = time.time()
            try:
                stats = scraper.run_today(target)
                _mark_done("JRA", target, None, stats, time.time()-t0)
                with lock:
                    cnt["done"] += 1
                    done = cnt["done"]; total = cnt["total"]
                    elapsed = time.time()-t_start
                    rate = done/elapsed*60
                    eta  = (total-done)/(done/elapsed)/60 if done>0 else 0
                    print(f"  [JRA W{wid}] {target} ({done}/{total}) {rate:.1f}日/分 残~{eta:.0f}分")
            except Exception as e:
                with lock: cnt["fail"] += 1
                _mark_error("JRA", target, str(e))

            dq.task_done()

    threads = []
    for i in range(workers):
        t = threading.Thread(target=worker, args=(i,), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(1.0)
    for t in threads: t.join()

    elapsed = (time.time() - t_start) / 60
    print(f"\n=== JRA完了: {cnt['done']}日成功/{cnt['fail']}日失敗 ({elapsed:.1f}分) ===")


# ──────────────────────────────────────────────────
# マイルストーン処理
# ──────────────────────────────────────────────────
def _run_milestone(races_count: int) -> None:
    """50,000件ごとに特徴量計算を実行。"""
    print(f"\n🎯 マイルストーン: {races_count:,}件達成 → 特徴量計算開始")
    try:
        from pipeline.train_pipeline import step2_build_features
        step2_build_features()
        print(f"✅ マイルストーン完了: {races_count:,}件")
    except Exception as e:
        print(f"❌ マイルストーンエラー: {e}")


# ──────────────────────────────────────────────────
# 進捗管理
# ──────────────────────────────────────────────────
def _day_completed(source: str, d: date, tf: Optional[list[str]]) -> bool:
    key = _day_key(source, d.isoformat(), tf)
    with get_conn() as conn:
        row = conn.execute(
            "SELECT status FROM scrape_progress WHERE source=? AND race_date=? AND track=?",
            (source, d.isoformat(), key)
        ).fetchone()
    return row is not None and row["status"] == "done"


def _mark_done(source, target_date, tf, stats, dur):
    key = _day_key(source, str(target_date), tf)
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO scrape_progress
              (source, race_date, track, status,
               races_fetched, results_fetched, payouts_fetched, duration_sec)
            VALUES (?,?,?,?,?,?,?,?)
        """, (source, str(target_date), key, "done",
              stats.get("success",0), stats.get("success",0)*10, 0, dur))


def _mark_error(source, target_date, err):
    key = _day_key(source, str(target_date), None)
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO scrape_progress
              (source, race_date, track, status, error_message)
            VALUES (?,?,?,'error',?)
        """, (source, str(target_date), key, err))


def _day_key(source, date_str, tf):
    return f"{source}|{date_str}|{'_'.join(sorted(tf)) if tf else 'ALL'}"


def _date_range(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _get_total_races() -> int:
    with get_conn() as conn:
        r = conn.execute(
            "SELECT (SELECT COUNT(*) FROM nar_races)+(SELECT COUNT(*) FROM jra_races) as total"
        ).fetchone()
    return r["total"] if r else 0


# ──────────────────────────────────────────────────
# ステータス表示
# ──────────────────────────────────────────────────
def print_status() -> None:
    with get_conn() as conn:
        nar_r  = conn.execute("SELECT COUNT(*) as c FROM nar_races").fetchone()["c"]
        nar_rs = conn.execute("SELECT COUNT(*) as c FROM nar_results").fetchone()["c"]
        jra_r  = conn.execute("SELECT COUNT(*) as c FROM jra_races").fetchone()["c"] if _table_exists("jra_races", conn) else 0
        jra_rs = conn.execute("SELECT COUNT(*) as c FROM jra_results").fetchone()["c"] if _table_exists("jra_results", conn) else 0
        nar_done = conn.execute("SELECT COUNT(*) as c FROM scrape_progress WHERE source='NAR' AND status='done'").fetchone()["c"]
        jra_done = conn.execute("SELECT COUNT(*) as c FROM scrape_progress WHERE source='JRA' AND status='done'").fetchone()["c"]
        hp_done  = conn.execute("SELECT COUNT(*) as c FROM historical_progress WHERE status='done'").fetchone()["c"]
        latest_nar = conn.execute("""
            SELECT race_date FROM scrape_progress WHERE source='NAR' AND status='done'
            ORDER BY race_date DESC LIMIT 1""").fetchone()
        oldest_nar = conn.execute("""
            SELECT race_date FROM scrape_progress WHERE source='NAR' AND status='done'
            ORDER BY race_date ASC LIMIT 1""").fetchone()
        feat_cnt = conn.execute("SELECT COUNT(*) as c FROM training_features").fetchone()["c"] if _table_exists("training_features", conn) else 0

    total_races = nar_r + jra_r
    pct = min(total_races / 50_000 * 100, 100)

    print(f"""
╔══════════════════════════════════════════════╗
║  SANRENPUKU AI — データ取得ステータス
╠══════════════════════════════════════════════╣
║  取得レース数 (NAR): {nar_r:>8,} 件
║  取得着分数  (NAR): {nar_rs:>8,} 件
║  取得レース数 (JRA): {jra_r:>8,} 件
║  取得着分数  (JRA): {jra_rs:>8,} 件
║  ─────────────────────────────────────────
║  合計レース:  {total_races:>8,} 件  [{pct:.0f}% / 50,000目標]
║  特徴量計算: {feat_cnt:>8,} 件
╠══════════════════════════════════════════════╣
║  NAR完了日数: {nar_done:>5} 日  (旧管理: {hp_done}日)
║  JRA完了日数: {jra_done:>5} 日
║  NAR期間: {oldest_nar['race_date'] if oldest_nar else '-':>12} 〜 {latest_nar['race_date'] if latest_nar else '-':>12}
╚══════════════════════════════════════════════╝
""")


def _table_exists(name: str, conn) -> bool:
    return conn.execute(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name='{name}'"
    ).fetchone() is not None


# ──────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from db.database import init_db
    init_db()
    init_extended_schema()

    parser = argparse.ArgumentParser(description="NAR+JRA 歴史データ取得")
    parser.add_argument("--test",    action="store_true", help="今日の園田・名古屋テスト")
    parser.add_argument("--start",   default="2023-01-01")
    parser.add_argument("--end",     default="2026-06-03")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--status",  action="store_true", help="進捗確認")
    parser.add_argument("--jra",     action="store_true", help="JRAも取得")
    parser.add_argument("--tracks",  nargs="+", help="NAR場コード")
    parser.add_argument("--priority", choices=["nar_recent","nar_all","jra","all"],
                        default="nar_all", help="取得優先順序")
    args = parser.parse_args()

    if args.status:
        print_status()
        raise SystemExit(0)

    if args.test:
        print("=== テスト: 今日の園田(50)・名古屋(48) ===")
        scraper = NARScraper(sleep_sec=SLEEP_SEC)
        for code in ["50","48"]:
            name = TRACK_MAP[code]["name"]
            print(f"\n--- {name} ---")
            print(scraper.run_today(track_filter=[code]))
        print_status()
        raise SystemExit(0)

    start   = date.fromisoformat(args.start)
    end     = date.fromisoformat(args.end)
    workers = min(args.workers, MAX_WORKERS)

    if args.priority == "nar_recent":
        # 優先1: NAR直近1年
        recent_start = max(start, date(2025, 6, 1))
        fetch_date_range_nar(recent_start, end, args.tracks, workers)
        # 残り
        if start < recent_start:
            fetch_date_range_nar(start, recent_start - timedelta(days=1), args.tracks, workers)
    elif args.priority == "jra":
        fetch_date_range_jra(start, end, min(workers, 2))
    elif args.priority == "all":
        # NAR先行、JRAは並行スレッドで
        import threading
        jra_thread = threading.Thread(
            target=fetch_date_range_jra,
            args=(start, end, 2),
            daemon=True
        )
        jra_thread.start()
        fetch_date_range_nar(start, end, args.tracks, workers)
        jra_thread.join()
    else:
        # デフォルト: NAR全期間
        fetch_date_range_nar(start, end, args.tracks, workers)
        if args.jra:
            fetch_date_range_jra(start, end, min(workers, 2))

    print_status()
