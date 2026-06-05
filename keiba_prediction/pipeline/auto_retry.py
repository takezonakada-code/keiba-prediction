"""
「出走表取得中」レースの自動リトライ処理。

未完了レース（nar_results/nar_entries が0頭）を検出し、
複数の取得手段で順番にリトライする。

実行方法:
  python -m pipeline.auto_retry            # 本日分をリトライ
  python -m pipeline.auto_retry --date 2026-06-05
  python -m pipeline.auto_retry --watch    # 30分ごとに監視・リトライ
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import get_conn
from data.nar_scraper import NARScraper


# ──────────────────────────────────────────────────
# 未完了レースの検出
# ──────────────────────────────────────────────────
def find_incomplete_races(target_date: str) -> list[dict]:
    """
    nar_races に存在するが nar_results も nar_entries も 0頭 のレースを返す。
    kg_ プレフィックスの不正IDも検出する。
    """
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT rc.race_id, rc.track, rc.race_no, rc.field_size,
                   rc.race_id LIKE 'kg_%%' as is_invalid_id,
                   COALESCE((SELECT COUNT(*) FROM nar_results nr WHERE nr.race_id=rc.race_id), 0) as res_cnt,
                   COALESCE((SELECT COUNT(*) FROM nar_entries ne WHERE ne.race_id=rc.race_id), 0) as ent_cnt
            FROM nar_races rc
            WHERE rc.race_date=? AND rc.race_type != 'banei'
            ORDER BY rc.track, rc.race_no
        """, (target_date,)).fetchall()

    incomplete = [dict(r) for r in rows
                  if (r["res_cnt"] == 0 and r["ent_cnt"] == 0) or r["is_invalid_id"]]
    return incomplete


# ──────────────────────────────────────────────────
# kg_ 不正IDの修復
# ──────────────────────────────────────────────────
def fix_invalid_ids(target_date: str) -> int:
    """
    kg_ プレフィックスの race_id を削除し、正規IDに統合する。
    Returns: 削除件数
    """
    with get_conn() as conn:
        normal = conn.execute(
            "SELECT COUNT(*) as c FROM nar_races WHERE race_date=? AND race_id NOT LIKE 'kg_%'",
            (target_date,)
        ).fetchone()["c"]

        kg = conn.execute(
            "SELECT COUNT(*) as c FROM nar_races WHERE race_date=? AND race_id LIKE 'kg_%'",
            (target_date,)
        ).fetchone()["c"]

        if kg == 0:
            return 0

        if normal >= 10:
            # 正規IDが十分あればkg_を削除
            for tbl in ("nar_results", "nar_entries", "nar_races"):
                conn.execute(
                    f"DELETE FROM {tbl} WHERE race_date=? AND race_id LIKE 'kg_%'",
                    (target_date,)
                )
            print(f"  kg_プレフィックスID {kg}件を削除（正規ID {normal}件あり）")
            return kg
        else:
            print(f"  kg_ID {kg}件あるが正規ID {normal}件のみ → スキップ")
            return 0


# ──────────────────────────────────────────────────
# entries → results にコピー
# ──────────────────────────────────────────────────
def copy_entries_to_results(target_date: str) -> int:
    """nar_entries のデータを nar_results にコピーして馬データを補完する。"""
    with get_conn() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO nar_results
              (race_id, race_date, horse_id, horse_name, draw_number, frame_number,
               jockey_id, jockey_name, trainer_id, horse_weight, horse_weight_diff,
               win_odds, popular_rank, race_type)
            SELECT race_id, race_date, horse_id, horse_name, draw_number, frame_number,
                   jockey_id, jockey_name, trainer_id, horse_weight, horse_weight_diff,
                   win_odds, popular_rank, race_type
            FROM nar_entries WHERE race_date=?
        """, (target_date,))

        n = conn.execute(
            "SELECT COUNT(*) as c FROM nar_results WHERE race_date=?", (target_date,)
        ).fetchone()["c"]
    return n


# ──────────────────────────────────────────────────
# メインリトライ処理
# ──────────────────────────────────────────────────
def retry_incomplete(target_date: date) -> dict:
    """
    未完了レースを検出して全手段でリトライする。

    手順:
    1. kg_IDを修復
    2. netkeiba で再スクレイプ（出馬表）
    3. entries → results にコピー
    4. 予測を再生成して index.html を更新
    """
    date_str = target_date.isoformat()
    print(f"\n=== 自動リトライ: {date_str} ===")

    # Step 1: kg_ID修復
    fixed = fix_invalid_ids(date_str)

    # Step 2: 未完了確認
    incomplete = find_incomplete_races(date_str)
    print(f"  未完了レース: {len(incomplete)}件")

    if not incomplete and fixed == 0:
        print("  → 全レース取得済み。リトライ不要。")
        return {"status": "ok", "incomplete": 0}

    # Step 3: netkeiba で再スクレイプ
    if incomplete:
        print(f"  netkeiba 再スクレイプ開始...")
        nar = NARScraper(sleep_sec=2.0, max_retry=5)
        stats = nar.run_today(target_date)
        print(f"  netkeiba: 取得={stats['success']} スキップ={stats['skipped']} 失敗={stats['failure']}")

        # entries → results コピー
        n_results = copy_entries_to_results(date_str)
        print(f"  nar_results: {n_results}頭")

    # Step 4: 再確認
    still_incomplete = find_incomplete_races(date_str)
    print(f"  リトライ後の未完了: {len(still_incomplete)}件")

    # Step 5: 予測再生成
    print("  予測再生成...")
    try:
        from pipeline.predict_dual import run_dual_predict
        results = run_dual_predict(target_date=target_date, push=True)
        no_data = sum(1 for r in results if r.get("no_data"))
        print(f"  予測完了: {len(results)}レース (取得中={no_data})")
    except Exception as e:
        print(f"  予測エラー: {e}")

    return {
        "status": "retried",
        "fixed_kg": fixed,
        "incomplete_before": len(incomplete),
        "incomplete_after": len(still_incomplete),
    }


# ──────────────────────────────────────────────────
# 監視ループ（--watch モード）
# ──────────────────────────────────────────────────
def watch_loop(interval_min: int = 30) -> None:
    """30分ごとに未完了レースをチェックしてリトライする。"""
    print(f"=== 監視モード開始（{interval_min}分間隔）===")
    while True:
        result = retry_incomplete(date.today())
        if result["status"] == "ok" and result.get("incomplete", 0) == 0:
            print("  ✅ 全レース完了。監視継続中...")
        time.sleep(interval_min * 60)


# ──────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    from db.database import init_db
    init_db()

    parser = argparse.ArgumentParser(description="未完了レースの自動リトライ")
    parser.add_argument("--date",  default=None,  help="対象日 YYYY-MM-DD（省略=今日）")
    parser.add_argument("--watch", action="store_true", help="30分ごとに監視")
    parser.add_argument("--interval", type=int, default=30, help="監視間隔(分)")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()

    if args.watch:
        watch_loop(interval_min=args.interval)
    else:
        result = retry_incomplete(target)
        print(f"\n結果: {result}")
