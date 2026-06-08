"""
リアルタイム予測更新エンジン

発走60分前から起動し、
NAR公式サイトのオッズ更新を2分ごとに取得して
予測をリアルタイム更新する。

実行方法:
  python -m pipeline.realtime_engine               # 今日の全レース
  python -m pipeline.realtime_engine --race-id XXX # 特定レース
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import get_conn, init_db

LOG_DIR = Path(__file__).parent.parent.parent / "logs"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {line2 := msg}"
    print(line)
    (LOG_DIR / f"realtime_{date.today().strftime('%Y%m%d')}.log").open("a").write(line + "\n")


# ──────────────────────────────────────────────────
# オッズ取得（NAR公式サイト）
# ──────────────────────────────────────────────────
def fetch_live_odds(race_id: str) -> dict[str, float]:
    """
    NAR公式サイトからリアルタイム単勝オッズを取得する。
    Returns: {draw_number_str: odds}
    """
    import re
    import requests
    from bs4 import BeautifulSoup

    # race_idからkeiba.go.jp形式のURLを構築
    # race_id例: 202650060601 → 2026-06-06 園田(50) 1R
    if len(race_id) != 12:
        return {}

    year  = race_id[0:4]
    nb    = race_id[4:6]    # netkeiba場コード
    month = race_id[6:8]
    day   = race_id[8:10]
    rno   = int(race_id[10:12])

    # netkeiba→keiba.go.jp 場コード変換
    NB_TO_KG = {
        '65':'3','44':'10','45':'11','43':'18','42':'19',
        '35':'20','36':'21','46':'22','47':'23','48':'24',
        '50':'27','51':'28','54':'31','55':'32','30':'36',
    }
    kg_code = NB_TO_KG.get(nb)
    if not kg_code:
        return {}

    date_str = f"{year}%2F{month}%2F{day}"
    url = f"https://www.keiba.go.jp/KeibaWeb/TodayRaceInfo/OddsTop?k_raceDate={date_str}&k_raceNo={rno}&k_babaCode={kg_code}"

    try:
        s = requests.Session()
        s.headers['User-Agent'] = 'Mozilla/5.0'
        r = s.get(url, timeout=10)
        if r.status_code != 200:
            return {}

        soup = BeautifulSoup(r.text, "html.parser")
        odds = {}
        for row in soup.select("table tr"):
            cells = [td.get_text(strip=True) for td in row.select("td")]
            if len(cells) >= 3:
                try:
                    dn    = int(re.sub(r"[^\d]", "", cells[0]))
                    od    = float(cells[1].replace(",", ""))
                    odds[str(dn)] = od
                except (ValueError, IndexError):
                    pass
        return odds
    except Exception:
        return {}


# ──────────────────────────────────────────────────
# 予測更新
# ──────────────────────────────────────────────────
def update_prediction(race_id: str, race_date: str, new_odds: dict) -> bool:
    """最新オッズで予測を再計算してDBを更新する。"""
    if not new_odds:
        return False

    with get_conn() as conn:
        race_row = conn.execute(
            "SELECT * FROM nar_races WHERE race_id=?", (race_id,)
        ).fetchone()
    if not race_row:
        return False

    from pipeline.predict_dual import _predict_one_race
    rd = dict(race_row)

    # nar_resultsのオッズを更新
    with get_conn() as conn:
        for dn_str, odds in new_odds.items():
            conn.execute("""
                UPDATE nar_results SET win_odds=?
                WHERE race_id=? AND draw_number=?
            """, (odds, race_id, int(dn_str)))

        # オッズスナップショット保存
        for dn_str, odds in new_odds.items():
            conn.execute("""
                INSERT OR REPLACE INTO nar_odds_snapshot
                  (race_id, race_date, draw_number, snapshot_type, win_odds)
                VALUES (?, ?, ?, 'live', ?)
            """, (race_id, race_date, int(dn_str), odds))

    return True


# ──────────────────────────────────────────────────
# メインループ
# ──────────────────────────────────────────────────
def run_realtime(target_date: date = None, interval_sec: int = 120) -> None:
    """
    発走前のレースを対象に2分ごとにオッズを更新し、
    予測を再計算してindex.htmlを更新する。
    """
    if target_date is None:
        target_date = date.today()
    date_str = target_date.isoformat()

    init_db()
    log(f"=== リアルタイムエンジン起動: {date_str} (更新間隔: {interval_sec}秒) ===")

    while True:
        now = datetime.now()

        with get_conn() as conn:
            # 未発走レース（発走時刻が未来）を取得
            upcoming = conn.execute("""
                SELECT race_id, track, race_no, post_time
                FROM nar_races
                WHERE race_date=? AND race_type!='banei'
                  AND race_id NOT LIKE 'kg_%'
                ORDER BY race_no
            """, (date_str,)).fetchall()

        updated = 0
        for race in upcoming:
            rid = race["race_id"]
            # オッズ更新
            odds = fetch_live_odds(rid)
            if odds:
                update_prediction(rid, date_str, odds)
                updated += 1
                time.sleep(1.0)   # レース間の待機

        if updated > 0:
            log(f"  オッズ更新: {updated}レース → 予測再生成中...")
            try:
                from pipeline.predict_dual import run_dual_predict
                run_dual_predict(target_date=target_date, push=True)
                log(f"  ✅ サイト更新完了")
            except Exception as e:
                log(f"  ⚠️ 予測エラー: {e}")
        else:
            log(f"  オッズ更新なし（{interval_sec}秒後に再試行）")

        time.sleep(interval_sec)


# ──────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="リアルタイム予測更新エンジン")
    parser.add_argument("--date",     default=None, help="対象日 YYYY-MM-DD")
    parser.add_argument("--interval", type=int, default=120, help="更新間隔(秒)")
    parser.add_argument("--once",     action="store_true", help="1回だけ実行")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()

    if args.once:
        with get_conn() as conn:
            races = conn.execute(
                "SELECT race_id, race_date FROM nar_races WHERE race_date=? AND race_id NOT LIKE 'kg_%'",
                (target.isoformat(),)
            ).fetchall()
        updated = 0
        for r in races:
            odds = fetch_live_odds(r["race_id"])
            if update_prediction(r["race_id"], r["race_date"], odds):
                updated += 1
        print(f"1回実行完了: {updated}レース更新")
    else:
        run_realtime(target, interval_sec=args.interval)
