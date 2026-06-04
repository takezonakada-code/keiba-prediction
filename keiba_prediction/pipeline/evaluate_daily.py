"""
日次・週次・月次・累計の的中率・ROIを計算してDBに保存する。

毎朝5:30のメインパイプラインから呼ばれる。
"""
from __future__ import annotations

import json
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import get_conn
from pipeline.predict_dual import _predict_one_race
from features.race_chaos import compute_chaos_score


# ──────────────────────────────────────────────────
# DBスキーマ初期化
# ──────────────────────────────────────────────────
PERFORMANCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS race_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    race_date       TEXT NOT NULL,
    race_id         TEXT NOT NULL,
    track           TEXT,
    race_no         INTEGER,
    race_name       TEXT,
    race_class      TEXT,
    chaos_score     INTEGER,
    result_combo    TEXT,          -- "1-3-7" 形式（3連複的中組み合わせ）
    payout_trio     INTEGER,       -- 3連複払戻金額（円）
    payout_trifecta INTEGER,       -- 3連単払戻金額（円）
    popular_rank    INTEGER,       -- 3連複の人気順位
    a_combos        TEXT,          -- JSON: システムAの予測コンボリスト
    b_combos        TEXT,          -- JSON: システムBの予測コンボリスト
    a_hit           INTEGER,       -- 0/1
    b_hit           INTEGER,       -- 0/1
    a_payout        INTEGER,       -- システムA的中時の払戻
    b_payout        INTEGER,       -- システムB的中時の払戻
    a_stake         INTEGER,       -- システムA投資額（点数×100円）
    b_stake         INTEGER,       -- システムB投資額（点数×100円）
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(race_id)
);

CREATE TABLE IF NOT EXISTS daily_performance (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    perf_date       TEXT NOT NULL UNIQUE,
    track_filter    TEXT DEFAULT 'ALL',
    total_races     INTEGER,
    a_races         INTEGER,
    a_hits          INTEGER,
    a_hit_rate      REAL,
    a_invest        INTEGER,
    a_payout        INTEGER,
    a_roi           REAL,
    b_races         INTEGER,
    b_hits          INTEGER,
    b_hit_rate      REAL,
    b_invest        INTEGER,
    b_payout        INTEGER,
    b_roi           REAL,
    chaos_high_races  INTEGER,
    chaos_high_hits   INTEGER,
    best_combo      TEXT,
    best_payout     INTEGER,
    best_race       TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS cumulative_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of_date      TEXT NOT NULL UNIQUE,
    total_days      INTEGER,
    total_races     INTEGER,
    a_total         INTEGER,
    a_hits          INTEGER,
    a_hit_rate      REAL,
    a_roi           REAL,
    b_total         INTEGER,
    b_hits          INTEGER,
    b_hit_rate      REAL,
    b_roi           REAL,
    sharpe_weekly   REAL,
    max_drawdown    REAL,
    best_payout     INTEGER,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS hit_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    race_date       TEXT NOT NULL,
    race_id         TEXT NOT NULL,
    track           TEXT,
    race_no         INTEGER,
    race_name       TEXT,
    result_combo    TEXT,
    payout          INTEGER,
    popular_rank    INTEGER,
    chaos_score     INTEGER,
    system          TEXT,          -- 'A' or 'B'
    predicted_combo TEXT,
    shap_json       TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(race_id, system)
);
"""


def init_performance_tables():
    with get_conn() as conn:
        conn.executescript(PERFORMANCE_SCHEMA)


# ──────────────────────────────────────────────────
# 日次評価のメイン関数
# ──────────────────────────────────────────────────
def evaluate_date(target_date: date) -> dict:
    """
    指定日の全レースについて予測と結果を照合し、
    パフォーマンス指標を計算してDBに保存する。
    """
    init_performance_tables()
    date_str = target_date.isoformat()
    print(f"[evaluate] {date_str} の評価開始")

    with get_conn() as conn:
        races = conn.execute("""
            SELECT * FROM nar_races
            WHERE race_date=? AND race_type!='banei'
            ORDER BY track, race_no
        """, (date_str,)).fetchall()

    if not races:
        print(f"[evaluate] {date_str}: レースなし")
        return {}

    results = []
    a_invest = a_payout_total = 0
    b_invest = b_payout_total = 0
    chaos_high_races = chaos_high_hits = 0
    best_pay = 0
    best_combo_str = best_race_str = ""

    for race_row in races:
        rd = dict(race_row)
        rid = rd["race_id"]

        # ── 結果取得 ─────────────────────────────
        actual = _get_actual_result(rid)
        if actual is None:
            continue

        result_combo = actual["combo"]
        payout_trio  = actual.get("payout_trio", 0)

        # ── 予測生成 ──────────────────────────────
        pred = _predict_one_race(rd, date_str)
        if not pred:
            continue

        chaos = pred["chaos"]["chaos_score"]
        a_combos = [t["combo"] for t in pred["system_a"]]
        b_combos = [t["combo"] for t in pred["system_b"]]

        a_hit = int(result_combo in a_combos)
        b_hit = int(result_combo in b_combos and bool(b_combos))

        a_inv  = len(a_combos) * 100
        b_inv  = len(b_combos) * 100 if b_combos else 0
        a_pay  = payout_trio if a_hit else 0
        b_pay  = payout_trio if b_hit else 0

        a_invest        += a_inv
        a_payout_total  += a_pay
        b_invest        += b_inv
        b_payout_total  += b_pay

        if chaos >= 60:
            chaos_high_races += 1
            if a_hit or b_hit:
                chaos_high_hits += 1

        if payout_trio > best_pay and (a_hit or b_hit):
            best_pay      = payout_trio
            best_combo_str = result_combo
            best_race_str  = f"{rd['track']}{rd['race_no']}R {rd.get('race_name','')}"

        # race_results に保存
        _save_race_result(rd, actual, pred, a_hit, b_hit, a_inv, b_inv, a_pay, b_pay, chaos)

        # 的中事例に追加
        if a_hit:
            _save_hit_record(rd, actual, result_combo, "A", a_combos[0] if a_combos else "", chaos, pred)
        if b_hit:
            _save_hit_record(rd, actual, result_combo, "B", b_combos[0] if b_combos else "", chaos, pred)

        results.append({
            "race_id": rid, "a_hit": a_hit, "b_hit": b_hit,
            "a_inv": a_inv, "b_inv": b_inv,
            "a_pay": a_pay, "b_pay": b_pay, "chaos": chaos,
        })

    n = len(results)
    a_hits = sum(r["a_hit"] for r in results)
    b_hits = sum(r["b_hit"] for r in results)
    a_roi  = a_payout_total / a_invest * 100 if a_invest > 0 else 0
    b_roi  = b_payout_total / b_invest * 100 if b_invest > 0 else 0

    perf = {
        "perf_date": date_str,
        "total_races": n,
        "a_races": n, "a_hits": a_hits,
        "a_hit_rate": a_hits / n if n > 0 else 0,
        "a_invest": a_invest, "a_payout": a_payout_total, "a_roi": a_roi,
        "b_races": len([r for r in results if r["b_inv"] > 0]),
        "b_hits": b_hits,
        "b_hit_rate": b_hits / max(len([r for r in results if r["b_inv"] > 0]), 1),
        "b_invest": b_invest, "b_payout": b_payout_total, "b_roi": b_roi,
        "chaos_high_races": chaos_high_races, "chaos_high_hits": chaos_high_hits,
        "best_combo": best_combo_str, "best_payout": best_pay, "best_race": best_race_str,
    }

    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO daily_performance
              (perf_date, total_races, a_races, a_hits, a_hit_rate,
               a_invest, a_payout, a_roi,
               b_races, b_hits, b_hit_rate, b_invest, b_payout, b_roi,
               chaos_high_races, chaos_high_hits,
               best_combo, best_payout, best_race)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (date_str, n, n, a_hits, a_hits/n if n else 0,
              a_invest, a_payout_total, a_roi,
              perf["b_races"], b_hits, perf["b_hit_rate"],
              b_invest, b_payout_total, b_roi,
              chaos_high_races, chaos_high_hits,
              best_combo_str, best_pay, best_race_str))

    _update_cumulative(date_str)
    print(f"[evaluate] {date_str}: {n}R | A {a_hits}/{n}={a_hits/n*100:.0f}% ROI{a_roi:.0f}% | "
          f"B {b_hits}/{perf['b_races']}R ROI{b_roi:.0f}%")
    return perf


def evaluate_yesterday() -> dict:
    """前日の評価（5:30メインパイプラインから呼ばれる）。"""
    return evaluate_date(date.today() - timedelta(days=1))


# ──────────────────────────────────────────────────
# 累計統計更新
# ──────────────────────────────────────────────────
def _update_cumulative(as_of_date: str) -> None:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT a_invest, a_payout, b_invest, b_payout,
                   a_hits, a_races, b_hits, b_races, total_races,
                   best_payout
            FROM daily_performance
            ORDER BY perf_date
        """).fetchall()

    if not rows:
        return

    a_inv   = sum(r["a_invest"]  or 0 for r in rows)
    a_pay   = sum(r["a_payout"]  or 0 for r in rows)
    b_inv   = sum(r["b_invest"]  or 0 for r in rows)
    b_pay   = sum(r["b_payout"]  or 0 for r in rows)
    a_hits  = sum(r["a_hits"]    or 0 for r in rows)
    a_total = sum(r["a_races"]   or 0 for r in rows)
    b_hits  = sum(r["b_hits"]    or 0 for r in rows)
    b_total = sum(r["b_races"]   or 0 for r in rows)
    best    = max((r["best_payout"] or 0) for r in rows)

    a_roi = a_pay / a_inv * 100 if a_inv else 0
    b_roi = b_pay / b_inv * 100 if b_inv else 0

    # 週次Sharpe（日次PnLから計算）
    pnl = np.array([(r["a_payout"] or 0) - (r["a_invest"] or 0) for r in rows], dtype=float)
    sharpe = (pnl.mean() / pnl.std() * math.sqrt(52)) if len(pnl) > 1 and pnl.std() > 0 else 0

    # 最大ドローダウン
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    mdd = float((peak - cum).max()) if len(cum) > 0 else 0

    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO cumulative_stats
              (as_of_date, total_days, total_races,
               a_total, a_hits, a_hit_rate, a_roi,
               b_total, b_hits, b_hit_rate, b_roi,
               sharpe_weekly, max_drawdown, best_payout)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (as_of_date, len(rows), sum(r["total_races"] or 0 for r in rows),
              a_total, a_hits, a_hits/a_total if a_total else 0, a_roi,
              b_total, b_hits, b_hits/b_total if b_total else 0, b_roi,
              sharpe, mdd, best))


# ──────────────────────────────────────────────────
# ヘルパー
# ──────────────────────────────────────────────────
def _get_actual_result(race_id: str) -> dict | None:
    """DBから着順結果を取得。"""
    with get_conn() as conn:
        # まず race_results に保存済みか確認
        r = conn.execute("SELECT * FROM race_results WHERE race_id=?", (race_id,)).fetchone()
        if r and r["result_combo"]:
            return dict(r)

        # nar_results から組み立て
        rows = conn.execute("""
            SELECT draw_number, finish_position FROM nar_results
            WHERE race_id=? AND finish_position BETWEEN 1 AND 3
            ORDER BY finish_position
        """, (race_id,)).fetchall()

        if len(rows) < 3:
            return None

        top3 = [r["draw_number"] for r in rows]
        combo = "-".join(str(n) for n in sorted(top3))

        # 払戻
        pay_row = conn.execute("""
            SELECT payout FROM nar_payouts
            WHERE race_id=? AND bet_type='trio' LIMIT 1
        """, (race_id,)).fetchone()
        payout_trio = pay_row["payout"] if pay_row else 0

    return {"combo": combo, "top3": top3, "payout_trio": payout_trio}


def _save_race_result(rd, actual, pred, a_hit, b_hit, a_inv, b_inv, a_pay, b_pay, chaos):
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO race_results
              (race_date, race_id, track, race_no, race_name, race_class,
               chaos_score, result_combo, payout_trio,
               a_combos, b_combos, a_hit, b_hit,
               a_payout, b_payout, a_stake, b_stake)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            rd["race_date"], rd["race_id"], rd["track"], rd["race_no"],
            rd.get("race_name",""), rd.get("race_class",""), chaos,
            actual["combo"], actual.get("payout_trio", 0),
            json.dumps([t["combo"] for t in pred["system_a"]]),
            json.dumps([t["combo"] for t in pred["system_b"]]),
            a_hit, b_hit, a_pay, b_pay, a_inv, b_inv,
        ))


def _save_hit_record(rd, actual, result_combo, system, predicted_combo, chaos, pred):
    with get_conn() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO hit_records
              (race_date, race_id, track, race_no, race_name,
               result_combo, payout, chaos_score, system, predicted_combo)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            rd["race_date"], rd["race_id"], rd["track"], rd["race_no"],
            rd.get("race_name",""), result_combo,
            actual.get("payout_trio", 0), chaos, system, predicted_combo,
        ))


# ──────────────────────────────────────────────────
# ダッシュボード用データ取得
# ──────────────────────────────────────────────────
def get_dashboard_data(days: int = 30) -> dict:
    """index.html用のROIダッシュボードデータを取得。"""
    with get_conn() as conn:
        daily = conn.execute("""
            SELECT * FROM daily_performance
            ORDER BY perf_date DESC LIMIT ?
        """, (days,)).fetchall()

        cumul = conn.execute("""
            SELECT * FROM cumulative_stats ORDER BY as_of_date DESC LIMIT 1
        """).fetchone()

        hits = conn.execute("""
            SELECT * FROM hit_records ORDER BY race_date DESC LIMIT 20
        """).fetchall()

        # 競馬場別ROI
        track_roi = conn.execute("""
            SELECT rr.track,
                   COUNT(*) as n,
                   SUM(rr.a_hit) as a_hits,
                   SUM(rr.a_stake) as a_inv,
                   SUM(rr.a_payout) as a_pay,
                   ROUND(100.0*SUM(rr.a_payout)/NULLIF(SUM(rr.a_stake),0),1) as roi
            FROM race_results rr
            GROUP BY rr.track
            ORDER BY roi DESC NULLS LAST
        """).fetchall()

        # chaos帯別ROI
        chaos_roi = conn.execute("""
            SELECT
              CASE WHEN chaos_score>=60 THEN '🔥60点+'
                   WHEN chaos_score>=40 THEN '⚡40-59点'
                   ELSE '40点未満' END as band,
              COUNT(*) as n,
              SUM(a_hit) as a_hits,
              ROUND(100.0*SUM(a_payout)/NULLIF(SUM(a_stake),0),1) as roi
            FROM race_results
            GROUP BY band ORDER BY roi DESC NULLS LAST
        """).fetchall()

    return {
        "daily":    [dict(r) for r in daily],
        "cumul":    dict(cumul) if cumul else {},
        "hits":     [dict(r) for r in hits],
        "track_roi": [dict(r) for r in track_roi],
        "chaos_roi": [dict(r) for r in chaos_roi],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYY-MM-DD（省略時=昨日）")
    args = parser.parse_args()
    target = date.fromisoformat(args.date) if args.date else date.today() - timedelta(days=1)
    evaluate_date(target)
