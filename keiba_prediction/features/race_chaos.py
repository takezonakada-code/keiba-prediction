"""
「荒れやすさ」スコア計算（0〜100点満点）

高配当の条件を点数化して HIGH_CHAOS(60点以上) を検出する。

主要カテゴリ:
  クラス条件     30点
  馬場条件       25点
  オッズ構造     30点（20+10）
  頭数           10点
  レース条件     15点（最大）
  競馬場ボーナス  5点
  合計最大       115点 → 100点に cap
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

# ──────────────────────────────────────────────────
# 閾値
# ──────────────────────────────────────────────────
HIGH_CHAOS = 60   # 高配当チャンス
MID_CHAOS  = 40   # 中配当チャンス


# ──────────────────────────────────────────────────
# クラス正規化
# ──────────────────────────────────────────────────
def normalize_class(race_class: str) -> str:
    if not race_class:
        return "other"
    rc = race_class.strip()
    if "重賞" in rc:                    return "重賞"
    if "オープン" in rc or "OP" in rc: return "オープン"
    if "特別" in rc:                    return "特別"
    m = re.match(r"^([A-Za-z])", rc)
    if m:                               return m.group(1).upper()
    return "other"


def class_base_chaos(race_class: str) -> float:
    """後方互換: 0〜1の荒れ率を返す（旧インターフェース用）。"""
    pts = compute_chaos_score(
        race_class=race_class,
        track_condition="良",
        field_size=10,
        win_odds=None,
        race_no=1,
        total_races=12,
        distance=1500,
        track="名古屋",
    )["chaos_score"]
    return pts / 100.0


# ──────────────────────────────────────────────────
# メインスコアリング
# ──────────────────────────────────────────────────
def compute_chaos_score(
    race_class:      str,
    track_condition: str,
    field_size:      int,
    win_odds:        Optional[np.ndarray],
    race_no:         int,
    total_races:     int   = 12,
    distance:        int   = 1500,
    track:           str   = "",
    front_runners:   int   = 0,
    new_comers:      int   = 0,
) -> dict:
    """
    荒れスコアを0〜100点で計算する。

    Parameters
    ----------
    race_class      : レースクラス文字列
    track_condition : 馬場状態（良/稍重/重/不良）
    field_size      : 頭数
    win_odds        : 単勝オッズ配列（None可）
    race_no         : レース番号
    total_races     : 当日最終レース番号
    distance        : 距離（m）
    track           : 競馬場名
    front_runners   : 逃げ馬候補の推定頭数（オプション）
    new_comers      : 転入馬・初コース馬の頭数（オプション）

    Returns
    -------
    dict:
        chaos_score  : 総合スコア（0〜100）
        is_high_chaos: chaos_score >= 60
        is_mid_chaos : chaos_score >= 40
        breakdown    : カテゴリ別内訳
    """
    score = 0
    breakdown = {}

    # ── 1. クラス条件（最重要・30点） ────────────
    cls = normalize_class(race_class)
    class_pts = 0
    if cls in ("C", "D"):
        class_pts = 30
    elif cls == "B":
        class_pts = 20
    elif cls in ("other",):
        class_pts = 15   # 条件戦・不明
    elif cls == "A":
        class_pts = 5
    score += class_pts
    breakdown["class"] = class_pts

    # ── 2. 馬場条件（25点） ───────────────────────
    tc = track_condition.strip() if track_condition else "良"
    going_pts = 0
    if "不" in tc or tc == "不良":
        going_pts = 25
    elif tc in ("重", "重馬場"):
        going_pts = 18
    elif tc in ("稍", "稍重", "やや重"):
        going_pts = 8
    score += going_pts
    breakdown["going"] = going_pts

    # ── 3. オッズ構造（最大30点） ─────────────────
    odds_pts = 0
    top3_share = 1.0
    fav_odds   = 1.5

    if win_odds is not None and len(win_odds) >= 3:
        sorted_odds = np.sort(win_odds)
        fav_odds    = float(sorted_odds[0])

        if fav_odds >= 3.0:
            odds_pts += 20
        elif fav_odds >= 2.5:
            odds_pts += 15
        elif fav_odds >= 2.0:
            odds_pts += 8

        # 上位3頭シェア
        inv = 1.0 / np.maximum(sorted_odds[:3], 0.1)
        total_inv = (1.0 / np.maximum(win_odds, 0.1)).sum()
        top3_share = float(inv.sum() / total_inv) if total_inv > 0 else 1.0

        if top3_share < 0.45:
            odds_pts += 15   # 10 + 5（超混戦ボーナス）
        elif top3_share < 0.55:
            odds_pts += 10
    else:
        # オッズ未取得時は中間値
        odds_pts += 8

    score += odds_pts
    breakdown["odds"] = odds_pts
    breakdown["fav_odds"]   = round(fav_odds, 2)
    breakdown["top3_share"] = round(top3_share, 3)

    # ── 4. 頭数（10点） ──────────────────────────
    field_pts = 0
    if field_size >= 12:
        field_pts = 10
    elif field_size >= 10:
        field_pts = 7
    elif field_size >= 8:
        field_pts = 3
    score += field_pts
    breakdown["field"] = field_pts

    # ── 5. レース条件（最大15点） ─────────────────
    race_pts = 0

    # 最終レース
    if race_no >= total_races:
        race_pts += 5

    # 短距離
    if distance <= 1400:
        race_pts += 5

    # 逃げ馬候補（複数いると乱ペースになりやすい）
    if front_runners >= 3:
        race_pts += 5

    # 転入・初コース馬（実力未知）
    if new_comers >= 2:
        race_pts += 5

    score += race_pts
    breakdown["race_cond"] = race_pts

    # ── 6. 競馬場ボーナス（5点） ─────────────────
    venue_pts = 0
    if track in ("高知", "佐賀"):
        venue_pts = 5
    elif track in ("笠松", "園田"):
        venue_pts = 4
    elif track in ("名古屋", "金沢", "浦和"):
        venue_pts = 3
    elif track in ("川崎", "水沢"):
        venue_pts = 2
    score += venue_pts
    breakdown["venue"] = venue_pts

    # 100点にキャップ
    final_score = min(int(score), 100)

    return {
        "chaos_score":   final_score,
        "is_high_chaos": final_score >= HIGH_CHAOS,
        "is_mid_chaos":  final_score >= MID_CHAOS,
        "breakdown":     breakdown,
        "level":         "🔥高配当" if final_score >= HIGH_CHAOS
                         else ("⚡中配当" if final_score >= MID_CHAOS else "普通"),
    }


# ──────────────────────────────────────────────────
# 高配当チャンスレース選別フィルター（後方互換）
# ──────────────────────────────────────────────────
def is_high_odds_target_race(
    race_class:      str,
    field_size:      int,
    fav_win_odds:    float,
    top3_odds_share: float,
    chaos_score:     int,
) -> tuple[bool, list[str]]:
    """
    高配当チャンスレースの選別フィルター。
    chaos_score >= HIGH_CHAOS ならTrue を返す。
    """
    reasons = []
    cls = normalize_class(race_class)

    if cls in ("C", "B", "D", "other"):
        reasons.append(f"C〜B級({cls})")
    if field_size >= 10:
        reasons.append(f"頭数{field_size}頭")
    if chaos_score >= HIGH_CHAOS:
        reasons.append(f"荒れ{chaos_score}点")
    if fav_win_odds > 2.0:
        reasons.append(f"1番人気{fav_win_odds:.1f}倍")
    if top3_odds_share < 0.60:
        reasons.append(f"上位3頭シェア{top3_odds_share:.0%}")

    passed = chaos_score >= HIGH_CHAOS and field_size >= 8
    return passed, reasons


# ──────────────────────────────────────────────────
# バックテスト: DB から chaos >= 60 の出現率計算
# ──────────────────────────────────────────────────
def backtest_chaos_accuracy(
    min_score:   int   = 60,
    min_payout:  int   = 5000,   # 50倍以上（100円単位）
    as_of_date:  str   = "2099-12-31",
) -> dict:
    """
    chaos_score >= min_score のレースで
    3連複50倍以上が出た割合を計算する。
    """
    from db.database import get_conn

    with get_conn() as conn:
        races = conn.execute("""
            SELECT rc.race_id, rc.race_class, rc.track_condition,
                   rc.field_size, rc.distance, rc.race_no, rc.track,
                   rc.race_date,
                   COUNT(DISTINCT rc2.race_id) as total_day_races
            FROM nar_races rc
            JOIN nar_races rc2 ON rc2.race_date = rc.race_date AND rc2.track = rc.track
            WHERE rc.race_date < ? AND rc.race_type != 'banei'
            GROUP BY rc.race_id
        """, (as_of_date,)).fetchall()

        # 各レースの単勝オッズ
        payouts = conn.execute("""
            SELECT race_id, payout FROM nar_payouts
            WHERE bet_type = 'trio' AND payout >= ?
        """, (min_payout,)).fetchall()

    high_payout_ids = {p["race_id"] for p in payouts}

    total = hit = 0
    for r in races:
        with __import__("contextlib").suppress(Exception):
            from db.database import get_conn as gc
            with gc() as c:
                odds_rows = c.execute(
                    "SELECT win_odds FROM nar_results WHERE race_id = ? AND win_odds IS NOT NULL",
                    (r["race_id"],)
                ).fetchall()

            win_odds = np.array([float(x["win_odds"]) for x in odds_rows]) if odds_rows else None

            cs = compute_chaos_score(
                race_class=r["race_class"] or "",
                track_condition=r["track_condition"] or "良",
                field_size=r["field_size"] or 8,
                win_odds=win_odds,
                race_no=r["race_no"],
                total_races=max(r["total_day_races"], r["race_no"]),
                distance=r["distance"] or 1500,
                track=r["track"] or "",
            )

            if cs["chaos_score"] >= min_score:
                total += 1
                if r["race_id"] in high_payout_ids:
                    hit += 1

    rate = hit / total if total > 0 else 0.0
    return {
        "min_score":      min_score,
        "total_races":    total,
        "high_payout_hit": hit,
        "hit_rate":       round(rate, 4),
        "n_payouts_in_db": len(high_payout_ids),
    }
