"""
騎手×競馬場の相性スコア（地元騎手ボーナス）

騎手ごとの競馬場別勝率・複勝率を算出。
ベイズ収縮で小サンプルを補正。
高配当モード用に「穴馬騎乗時の高配当率」も計算。
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def jockey_venue_stats(
    jockey_id: str,
    track: str,
    as_of_date: str,
    prior_m: float = 50.0,
    conn=None,
) -> dict[str, Optional[float]]:
    """
    騎手×競馬場の成績をベイズ収縮で計算。

    Returns
    -------
    dict:
        win_rate       : 勝率（m-estimate）
        top3_rate      : 複勝率（m-estimate）
        local_bonus    : 地元ボーナス（このコース勝率 / 全体勝率）
        high_odds_rate : 人気薄（5番人気以下）での3着以内率
        n_rides        : このコースでの騎乗数
    """
    if not jockey_id or jockey_id in ("recent", "", None):
        return {
            "win_rate":       0.10,
            "top3_rate":      0.30,
            "local_bonus":    1.0,
            "high_odds_rate": 0.10,
            "n_rides":        0,
        }

    from db.database import get_conn as _gc
    with _gc() as ctx:
        # このコースでの成績
        local = ctx.execute("""
            SELECT COUNT(*) as cnt,
                   SUM(CASE WHEN nr.finish_position = 1 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN nr.finish_position <= 3 THEN 1 ELSE 0 END) as top3,
                   SUM(CASE WHEN nr.popular_rank >= 5 AND nr.finish_position <= 3
                            THEN 1 ELSE 0 END) as high_odds_top3,
                   SUM(CASE WHEN nr.popular_rank >= 5 THEN 1 ELSE 0 END) as high_odds_rides
            FROM nar_results nr
            JOIN nar_races rc ON nr.race_id = rc.race_id
            WHERE nr.jockey_id = ? AND rc.track = ?
              AND nr.race_date < ? AND nr.finish_position IS NOT NULL
        """, (jockey_id, track, as_of_date)).fetchone()

        # 全コース通算（事前分布）
        total = ctx.execute("""
            SELECT COUNT(*) as cnt,
                   SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN finish_position <= 3 THEN 1 ELSE 0 END) as top3
            FROM nar_results
            WHERE jockey_id = ? AND race_date < ?
              AND finish_position IS NOT NULL
        """, (jockey_id, as_of_date)).fetchone()

    lcnt    = local["cnt"]  or 0
    lwins   = local["wins"] or 0
    ltop3   = local["top3"] or 0
    tcnt    = total["cnt"]  or 1
    twins   = total["wins"] or 0
    ttop3   = total["top3"] or 0

    g_wr  = twins / tcnt
    g_t3r = ttop3 / tcnt

    # m-estimate
    win_rate  = (lwins + prior_m * g_wr)  / (lcnt + prior_m)
    top3_rate = (ltop3 + prior_m * g_t3r) / (lcnt + prior_m)

    # 地元ボーナス（1.0超 = このコースが得意）
    local_bonus = (lwins / lcnt) / g_wr if lcnt >= 10 and g_wr > 0 else 1.0

    # 人気薄高配当率
    hor = local["high_odds_rides"] or 0
    hot = local["high_odds_top3"]  or 0
    high_odds_rate = (hot + 10 * 0.12) / (hor + 10) if hor >= 0 else 0.12

    return {
        "win_rate":       round(win_rate,  4),
        "top3_rate":      round(top3_rate, 4),
        "local_bonus":    round(local_bonus, 3),
        "high_odds_rate": round(high_odds_rate, 4),
        "n_rides":        lcnt,
    }


def jockey_venue_score(
    jockey_id: str,
    track: str,
    as_of_date: str,
    conn=None,
) -> float:
    """
    騎手×競馬場の総合相性スコアを0〜1で返す。
    勝率・複勝率・地元ボーナスを組み合わせた合成スコア。
    """
    stats = jockey_venue_stats(jockey_id, track, as_of_date, conn=conn)
    # 重み付き合成
    score = (
        0.5 * stats["top3_rate"]
        + 0.3 * stats["win_rate"]
        + 0.2 * min(stats["local_bonus"] / 2.0, 1.0)
    )
    return round(min(score, 1.0), 4)


def high_odds_jockey_score(
    jockey_id: str,
    track: str,
    as_of_date: str,
    conn=None,
) -> float:
    """
    高配当狙いモード用: 人気薄での爆発力スコア。
    过去3連複100倍以上での3着内率を含む。
    """
    stats = jockey_venue_stats(jockey_id, track, as_of_date, conn=conn)
    return stats.get("high_odds_rate", 0.10)
