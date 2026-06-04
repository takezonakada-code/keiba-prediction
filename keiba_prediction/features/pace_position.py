"""
先行力スコア（地方競馬最重要特徴量）

直近5走の1角・2角通過順位の加重平均から先行力を算出。
小回りコース（名古屋・園田・笠松等）では重みを2倍にする。
"""
from __future__ import annotations

import numpy as np

# コース別「先行有利度」（0=差し有利, 1=完全先行有利）
# NAR公式コースガイドより実測・実績ベースで設定
COURSE_FRONT_ADVANTAGE: dict[str, float] = {
    "園田":   0.88,   # 超小回り1051m・上り坂 → 逃げ圧倒的有利
    "笠松":   0.85,   # 小回り1100m・追込困難
    "名古屋": 0.72,   # スパイラルカーブ1180m → 先行有利
    "川崎":   0.79,   # タイト1200m → 先行有利
    "水沢":   0.76,   # 小回り1200m平坦
    "佐賀":   0.75,   # 小回り1100m砂深い
    "高知":   0.74,   # 小回り1100m内砂深い
    "浦和":   0.73,   # 極端小回り短直線
    "金沢":   0.72,   # 小回り1200m
    "船橋":   0.60,   # スパイラル1400m → やや中間
    "大井":   0.55,   # 外回り486m直線 → 差しも届く
    "門別":   0.58,   # 外回り400m直線
    "盛岡":   0.50,   # 左回り・長直線400m → バランス型
    "姫路":   0.65,   # 平坦1200m
    "帯広":   0.50,   # ばんえい（別競技）
}
DEFAULT_FRONT_ADVANTAGE = 0.65

# 小回りコース（先行重みを2倍にする閾値）
TIGHT_COURSE_THRESHOLD = 0.70


def get_front_advantage(track: str) -> float:
    """競馬場の先行有利度を返す（0〜1）。"""
    return COURSE_FRONT_ADVANTAGE.get(track, DEFAULT_FRONT_ADVANTAGE)


def is_tight_course(track: str) -> bool:
    """小回りコース（先行重み2倍の対象）か判定。"""
    return get_front_advantage(track) >= TIGHT_COURSE_THRESHOLD


def pace_position_score(
    horse_id: str,
    as_of_date: str,
    race_track: str,
    n_recent: int = 5,
    weights: tuple[float, ...] = (0.4, 0.25, 0.17, 0.11, 0.07),
    conn=None,
) -> dict[str, float | None]:
    """
    先行力スコアを計算する。

    Parameters
    ----------
    horse_id      : 馬ID
    as_of_date    : この日より前のデータのみ使用
    race_track    : レースの競馬場（コース補正に使用）
    n_recent      : 直近N走を使用
    weights       : 直近走から古い順への加重

    Returns
    -------
    dict:
        pace_score     : 先行力スコア（0=先頭, 1=最後方）
        pace_score_adj : コース補正後スコア（小回りは先行有利度を掛ける）
        pace_vol       : 安定度（低=スタイル安定）
        front_rate     : 1〜3番手以内の割合（先行率）
        front_advantage: このコースの先行有利度
    """
    from db.database import get_conn as _gc

    def _query():
        with _gc() as c:
            return c.execute("""
                SELECT nr.corner1_pos, nr.corner2_pos, rc.field_size
                FROM nar_results nr
                JOIN nar_races rc ON nr.race_id = rc.race_id
                WHERE nr.horse_id = ? AND nr.race_date < ?
                  AND (nr.corner1_pos IS NOT NULL OR nr.corner2_pos IS NOT NULL)
                  AND rc.field_size > 1
                ORDER BY nr.race_date DESC
                LIMIT ?
            """, (horse_id, as_of_date, n_recent)).fetchall()

    rows = _query() if conn is None else conn.execute("""
        SELECT nr.corner1_pos, nr.corner2_pos, rc.field_size
        FROM nar_results nr
        JOIN nar_races rc ON nr.race_id = rc.race_id
        WHERE nr.horse_id = ? AND nr.race_date < ?
          AND (nr.corner1_pos IS NOT NULL OR nr.corner2_pos IS NOT NULL)
          AND rc.field_size > 1
        ORDER BY nr.race_date DESC
        LIMIT ?
    """, (horse_id, as_of_date, n_recent)).fetchall()

    if not rows:
        return {
            "pace_score":      None,
            "pace_score_adj":  None,
            "pace_vol":        None,
            "front_rate":      None,
            "front_advantage": get_front_advantage(race_track),
        }

    rel_positions = []
    front_count   = 0
    for r in rows:
        fs = r["field_size"]
        # 1角優先、なければ2角を使用
        pos = r["corner1_pos"] if r["corner1_pos"] else r["corner2_pos"]
        if pos and fs > 1:
            rel = (pos - 1) / (fs - 1)   # 0=先頭, 1=最後方
            rel_positions.append(rel)
            if pos <= 3:
                front_count += 1

    if not rel_positions:
        return {
            "pace_score":      None,
            "pace_score_adj":  None,
            "pace_vol":        None,
            "front_rate":      None,
            "front_advantage": get_front_advantage(race_track),
        }

    w = np.array(weights[:len(rel_positions)], dtype=float)
    w /= w.sum()
    pace_score = float(np.dot(w, rel_positions))
    pace_vol   = float(np.std(rel_positions)) if len(rel_positions) > 1 else 0.0
    front_rate = front_count / len(rel_positions)

    # コース補正: 小回りほど先行スコアの逆数（先行力）を重視
    adv = get_front_advantage(race_track)
    # 先行力を「低いスコア = 有利」として、コース有利度で増幅
    # pace_score_adj: 小回りで先行馬ほど低い値 = 有利
    multiplier = 2.0 if is_tight_course(race_track) else 1.0
    pace_score_adj = pace_score * (1.0 + (adv - 0.5) * multiplier)

    return {
        "pace_score":      round(pace_score, 4),
        "pace_score_adj":  round(pace_score_adj, 4),
        "pace_vol":        round(pace_vol, 4),
        "front_rate":      round(front_rate, 4),
        "front_advantage": adv,
    }
