"""
先行力スコア（地方競馬最重要特徴量）

直近5走の1角・2角通過順位の加重平均から先行力を算出。
コース別先行有利度との積でコース補正を適用する。
"""
from __future__ import annotations

import numpy as np

# コース別「先行有利度」（NAR公式コースガイド・実績ベース）
COURSE_FRONT_ADVANTAGE: dict[str, float] = {
    "園田":   0.88,   # 超小回り1051m・上り坂 → 逃げ圧倒的有利
    "笠松":   0.85,   # 小回り1100m・追込困難
    "浦和":   0.82,   # 極小回り短直線220m
    "高知":   0.80,   # 内砂深い・先行残り
    "川崎":   0.79,   # タイトコーナー300m
    "佐賀":   0.78,   # パワー型砂・先行有利
    "金沢":   0.75,   # 小回り1200m
    "名古屋": 0.72,   # スパイラルカーブ1180m
    "水沢":   0.72,   # 小回り1200m平坦
    "姫路":   0.65,   # 平坦1200m
    "大井":   0.45,   # 外回り486m直線 → 差し届く
    "船橋":   0.55,   # スパイラル1400m・中間
    "門別":   0.50,   # 外回り400m直線
    "盛岡":   0.48,   # 左回り長直線・差し有利
    "帯広":   0.50,   # ばんえい（別競技）
}
DEFAULT_FRONT_ADVANTAGE = 0.65
TIGHT_COURSE_THRESHOLD  = 0.72   # 小回りコース（先行重み2倍の境界）

PACE_WEIGHTS = (0.40, 0.25, 0.15, 0.10, 0.10)   # 直近から古い順


def get_front_advantage(track: str) -> float:
    return COURSE_FRONT_ADVANTAGE.get(track, DEFAULT_FRONT_ADVANTAGE)


def is_tight_course(track: str) -> bool:
    return get_front_advantage(track) >= TIGHT_COURSE_THRESHOLD


def pace_position_score(
    horse_id:   str,
    as_of_date: str,
    race_track: str,
    n_recent:   int   = 5,
    weights:    tuple = PACE_WEIGHTS,
    conn              = None,
) -> dict:
    """
    先行力スコアを計算する。

    Returns
    -------
    dict:
        pace_score     : 0〜1（0=先頭/1=最後方）
        pace_score_adj : コース補正後スコア
        pace_vol       : 安定度（低=スタイル安定）
        front_rate     : 直近N走で1〜3番手以内の割合
        front_advantage: このコースの先行有利度
    """
    from db.database import get_conn as _gc

    def _fetch():
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

    rows = _fetch() if conn is None else conn.execute("""
        SELECT nr.corner1_pos, nr.corner2_pos, rc.field_size
        FROM nar_results nr
        JOIN nar_races rc ON nr.race_id = rc.race_id
        WHERE nr.horse_id = ? AND nr.race_date < ?
          AND (nr.corner1_pos IS NOT NULL OR nr.corner2_pos IS NOT NULL)
          AND rc.field_size > 1
        ORDER BY nr.race_date DESC
        LIMIT ?
    """, (horse_id, as_of_date, n_recent)).fetchall()

    adv = get_front_advantage(race_track)

    if not rows:
        return {
            "pace_score":      None,
            "pace_score_adj":  None,
            "pace_vol":        None,
            "front_rate":      None,
            "front_advantage": adv,
        }

    rel_positions = []
    front_count   = 0
    for r in rows:
        fs  = r["field_size"]
        pos = r["corner1_pos"] if r["corner1_pos"] else r["corner2_pos"]
        if pos and fs > 1:
            rel = (pos - 1) / (fs - 1)   # 0=先頭, 1=最後方
            rel_positions.append(rel)
            if pos <= 3:
                front_count += 1

    if not rel_positions:
        return {"pace_score": None, "pace_score_adj": None,
                "pace_vol": None, "front_rate": None, "front_advantage": adv}

    w = np.array(weights[:len(rel_positions)], dtype=float)
    w /= w.sum()
    pace_score = float(np.dot(w, rel_positions))
    pace_vol   = float(np.std(rel_positions)) if len(rel_positions) > 1 else 0.0
    front_rate = front_count / len(rel_positions)

    # コース補正: 小回り（先行有利度高）ほど先行スコア(低値)が重要
    multiplier = 2.0 if is_tight_course(race_track) else 1.0
    pace_score_adj = pace_score * (1.0 + (adv - 0.5) * multiplier)

    return {
        "pace_score":      round(pace_score, 4),
        "pace_score_adj":  round(pace_score_adj, 4),
        "pace_vol":        round(pace_vol, 4),
        "front_rate":      round(front_rate, 4),
        "front_advantage": adv,
    }
