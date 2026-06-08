"""
気象スコア（OpenWeatherMap APIなし版）
NAR公式サイトから天候・馬場状態を取得して
馬場状態変化の影響を特徴量化する
"""
from __future__ import annotations

# 馬場状態 → 数値エンコード（重いほど高い）
GOING_ENC = {"良": 0, "稍重": 1, "稍": 1, "重": 2, "不良": 3, "不": 3}

# 馬場状態別の距離補正係数（重馬場ほど長距離で不利）
GOING_DIST_COEFF = {
    0: 1.00,   # 良
    1: 0.98,   # 稍重
    2: 0.95,   # 重
    3: 0.90,   # 不良
}

def going_score(track_condition: str, distance: int) -> dict:
    """馬場状態スコアを返す。"""
    enc = GOING_ENC.get(track_condition, 0)
    dist_coeff = GOING_DIST_COEFF.get(enc, 1.0)

    # 距離1600m超は重馬場の影響大
    dist_penalty = 0.0
    if distance > 1600 and enc >= 2:
        dist_penalty = (enc - 1) * 0.05 * ((distance - 1600) / 400)

    return {
        "going_enc":      enc,
        "going_dist_adj": round(dist_coeff - dist_penalty, 4),
    }


def horse_going_affinity(
    horse_id: str, track_condition: str, as_of_date: str,
) -> dict:
    """馬の馬場状態適性（過去成績ベース）"""
    enc_target = GOING_ENC.get(track_condition, 0)

    from db.database import get_conn as _gc
    with _gc() as c:
        rows = c.execute("""
            SELECT nr.finish_position, rc.field_size, rc.track_condition
            FROM nar_results nr JOIN nar_races rc ON nr.race_id=rc.race_id
            WHERE nr.horse_id=? AND nr.race_date < ?
              AND nr.finish_position IS NOT NULL AND rc.field_size > 1
            ORDER BY nr.race_date DESC LIMIT 20
        """, (horse_id, as_of_date)).fetchall()

    if not rows:
        return {"going_affinity": 0.5, "going_n": 0}

    # 同馬場状態での成績
    same_going = [r for r in rows if GOING_ENC.get(r["track_condition"], 0) == enc_target]
    if not same_going:
        return {"going_affinity": 0.5, "going_n": 0}

    top3 = sum(1 for r in same_going if r["finish_position"] <= 3) / len(same_going)
    return {
        "going_affinity": round(top3, 4),
        "going_n":        len(same_going),
    }
