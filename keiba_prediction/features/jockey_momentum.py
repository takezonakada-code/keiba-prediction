"""
騎手モメンタム特徴量
- 直近7日勝率
- 直近3走成績トレンド（上昇/下降）
- 連続好走・連続不振の検出
"""
from __future__ import annotations
import numpy as np


def jockey_momentum(jockey_id: str, as_of_date: str) -> dict:
    if not jockey_id or jockey_id in ("", "recent", None):
        return {"jk_win7d": 0.10, "jk_top3_7d": 0.30,
                "jk_trend": 0.0, "jk_streak": 0}

    from db.database import get_conn as _gc
    with _gc() as c:
        # 直近7日成績
        r7 = c.execute("""
            SELECT COUNT(*) as rides,
                   SUM(CASE WHEN finish_position=1 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN finish_position<=3 THEN 1 ELSE 0 END) as top3
            FROM nar_results
            WHERE jockey_id=? AND race_date BETWEEN date(?,-7) AND date(?,-1)
              AND finish_position IS NOT NULL
        """, (jockey_id, as_of_date, as_of_date)).fetchone()

        # 直近30走の着順（トレンド計算用）
        recent = c.execute("""
            SELECT finish_position FROM nar_results
            WHERE jockey_id=? AND race_date < ?
              AND finish_position IS NOT NULL
            ORDER BY race_date DESC, race_id DESC
            LIMIT 30
        """, (jockey_id, as_of_date)).fetchall()

    rides = r7["rides"] or 0
    win7d  = (r7["wins"]  or 0) / max(rides, 1)
    top37d = (r7["top3"]  or 0) / max(rides, 1)

    # トレンド: 前15走 vs 直近15走の3着内率差
    positions = [r["finish_position"] for r in recent]
    trend = 0.0
    if len(positions) >= 10:
        half = len(positions) // 2
        old_rate = sum(1 for p in positions[half:] if p <= 3) / max(len(positions[half:]), 1)
        new_rate = sum(1 for p in positions[:half] if p <= 3) / max(half, 1)
        trend = new_rate - old_rate   # 正=上昇傾向

    # 連続好走/不振ストリーク
    streak = 0
    if positions:
        is_good = positions[0] <= 3
        for p in positions:
            if (p <= 3) == is_good:
                streak += 1 if is_good else -1
            else:
                break

    return {
        "jk_win7d":  round(win7d,  4),
        "jk_top3_7d": round(top37d, 4),
        "jk_trend":  round(trend,  4),
        "jk_streak": streak,
    }


def horse_recent_form(horse_id: str, as_of_date: str, n: int = 5) -> dict:
    """直近N走の成績トレンド（馬版）"""
    from db.database import get_conn as _gc
    with _gc() as c:
        rows = c.execute("""
            SELECT finish_position, field_size FROM nar_results nr
            JOIN nar_races rc ON nr.race_id=rc.race_id
            WHERE nr.horse_id=? AND nr.race_date < ?
              AND nr.finish_position IS NOT NULL AND rc.field_size > 1
            ORDER BY nr.race_date DESC LIMIT ?
        """, (horse_id, as_of_date, n)).fetchall()

    if not rows:
        return {"form_score": 0.5, "form_trend": 0.0, "top3_rate_5r": 0.0}

    rel = [(r["finish_position"]-1)/(r["field_size"]-1) for r in rows]
    weights = np.array([0.4, 0.25, 0.15, 0.1, 0.1][:len(rel)])
    weights /= weights.sum()
    form_score = 1.0 - float(np.dot(weights, rel))   # 高いほど好調

    top3_rate = sum(1 for r in rows if r["finish_position"] <= 3) / len(rows)
    trend = (rel[0] - np.mean(rel)) * -1   # 直近が平均より良ければ正

    return {
        "form_score":    round(form_score, 4),
        "form_trend":    round(trend, 4),
        "top3_rate_5r":  round(top3_rate, 4),
    }
