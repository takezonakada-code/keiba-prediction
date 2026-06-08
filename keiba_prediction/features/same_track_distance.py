"""
同場同距離の過去成績
地方競馬はリピーターが多く効果大
"""
from __future__ import annotations
import numpy as np


def same_track_distance_score(
    horse_id: str, track: str, distance: int, as_of_date: str,
    dist_margin: int = 200,
) -> dict:
    from db.database import get_conn as _gc
    with _gc() as c:
        rows = c.execute("""
            SELECT nr.finish_position, rc.field_size, rc.distance
            FROM nar_results nr JOIN nar_races rc ON nr.race_id=rc.race_id
            WHERE nr.horse_id=? AND rc.track=?
              AND rc.distance BETWEEN ? AND ?
              AND nr.race_date < ? AND nr.finish_position IS NOT NULL
            ORDER BY nr.race_date DESC LIMIT 10
        """, (horse_id, track, distance-dist_margin, distance+dist_margin,
              as_of_date)).fetchall()

    if not rows:
        return {"same_td_top3": 0.0, "same_td_score": 0.5, "same_td_n": 0}

    top3_rate = sum(1 for r in rows if r["finish_position"] <= 3) / len(rows)
    rel = [(r["finish_position"]-1)/max(r["field_size"]-1, 1) for r in rows]
    w = np.array([0.4, 0.25, 0.15, 0.1, 0.05, 0.05][:len(rel)])
    w /= w.sum()
    score = 1.0 - float(np.dot(w, rel[:len(w)]))

    return {
        "same_td_top3":  round(top3_rate, 4),
        "same_td_score": round(score, 4),
        "same_td_n":     len(rows),
    }
