"""
「荒れやすさ」スコア計算（0〜1）

荒れ = 1番人気が3着以内に入らない確率

以下の複数指標を合成する:
1. クラス別過去荒れ率（C > B > A > 重賞）
2. 頭数（多いほど荒れやすい）
3. 1番人気のオッズ（高いほど不確実）
4. 上位3頭のオッズシェア（低いほど混戦）
5. 近走クラス変動（昇降級はバイアス不安定）
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np


# ──────────────────────────────────────────────────
# クラス正規化
# ──────────────────────────────────────────────────
_CLASS_ORDER = {"重賞": 0, "オープン": 1, "特別": 2,
                "A": 3, "B": 4, "C": 5, "D": 6, "other": 5}


def normalize_class(race_class: str) -> str:
    if not race_class:
        return "other"
    rc = race_class.strip()
    if "重賞" in rc:        return "重賞"
    if "オープン" in rc:    return "オープン"
    if "特別" in rc:        return "特別"
    m = re.match(r"^([A-Z])", rc)
    if m:                   return m.group(1)
    return "other"


def class_base_chaos_rate(race_class: str) -> float:
    """クラスだけから基準荒れ率を返す（経験則）。"""
    cls = normalize_class(race_class)
    base = {
        "C": 0.68, "D": 0.70,   # 最も荒れやすい
        "B": 0.62,
        "A": 0.55,
        "特別": 0.50,
        "オープン": 0.45,
        "重賞": 0.38,
        "other": 0.60,
    }
    return base.get(cls, 0.60)


# ──────────────────────────────────────────────────
# DB ベースの荒れ率
# ──────────────────────────────────────────────────
def historical_chaos_rate(
    race_class: str,
    track: str,
    as_of_date: str,
    min_samples: int = 20,
    conn=None,
) -> dict[str, float]:
    """
    過去データから同クラス×同競馬場の荒れ率を計算。

    Returns
    -------
    dict: {chaos_rate, fav_top3_rate, n_races}
    """
    from db.database import get_conn as _gc
    cls = normalize_class(race_class)
    with _gc() as ctx:
        # 同クラス・同競馬場
        row = ctx.execute("""
            SELECT COUNT(DISTINCT rc.race_id) as n_races,
                   SUM(CASE WHEN nr.popular_rank = 1 AND nr.finish_position <= 3
                            THEN 1 ELSE 0 END) as fav_top3,
                   COUNT(DISTINCT CASE WHEN nr.popular_rank = 1 THEN rc.race_id END) as races_with_fav
            FROM nar_races rc
            JOIN nar_results nr ON rc.race_id = nr.race_id
            WHERE rc.race_date < ?
              AND rc.track = ?
              AND nr.finish_position IS NOT NULL
              AND CASE WHEN ? = 'C' THEN rc.race_class LIKE 'C%'
                       WHEN ? = 'B' THEN rc.race_class LIKE 'B%'
                       WHEN ? = 'A' THEN rc.race_class LIKE 'A%'
                       WHEN ? = '重賞' THEN rc.race_class LIKE '%賞%'
                       ELSE 1=1 END
        """, (as_of_date, track, cls, cls, cls, cls)).fetchone()

    n_races    = row["n_races"]    or 0
    fav_top3   = row["fav_top3"]   or 0
    races_wfav = row["races_with_fav"] or 0

    if races_wfav < min_samples:
        # サンプル不足 → クラスベース推定
        return {
            "chaos_rate":     class_base_chaos_rate(race_class),
            "fav_top3_rate":  1.0 - class_base_chaos_rate(race_class),
            "n_races":        n_races,
            "source":         "prior",
        }

    fav_rate   = fav_top3 / races_wfav
    chaos_rate = 1.0 - fav_rate
    return {
        "chaos_rate":     round(chaos_rate, 4),
        "fav_top3_rate":  round(fav_rate, 4),
        "n_races":        n_races,
        "source":         "historical",
    }


# ──────────────────────────────────────────────────
# 総合荒れやすさスコア
# ──────────────────────────────────────────────────
def race_chaos_score(
    race_id: str,
    as_of_date: str,
    race_class: str,
    track: str,
    field_size: int,
    win_odds: Optional[np.ndarray] = None,
    conn=None,
) -> dict[str, float]:
    """
    レースの総合「荒れやすさ」スコアを0〜1で返す。

    5つの指標を重み付き合成:
    1. クラス別過去荒れ率 (40%)
    2. 頭数スコア         (20%)
    3. 1番人気オッズ      (15%)
    4. 上位3頭シェア      (15%)
    5. 競馬場特性         (10%)

    Returns
    -------
    dict: {chaos_score, is_high_chaos, details}
    """
    from features.pace_position import get_front_advantage

    # 1. クラス別荒れ率
    hist = historical_chaos_rate(race_class, track, as_of_date, conn=conn)
    chaos_class = hist["chaos_rate"]

    # 2. 頭数スコア（8〜18頭の範囲で正規化）
    chaos_field = np.clip((field_size - 6) / 12, 0.0, 1.0)

    # 3. 1番人気オッズスコア（高いほど不確実）
    chaos_fav = 0.5
    chaos_share = 0.5
    if win_odds is not None and len(win_odds) >= 3:
        sorted_odds = np.sort(win_odds)
        fav_odds = sorted_odds[0]
        # 1倍台 → 0, 4倍以上 → 1
        chaos_fav = np.clip((fav_odds - 1.0) / 3.0, 0.0, 1.0)

        # 上位3頭オッズシェア（低いほど混戦）
        inv = 1.0 / np.maximum(sorted_odds[:3], 0.1)
        total_inv = (1.0 / np.maximum(win_odds, 0.1)).sum()
        share_top3 = inv.sum() / total_inv if total_inv > 0 else 1.0
        chaos_share = 1.0 - np.clip(share_top3, 0.0, 1.0)

    # 4. 競馬場の先行有利度（小回りほど荒れにくい傾向？実は逆: C級小回りは荒れる）
    front_adv = get_front_advantage(track)
    # 小回りC級は荒れやすい（差し馬の台頭が読めない）
    chaos_course = 0.6 if front_adv >= 0.70 else 0.4

    # 重み付き合成
    chaos_score = (
        0.40 * chaos_class
        + 0.20 * chaos_field
        + 0.15 * chaos_fav
        + 0.15 * chaos_share
        + 0.10 * chaos_course
    )
    chaos_score = round(float(np.clip(chaos_score, 0.0, 1.0)), 4)

    return {
        "chaos_score":    chaos_score,
        "is_high_chaos":  chaos_score >= 0.60,
        "chaos_class":    round(chaos_class, 4),
        "chaos_field":    round(chaos_field, 4),
        "chaos_fav":      round(chaos_fav, 4),
        "chaos_share":    round(chaos_share, 4),
        "n_historical":   hist["n_races"],
    }


# ──────────────────────────────────────────────────
# 高配当チャンスレース選別フィルター
# ──────────────────────────────────────────────────
def is_high_odds_target_race(
    race_class: str,
    field_size: int,
    fav_win_odds: float,
    top3_odds_share: float,
    chaos_score: float,
) -> tuple[bool, list[str]]:
    """
    高配当チャンスレースの選別フィルター。

    全5条件を確認して (通過可否, 通過条件リスト) を返す。
    """
    reasons = []
    cls = normalize_class(race_class)

    # 条件1: クラスがC〜B級
    cond1 = cls in ("C", "B", "D", "other")
    if cond1:
        reasons.append(f"C〜B級({cls})")

    # 条件2: 頭数10頭以上
    cond2 = field_size >= 10
    if cond2:
        reasons.append(f"頭数{field_size}頭")

    # 条件3: 過去同条件荒れ率 > 40%
    cond3 = chaos_score >= 0.40
    if cond3:
        reasons.append(f"荒れ率{chaos_score:.0%}")

    # 条件4: 1番人気単勝 > 2.0倍
    cond4 = fav_win_odds > 2.0
    if cond4:
        reasons.append(f"1番人気{fav_win_odds:.1f}倍")

    # 条件5: 上位3頭シェア < 60%
    cond5 = top3_odds_share < 0.60
    if cond5:
        reasons.append(f"上位3頭シェア{top3_odds_share:.0%}")

    passed = all([cond1, cond2, cond3, cond4, cond5])
    return passed, reasons
