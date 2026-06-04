"""
地方競馬専用特徴量

1. 騎手×競馬場の過去勝率（地元騎手ボーナス）
2. 斤量の相対軽さスコア（同レース内比較）
3. 先行力スコア（コーナー通過順の加重平均）
4. C級・B級など条件クラス別の荒れ率
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────
# 1. 騎手×競馬場 勝率（地元騎手ボーナス）
# ──────────────────────────────────────────────────
def jockey_track_winrate(
    jockey_id: str,
    track: str,
    as_of_date: str,
    prior_m: float = 30.0,
    conn=None,
) -> float:
    """
    騎手×競馬場の過去勝率をm-estimate（ベイズ収縮）で計算。
    地元騎手ボーナス検出に使う。

    地方競馬では同一競馬場専属騎手が多く、
    コース熟知度が勝率に直結する（例: 園田・笠松の地元騎手）。

    Parameters
    ----------
    jockey_id  : 騎手ID
    track      : 競馬場名
    as_of_date : この日より前のデータのみ使用（リーク防止）
    prior_m    : 収縮強度
    conn       : DBコネクション（None の場合は新規接続）
    """
    if not jockey_id or jockey_id == "recent":
        return _global_jockey_winrate(as_of_date, prior_m, conn)

    from db.database import get_conn as _get_conn
    ctx = conn if conn is not None else _get_conn().__enter__()

    try:
        # このコース×この騎手の成績
        row = ctx.execute("""
            SELECT COUNT(*) as cnt,
                   SUM(CASE WHEN nr.finish_position = 1 THEN 1 ELSE 0 END) as wins
            FROM nar_results nr
            JOIN nar_races rc ON nr.race_id = rc.race_id
            WHERE nr.jockey_id = ? AND rc.track = ? AND nr.race_date < ?
              AND nr.finish_position IS NOT NULL
        """, (jockey_id, track, as_of_date)).fetchone()

        # 全コースでの全体勝率（事前分布）
        g_row = ctx.execute("""
            SELECT COUNT(*) as cnt,
                   SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) as wins
            FROM nar_results
            WHERE jockey_id = ? AND race_date < ? AND finish_position IS NOT NULL
        """, (jockey_id, as_of_date)).fetchone()

    finally:
        if conn is None:
            ctx.__exit__(None, None, None)

    cnt    = row["cnt"]    or 0
    wins   = row["wins"]   or 0
    g_cnt  = g_row["cnt"]  or 1
    g_wins = g_row["wins"] or 0
    g_rate = g_wins / g_cnt

    return (wins + prior_m * g_rate) / (cnt + prior_m)


def _global_jockey_winrate(as_of_date: str, m: float, conn=None) -> float:
    """jockey_id が不明なときの全体平均を返す。"""
    from db.database import get_conn as _get_conn
    ctx = conn if conn is not None else _get_conn().__enter__()
    try:
        row = ctx.execute("""
            SELECT AVG(CASE WHEN finish_position = 1 THEN 1.0 ELSE 0.0 END) as wr
            FROM nar_results WHERE race_date < ? AND finish_position IS NOT NULL
        """, (as_of_date,)).fetchone()
    finally:
        if conn is None:
            ctx.__exit__(None, None, None)
    return float(row["wr"] or 0.10)


# ──────────────────────────────────────────────────
# 2. 斤量の相対軽さスコア（同レース内）
# ──────────────────────────────────────────────────
def relative_weight_score(
    race_id: str,
    conn=None,
) -> pd.DataFrame:
    """
    同レース内の斤量を比較して相対軽さスコアを計算する。

    - weight_z: 同レース内 z-score（正 = 他馬より軽い）
    - weight_rank_pct: 同レース内軽さ順位% (0=最軽量, 1=最重量)

    NAR は牝馬減量・減量騎手の恩恵が大きく、
    軽ハンデは距離が長くなるほど有利になる傾向がある。

    Returns
    -------
    DataFrame: race_id, draw_number, weight_z, weight_rank_pct
    """
    from db.database import get_conn as _get_conn
    ctx = conn if conn is not None else _get_conn().__enter__()
    try:
        rows = ctx.execute("""
            SELECT draw_number, weight_carried
            FROM nar_results
            WHERE race_id = ? AND weight_carried IS NOT NULL
        """, (race_id,)).fetchall()
    finally:
        if conn is None:
            ctx.__exit__(None, None, None)

    if not rows:
        return pd.DataFrame(columns=["race_id", "draw_number",
                                     "weight_z", "weight_rank_pct"])

    df = pd.DataFrame([dict(r) for r in rows])
    wts = df["weight_carried"].values

    mean_w = wts.mean()
    std_w  = wts.std() if wts.std() > 0 else 1.0

    # 軽い方が有利 → 符号を反転（軽いほど正）
    df["weight_z"] = -(wts - mean_w) / std_w

    # 軽さ順位%: 軽いほど0（最軽量=0.0, 最重量=1.0）
    df["weight_rank_pct"] = pd.Series(wts).rank(
        method="average", ascending=True, pct=True
    ).values - 1.0   # 0-origin に調整 (1-origin → shift)

    df["race_id"] = race_id
    return df[["race_id", "draw_number", "weight_z", "weight_rank_pct"]]


def batch_relative_weight(
    race_ids: list[str],
    conn=None,
) -> pd.DataFrame:
    """複数レース一括で相対斤量スコアを計算。"""
    from db.database import get_conn as _get_conn
    ctx = conn if conn is not None else _get_conn().__enter__()
    try:
        placeholders = ",".join("?" * len(race_ids))
        rows = ctx.execute(f"""
            SELECT race_id, draw_number, weight_carried
            FROM nar_results
            WHERE race_id IN ({placeholders}) AND weight_carried IS NOT NULL
        """, race_ids).fetchall()
    finally:
        if conn is None:
            ctx.__exit__(None, None, None)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    df["weight_carried"] = pd.to_numeric(df["weight_carried"], errors="coerce")

    def _score_group(g):
        wts = g["weight_carried"].values
        std = wts.std() if wts.std() > 0 else 1.0
        g = g.copy()
        g["weight_z"]        = -(wts - wts.mean()) / std
        g["weight_rank_pct"] = pd.Series(wts).rank(
            method="average", ascending=True, pct=True
        ).values - 1.0
        return g

    return df.groupby("race_id", group_keys=False).apply(_score_group)


# ──────────────────────────────────────────────────
# 3. 先行力スコア（コーナー通過順加重平均）
# ──────────────────────────────────────────────────
def frontrun_score(
    horse_id: str,
    as_of_date: str,
    n_recent: int = 5,
    weights: tuple = (0.5, 0.3, 0.2),
    conn=None,
) -> dict[str, Optional[float]]:
    """
    過去走の4角通過順をもとに先行力スコアを計算する。

    4角相対位置 = (corner4_pos - 1) / (field_size - 1)
      → 0 = 逃げ（先頭）/ 1 = 最後方

    先行力スコア（小さいほど先行）:
      - style_score_c4: 直近N走加重平均（逃げ寄り=0, 追い込み=1）
      - style_vol_c4:   安定度（低=スタイル安定）
      - front_pct:      N走中で4角3番手以内の割合（先行率）

    NAR 小回り競馬場では先行有利バイアスが強く、
    style_score_c4 は予測精度に直結する。
    """
    from db.database import get_conn as _get_conn
    ctx = conn if conn is not None else _get_conn().__enter__()
    try:
        rows = ctx.execute("""
            SELECT nr.corner4_pos, rc.field_size
            FROM nar_results nr
            JOIN nar_races rc ON nr.race_id = rc.race_id
            WHERE nr.horse_id = ? AND nr.race_date < ?
              AND nr.corner4_pos IS NOT NULL AND nr.corner4_pos > 0
              AND rc.field_size > 1
            ORDER BY nr.race_date DESC
            LIMIT ?
        """, (horse_id, as_of_date, n_recent)).fetchall()
    finally:
        if conn is None:
            ctx.__exit__(None, None, None)

    if not rows:
        return {"style_score_c4": None, "style_vol_c4": None, "front_pct": None}

    rel_positions = [
        (r["corner4_pos"] - 1) / (r["field_size"] - 1)
        for r in rows
    ]

    w = np.array(weights[:len(rel_positions)], dtype=float)
    w /= w.sum()
    style_score = float(np.dot(w, rel_positions))
    style_vol   = float(np.std(rel_positions)) if len(rel_positions) > 1 else 0.0
    front_pct   = float(sum(1 for r in rows if r["corner4_pos"] <= 3) / len(rows))

    return {
        "style_score_c4": style_score,
        "style_vol_c4":   style_vol,
        "front_pct":      front_pct,
    }


# ──────────────────────────────────────────────────
# 4. 条件クラス別の荒れ率
# ──────────────────────────────────────────────────
def _normalize_class(race_class: str) -> str:
    """
    'C310', 'B2四', '重賞', 'A1' → 'C', 'B', '重賞', 'A' に正規化。
    """
    if not race_class:
        return "unknown"
    rc = race_class.strip()
    if "重賞" in rc:
        return "重賞"
    if "オープン" in rc or "OP" in rc:
        return "オープン"
    if "特別" in rc:
        return "特別"
    # 先頭のアルファベットで大分類
    m = re.match(r"^([A-Z])", rc)
    if m:
        return m.group(1)
    return "other"


def class_upset_rates(
    as_of_date: str,
    min_samples: int = 30,
    conn=None,
) -> dict[str, dict]:
    """
    条件クラス別の荒れ率を計算する。

    荒れ = 1番人気が3着以内に入らない確率（人気裏切り率）

    NAR では C・D 級は荒れやすく、重賞は本命が来やすい傾向がある。
    この情報を特徴量として使うことで、
    「このクラスは本命を信頼すべきか」を学習できる。

    Returns
    -------
    dict: {class_name: {"upset_rate": float, "n_races": int}}
    """
    from db.database import get_conn as _get_conn
    ctx = conn if conn is not None else _get_conn().__enter__()
    try:
        rows = ctx.execute("""
            SELECT rc.race_class,
                   COUNT(DISTINCT rc.race_id) as n_races,
                   SUM(CASE
                       WHEN nr.popular_rank = 1 AND nr.finish_position <= 3 THEN 1
                       ELSE 0 END) as fav_in_top3,
                   COUNT(DISTINCT CASE WHEN nr.popular_rank = 1 THEN rc.race_id END) as races_with_fav
            FROM nar_races rc
            JOIN nar_results nr ON rc.race_id = nr.race_id
            WHERE rc.race_date < ? AND nr.finish_position IS NOT NULL
            GROUP BY rc.race_class
        """, (as_of_date,)).fetchall()
    finally:
        if conn is None:
            ctx.__exit__(None, None, None)

    result: dict[str, dict] = {}
    class_buckets: dict[str, list] = {}

    for row in rows:
        cls_raw = row["race_class"] or "unknown"
        cls     = _normalize_class(cls_raw)
        rfav    = row["races_with_fav"] or 0
        fin3    = row["fav_in_top3"]    or 0

        if cls not in class_buckets:
            class_buckets[cls] = []
        class_buckets[cls].append((rfav, fin3))

    for cls, items in class_buckets.items():
        total_races = sum(i[0] for i in items)
        total_fav3  = sum(i[1] for i in items)
        if total_races < min_samples:
            continue
        fav_rate    = total_fav3 / total_races
        upset_rate  = 1.0 - fav_rate
        result[cls] = {
            "upset_rate":  round(upset_rate, 4),
            "fav_top3_rate": round(fav_rate, 4),
            "n_races":     total_races,
        }

    return result


def get_class_upset_rate(
    race_class: str,
    upset_rates: dict[str, dict],
    default: float = 0.65,
) -> float:
    """
    1レースの race_class から荒れ率を返す。
    クラス不明・サンプル不足の場合は default を使用。
    """
    cls = _normalize_class(race_class)
    info = upset_rates.get(cls)
    return info["upset_rate"] if info else default
