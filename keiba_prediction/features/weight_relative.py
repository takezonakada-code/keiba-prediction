"""
斤量の相対軽さスコア

同レース内での斤量順位と差分を計算。
C級では斤量差の影響が大きく予測精度に寄与する。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def relative_weight_features(
    race_id: str,
    conn=None,
) -> pd.DataFrame:
    """
    1レース内の全馬の相対斤量スコアを計算する。

    Returns
    -------
    DataFrame: draw_number, weight_carried,
               weight_diff_from_mean, weight_z, weight_rank_pct,
               is_lightest, weight_vs_fav
    """
    from db.database import get_conn as _gc
    with _gc() as ctx:
        rows = ctx.execute("""
            SELECT draw_number, weight_carried, popular_rank
            FROM nar_results
            WHERE race_id = ? AND weight_carried IS NOT NULL
            ORDER BY draw_number
        """, (race_id,)).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    df["weight_carried"] = pd.to_numeric(df["weight_carried"], errors="coerce")
    df = df.dropna(subset=["weight_carried"])

    if len(df) < 2:
        return df

    wts  = df["weight_carried"].values
    mean = wts.mean()
    std  = wts.std() if wts.std() > 0 else 1.0

    # 軽いほど有利 → 差分は「平均 - 自分」（正=軽い）
    df["weight_diff_from_mean"] = mean - wts

    # z-score（軽いほど正）
    df["weight_z"] = (mean - wts) / std

    # 順位%（0=最軽量, 1=最重量）
    df["weight_rank_pct"] = pd.Series(wts).rank(
        method="average", ascending=True, pct=True
    ).values

    # 最軽量フラグ
    df["is_lightest"] = (wts == wts.min()).astype(int)

    # 1番人気との斤量差（人気馬との差 = プラスが軽い）
    fav_row = df[df["popular_rank"] == 1]
    fav_weight = fav_row["weight_carried"].values[0] if len(fav_row) > 0 else mean
    df["weight_vs_fav"] = fav_weight - df["weight_carried"]

    return df[["draw_number", "weight_carried", "weight_diff_from_mean",
               "weight_z", "weight_rank_pct", "is_lightest", "weight_vs_fav"]]


def batch_relative_weight(
    race_ids: list[str],
    conn=None,
) -> pd.DataFrame:
    """複数レース一括計算。"""
    from db.database import get_conn as _gc
    with _gc() as ctx:
        ph = ",".join("?" * len(race_ids))
        rows = ctx.execute(f"""
            SELECT race_id, draw_number, weight_carried, popular_rank
            FROM nar_results
            WHERE race_id IN ({ph}) AND weight_carried IS NOT NULL
        """, race_ids).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])
    df["weight_carried"] = pd.to_numeric(df["weight_carried"], errors="coerce")

    def _score_group(g: pd.DataFrame) -> pd.DataFrame:
        wts  = g["weight_carried"].values
        mean = wts.mean()
        std  = wts.std() if len(wts) > 1 and wts.std() > 0 else 1.0
        g = g.copy()
        g["weight_diff_from_mean"] = mean - wts
        g["weight_z"]              = (mean - wts) / std
        g["weight_rank_pct"]       = pd.Series(wts).rank(
            method="average", ascending=True, pct=True
        ).values
        g["is_lightest"] = (wts == wts.min()).astype(int)
        fav = g[g["popular_rank"] == 1]["weight_carried"]
        fw  = fav.values[0] if len(fav) > 0 else mean
        g["weight_vs_fav"] = fw - g["weight_carried"]
        return g

    return df.groupby("race_id", group_keys=False).apply(_score_group)
