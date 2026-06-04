"""
脚質スコアの計算（3走加重平均 + 安定度）。
直線長との交互作用項も生成する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import STYLE_WEIGHTS


def _relative_pos(corner_pos: float, field_size: int) -> float:
    """
    コーナー通過順を相対位置に変換（0=先頭、1=最後方）。
    """
    if field_size <= 1:
        return 0.0
    return (corner_pos - 1) / (field_size - 1)


def compute_style_score_single(
    corner4_pos: float,
    field_size: int,
) -> float:
    """
    1走分の脚質スコア（4角相対位置）。
    小さいほど先行、大きいほど追い込み。
    """
    return _relative_pos(corner4_pos, field_size)


def aggregate_style_features(
    past_races: pd.DataFrame,
    target_entries: pd.DataFrame,
    as_of_date_col: str = "race_date",
    weights: list[float] | None = None,
    straight_length_col: str = "straight_length",
) -> pd.DataFrame:
    """
    target_entries に脚質特徴量を追加して返す。

    past_races に必要なカラム:
      - horse_id, race_date
      - corner4_pos  : 4角通過順
      - field_size   : 頭数

    target_entries に必要なカラム:
      - horse_id, race_date (as_of_date_col)
      - straight_length : コース直線長 (m)

    追加されるカラム:
      - style_score_wavg3  : 直近3走の加重平均脚質スコア
      - style_vol3         : 直近3走の標準偏差（安定度）
      - style_x_straight   : style_score_wavg3 × straight_length（交互作用）
      - style_hist_n       : 実績走数
    """
    if weights is None:
        weights = STYLE_WEIGHTS

    required_past = {"horse_id", "race_date", "corner4_pos", "field_size"}
    missing = required_past - set(past_races.columns)
    if missing:
        raise ValueError(f"past_races に必要カラムが不足: {missing}")

    past = past_races.copy()
    past["_style_score"] = past.apply(
        lambda r: compute_style_score_single(r["corner4_pos"], r["field_size"]),
        axis=1,
    )

    results = []
    for _, entry in target_entries.iterrows():
        horse_id = entry["horse_id"]
        cutoff   = entry[as_of_date_col]

        hist = past[
            (past["horse_id"] == horse_id) &
            (past["race_date"] < cutoff)
        ].sort_values("race_date", ascending=False).head(len(weights))

        scores = hist["_style_score"].values  # 直近順

        if len(scores) == 0:
            wavg = np.nan
            vol  = np.nan
        else:
            w = np.array(weights[:len(scores)])
            w = w / w.sum()
            wavg = np.dot(w, scores)
            vol  = scores.std() if len(scores) > 1 else 0.0

        straight = entry.get(straight_length_col, np.nan)
        style_x_straight = wavg * straight if (not np.isnan(wavg) and not np.isnan(straight)) else np.nan

        results.append({
            "horse_id":          horse_id,
            "style_score_wavg3": wavg,
            "style_vol3":        vol,
            "style_x_straight":  style_x_straight,
            "style_hist_n":      len(scores),
        })

    agg = pd.DataFrame(results)
    return target_entries.merge(agg, on="horse_id", how="left")
