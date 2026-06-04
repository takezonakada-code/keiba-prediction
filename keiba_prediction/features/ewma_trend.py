"""
直近トレンド特徴量（EWMA + 傾き）。
直近5走のEWMAと前3走平均との差、傾きを生成する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ewma(values: np.ndarray, span: int = 3) -> float:
    """直近N走のEWMAを計算（最新が末尾）。"""
    if len(values) == 0:
        return np.nan
    alpha = 2.0 / (span + 1)
    result = values[0]
    for v in values[1:]:
        result = alpha * v + (1 - alpha) * result
    return result


def _trend_slope(values: np.ndarray) -> float:
    """時系列の線形回帰傾き（正=上昇傾向）。"""
    n = len(values)
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=float)
    slope = np.polyfit(x, values, 1)[0]
    return slope


def aggregate_ewma_features(
    past_races: pd.DataFrame,
    target_entries: pd.DataFrame,
    score_col: str = "speed_index_raw",
    as_of_date_col: str = "race_date",
    n_recent: int = 5,
    ewma_span: int = 3,
) -> pd.DataFrame:
    """
    target_entries にEWMAトレンド特徴量を追加して返す。

    past_races に必要なカラム:
      - horse_id, race_date, {score_col}

    追加カラム:
      - sf_ewma          : 直近N走のEWMA
      - sf_trend_slope   : 直近N走の傾き
      - sf_latest_vs_mean: 最新走 - 前3走平均（上昇馬検出）
      - sf_trend_n       : 実績走数
    """
    results = []
    for _, entry in target_entries.iterrows():
        horse_id = entry["horse_id"]
        cutoff   = entry[as_of_date_col]

        hist = past_races[
            (past_races["horse_id"] == horse_id) &
            (past_races["race_date"] < cutoff) &
            (past_races[score_col].notna())
        ].sort_values("race_date", ascending=True).tail(n_recent)

        vals = hist[score_col].values

        if len(vals) == 0:
            results.append({
                "horse_id":           horse_id,
                "sf_ewma":            np.nan,
                "sf_trend_slope":     np.nan,
                "sf_latest_vs_mean":  np.nan,
                "sf_trend_n":         0,
            })
        else:
            ewma  = _ewma(vals, span=ewma_span)
            slope = _trend_slope(vals)
            latest = vals[-1]
            mean3  = vals[-3:].mean() if len(vals) >= 3 else vals.mean()
            results.append({
                "horse_id":           horse_id,
                "sf_ewma":            ewma,
                "sf_trend_slope":     slope,
                "sf_latest_vs_mean":  latest - mean3,
                "sf_trend_n":         len(vals),
            })

    agg = pd.DataFrame(results)
    return target_entries.merge(agg, on="horse_id", how="left")
