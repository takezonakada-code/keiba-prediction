"""
スピード指数の計算。
= クラス内偏差値 × 距離帯補正 × going別補正
going別補正は 良/稍重/重/不良 で独立計算（線形補間禁止）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import DISTANCE_BANDS


GOING_TYPES = ["良", "稍重", "重", "不良"]


def _distance_band(distance: int) -> str:
    for band, (lo, hi) in DISTANCE_BANDS.items():
        if lo <= distance <= hi:
            return band
    return "unknown"


def compute_speed_index(
    past_races: pd.DataFrame,
) -> pd.DataFrame:
    """
    過去走レコードにスピード指数を追加して返す。

    past_races に必要なカラム:
      - race_date, horse_id, race_class, distance, track_condition
      - race_time_seconds : ゴールタイム（秒）
      - field_size

    追加カラム:
      - distance_band      : 距離帯
      - speed_index_raw    : クラス × 距離帯 × going 内の偏差値（平均50、σ10）
      - speed_index_norm   : 0〜1正規化版（モデル入力用）
    """
    required = {"race_date", "horse_id", "race_class", "distance",
                "track_condition", "race_time_seconds"}
    missing = required - set(past_races.columns)
    if missing:
        raise ValueError(f"必要カラムが不足: {missing}")

    df = past_races.copy()
    df["distance_band"] = df["distance"].apply(_distance_band)

    # going は4分類に統一
    df["going_cat"] = df["track_condition"].apply(
        lambda x: x if x in GOING_TYPES else "良"
    )

    group_keys = ["race_class", "distance_band", "going_cat"]

    group_mean = df.groupby(group_keys)["race_time_seconds"].transform("mean")
    group_std  = df.groupby(group_keys)["race_time_seconds"].transform("std").replace(0, np.nan)

    # タイムは小さいほど速い → z-scoreの符号を反転
    z = -(df["race_time_seconds"] - group_mean) / group_std

    # 偏差値スケール（平均50、σ10）
    df["speed_index_raw"] = 50 + 10 * z

    # 0〜1正規化（グループ内）
    group_min = df.groupby(group_keys)["speed_index_raw"].transform("min")
    group_max = df.groupby(group_keys)["speed_index_raw"].transform("max")
    denom = (group_max - group_min).replace(0, np.nan)
    df["speed_index_norm"] = (df["speed_index_raw"] - group_min) / denom

    return df


def aggregate_speed_index(
    past_races: pd.DataFrame,
    target_entries: pd.DataFrame,
    as_of_date_col: str = "race_date",
    n_recent: int = 5,
) -> pd.DataFrame:
    """
    target_entries に馬ごとのスピード指数集計値を追加。

    Returns
    -------
    target_entries + 以下カラム:
      - speed_index_mean  : 直近N走の平均
      - speed_index_max   : 直近N走の最高値
      - speed_index_std   : 直近N走のばらつき
      - speed_index_n     : 実績走数
    """
    if "speed_index_raw" not in past_races.columns:
        past = compute_speed_index(past_races)
    else:
        past = past_races.copy()

    results = []
    for _, entry in target_entries.iterrows():
        horse_id = entry["horse_id"]
        cutoff   = entry[as_of_date_col]
        dist     = entry.get("distance")
        going    = entry.get("track_condition", "良")

        # 同距離帯・同going でフィルタ（クロス分析のため）
        dist_band = _distance_band(dist) if dist is not None else None
        going_cat = going if going in GOING_TYPES else "良"

        mask = (past["horse_id"] == horse_id) & (past["race_date"] < cutoff)
        if dist_band:
            mask &= (past["distance_band"] == dist_band)

        hist = past[mask].sort_values("race_date", ascending=False).head(n_recent)

        if len(hist) == 0:
            # 距離帯フィルタなしでフォールバック
            hist = past[
                (past["horse_id"] == horse_id) & (past["race_date"] < cutoff)
            ].sort_values("race_date", ascending=False).head(n_recent)

        if len(hist) == 0:
            row = {
                "horse_id":         horse_id,
                "speed_index_mean": np.nan,
                "speed_index_max":  np.nan,
                "speed_index_std":  np.nan,
                "speed_index_n":    0,
            }
        else:
            row = {
                "horse_id":         horse_id,
                "speed_index_mean": hist["speed_index_raw"].mean(),
                "speed_index_max":  hist["speed_index_raw"].max(),
                "speed_index_std":  hist["speed_index_raw"].std(),
                "speed_index_n":    len(hist),
            }
        results.append(row)

    agg = pd.DataFrame(results)
    return target_entries.merge(agg, on="horse_id", how="left")
