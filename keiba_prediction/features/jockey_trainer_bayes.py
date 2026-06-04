"""
騎手・調教師の条件別成績をベイズ収縮（m-estimate）で推定。
サンプル不足による過学習を防ぐ。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def m_estimate_rate(
    count: float,
    success: float,
    global_rate: float,
    m: float = 100.0,
) -> float:
    """
    m-estimate（ベイズ収縮）による成功率推定。

    formula: (success + m * global_rate) / (count + m)

    Parameters
    ----------
    count       : 出走回数
    success     : 3着以内回数
    global_rate : 全体平均3着内率（事前分布の中心）
    m           : 収縮強度（大きいほど全体平均に寄せる）

    Returns
    -------
    float : 収縮後の3着内率
    """
    return (success + m * global_rate) / (count + m)


def compute_jockey_bayes_rates(
    past_races: pd.DataFrame,
    target_entries: pd.DataFrame,
    as_of_date_col: str = "race_date",
    m: float = 100.0,
    condition_keys: list[str] | None = None,
) -> pd.DataFrame:
    """
    騎手の条件別（コース×距離帯×馬場）ベイズ収縮3着内率を計算。

    past_races に必要なカラム:
      - jockey_id, race_date, track, distance_band, track_condition
      - finish_position

    target_entries に必要なカラム:
      - jockey_id, race_date (as_of_date_col)
      - track, distance_band, track_condition

    Returns
    -------
    target_entries + jockey_bayes_top3_rate カラム
    """
    if condition_keys is None:
        condition_keys = ["track", "distance_band", "track_condition"]

    past = past_races.copy()
    past["_top3"] = (past["finish_position"] <= 3).astype(int)

    # 全体平均（事前分布）
    global_top3_rate = past["_top3"].mean()

    results = []
    for _, entry in target_entries.iterrows():
        jockey_id = entry.get("jockey_id")
        cutoff    = entry[as_of_date_col]

        # as_of_date より前のデータのみ使用（リーク防止）
        hist = past[(past["jockey_id"] == jockey_id) & (past["race_date"] < cutoff)]

        # 条件フィルタ
        for key in condition_keys:
            val = entry.get(key)
            if val is not None:
                hist = hist[hist[key] == val]

        count   = len(hist)
        success = hist["_top3"].sum()
        rate    = m_estimate_rate(count, success, global_top3_rate, m)

        results.append({
            "jockey_id":                jockey_id,
            "_entry_idx":               entry.name,
            "jockey_bayes_top3_rate":   rate,
            "jockey_cond_count":        count,
        })

    agg = pd.DataFrame(results).set_index("_entry_idx")
    result_df = target_entries.copy()
    result_df["jockey_bayes_top3_rate"] = agg["jockey_bayes_top3_rate"]
    result_df["jockey_cond_count"]      = agg["jockey_cond_count"]
    return result_df


def compute_trainer_bayes_rates(
    past_races: pd.DataFrame,
    target_entries: pd.DataFrame,
    as_of_date_col: str = "race_date",
    m: float = 50.0,
    condition_keys: list[str] | None = None,
) -> pd.DataFrame:
    """
    調教師の条件別ベイズ収縮3着内率。（騎手版と同構造）

    Returns
    -------
    target_entries + trainer_bayes_top3_rate カラム
    """
    if condition_keys is None:
        condition_keys = ["track", "distance_band"]

    past = past_races.copy()
    past["_top3"] = (past["finish_position"] <= 3).astype(int)
    global_top3_rate = past["_top3"].mean()

    results = []
    for _, entry in target_entries.iterrows():
        trainer_id = entry.get("trainer_id")
        cutoff     = entry[as_of_date_col]

        hist = past[(past["trainer_id"] == trainer_id) & (past["race_date"] < cutoff)]
        for key in condition_keys:
            val = entry.get(key)
            if val is not None:
                hist = hist[hist[key] == val]

        count   = len(hist)
        success = hist["_top3"].sum()
        rate    = m_estimate_rate(count, success, global_top3_rate, m)

        results.append({
            "trainer_id":                trainer_id,
            "_entry_idx":                entry.name,
            "trainer_bayes_top3_rate":   rate,
        })

    agg = pd.DataFrame(results).set_index("_entry_idx")
    result_df = target_entries.copy()
    result_df["trainer_bayes_top3_rate"] = agg["trainer_bayes_top3_rate"]
    return result_df
