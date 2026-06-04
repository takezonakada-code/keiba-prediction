"""
相対上がり3F特徴量の計算。
当走の値は使わない — 過去走の履歴のみ使用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


GROUP_KEYS = ["race_date", "track", "surface", "course_dir", "distance", "track_condition"]


def compute_relative_agari(df: pd.DataFrame) -> pd.DataFrame:
    """
    過去走レコードから相対上がり3Fを計算してDataFrameに追加。

    入力DataFrame (過去走テーブル) に必要なカラム:
      - race_date, track, surface, course_dir, distance, track_condition
      - agari3f_seconds : 当走の上がり3F秒数（過去走レコードにのみ存在）
      - horse_id

    出力: 元のDataFrameに以下を追加
      - agari3f_rank_pct : 同グループ内順位%（0=最速、1=最遅）
      - agari3f_z        : 同グループ内z-score（大きいほど遅い → 符号反転して使う）
    """
    required = {"agari3f_seconds"} | set(GROUP_KEYS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"必要カラムが不足: {missing}")

    df = df.copy()

    # グループ内でのランク（agari3f_seconds は小さいほど速い）
    df["agari3f_rank_pct"] = df.groupby(GROUP_KEYS)["agari3f_seconds"].rank(
        method="average", ascending=True, pct=True
    )

    # グループ内z-score
    group_mean = df.groupby(GROUP_KEYS)["agari3f_seconds"].transform("mean")
    group_std  = df.groupby(GROUP_KEYS)["agari3f_seconds"].transform("std").replace(0, np.nan)
    df["agari3f_z"] = (df["agari3f_seconds"] - group_mean) / group_std

    return df


def aggregate_hist_agari(
    past_races: pd.DataFrame,
    target_entries: pd.DataFrame,
    as_of_date_col: str = "race_date",
    n_recent: int = 5,
) -> pd.DataFrame:
    """
    target_entries（予測対象エントリ）に対し、
    past_races（過去走履歴）から馬ごとの上がり3F特徴量を集計して結合する。

    過去走テーブルには compute_relative_agari() 適用済みであること。

    Parameters
    ----------
    past_races : 全過去走レコード（agari3f_rank_pct / agari3f_z を含む）
    target_entries : 予測対象レース × 馬のエントリ一覧
    as_of_date_col : target_entries の日付カラム名
    n_recent : 集計に使う最近N走

    Returns
    -------
    target_entries に以下カラムを追加した DataFrame:
      - agari3f_rank_pct_hist_mean  : 直近N走の rank_pct 平均（小さいほど速い）
      - agari3f_rank_pct_hist_min   : 直近N走の最高（最速）rank_pct
      - agari3f_z_hist_mean         : 直近N走の z-score 平均（大きいほど遅い）
    """
    past = compute_relative_agari(past_races) if "agari3f_rank_pct" not in past_races.columns \
           else past_races.copy()

    results = []
    for _, entry in target_entries.iterrows():
        horse_id  = entry["horse_id"]
        cutoff    = entry[as_of_date_col]

        hist = past[
            (past["horse_id"] == horse_id) &
            (past["race_date"] < cutoff)
        ].sort_values("race_date", ascending=False).head(n_recent)

        if len(hist) == 0:
            row = {
                "horse_id":                   horse_id,
                "agari3f_rank_pct_hist_mean": np.nan,
                "agari3f_rank_pct_hist_min":  np.nan,
                "agari3f_z_hist_mean":        np.nan,
                "agari3f_hist_n":             0,
            }
        else:
            row = {
                "horse_id":                   horse_id,
                "agari3f_rank_pct_hist_mean": hist["agari3f_rank_pct"].mean(),
                "agari3f_rank_pct_hist_min":  hist["agari3f_rank_pct"].min(),
                "agari3f_z_hist_mean":        hist["agari3f_z"].mean(),
                "agari3f_hist_n":             len(hist),
            }
        results.append(row)

    agg = pd.DataFrame(results)
    return target_entries.merge(agg, on="horse_id", how="left")
