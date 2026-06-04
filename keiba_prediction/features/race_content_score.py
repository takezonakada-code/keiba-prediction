"""
レース内容スコア（Race Content Score = RCS）。
着順より「負け方の質」を評価する合成指標。
RCS = 0.35*SF + 0.20*late_speed + 0.20*position_gain + 0.15*field_strength + 0.10*trip_proxy
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rcs_single(
    speed_index: float,
    agari3f_rank_pct: float,  # 0=最速, 1=最遅
    corner4_pos: float,
    corner1_pos: float,
    field_size: int,
    field_mean_speed_index: float,
    trip_penalty: float = 0.0,
) -> float:
    """
    1走分のRCSを計算する。

    Parameters
    ----------
    speed_index         : 正規化スピード指数（大きいほど速い）
    agari3f_rank_pct    : 上がり順位%（小さいほど速い → 反転してプラス評価）
    corner4_pos         : 4角通過順
    corner1_pos         : 1角通過順
    field_size          : 頭数
    field_mean_speed_index : 同レースの馬群平均スピード指数（相手強度）
    trip_penalty        : 不利の大きさ（0=不利なし、正=不利あり）

    Returns
    -------
    float : RCS スコア
    """
    # late_speed: 上がり順位を反転（0=最速→1.0, 1=最遅→0.0）
    late_speed = 1.0 - agari3f_rank_pct

    # position_gain: 後方から追い込んだほど高評価（4角より1角が前方だと差し）
    rel_c1 = (corner1_pos - 1) / max(field_size - 1, 1)
    rel_c4 = (corner4_pos - 1) / max(field_size - 1, 1)
    position_gain = rel_c1 - rel_c4   # 正=後方から追い上げ

    # field_strength: 相手の強さ（自分のSFから馬群平均を引いた残差）
    field_strength = speed_index - field_mean_speed_index

    # trip_proxy: 不利補正（スタミナ消耗等の推定）
    trip_proxy = -trip_penalty   # 不利があったほどボーナス

    # Benter型加重和
    rcs = (
        0.35 * speed_index
        + 0.20 * late_speed
        + 0.20 * position_gain
        + 0.15 * field_strength
        + 0.10 * trip_proxy
    )
    return rcs


def aggregate_rcs(
    past_races: pd.DataFrame,
    target_entries: pd.DataFrame,
    as_of_date_col: str = "race_date",
    n_recent: int = 5,
) -> pd.DataFrame:
    """
    target_entries に馬ごとのRCS集計値を追加。

    past_races に必要なカラム:
      - horse_id, race_date, race_id
      - speed_index_raw, agari3f_rank_pct
      - corner4_pos, corner1_pos (なければ0埋め)
      - field_size

    追加カラム:
      - rcs_mean   : 直近N走の平均
      - rcs_max    : 直近N走の最高値
      - rcs_trend  : 最新 - 前3走平均（上昇傾向の検出）
    """
    # 同レースの馬群平均SFを計算
    if "field_mean_sf" not in past_races.columns:
        past_races = past_races.copy()
        past_races["field_mean_sf"] = past_races.groupby("race_id")["speed_index_raw"].transform("mean")

    def _rcs_row(row):
        sf   = row.get("speed_index_raw", 0) or 0
        ap   = row.get("agari3f_rank_pct", 0.5) or 0.5
        c4   = row.get("corner4_pos", 1) or 1
        c1   = row.get("corner1_pos", c4) or c4
        fs   = row.get("field_size", 10) or 10
        fmsf = row.get("field_mean_sf", sf) or sf
        return compute_rcs_single(sf, ap, c4, c1, int(fs), fmsf)

    past = past_races.copy()
    if "rcs" not in past.columns:
        past["rcs"] = past.apply(_rcs_row, axis=1)

    results = []
    for _, entry in target_entries.iterrows():
        horse_id = entry["horse_id"]
        cutoff   = entry[as_of_date_col]

        hist = past[
            (past["horse_id"] == horse_id) &
            (past["race_date"] < cutoff)
        ].sort_values("race_date", ascending=True).tail(n_recent)

        vals = hist["rcs"].values
        if len(vals) == 0:
            results.append({"horse_id": horse_id, "rcs_mean": np.nan,
                            "rcs_max": np.nan, "rcs_trend": np.nan})
        else:
            mean3  = vals[-3:].mean() if len(vals) >= 3 else vals.mean()
            results.append({
                "horse_id":  horse_id,
                "rcs_mean":  vals.mean(),
                "rcs_max":   vals.max(),
                "rcs_trend": vals[-1] - mean3,
            })

    agg = pd.DataFrame(results)
    return target_entries.merge(agg, on="horse_id", how="left")
