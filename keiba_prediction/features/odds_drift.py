"""
締切前オッズドリフト特徴量。
NAR公式CSVの中間オッズ（約2分更新）とJRA-VANの速報オッズを活用。
「締切30分前」「10分前」「2分前」「最終」の差分を特徴量化する。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


SNAPSHOT_WINDOWS = {
    "30m": 30,
    "10m": 10,
    "2m":  2,
}


def compute_odds_drift_features(
    odds_snapshots: pd.DataFrame,
    target_race_id: str,
    target_horse_id: str | None = None,
) -> dict[str, float]:
    """
    1レース分のオッズスナップショット履歴からドリフト特徴量を計算。

    Parameters
    ----------
    odds_snapshots : 以下のカラムを持つDataFrame
        - race_id, horse_id, snapshot_time (datetime), win_odds
    target_race_id : 対象レースID
    target_horse_id : 対象馬ID（None の場合はレース全馬の集計のみ）

    Returns
    -------
    dict: {
        win_odds_final, win_odds_30m, win_odds_10m, win_odds_2m,
        drift_30m_to_final, drift_10m_to_final, drift_2m_to_final,
        drift_acceleration (10m→2m の変化 / 30m→10m の変化),
        is_drifter (最終オッズが30分前より10%以上上昇)
    }
    """
    race_snaps = odds_snapshots[odds_snapshots["race_id"] == target_race_id].copy()
    if len(race_snaps) == 0:
        return {}

    race_snaps["snapshot_time"] = pd.to_datetime(race_snaps["snapshot_time"])

    # 最終スナップショット時刻
    t_final = race_snaps["snapshot_time"].max()

    def _get_odds_near(minutes_before: int, horse_id: Optional[str]) -> float | None:
        cutoff = t_final - pd.Timedelta(minutes=minutes_before)
        window = race_snaps[race_snaps["snapshot_time"] <= cutoff]
        if horse_id:
            window = window[window["horse_id"] == horse_id]
        if len(window) == 0:
            return None
        return window.sort_values("snapshot_time").iloc[-1]["win_odds"]

    def _get_final_odds(horse_id: Optional[str]) -> float | None:
        snaps = race_snaps
        if horse_id:
            snaps = snaps[snaps["horse_id"] == horse_id]
        if len(snaps) == 0:
            return None
        return snaps.sort_values("snapshot_time").iloc[-1]["win_odds"]

    hid = target_horse_id
    final = _get_final_odds(hid)
    o30   = _get_odds_near(30, hid)
    o10   = _get_odds_near(10, hid)
    o2    = _get_odds_near(2,  hid)

    def _drift(prev, curr):
        if prev is None or curr is None or prev == 0:
            return np.nan
        return (curr - prev) / prev   # 正=上昇（人気薄化）、負=下降（人気上昇）

    drift_30 = _drift(o30, final)
    drift_10 = _drift(o10, final)
    drift_2  = _drift(o2,  final)

    drift_30_10 = _drift(o30, o10)
    drift_10_2  = _drift(o10, o2)
    acceleration = drift_10_2 / drift_30_10 if drift_30_10 and drift_30_10 != 0 else np.nan

    is_drifter = int(drift_30 is not None and drift_30 > 0.10)   # 30分前比+10%以上上昇

    return {
        "win_odds_final":       final,
        "win_odds_30m":         o30,
        "win_odds_10m":         o10,
        "win_odds_2m":          o2,
        "drift_30m_to_final":   drift_30,
        "drift_10m_to_final":   drift_10,
        "drift_2m_to_final":    drift_2,
        "drift_acceleration":   acceleration,
        "is_drifter":           is_drifter,
    }


def add_drift_features(
    target_entries: pd.DataFrame,
    odds_snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """
    target_entries の全馬にオッズドリフト特徴量を追加。

    Parameters
    ----------
    target_entries : horse_id, race_id を持つDataFrame
    odds_snapshots : race_id, horse_id, snapshot_time, win_odds を持つDataFrame

    Returns
    -------
    target_entries + drift特徴量
    """
    drift_records = []
    for _, row in target_entries.iterrows():
        drift = compute_odds_drift_features(
            odds_snapshots, row["race_id"], row["horse_id"]
        )
        drift["horse_id"] = row["horse_id"]
        drift["race_id"]  = row["race_id"]
        drift_records.append(drift)

    drift_df = pd.DataFrame(drift_records)
    merge_keys = [k for k in ["race_id", "horse_id"] if k in drift_df.columns]
    return target_entries.merge(drift_df, on=merge_keys, how="left")
