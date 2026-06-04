"""
出走間隔の非線形基底展開（ガウスRBF）。
days_since_last の単純な1列使用は禁止。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import REST_RBF_CENTERS, REST_RBF_SIGMA


def rbf(x: float, center: float, sigma: float = REST_RBF_SIGMA) -> float:
    """ガウスRBF基底関数。"""
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


def compute_rest_features(
    days_since_last: float | np.ndarray,
    centers: list[int] | None = None,
    sigma: float = REST_RBF_SIGMA,
) -> dict[str, float | np.ndarray]:
    """
    出走間隔をRBF基底に展開する。

    Parameters
    ----------
    days_since_last : float or array
        前走からの日数
    centers : list[int]
        RBFセンター（デフォルトは config の REST_RBF_CENTERS）
    sigma : float
        RBF幅

    Returns
    -------
    dict: {"rest_rbf_{c}": value, ..., "layoff_150plus": 0/1}
    """
    if centers is None:
        centers = REST_RBF_CENTERS

    features = {}
    for c in centers:
        features[f"rest_rbf_{c}"] = rbf(days_since_last, center=c, sigma=sigma)

    # 長期休養ダミー（150日以上）
    features["layoff_150plus"] = (np.array(days_since_last) >= 150).astype(int) \
        if hasattr(days_since_last, "__len__") else int(days_since_last >= 150)

    return features


def add_rest_features(
    df: pd.DataFrame,
    days_col: str = "days_since_last",
    centers: list[int] | None = None,
    sigma: float = REST_RBF_SIGMA,
) -> pd.DataFrame:
    """
    DataFrame に出走間隔RBF特徴量を追加して返す。

    Parameters
    ----------
    df : エントリ DataFrame（days_since_last 列を持つ）
    days_col : 前走からの日数カラム名
    centers : RBFセンター
    sigma : RBF幅

    Returns
    -------
    DataFrame with rest_rbf_* and layoff_150plus columns added
    """
    if centers is None:
        centers = REST_RBF_CENTERS

    days = df[days_col].values

    result = df.copy()
    for c in centers:
        result[f"rest_rbf_{c}"] = rbf(days, center=c, sigma=sigma)

    result["layoff_150plus"] = (days >= 150).astype(int)

    return result


def compute_days_since_last(
    target_entries: pd.DataFrame,
    past_races: pd.DataFrame,
    as_of_date_col: str = "race_date",
) -> pd.DataFrame:
    """
    target_entries に days_since_last を計算して追加。

    Parameters
    ----------
    target_entries : 予測対象エントリ（horse_id, race_date 必須）
    past_races     : 全過去走記録（horse_id, race_date 必須）

    Returns
    -------
    target_entries + days_since_last カラム
    """
    past = past_races[["horse_id", "race_date"]].copy()
    past["race_date"] = pd.to_datetime(past["race_date"])

    results = []
    for _, entry in target_entries.iterrows():
        horse_id = entry["horse_id"]
        cutoff   = pd.to_datetime(entry[as_of_date_col])

        hist = past[
            (past["horse_id"] == horse_id) &
            (past["race_date"] < cutoff)
        ]
        if len(hist) == 0:
            days = np.nan
        else:
            last_race = hist["race_date"].max()
            days = (cutoff - last_race).days

        results.append({"horse_id": horse_id, "days_since_last": days})

    agg = pd.DataFrame(results)
    merged = target_entries.merge(agg, on="horse_id", how="left")
    return add_rest_features(merged)
