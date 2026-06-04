"""
コンボレベルの残差モデル（二段目）。
Harville/PL近似の誤差を補正する。
入力: 組み合わせごとの特徴量（スタイル相性・枠バイアス等）
出力: p_hit の補正係数
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from features.feature_specs import assert_no_forbidden


def build_combo_features(
    entries_df: pd.DataFrame,
    race_id: str,
) -> pd.DataFrame:
    """
    1レース分の全C(n,3)組み合わせの特徴量を生成する。

    entries_df : 1レース分のエントリ（特徴量付き）
    race_id    : レースID

    追加する組み合わせ特徴量:
      - style_variance      : 3頭の脚質スコア分散（逃げ混在 or 全差しか）
      - speed_std           : 3頭のスピード指数標準偏差
      - min_bayes_jockey    : 3頭のjockey_bayes_top3_rate 最小値
      - max_bayes_jockey    : 最大値
      - sum_p_top3          : 3頭のp_top3合計（能力集中度）
      - max_drift           : 3頭の最大オッズドリフト（絶対値）
    """
    n = len(entries_df)
    if n < 3:
        return pd.DataFrame()

    records = []
    for combo in combinations(range(n), 3):
        trio = entries_df.iloc[list(combo)]

        style_var = trio["style_score_wavg3"].var() if "style_score_wavg3" in trio.columns else np.nan
        speed_std = trio["speed_index_mean"].std()  if "speed_index_mean"  in trio.columns else np.nan
        min_jk    = trio["jockey_bayes_top3_rate"].min() if "jockey_bayes_top3_rate" in trio.columns else np.nan
        max_jk    = trio["jockey_bayes_top3_rate"].max() if "jockey_bayes_top3_rate" in trio.columns else np.nan
        sum_p3    = trio["p_top3"].sum()                  if "p_top3"                in trio.columns else np.nan
        max_drift = trio["drift_30m_to_final"].abs().max() if "drift_30m_to_final"   in trio.columns else np.nan

        records.append({
            "race_id":       race_id,
            "combo_idx":     combo,
            "combo_ids":     tuple(entries_df.iloc[list(combo)]["horse_id"].values),
            "style_variance": style_var,
            "speed_std":      speed_std,
            "min_bayes_jk":   min_jk,
            "max_bayes_jk":   max_jk,
            "sum_p_top3":     sum_p3,
            "max_drift":      max_drift,
        })

    return pd.DataFrame(records)


COMBO_FEATURE_COLS = [
    "style_variance", "speed_std", "min_bayes_jk", "max_bayes_jk",
    "sum_p_top3", "max_drift",
]


def train_combo_residual(
    combo_df: pd.DataFrame,
    y_col: str = "is_hit",
    n_estimators: int = 300,
    learning_rate: float = 0.05,
) -> lgb.LGBMClassifier:
    """
    コンボ特徴量で的中/外れを予測するLGBM分類器。
    一段目のp_hitを補正するための残差モデル。

    Parameters
    ----------
    combo_df : build_combo_features() の出力 + is_hit(0/1) と p_hit_base
    """
    feature_cols = [c for c in COMBO_FEATURE_COLS if c in combo_df.columns]
    if "p_hit_base" in combo_df.columns:
        feature_cols = ["p_hit_base"] + feature_cols

    X = combo_df[feature_cols].fillna(0)
    y = combo_df[y_col].values

    clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=31,
        min_child_samples=20,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def predict_combo_residual(
    clf: lgb.LGBMClassifier,
    combo_df: pd.DataFrame,
) -> np.ndarray:
    """補正後のp_hitを返す。"""
    feature_cols = [c for c in COMBO_FEATURE_COLS if c in combo_df.columns]
    if "p_hit_base" in combo_df.columns:
        feature_cols = ["p_hit_base"] + feature_cols
    X = combo_df[feature_cols].fillna(0)
    return clf.predict_proba(X)[:, 1]
