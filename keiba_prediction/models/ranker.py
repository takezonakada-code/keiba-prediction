"""
LGBMRanker（lambdarank / NDCG@3）。
目的変数: relevance = max(0, 4 - finish_position)
  1着=3, 2着=2, 3着=1, 4着以下=0
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import MODEL_DIR, RELEVANCE_MAP
from features.feature_specs import assert_no_forbidden


def build_relevance(finish_positions: pd.Series) -> pd.Series:
    """着順から relevance ラベルを生成。"""
    return finish_positions.map(lambda p: RELEVANCE_MAP.get(p, 0))


def make_group_array(df: pd.DataFrame, race_id_col: str = "race_id") -> np.ndarray:
    """race_id ごとの頭数リスト（LGB group引数用）。"""
    return df.groupby(race_id_col, sort=False).size().values


def create_ranker(
    n_estimators: int = 800,
    learning_rate: float = 0.03,
    num_leaves: int = 63,
    min_child_samples: int = 30,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
) -> lgb.LGBMRanker:
    return lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[3],
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        n_jobs=-1,
        importance_type="gain",
    )


def train_ranker(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    group_train: np.ndarray,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    group_val: np.ndarray | None = None,
    early_stopping_rounds: int = 100,
    **ranker_kwargs,
) -> lgb.LGBMRanker:
    """
    LGBMRanker を訓練して返す。

    Parameters
    ----------
    X_train, y_train : 訓練特徴量・relevanceラベル
    group_train      : 訓練データのレースごと頭数
    X_val, y_val     : 検証データ（early stopping用、省略可）
    group_val        : 検証データの頭数
    """
    assert_no_forbidden(list(X_train.columns))

    ranker = create_ranker(**ranker_kwargs)

    callbacks = [lgb.log_evaluation(period=50)]
    if X_val is not None and early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
        ranker.fit(
            X_train, y_train,
            group=group_train,
            eval_set=[(X_val, y_val)],
            eval_group=[group_val],
            callbacks=callbacks,
        )
    else:
        ranker.fit(X_train, y_train, group=group_train, callbacks=callbacks)

    return ranker


def predict_scores(ranker: lgb.LGBMRanker, X: pd.DataFrame) -> np.ndarray:
    """予測スコア（生スコア）を返す。高いほど上位着順の期待。"""
    return ranker.predict(X)


def save_ranker(ranker: lgb.LGBMRanker, name: str = "ranker") -> Path:
    path = MODEL_DIR / f"{name}.txt"
    ranker.booster_.save_model(str(path))
    print(f"モデル保存: {path}")
    return path


def load_ranker(name: str = "ranker") -> lgb.LGBMRanker:
    path = MODEL_DIR / f"{name}.txt"
    booster = lgb.Booster(model_file=str(path))
    ranker = lgb.LGBMRanker()
    ranker._Booster = booster
    return ranker
