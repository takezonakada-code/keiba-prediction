"""
確率校正モジュール。
Isotonic Regression / Platt Scaling / Temperature Scaling。
EV計算前に必ず校正済み確率を使う。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class IsotonicCalibrator:
    """
    Isotonic Regressionによる確率校正。
    モノトーン制約があるため、競馬の順序予測と相性が良い。
    """

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip", increasing=True)
        self._fitted = False

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        """
        Parameters
        ----------
        scores : モデルの生スコア（0〜1でなくてもOK）
        y_true : 二値ラベル（1=3着以内, 0=4着以下）
        """
        self.model.fit(scores, y_true)
        self._fitted = True
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit()を先に呼んでください。")
        return self.model.predict(scores)

    def calibration_error(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """期待校正誤差（ECE）を計算。"""
        prob_true, prob_pred = calibration_curve(y_true, self.predict(scores), n_bins=n_bins)
        return float(np.mean(np.abs(prob_true - prob_pred)))


class TemperatureScaling:
    """
    Temperature Scaling: スコアを温度パラメータTで割って校正。
    T > 1 → ソフトにする（不確実性を増やす）
    T < 1 → ハードにする（確信度を増やす）
    """

    def __init__(self):
        self.T = 1.0

    def fit(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 1000,
    ) -> "TemperatureScaling":
        """
        NLL最小化でTを最適化（単純勾配降下）。

        Parameters
        ----------
        logits : 未校正ロジット
        y_true : 二値ラベル
        """
        T = 1.0
        eps = 1e-7
        for _ in range(max_iter):
            p = 1.0 / (1.0 + np.exp(-logits / T))
            p = np.clip(p, eps, 1 - eps)
            nll = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
            # 勾配 dNLL/dT
            dp_dT = p * (1 - p) * logits / (T ** 2)
            grad = -np.mean(y_true * dp_dT / p - (1 - y_true) * dp_dT / (1 - p))
            T = T - lr * grad
            T = max(0.01, T)  # 正値制約
        self.T = T
        return self

    def predict(self, logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-logits / self.T))


class PlattScaling:
    """
    Platt Scaling: シグモイドフィッティングによる校正。
    """

    def __init__(self):
        self.lr = LogisticRegression(C=1e10, fit_intercept=True)

    def fit(self, scores: np.ndarray, y_true: np.ndarray) -> "PlattScaling":
        self.lr.fit(scores.reshape(-1, 1), y_true)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(scores.reshape(-1, 1))[:, 1]


def calibrate_and_normalize(
    probs: np.ndarray,
    race_ids: np.ndarray,
) -> np.ndarray:
    """
    校正後の確率をレース内で正規化（合計を1にしない — 各馬独立の3着内確率）。
    同レース内のmax確率が1を超えないようスケールだけ調整する。
    """
    result = probs.copy().astype(float)
    for race_id in np.unique(race_ids):
        mask = race_ids == race_id
        race_probs = result[mask]
        max_p = race_probs.max()
        if max_p > 1.0:
            result[mask] = race_probs / max_p
    return result
