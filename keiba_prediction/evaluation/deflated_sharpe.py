"""
Deflated Sharpe Ratio（DSR）。
バックテストの多重テスト問題を考慮した戦略評価指標。
偶然勝った戦略を弾くために使う。
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def sharpe_ratio(returns: np.ndarray) -> float:
    """週次リターンのSharpe比（年率化）。"""
    if returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * math.sqrt(52)


def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    sharpe_ref: float = 0.0,
) -> float:
    """
    Probabilistic Sharpe Ratio (PSR)。
    観測されたSR >= sharpe_ref である確率。

    Parameters
    ----------
    returns    : 期間リターン系列
    sharpe_ref : 基準SR（デフォルト0=勝てるかどうか）

    Returns
    -------
    float : 0〜1の確率
    """
    n  = len(returns)
    sr = returns.mean() / (returns.std() + 1e-10)

    skew  = _skewness(returns)
    kurt  = _kurtosis(returns)  # excess kurtosis

    # PSR = Φ( (SR - SR_ref) * sqrt(n-1) / sqrt(1 - skew*SR + (kurt-1)/4 * SR^2) )
    var_sr = (1 - skew * sr + (kurt + 1) / 4 * sr ** 2)
    if var_sr <= 0:
        return float(norm.cdf(0.0))

    z = (sr - sharpe_ref) * math.sqrt(n - 1) / math.sqrt(var_sr)
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    expected_max_sr: float | None = None,
) -> float:
    """
    Deflated Sharpe Ratio (DSR)。

    複数の戦略を試したとき、その中の最良の結果が偶然であるかを補正する。

    Parameters
    ----------
    returns         : 最良戦略の期間リターン系列
    n_trials        : 試した戦略/パラメータセットの総数
    expected_max_sr : n_trials回の試行から期待される最大SR
                      (None の場合は近似式で自動計算)

    Returns
    -------
    float : DSR（0〜1の確率）。0.95以上が「有意に勝てている」目安。
    """
    n = len(returns)

    if expected_max_sr is None:
        # Marcos Lopez de Prado の近似式
        # E[max SR] ≈ (1 - γ) Z^{-1}(1 - 1/n_trials) + γ Z^{-1}(1 - 1/(n_trials*e))
        gamma = 0.5772  # Euler-Mascheroni定数
        z1 = norm.ppf(1 - 1.0 / n_trials)
        z2 = norm.ppf(1 - 1.0 / (n_trials * math.e))
        expected_max_sr = (1 - gamma) * z1 + gamma * z2

    return probabilistic_sharpe_ratio(returns, sharpe_ref=expected_max_sr)


def _skewness(x: np.ndarray) -> float:
    """歪度。"""
    n = len(x)
    if n < 3:
        return 0.0
    m = x.mean()
    std = x.std()
    if std == 0:
        return 0.0
    return float(np.mean(((x - m) / std) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    """超過尖度（excess kurtosis）。正規分布=0。"""
    n = len(x)
    if n < 4:
        return 0.0
    m = x.mean()
    std = x.std()
    if std == 0:
        return 0.0
    return float(np.mean(((x - m) / std) ** 4) - 3)


def evaluate_strategy(
    stakes: np.ndarray,
    payouts: np.ndarray,
    n_trials: int = 1,
    periods_per_year: int = 52,
) -> dict:
    """
    戦略の総合評価指標を返す。

    Returns
    -------
    dict: {sharpe, psr, dsr, skew, kurtosis, n_periods}
    """
    pnl = payouts - stakes
    weekly = _aggregate_weekly(pnl)

    sr  = sharpe_ratio(weekly)
    psr = probabilistic_sharpe_ratio(weekly)
    dsr = deflated_sharpe_ratio(weekly, n_trials=n_trials)

    return {
        "sharpe":    round(sr, 4),
        "psr":       round(psr, 4),
        "dsr":       round(dsr, 4),
        "skew":      round(_skewness(weekly), 4),
        "kurtosis":  round(_kurtosis(weekly), 4),
        "n_periods": len(weekly),
        "n_trials":  n_trials,
        "verdict":   "有意" if dsr >= 0.95 else "不明" if dsr >= 0.80 else "非有意",
    }


def _aggregate_weekly(pnl: np.ndarray) -> np.ndarray:
    """日次PnLを週次に集約（簡易版: 7日ごとに合計）。"""
    n = len(pnl)
    if n == 0:
        return np.array([])
    weeks = max(1, n // 7)
    return np.array([pnl[i*7:(i+1)*7].sum() for i in range(weeks)])
