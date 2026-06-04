"""
評価指標: LogLoss / Brier / ROI / Racing Sharpe Ratio。
AUCは参考値として残すが主KPIから外す。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


# ────────────────────────────────────────────────
# 基本指標
# ────────────────────────────────────────────────
def compute_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """LogLoss（主指標）。低いほど良い。"""
    return log_loss(y_true, y_prob)


def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier Score（主指標）。低いほど良い。"""
    return brier_score_loss(y_true, y_prob)


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC（参考値）。walk-forward OOF ベースで計算すること。"""
    return roc_auc_score(y_true, y_score)


# ────────────────────────────────────────────────
# 投資指標
# ────────────────────────────────────────────────
def compute_roi(
    stakes: np.ndarray | pd.Series,
    payouts: np.ndarray | pd.Series,
) -> float:
    """
    ROI（回収率%）。
    ROI = 100 × Σpayout / Σstake
    """
    total_stake = np.sum(stakes)
    if total_stake == 0:
        return 0.0
    return 100.0 * np.sum(payouts) / total_stake


def compute_racing_sharpe(
    stakes: np.ndarray | pd.Series,
    payouts: np.ndarray | pd.Series,
    periods_per_year: int = 52,
) -> float:
    """
    Racing Sharpe Ratio（週次ベース）。

    各ベットの損益 = payout - stake。
    Sharpe = mean(pnl) / std(pnl) × sqrt(periods_per_year)

    Parameters
    ----------
    stakes, payouts : 各ベットの賭け金・払戻金
    periods_per_year : 1年あたりの期間数（週次=52、日次=365）
    """
    pnl = np.asarray(payouts, dtype=float) - np.asarray(stakes, dtype=float)
    if pnl.std() == 0:
        return 0.0
    return (pnl.mean() / pnl.std()) * math.sqrt(periods_per_year)


def compute_max_drawdown(
    stakes: np.ndarray | pd.Series,
    payouts: np.ndarray | pd.Series,
) -> float:
    """
    最大ドローダウン（円）。累積損益の最高点からの最大下落。
    """
    pnl = np.asarray(payouts, dtype=float) - np.asarray(stakes, dtype=float)
    cumulative = np.cumsum(pnl)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    return float(drawdown.max())


# ────────────────────────────────────────────────
# 全指標をまとめて計算
# ────────────────────────────────────────────────
def full_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    stakes: np.ndarray | None = None,
    payouts: np.ndarray | None = None,
) -> dict[str, float]:
    """
    全評価指標をまとめて返す。

    Returns
    -------
    dict: {logloss, brier, auc, roi_pct, sharpe, max_drawdown_yen}
    """
    result = {
        "logloss": compute_logloss(y_true, y_prob),
        "brier":   compute_brier(y_true, y_prob),
        "auc":     compute_auc(y_true, y_prob),
    }
    if stakes is not None and payouts is not None:
        result["roi_pct"]          = compute_roi(stakes, payouts)
        result["sharpe"]           = compute_racing_sharpe(stakes, payouts)
        result["max_drawdown_yen"] = compute_max_drawdown(stakes, payouts)

    return result


def weekly_roi_series(
    df: pd.DataFrame,
    date_col: str = "race_date",
    stake_col: str = "stake",
    payout_col: str = "payout",
) -> pd.DataFrame:
    """
    週次ROIの時系列を返す。Streamlit チャート用。
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["week"] = df[date_col].dt.to_period("W")

    weekly = df.groupby("week").apply(
        lambda g: pd.Series({
            "total_stake":  g[stake_col].sum(),
            "total_payout": g[payout_col].sum(),
            "tickets":      len(g),
            "hits":         g["is_hit"].sum() if "is_hit" in g.columns else 0,
        })
    ).reset_index()

    weekly["roi_pct"] = 100.0 * weekly["total_payout"] / weekly["total_stake"].replace(0, np.nan)
    return weekly
