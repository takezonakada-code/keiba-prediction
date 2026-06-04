"""
Harville近似による三連複確率計算 + Benter型市場統合。

- 単勝オッズから市場暗黙勝率を逆数正規化で復元
- モデル勝率と市場勝率をロジット空間でブレンド（Benter型）
- Harville近似で3連複候補確率を高速生成（候補絞り込み用）
- PL exact はplackett_luce.pyで補完
"""
from __future__ import annotations

from itertools import permutations
from typing import Sequence, Tuple

import numpy as np


# ────────────────────────────────────────────────
# 市場勝率の復元
# ────────────────────────────────────────────────
def market_win_probs(win_odds: np.ndarray) -> np.ndarray:
    """
    単勝オッズから市場暗黙勝率を逆数正規化で復元。
    takeout と丸め誤差を吸収する。

    Parameters
    ----------
    win_odds : 各馬の単勝オッズ（JRA表示）

    Returns
    -------
    np.ndarray : 合計1に正規化された勝率
    """
    inv = 1.0 / np.array(win_odds, dtype=float)
    return inv / inv.sum()


# ────────────────────────────────────────────────
# Benter型ロジットブレンド
# ────────────────────────────────────────────────
def blend_model_and_market(
    p_model: np.ndarray,
    p_market: np.ndarray,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> np.ndarray:
    """
    モデル勝率と市場勝率をロジット空間でブレンド（Benter 2段階統合）。

    logit(p*) = alpha * logit(p_model) + beta * logit(p_market)

    Parameters
    ----------
    p_model  : モデルの勝率（Plackett-Luceのworth推定値）
    p_market : 市場の暗黙勝率（market_win_probs の出力）
    alpha    : モデル重み（デフォルト0.7）
    beta     : 市場重み（デフォルト0.3）

    Returns
    -------
    np.ndarray : ブレンド後の正規化勝率
    """
    eps = 1e-6
    p_m = np.clip(p_model,  eps, 1 - eps)
    p_k = np.clip(p_market, eps, 1 - eps)

    logit_m = np.log(p_m / (1 - p_m))
    logit_k = np.log(p_k / (1 - p_k))

    blended_logit = alpha * logit_m + beta * logit_k
    blended = 1.0 / (1.0 + np.exp(-blended_logit))
    return blended / blended.sum()


# ────────────────────────────────────────────────
# Harville近似
# ────────────────────────────────────────────────
def harville_sequence_prob(
    p: np.ndarray,
    order: Tuple[int, ...],
) -> float:
    """
    Harville近似による順序付き着順確率。
    P(i>j>k) = p[i] * p[j]/(1-p[i]) * p[k]/(1-p[i]-p[j])

    Parameters
    ----------
    p     : 各馬の勝率（合計1に正規化済み）
    order : 着順インデックス (i, j, k, ...)

    Returns
    -------
    float
    """
    remaining = 1.0
    prob = 1.0
    for idx in order:
        prob *= p[idx] / remaining
        remaining -= p[idx]
        if remaining <= 0:
            break
    return prob


def harville_trio_prob(p: np.ndarray, combo: Tuple[int, int, int]) -> float:
    """
    Harville近似による3連複（box）確率。
    全6順列の合計。
    """
    total = 0.0
    for perm in permutations(combo):
        total += harville_sequence_prob(p, perm)
    return total


def harville_all_trio_probs(p: np.ndarray) -> dict[Tuple[int, int, int], float]:
    """全C(n,3)の3連複確率をHarville近似で計算。"""
    n = len(p)
    results = {}
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                combo = (i, j, k)
                results[combo] = harville_trio_prob(p, combo)
    return results


# ────────────────────────────────────────────────
# 市場との乖離（Gap）計算
# ────────────────────────────────────────────────
def compute_model_market_gap(
    p_model: np.ndarray,
    p_market: np.ndarray,
) -> np.ndarray:
    """
    馬ごとのモデル vs 市場の乖離（ロジット差）。

    gap_i = logit(p_model_i) - logit(p_market_i)

    正 → モデルが市場より高く評価（妙味あり候補）
    負 → 過剰人気（モデルより市場が過大評価）
    """
    eps = 1e-6
    pm = np.clip(p_model,  eps, 1 - eps)
    pk = np.clip(p_market, eps, 1 - eps)
    return np.log(pm / (1 - pm)) - np.log(pk / (1 - pk))


def combo_model_market_gap(
    p_model: np.ndarray,
    p_market: np.ndarray,
    combo: Tuple[int, int, int],
) -> float:
    """
    3連複組み合わせレベルのモデル vs 市場乖離。

    gap_{ijk} = log(P_model({i,j,k})) - log(P_market({i,j,k}))
    """
    p_trio_model  = harville_trio_prob(p_model,  combo)
    p_trio_market = harville_trio_prob(p_market, combo)
    eps = 1e-10
    return np.log(p_trio_model + eps) - np.log(p_trio_market + eps)


def top_k_from_win_probs(
    p: np.ndarray,
    horse_ids: Sequence,
    win_odds: np.ndarray,
    top_k: int = 10,
    use_blend: bool = True,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> list[dict]:
    """
    市場統合後の3連複候補をEV計算のために上位K件返す。

    Parameters
    ----------
    p         : モデル勝率（softmax worth）
    horse_ids : 馬IDリスト
    win_odds  : 単勝オッズ
    top_k     : 上位何件
    use_blend : Benterブレンドを使うか

    Returns
    -------
    list of {combo_idx, combo_ids, p_hit, gap}
    """
    p_mkt = market_win_probs(win_odds)

    if use_blend:
        p_blended = blend_model_and_market(p, p_mkt, alpha=alpha, beta=beta)
    else:
        p_blended = p

    probs = harville_all_trio_probs(p_blended)
    sorted_combos = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [
        {
            "combo_idx":  combo,
            "combo_ids":  tuple(horse_ids[i] for i in combo),
            "p_hit":      prob,
            "gap":        combo_model_market_gap(p, p_mkt, combo),
        }
        for combo, prob in sorted_combos
    ]
