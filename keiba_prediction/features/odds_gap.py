"""
オッズ歪み検出（市場の過小評価を検出する）

gap = log(model_prob) - log(market_prob)
gap > 0 → モデルが市場より高く評価している穴馬
gap > 0.5 → 強い過小評価シグナル

Benter型の市場統合とHarville近似も含む。
"""
from __future__ import annotations

from itertools import permutations
from typing import Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────
# 市場確率の復元
# ──────────────────────────────────────────────────
def market_win_probs(win_odds: np.ndarray) -> np.ndarray:
    """
    単勝オッズから市場暗黙勝率を逆数正規化で復元。
    控除率と丸め誤差を吸収する。
    """
    inv = 1.0 / np.maximum(win_odds, 0.1)
    return inv / inv.sum()


def harville_trio_prob(
    win_probs: np.ndarray,
    combo: Tuple[int, int, int],
) -> float:
    """Harville近似で3連複確率を高速計算。"""
    total = 0.0
    for perm in permutations(combo):
        i, j, k = perm
        p_i = win_probs[i]
        p_j = win_probs[j]
        p_k = win_probs[k]
        denom_j = 1.0 - p_i
        denom_k = 1.0 - p_i - p_j
        if denom_j > 0 and denom_k > 0:
            total += p_i * (p_j / denom_j) * (p_k / denom_k)
    return total


# ──────────────────────────────────────────────────
# オッズ乖離（gap）計算
# ──────────────────────────────────────────────────
def compute_odds_gap(
    model_scores: np.ndarray,
    win_odds: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    馬ごとのオッズ乖離スコアを計算する。

    gap_i = logit(p_model_i) - logit(p_market_i)
    正 → モデルが市場より高評価（過小評価された穴馬）
    負 → モデルが市場より低評価（過剰人気）

    Parameters
    ----------
    model_scores : モデルスコア（softmaxで正規化済み）
    win_odds     : 各馬の単勝オッズ

    Returns
    -------
    np.ndarray : 各馬のgap値
    """
    p_model  = model_scores / model_scores.sum()
    p_market = market_win_probs(win_odds)

    p_model  = np.clip(p_model,  eps, 1 - eps)
    p_market = np.clip(p_market, eps, 1 - eps)

    logit_model  = np.log(p_model  / (1 - p_model))
    logit_market = np.log(p_market / (1 - p_market))

    return logit_model - logit_market


def compute_trio_gap(
    model_scores: np.ndarray,
    win_odds: np.ndarray,
    combo: Tuple[int, int, int],
    payback_rate: float = 0.725,
    eps: float = 1e-9,
) -> dict:
    """
    3連複組み合わせのオッズ乖離を計算する。

    gap_{ijk} = log(P_model) - log(P_market)

    Parameters
    ----------
    model_scores  : モデルスコア配列
    win_odds      : 単勝オッズ配列
    combo         : (i, j, k) インデックス
    payback_rate  : 払戻率（NAR南関東: 0.725, その他: 0.70）

    Returns
    -------
    dict:
        p_model      : モデルが計算した3連複確率
        p_market     : 単勝オッズから推定した市場確率
        gap          : log乖離（正=過小評価）
        est_odds     : 推定オッズ（payback_rate / p_model）
        ev_estimate  : 期待値推定（payback_rate / p_market - 1）
    """
    from probability.plackett_luce import softmax_worth, trifecta_box_prob

    worth     = softmax_worth(model_scores)
    p_model   = trifecta_box_prob(model_scores, combo)

    p_market  = win_odds_to_trio_prob(win_odds, combo)

    p_model_c  = max(p_model,  eps)
    p_market_c = max(p_market, eps)

    gap = np.log(p_model_c) - np.log(p_market_c)

    est_odds    = payback_rate / p_model_c
    ev_estimate = payback_rate / p_market_c - 1.0   # 市場確率に基づくEV

    return {
        "p_model":    round(p_model,  6),
        "p_market":   round(p_market, 6),
        "gap":        round(float(gap), 4),
        "est_odds":   round(est_odds, 1),
        "ev_estimate": round(ev_estimate, 4),
    }


def win_odds_to_trio_prob(
    win_odds: np.ndarray,
    combo: Tuple[int, int, int],
) -> float:
    """単勝オッズからHarville近似で3連複確率を推定。"""
    p = market_win_probs(win_odds)
    return harville_trio_prob(p, combo)


# ──────────────────────────────────────────────────
# 高配当候補の生成
# ──────────────────────────────────────────────────
def find_high_odds_candidates(
    model_scores: np.ndarray,
    win_odds: np.ndarray,
    horse_ids: list,
    payback_rate: float = 0.725,
    min_gap: float = 0.5,
    min_est_odds: float = 50.0,
    top_n: int = 20,
) -> list[dict]:
    """
    gap > min_gap かつ推定オッズ > min_est_odds の組み合わせを返す。

    高配当狙いモードのコア関数。

    Returns
    -------
    list of dict: [{combo_idx, combo_ids, gap, p_model, est_odds, ev_estimate}, ...]
                  gap降順ソート
    """
    from itertools import combinations

    n = len(model_scores)
    candidates = []

    for combo in combinations(range(n), 3):
        info = compute_trio_gap(model_scores, win_odds, combo, payback_rate)
        if info["gap"] >= min_gap and info["est_odds"] >= min_est_odds:
            candidates.append({
                "combo_idx":   combo,
                "combo_ids":   tuple(horse_ids[i] for i in combo),
                "gap":         info["gap"],
                "p_model":     info["p_model"],
                "est_odds":    info["est_odds"],
                "ev_estimate": info["ev_estimate"],
            })

    return sorted(candidates, key=lambda x: x["gap"], reverse=True)[:top_n]
