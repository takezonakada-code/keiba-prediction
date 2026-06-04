"""
Plackett-Luce モデルによる 3連複（box）確率計算。

重要:
  - JRAの表示オッズは「払戻倍率（元本込み・控除済み）」
  - EV = p_hit × jra_display_odds - 1.0
  - (1 - 控除率) の再乗算は二重控除になるため禁止
"""
from __future__ import annotations

from itertools import permutations
from typing import Sequence, Tuple

import numpy as np


def softmax_worth(scores: np.ndarray) -> np.ndarray:
    """スコアをPL worthに変換（softmax）。合計=1を保証。"""
    s = scores - scores.max()   # 数値安定化
    exp_s = np.exp(s)
    return exp_s / exp_s.sum()


def pl_sequence_prob(worth: np.ndarray, order: Tuple[int, ...]) -> float:
    """
    Plackett-Luce: 指定した着順（order）の確率。
    order = (i0, i1, i2, ...) のインデックス列
    P = w[i0]/(Σw) × w[i1]/(Σw - w[i0]) × ...
    """
    remaining_sum = 1.0   # softmax後は合計=1
    prob = 1.0
    for idx in order:
        prob *= worth[idx] / remaining_sum
        remaining_sum -= worth[idx]
    return prob


def trifecta_box_prob(
    scores: np.ndarray,
    combo: Tuple[int, int, int],
) -> float:
    """
    3連複（box）の確率 = 指定3頭の全6順列確率の合計。

    Parameters
    ----------
    scores : np.ndarray
        同レース内の各馬のモデルスコア（生スコアでOK）
    combo : Tuple[int, int, int]
        馬のインデックス（0始まり）の組み合わせ

    Returns
    -------
    float : 0〜1の確率
    """
    worth = softmax_worth(scores)
    total = 0.0
    for perm in permutations(combo):
        total += pl_sequence_prob(worth, perm)
    return total


def all_trifecta_box_probs(scores: np.ndarray) -> dict[Tuple[int, int, int], float]:
    """
    全C(n,3)組み合わせの3連複確率を計算して返す。
    16頭 → 560通り、exact計算で十分高速。
    """
    n = len(scores)
    worth = softmax_worth(scores)
    results: dict[Tuple[int, int, int], float] = {}

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                combo = (i, j, k)
                total = 0.0
                for perm in permutations(combo):
                    total += pl_sequence_prob(worth, perm)
                results[combo] = total

    return results


def top_n_combos(
    scores: np.ndarray,
    horse_ids: Sequence,
    top_n: int = 10,
) -> list[dict]:
    """
    スコア上位の組み合わせをEV順に返す。

    Parameters
    ----------
    scores : np.ndarray
    horse_ids : 馬IDのリスト（インデックスに対応）
    top_n : 上位何件を返すか

    Returns
    -------
    list of dict: [{"combo_ids": [...], "p_hit": float}, ...]
    """
    probs = all_trifecta_box_probs(scores)
    sorted_combos = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    return [
        {
            "combo_idx":  combo,
            "combo_ids":  tuple(horse_ids[i] for i in combo),
            "p_hit":      p,
        }
        for combo, p in sorted_combos[:top_n]
    ]
