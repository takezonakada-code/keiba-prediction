"""
Monte Carlo 近似による 3連複確率計算（フォールバック用）。
頭数が多い・PL exact が遅い場合に使用。
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from probability.plackett_luce import softmax_worth


def mc_trifecta_box_prob(
    scores: np.ndarray,
    combo: Tuple[int, int, int],
    n_samples: int = 100_000,
    seed: int = 42,
) -> float:
    """
    Monte Carlo サンプリングで3連複確率を推定。

    Parameters
    ----------
    scores : np.ndarray
    combo : Tuple[int, int, int]
    n_samples : int
    seed : int

    Returns
    -------
    float
    """
    rng = np.random.default_rng(seed)
    worth = softmax_worth(scores)
    n_horses = len(scores)
    combo_set = set(combo)

    hits = 0
    for _ in range(n_samples):
        # PL サンプリング: 重み付きでシャッフル
        order = _pl_sample(rng, worth, n_horses)
        top3 = set(order[:3])
        if top3 == combo_set:
            hits += 1

    return hits / n_samples


def _pl_sample(rng: np.random.Generator, worth: np.ndarray, n: int) -> np.ndarray:
    """PL分布から1つの順列をサンプリング（Gumbel trick）。"""
    gumbel = rng.gumbel(size=n)
    perturbed = np.log(worth + 1e-12) + gumbel
    return np.argsort(-perturbed)   # 降順インデックス


def mc_all_trifecta_probs(
    scores: np.ndarray,
    horse_ids: Sequence,
    n_samples: int = 200_000,
    seed: int = 42,
) -> list[dict]:
    """
    全組み合わせの確率をMCで一括推定（Gumbel trick）。

    Returns
    -------
    list of dict sorted by p_hit desc
    """
    from itertools import combinations

    rng = np.random.default_rng(seed)
    worth = softmax_worth(scores)
    n_horses = len(scores)

    counter: dict[frozenset, int] = {}
    for _ in range(n_samples):
        order = _pl_sample(rng, worth, n_horses)
        key = frozenset(order[:3])
        counter[key] = counter.get(key, 0) + 1

    results = []
    for combo in combinations(range(n_horses), 3):
        key = frozenset(combo)
        p = counter.get(key, 0) / n_samples
        results.append({
            "combo_idx": combo,
            "combo_ids": tuple(horse_ids[i] for i in combo),
            "p_hit":     p,
        })

    return sorted(results, key=lambda x: x["p_hit"], reverse=True)
