"""
高配当買い目生成エンジン（システムB）

gap × 先行力 × 推定オッズ × EV の4軸で総合スコアを計算し、
上位6点を高配当候補として出力する。
"""
from __future__ import annotations

import math
from itertools import combinations, permutations
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────
# オッズ帯ラベル
# ──────────────────────────────────────────────────
def get_odds_band(est_odds: float) -> str:
    if est_odds >= 200:  return "200倍〜"
    if est_odds >= 100:  return "100〜200倍"
    if est_odds >= 50:   return "50〜100倍"
    return "〜50倍"


# ──────────────────────────────────────────────────
# Harville 近似
# ──────────────────────────────────────────────────
def _harville_trio(p: np.ndarray, combo: Tuple[int, int, int]) -> float:
    total = 0.0
    for perm in permutations(combo):
        i, j, k = perm
        dj = 1.0 - p[i]
        dk = 1.0 - p[i] - p[j]
        if dj > 0 and dk > 0:
            total += p[i] * (p[j] / dj) * (p[k] / dk)
    return total


def _market_win_probs(win_odds: np.ndarray) -> np.ndarray:
    inv = 1.0 / np.maximum(win_odds, 0.01)
    return inv / inv.sum()


# ──────────────────────────────────────────────────
# メイン選択関数
# ──────────────────────────────────────────────────
def select_high_odds_tickets(
    race_id:      str,
    scores:       np.ndarray,        # LGBMRanker or 市場確率スコア
    win_odds:     np.ndarray,        # 単勝オッズ
    chaos_score:  int,               # 荒れスコア（0〜100）
    feature_df:   pd.DataFrame,      # 各馬の特徴量（pace_score等）
    draw_numbers: np.ndarray,        # 馬番
    payback_rate: float = 0.725,     # 払戻率（NAR南関東）
    max_tickets:  int   = 6,
    min_gap:      float = 0.3,
    min_est_odds: float = 30.0,
) -> list[dict]:
    """
    高配当スコア順で上位N点の買い目を返す。

    高配当スコア =
      gap      × 0.40    # 市場との乖離（最重要）
      pace_combo × 0.30  # 先行力の組み合わせ
      odds_norm  × 0.20  # 高オッズ優先
      ev         × 0.10  # 期待値

    Parameters
    ----------
    scores      : モデルスコア配列（正規化前でOK）
    win_odds    : 単勝オッズ配列
    chaos_score : 荒れスコア（0〜100）
    feature_df  : index が馬の0始まりインデックス, 'pace_score' 列を持つ
    draw_numbers: 各インデックスに対応する馬番配列
    payback_rate: NAR払戻率
    """
    from probability.plackett_luce import softmax_worth, trifecta_box_prob

    n = len(scores)
    if n < 3:
        return []

    # Plackett-Luce worth
    worth     = softmax_worth(scores)
    p_market  = _market_win_probs(win_odds)
    eps       = 1e-9

    candidates = []
    for combo in combinations(range(n), 3):
        # 1. モデル確率
        p_model = trifecta_box_prob(scores, combo)

        # 2. 市場確率（Harville近似）
        p_mkt = _harville_trio(p_market, combo)

        # 3. オッズ乖離
        gap = math.log(p_model + eps) - math.log(p_mkt + eps)

        # 4. 推定オッズ
        est_odds = payback_rate / max(p_model, eps)

        # 5. EV（推定）
        ev = payback_rate / max(p_mkt, eps) - 1.0

        # フィルター
        if gap < min_gap or est_odds < min_est_odds:
            continue

        # 6. 先行力スコアの組み合わせ（3頭の合計）
        pace_sum = 0.0
        for idx in combo:
            if idx < len(feature_df) and "pace_score" in feature_df.columns:
                ps = feature_df.iloc[idx].get("pace_score")
                if ps is not None and not (isinstance(ps, float) and math.isnan(ps)):
                    # 先行力 = 1 - pace_score（低いほど先行 = 高い先行力）
                    pace_sum += (1.0 - float(ps))
        pace_combo = pace_sum / 3.0   # 0〜1平均

        # 7. 高配当スコア（総合）
        odds_norm = min(est_odds / 200.0, 1.0)   # 200倍を1.0とする
        high_odds_score = (
            gap       * 0.40
            + pace_combo * 0.30
            + odds_norm  * 0.20
            + min(max(ev, -1.0), 2.0) * 0.10
        )

        nums = sorted(int(draw_numbers[i]) for i in combo)
        candidates.append({
            "combo":            "-".join(str(n) for n in nums),
            "combo_idx":        combo,
            "p_model":          round(p_model, 6),
            "p_market":         round(p_mkt,   6),
            "gap":              round(gap,      3),
            "est_odds":         round(est_odds, 1),
            "ev":               round(ev,       4),
            "pace_combo":       round(pace_combo, 3),
            "high_odds_score":  round(high_odds_score, 4),
            "odds_band":        get_odds_band(est_odds),
            "mode":             "B",
        })

    # 高配当スコア降順ソート
    candidates.sort(key=lambda x: x["high_odds_score"], reverse=True)
    return candidates[:max_tickets]


# ──────────────────────────────────────────────────
# 名古屋12R シミュレーション
# ──────────────────────────────────────────────────
def simulate_nagoya12r(win_odds_list: list[float]) -> dict:
    """
    名古屋12Rの単勝オッズを入力として
    高配当候補を生成するシミュレーション。

    Parameters
    ----------
    win_odds_list : 馬番1番から順の単勝オッズリスト

    Returns
    -------
    dict: top3_candidates, is_156_in_top6
    """
    win_odds    = np.array(win_odds_list)
    # モデルスコア = 市場確率ベース（学習前の代理）
    inv_odds    = 1.0 / win_odds
    scores      = inv_odds / inv_odds.sum()

    n           = len(win_odds)
    draw_nums   = np.arange(1, n + 1)

    # ダミーfeature_df（pace_score なし）
    feature_df  = pd.DataFrame({"pace_score": [0.5] * n})

    candidates  = select_high_odds_tickets(
        race_id      = "202648060412",
        scores       = scores,
        win_odds     = win_odds,
        chaos_score  = 73,    # 名古屋12R の推定値
        feature_df   = feature_df,
        draw_numbers = draw_nums,
        payback_rate = 0.70,  # 名古屋（NAR非南関東）
        max_tickets  = 10,
        min_gap      = 0.2,
        min_est_odds = 20.0,
    )

    # 1-5-6 が上位6点に入っているか確認
    top6_combos = [c["combo"] for c in candidates[:6]]
    is_in_top6  = "1-5-6" in top6_combos

    return {
        "top_candidates":  candidates[:6],
        "is_156_in_top6":  is_in_top6,
        "total_candidates": len(candidates),
    }
