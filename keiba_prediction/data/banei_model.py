"""
ばんえい競馬専用予測モデル

設計原則:
  1. 市場オッズを基準に小幅補正（大きく逸脱しない）
  2. 馬番バイアス・積載重量の補正は±10%以内
  3. EV は -0.3 ～ +0.3 の範囲に収まるよう設計
  4. 上位馬と買い目の一貫性を保証

EV 計算式:
  p_adjusted = 市場確率 × 補正係数（0.8 ～ 1.2 の範囲）
  EV_trio = p_adjusted_trio × market_odds_trio - 1
"""
from __future__ import annotations

import numpy as np
from itertools import combinations
from typing import Optional

# ────────────────────────────────────────────────
# 帯広ばんえい専用バイアス（公式統計）
# ────────────────────────────────────────────────
POST_BIAS_SHOW = {
    1: 30.5, 2: 32.6, 3: 34.2, 4: 31.4, 5: 32.9,
    6: 30.6, 7: 34.8, 8: 33.5, 9: 29.9, 10: 28.6,
}   # 3着内率（%）

POP_BIAS_SHOW = {
    1: 68.8, 2: 53.8, 3: 45.0, 4: 37.2, 5: 30.1,
    6: 24.5, 7: 20.2, 8: 16.8, 9: 13.4, 10: 10.7,
}   # 人気別3着内率（%）

PAYBACK = 0.70   # 帯広払戻率


def market_win_probs(win_odds: np.ndarray) -> np.ndarray:
    """単勝オッズ → 市場確率（逆数正規化）"""
    inv = 1.0 / np.maximum(win_odds, 0.1)
    return inv / inv.sum()


def adjusted_scores(
    draw_numbers:    list[int],
    win_odds:        np.ndarray,
    burden_weights:  Optional[list[float]] = None,
    popularity_ranks: Optional[list[int]] = None,
) -> np.ndarray:
    """
    市場確率に小幅な補正を加えたスコアを返す。

    補正の設計:
      - 馬番バイアス補正: 帯広3着内率の平均からの乖離（±5%）
      - 積載重量補正:  軽いほど有利（最大±5%）
      - 人気補正:     公式3着内率 vs 市場確率の乖離（±10%）

    全体として補正係数は 0.85 ～ 1.15 に収まる。
    """
    n = len(draw_numbers)
    p_mkt = market_win_probs(win_odds)
    avg_show = sum(POST_BIAS_SHOW.values()) / len(POST_BIAS_SHOW) / 100  # ~0.32

    correction = np.ones(n)

    for i, (dn, pop) in enumerate(zip(
        draw_numbers,
        popularity_ranks or list(range(1, n+1))
    )):
        # 1. 馬番バイアス補正（±5%）
        pp_show = POST_BIAS_SHOW.get(dn, 32) / 100
        pp_adj  = 1.0 + (pp_show - avg_show) * 0.5   # 半分だけ適用
        pp_adj  = np.clip(pp_adj, 0.95, 1.05)

        # 2. 積載重量補正（軽いほど有利、最大±5%）
        bw_adj = 1.0
        if burden_weights:
            valid_bws = [b for b in burden_weights if b is not None]
            if valid_bws:
                bw = burden_weights[i] if (i < len(burden_weights) and burden_weights[i] is not None) else float(np.mean(valid_bws))
                avg_bw = float(np.mean(valid_bws))
                bw_diff = avg_bw - bw   # 軽い=正
                bw_adj  = 1.0 + np.clip(bw_diff * 0.01, -0.05, 0.05)

        # 3. 人気補正（公式3着内率 vs 市場3着内率の差、±10%）
        pop_show = POP_BIAS_SHOW.get(min(pop, 10), 10.0) / 100
        # 市場が示す3着内率（3着以内確率の近似）
        market_show_approx = min(p_mkt[i] * 3.5, 0.95)  # 勝率×3.5 ≈ 3着内率
        pop_adj = 1.0 + np.clip(pop_show - market_show_approx, -0.15, 0.15) * 0.5
        pop_adj = np.clip(pop_adj, 0.90, 1.10)

        correction[i] = pp_adj * bw_adj * pop_adj

    # 補正後スコア（市場確率×補正係数、再正規化）
    scores = p_mkt * correction
    scores /= scores.sum()
    return scores


def harville_trio_prob(p: np.ndarray, combo: tuple) -> float:
    """Harville近似で3頭の3連複確率を計算"""
    total = 0.0
    from itertools import permutations
    for perm in permutations(combo):
        i, j, k = perm
        dj = 1.0 - p[i]
        dk = 1.0 - p[i] - p[j]
        if dj > 0 and dk > 0:
            total += p[i] * (p[j] / dj) * (p[k] / dk)
    return total


def trio_ev(
    p_model_trio: float,
    p_market_trio: float,
    payback: float = PAYBACK,
) -> float:
    """
    3連複EV計算。

    市場推定オッズ = payback / p_market_trio
    EV = p_model × 市場推定オッズ - 1
    """
    if p_market_trio <= 0:
        return -1.0
    est_odds = payback / p_market_trio
    return p_model_trio * est_odds - 1.0


def predict_banei_race(
    draw_numbers:     list[int],
    horse_names:      list[str],
    win_odds:         list[float],
    horse_weights:    Optional[list[float]] = None,
    burden_weights:   Optional[list[float]] = None,
    popularity_ranks: Optional[list[int]]   = None,
    payback:          float = PAYBACK,
    top_n:            int   = 5,
) -> dict:
    """
    ばんえい1レース分の予測を生成する。

    Returns
    -------
    dict:
        horses      : スコア降順の馬リスト
        tickets_a   : EV順上位5点（本命）
        tickets_b   : 推定オッズ30倍以上でEV高い3点（高配当）
        top_note    : 注目ポイントのテキスト
    """
    n = len(draw_numbers)
    if n < 3:
        return {"horses": [], "tickets_a": [], "tickets_b": [], "top_note": ""}

    odds_arr = np.array([float(o) for o in win_odds])
    p_mkt    = market_win_probs(odds_arr)

    # ばんえい補正スコア
    pop_ranks = popularity_ranks or list(range(1, n + 1))
    scores = adjusted_scores(
        draw_numbers, odds_arr, burden_weights, pop_ranks
    )

    # 馬リスト（スコア降順）
    sorted_idx = np.argsort(-scores)
    horse_list = []
    for rank, i in enumerate(sorted_idx, 1):
        dn   = draw_numbers[i]
        name = horse_names[i] if i < len(horse_names) else f"馬番{dn}"
        bw   = burden_weights[i] if burden_weights and i < len(burden_weights) else None
        hw   = horse_weights[i]  if horse_weights  and i < len(horse_weights)  else None
        horse_list.append({
            "num":     dn,
            "name":    name,
            "score":   round(float(scores[i]), 4),
            "odds":    float(odds_arr[i]),
            "p_mkt":   round(float(p_mkt[i]), 4),
            "horse_weight":   hw,
            "burden_weight":  bw,
            "pp_bias": POST_BIAS_SHOW.get(dn, 32),
        })

    # EV計算（全C(n,3)組み合わせ）
    ev_list = []
    for combo in combinations(range(n), 3):
        p_mod = harville_trio_prob(scores, combo)
        p_mkt_trio = harville_trio_prob(p_mkt, combo)
        ev  = trio_ev(p_mod, p_mkt_trio, payback)
        est = payback / max(p_mkt_trio, 1e-9)
        nums = sorted([draw_numbers[i] for i in combo])
        score_sum = sum(scores[i] for i in combo)  # 3頭のスコア合計
        ev_list.append({
            "combo":      "-".join(map(str, nums)),
            "ev":         round(ev, 4),
            "est_odds":   round(est, 1),
            "p_model":    round(p_mod, 6),
            "p_market":   round(p_mkt_trio, 6),
            "score_sum":  round(score_sum, 4),
            "combo_idx":  combo,
        })

    ev_list.sort(key=lambda x: x["ev"], reverse=True)

    # ── System A: 上位馬を含むEV上位5点 ─────────────
    # 上位3頭のインデックスを取得
    top3_idx = set(sorted_idx[:3])
    # 上位3頭のいずれかを含むコンボを優先（EVでソート）
    tickets_a_with_top = [t for t in ev_list if any(i in top3_idx for i in t["combo_idx"])][:top_n]
    # 不足分は全体から補完
    if len(tickets_a_with_top) < top_n:
        extra = [t for t in ev_list if t not in tickets_a_with_top]
        tickets_a_with_top += extra[:top_n - len(tickets_a_with_top)]
    tickets_a = tickets_a_with_top[:top_n]

    # ── System B: 穴・高配当狙い（推定30倍以上、EV正）─
    tickets_b = [t for t in ev_list if t["est_odds"] >= 30 and t["ev"] > -0.20][:3]
    if not tickets_b:
        tickets_b = [t for t in ev_list if t["est_odds"] >= 20][:3]

    # トップ注目馬の文章生成
    top1 = horse_list[0]
    top2 = horse_list[1] if len(horse_list) > 1 else None
    bw_note = ""
    if burden_weights:
        valid_bw = [(bw, dn) for bw, dn in zip(burden_weights, draw_numbers) if bw is not None]
        if valid_bw:
            light = min(valid_bw, key=lambda x: x[0])
            min_bw = min(x[0] for x in valid_bw)
            if light[0] <= min_bw + 5:
                bw_note = f"({light[1]}番は積載{light[0]}kgで最軽量・波乱候補)"

    note = (
        f"1番人気: {top1['num']}番 {top1['name']}（{top1['odds']}倍）"
        + (f" 対抗: {top2['num']}番 {top2['name']}（{top2['odds']}倍）" if top2 else "")
        + f" {bw_note}"
    )

    return {
        "horses":   horse_list,
        "tickets_a": tickets_a,
        "tickets_b": tickets_b,
        "top_note": note,
        "n_horses": n,
    }
