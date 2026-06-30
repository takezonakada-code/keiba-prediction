"""
3連単（exacta trifecta）15点賭けシステム。

選択ロジック:
  Part A 本命  8点: Harville EV降順（順当予想）
  Part B 穴馬  4点: GAP上位（市場過小評価の組み合わせ）
  Part C 荒れ  3点: 高オッズ馬が絡むアップセット組み合わせ
  ※重複は削除して合計15点以内に収める
"""
from __future__ import annotations

from itertools import permutations
from typing import Sequence

import numpy as np


def harville_sanrentan_prob(p: np.ndarray, perm: tuple) -> float:
    """P(A 1着, B 2着, C 3着) をHarville近似で計算。"""
    i, j, k = perm
    denom_j = max(1.0 - p[i], 1e-9)
    denom_k = max(1.0 - p[i] - p[j], 1e-9)
    return float(p[i] * (p[j] / denom_j) * (p[k] / denom_k))


def select_sanrentan_15(
    model_scores: np.ndarray,
    market_probs: np.ndarray,
    win_odds: np.ndarray,
    draw_numbers: Sequence[int],
    payback: float = 0.70,
    n_hon: int = 8,
    n_ana: int = 4,
    n_are: int = 3,
) -> list[dict]:
    """
    3連単15点を選択する。

    Parameters
    ----------
    model_scores  : モデルスコア（正規化済み）
    market_probs  : 市場勝率（単勝逆数正規化）
    win_odds      : 各馬の単勝オッズ
    draw_numbers  : 各馬の馬番
    payback       : 払戻率
    n_hon         : 本命チケット数
    n_ana         : 穴馬チケット数
    n_are         : 荒れチケット数

    Returns
    -------
    list of dict:
      {combo, label, p_model, p_market, est_odds, ev, gap}
    """
    n = len(model_scores)
    if n < 3:
        return []

    p_mod = np.array(model_scores, dtype=float)
    p_mkt = np.array(market_probs, dtype=float)
    p_mod = np.clip(p_mod, 1e-9, None); p_mod /= p_mod.sum()
    p_mkt = np.clip(p_mkt, 1e-9, None); p_mkt /= p_mkt.sum()

    dns = list(draw_numbers)

    all_perms = []
    for perm in permutations(range(n), 3):
        i, j, k = perm
        p_m   = harville_sanrentan_prob(p_mod, perm)
        p_mk  = harville_sanrentan_prob(p_mkt, perm)
        est   = payback / max(p_mk, 1e-9)
        ev_raw = p_m * est - 1.0
        # clamp EV
        ev = max(-1.0, min(3.0, ev_raw))
        # gap = log(p_mod / p_mkt) for the combo
        gap = float(np.log(max(p_m, 1e-12) / max(p_mk, 1e-12)))
        # 最大オッズ（荒れ度）
        max_odds = float(max(win_odds[i], win_odds[j], win_odds[k]))
        first_odds = float(win_odds[i])  # 1着馬オッズ

        all_perms.append({
            "perm":       perm,
            "combo":      f"{dns[i]}-{dns[j]}-{dns[k]}",
            "p_model":    float(p_m),
            "p_market":   float(p_mk),
            "est_odds":   float(est),
            "ev":         float(ev),
            "gap":        float(gap),
            "max_odds":   max_odds,
            "first_odds": first_odds,
            "label":      "",
        })

    # ── Part A: 本命（p_model降順 → 確率最大の順当予想） ──
    # EV順ではなく、モデル予測確率の高い順（本命的）を選ぶ
    hon_sorted = sorted(all_perms, key=lambda x: x["p_model"], reverse=True)
    selected_combos = set()
    tickets = []

    for t in hon_sorted:
        if len([x for x in tickets if x["label"]=="本命"]) >= n_hon:
            break
        if t["combo"] in selected_combos:
            continue
        t2 = dict(t); t2["label"] = "本命"
        tickets.append(t2)
        selected_combos.add(t["combo"])

    # ── Part B: 穴馬（GAP上位・穴馬が1着か2着に来る） ──
    # gap > 0 で max_odds > 5 倍の馬が絡む組み合わせ
    ana_sorted = sorted(
        [x for x in all_perms
         if x["gap"] > 0 and x["max_odds"] > 5.0
         and x["combo"] not in selected_combos],
        key=lambda x: x["gap"] + x["max_odds"] * 0.05,
        reverse=True,
    )
    for t in ana_sorted:
        if len([x for x in tickets if x["label"]=="穴馬"]) >= n_ana:
            break
        if t["combo"] in selected_combos:
            continue
        t2 = dict(t); t2["label"] = "穴馬"
        tickets.append(t2)
        selected_combos.add(t["combo"])

    # ── Part C: 荒れ（高オッズ馬が1着） ──────────
    # 1着馬のオッズが10倍以上（大荒れシナリオ）
    are_sorted = sorted(
        [x for x in all_perms
         if x["first_odds"] >= 8.0
         and x["combo"] not in selected_combos],
        key=lambda x: x["gap"] * 0.5 + x["ev"] * 0.5,
        reverse=True,
    )
    for t in are_sorted:
        if len([x for x in tickets if x["label"]=="荒れ"]) >= n_are:
            break
        if t["combo"] in selected_combos:
            continue
        t2 = dict(t); t2["label"] = "荒れ"
        tickets.append(t2)
        selected_combos.add(t["combo"])

    # ── 不足分をEV順で補完 ────────────────────────
    remaining = 15 - len(tickets)
    if remaining > 0:
        fill = sorted(
            [x for x in all_perms if x["combo"] not in selected_combos],
            key=lambda x: x["ev"], reverse=True,
        )
        for t in fill[:remaining]:
            t2 = dict(t); t2["label"] = "本命"
            tickets.append(t2)
            selected_combos.add(t["combo"])

    return tickets[:15]
