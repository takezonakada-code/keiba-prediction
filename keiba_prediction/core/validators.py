"""
予測システム全体のバリデーション。
異常値・NaN・Inf・範囲外の値を全て排除する。
"""
from __future__ import annotations
import math
import logging
from typing import Optional

log = logging.getLogger(__name__)

# ────────────────────────────────────────────────
# 正常範囲定数
# ────────────────────────────────────────────────
EV_MIN    = -1.0    # -100%
EV_MAX    =  3.0    # +300%（合理的な上限）
ODDS_MIN  =  1.0
ODDS_MAX  = 9999.0
PROB_MIN  =  0.0001
PROB_MAX  =  0.9999
SCORE_MIN =  0.0
SCORE_MAX =  1.0


def _is_bad(v) -> bool:
    """None / NaN / Inf を検出"""
    if v is None:
        return True
    try:
        return math.isnan(float(v)) or math.isinf(float(v))
    except (TypeError, ValueError):
        return True


def validate_ev(ev, label: str = "") -> Optional[float]:
    """
    EV を正常範囲にクランプして返す。
    異常値（NaN/Inf/None）は None を返す。
    """
    if _is_bad(ev):
        if ev is not None:
            log.warning(f"EV異常値 [{label}]: {ev} → None")
        return None
    ev = float(ev)
    if ev > EV_MAX:
        log.warning(f"EV上限超過 [{label}]: {ev:.2f} → {EV_MAX}")
        return EV_MAX
    if ev < EV_MIN:
        return EV_MIN
    return ev


def validate_odds(odds, label: str = "") -> Optional[float]:
    """オッズを正常範囲でチェック。異常は None。"""
    if _is_bad(odds):
        return None
    odds = float(odds)
    if odds < ODDS_MIN or odds > ODDS_MAX:
        log.warning(f"オッズ範囲外 [{label}]: {odds}")
        return None
    return odds


def validate_probability(prob) -> float:
    """確率を [PROB_MIN, PROB_MAX] にクランプ。"""
    if _is_bad(prob):
        return PROB_MIN
    return max(PROB_MIN, min(PROB_MAX, float(prob)))


def validate_scores(scores, label: str = "") -> list[float]:
    """
    スコアリスト全体をバリデート・正規化する。
    - 異常値は最小値に置換
    - 合計が1になるよう正規化
    """
    import numpy as np
    arr = np.array([float(s) if not _is_bad(s) else 0.0 for s in scores])
    arr = np.clip(arr, SCORE_MIN, SCORE_MAX)
    total = arr.sum()
    if total <= 0:
        log.warning(f"スコア合計=0 [{label}] → 均等分配")
        arr = np.ones(len(arr)) / len(arr)
    else:
        arr /= total
    return arr.tolist()


def validate_race_data(
    horses: list[dict],
    race_id: str = "",
) -> tuple[bool, list[str]]:
    """
    レースデータ全体の整合性チェック。

    Returns
    -------
    (is_valid, error_messages)
    """
    errors = []

    # 出走馬数
    if len(horses) < 3:
        errors.append(f"出走馬が3頭未満: {len(horses)}頭")

    # オッズ取得率
    missing = [h for h in horses if not h.get("win_odds") or h["win_odds"] <= 0]
    if len(missing) > len(horses) * 0.5:
        errors.append(f"オッズ未取得が50%超: {len(missing)}/{len(horses)}頭")

    # オッズの正常範囲
    for h in horses:
        odds = h.get("win_odds")
        if odds and validate_odds(odds) is None:
            errors.append(f"馬番{h.get('draw_number')}のオッズ異常: {odds}")

    is_valid = len(errors) == 0
    if not is_valid:
        log.warning(f"バリデーションエラー [{race_id}]: {errors}")
    return is_valid, errors
