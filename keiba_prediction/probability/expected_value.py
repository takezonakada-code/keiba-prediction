"""
EV（期待値）計算 と Half-Kelly 賭け金算出。

JRA表示オッズは「払戻倍率（元本込み・控除済み）」なので、
EV = p_hit × jra_display_odds - 1.0
(1 - 控除率) の再乗算は二重控除になるため禁止。
"""
from __future__ import annotations

from config import KELLY_CAP_PER_RACE


def expected_value(p_hit: float, jra_display_odds: float) -> float:
    """
    期待値（EV）を計算する。

    Parameters
    ----------
    p_hit : float
        3連複的中確率（Plackett-Luceで計算）
    jra_display_odds : float
        JRAの表示オッズ（例: 15.2倍）。元本込みの払戻倍率。

    Returns
    -------
    float
        EV > 0 なら買い有望。EV = 0.05 → 期待収益5%。
    """
    return p_hit * jra_display_odds - 1.0


def kelly_fraction(p_hit: float, jra_display_odds: float) -> float:
    """
    Half-Kelly比率を計算する（フルKelly禁止）。

    Kelly = (p × b - (1 - p)) / b   where b = odds - 1
    Half-Kelly = Kelly / 2

    Parameters
    ----------
    p_hit : float
    jra_display_odds : float

    Returns
    -------
    float : 賭け比率（0以上）。負の場合は0（ベットしない）。
    """
    b = jra_display_odds - 1.0   # 純利益倍率
    if b <= 0:
        return 0.0
    full_kelly = (p_hit * b - (1.0 - p_hit)) / b
    half_kelly = full_kelly / 2.0
    return max(0.0, half_kelly)


def kelly_stake(
    p_hit: float,
    jra_display_odds: float,
    bankroll: float,
    cap: float = KELLY_CAP_PER_RACE,
    min_ev: float = 0.0,
) -> float:
    """
    賭け金（円）を算出する。

    Parameters
    ----------
    p_hit : float
    jra_display_odds : float
    bankroll : float
        現在の資金（円）
    cap : float
        1レース上限比率（デフォルト1% = 0.01）
    min_ev : float
        EV下限（これ未満は0円を返す）

    Returns
    -------
    float : 賭け金（100円単位に切り捨て）
    """
    ev = expected_value(p_hit, jra_display_odds)
    if ev < min_ev:
        return 0.0

    frac = kelly_fraction(p_hit, jra_display_odds)
    frac = min(frac, cap)
    raw_stake = bankroll * frac

    # 100円単位に切り捨て
    return max(0.0, (raw_stake // 100) * 100)


def score_bet(
    p_hit: float,
    jra_display_odds: float,
    bankroll: float = 100_000,
) -> dict:
    """
    1組み合わせのベット情報をまとめて返す。

    Returns
    -------
    dict with keys: ev, kelly_frac, stake
    """
    ev = expected_value(p_hit, jra_display_odds)
    frac = kelly_fraction(p_hit, jra_display_odds)
    stake = kelly_stake(p_hit, jra_display_odds, bankroll)
    return {
        "ev":         round(ev, 4),
        "kelly_frac": round(frac, 4),
        "stake":      stake,
    }
