"""
確率計算の整合性テスト。
pytest tests/test_probability.py で実行。
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pytest

from probability.plackett_luce import (
    all_trifecta_box_probs,
    softmax_worth,
    trifecta_box_prob,
)
from probability.expected_value import expected_value, kelly_fraction, kelly_stake


# ───────────────────────────────────────────────
# softmax_worth
# ───────────────────────────────────────────────
class TestSoftmaxWorth:
    def test_sums_to_one(self):
        scores = np.array([1.0, 2.5, 0.3, 3.1, 1.7])
        worth = softmax_worth(scores)
        assert math.isclose(worth.sum(), 1.0, rel_tol=1e-9)

    def test_all_positive(self):
        scores = np.array([-10.0, 0.0, 10.0])
        worth = softmax_worth(scores)
        assert (worth > 0).all()

    def test_numerical_stability_large_values(self):
        scores = np.array([1000.0, 999.0, 998.0])
        worth = softmax_worth(scores)
        assert not np.isnan(worth).any()
        assert math.isclose(worth.sum(), 1.0, rel_tol=1e-9)


# ───────────────────────────────────────────────
# trifecta_box_prob
# ───────────────────────────────────────────────
class TestTrifectaBoxProb:
    def test_valid_probability_range(self):
        scores = np.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.5])
        p = trifecta_box_prob(scores, (0, 1, 2))
        assert 0.0 <= p <= 1.0

    def test_all_combos_sum_to_one(self):
        """全C(n,3)の確率合計が1になるか。PL exact の最重要テスト。"""
        scores = np.random.default_rng(42).normal(size=10)
        probs = all_trifecta_box_probs(scores)
        total = sum(probs.values())
        assert math.isclose(total, 1.0, rel_tol=1e-7), f"合計={total}"

    def test_all_combos_sum_to_one_16horses(self):
        """16頭（560通り）で合計=1の検証。"""
        scores = np.random.default_rng(0).normal(size=16)
        probs = all_trifecta_box_probs(scores)
        assert len(probs) == 560
        total = sum(probs.values())
        assert math.isclose(total, 1.0, rel_tol=1e-7), f"合計={total}"

    def test_highest_score_combo_has_highest_prob(self):
        """スコア上位3頭の組み合わせが最高確率を持つか。"""
        scores = np.array([5.0, 4.0, 3.0, 1.0, 0.5])
        probs = all_trifecta_box_probs(scores)
        best_combo = max(probs, key=probs.get)
        assert set(best_combo) == {0, 1, 2}

    def test_combo_count_correctness(self):
        for n in [5, 8, 10, 16, 18]:
            scores = np.ones(n)
            probs = all_trifecta_box_probs(scores)
            expected = math.comb(n, 3)
            assert len(probs) == expected, f"n={n}: expected {expected}, got {len(probs)}"


# ───────────────────────────────────────────────
# EV計算 – 控除率二重適用なし
# ───────────────────────────────────────────────
class TestExpectedValue:
    def test_ev_formula(self):
        """EV = p × odds - 1.0 の直接検証。"""
        p = 0.1
        odds = 12.0
        ev = expected_value(p, odds)
        assert math.isclose(ev, 0.1 * 12.0 - 1.0, rel_tol=1e-9)

    def test_ev_no_double_deduction(self):
        """
        JRAオッズは控除済みなので (1 - 0.25) を再乗算してはいけない。
        EV計算で控除率を二重適用していないか検証。
        """
        p = 0.1
        odds = 12.0
        ev_correct = expected_value(p, odds)          # 正しい
        ev_wrong   = p * (odds * (1 - 0.25)) - 1.0   # 二重控除（誤り）
        # 正しいEVの方が大きいはず
        assert ev_correct > ev_wrong, \
            f"EVが二重控除されている可能性: correct={ev_correct}, wrong={ev_wrong}"

    def test_negative_ev_below_fair_odds(self):
        """オッズが理論値より低ければEV < 0。"""
        p = 0.1
        fair_odds = 1.0 / p   # 10.0倍（控除なし）
        ev = expected_value(p, fair_odds * 0.75)   # JRA控除後
        assert ev < 0

    def test_positive_ev_above_fair_odds(self):
        """オッズが理論値より高ければEV > 0（過剰配当の場合）。"""
        p = 0.1
        ev = expected_value(p, 15.0)
        assert ev > 0


# ───────────────────────────────────────────────
# Kelly
# ───────────────────────────────────────────────
class TestKelly:
    def test_kelly_zero_for_negative_ev(self):
        """EV < 0 のとき Half-Kelly = 0。"""
        p = 0.05
        odds = 5.0   # EV = 0.05*5 - 1 = -0.75
        assert kelly_fraction(p, odds) == 0.0

    def test_kelly_positive_for_positive_ev(self):
        p = 0.2
        odds = 8.0   # EV = 0.2*8 - 1 = 0.6 > 0
        frac = kelly_fraction(p, odds)
        assert frac > 0.0

    def test_kelly_stake_respects_cap(self):
        """賭け金が bankroll × cap を超えないこと。"""
        p = 0.5
        odds = 10.0
        bankroll = 100_000
        cap = 0.01
        stake = kelly_stake(p, odds, bankroll, cap=cap)
        assert stake <= bankroll * cap

    def test_kelly_stake_is_100yen_unit(self):
        """賭け金が100円単位になっているか。"""
        p = 0.15
        odds = 9.0
        stake = kelly_stake(p, odds, bankroll=100_000)
        assert stake % 100 == 0

    def test_kelly_stake_zero_below_min_ev(self):
        """min_ev を下回るときは0円。"""
        p = 0.05
        odds = 6.0   # EV = -0.7
        stake = kelly_stake(p, odds, bankroll=100_000, min_ev=0.0)
        assert stake == 0.0
