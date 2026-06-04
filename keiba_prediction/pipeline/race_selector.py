"""
レース選別モジュール。
全レース参加をやめ「勝てるレースだけ打つ」。
EV・モデル信頼度・流動性・データ品質の4軸で評価する。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RaceSelectionConfig:
    """レース選別の設定。"""
    # モデル信頼度: 上位3頭のp_top3合計が閾値を超えるか
    min_top3_concentration: float = 0.50

    # 正のEV組み合わせ数の上限（多すぎると買い目が膨らむ）
    max_positive_ev_combos: int = 8

    # 最低プールサイズ（円）— 地方の薄いプールでの自分のbetによるimpactを防ぐ
    min_pool_size: int = 0  # 0=チェックなし

    # データ品質: 欠損率が高いレースを除外
    max_missing_rate: float = 0.30

    # 除外フラグ: 新馬・転入初戦が多いレースを下げる
    max_debut_ratio: float = 0.50

    # オッズ安定性: 締切前ドリフトが極端なレースを別管理
    max_drift_flag: bool = True

    # ばんえいは必ず除外（別タスク）
    exclude_banei: bool = True


@dataclass
class RaceScore:
    """レースのスコアリング結果。"""
    race_id: str
    score: float
    passed: bool
    reasons: list[str] = field(default_factory=list)
    positive_ev_combos: int = 0


def score_race(
    race_id: str,
    entries_df: pd.DataFrame,
    bet_candidates: list[dict],
    config: RaceSelectionConfig | None = None,
) -> RaceScore:
    """
    1レースの選別スコアを計算する。

    Parameters
    ----------
    race_id        : レースID
    entries_df     : 出走馬のエントリ（p_top3, pred_score等を含む）
    bet_candidates : score_bet()の結果リスト（ev, stake, p_hitを含む）
    config         : 選別設定

    Returns
    -------
    RaceScore
    """
    if config is None:
        config = RaceSelectionConfig()

    reasons = []
    score = 1.0   # 基礎スコア（高いほど良い）
    passed = True

    # ── 1. ばんえい除外 ─────────────────────────
    if config.exclude_banei and entries_df.get("is_banei", pd.Series([0])).max() > 0:
        return RaceScore(race_id, 0.0, False, ["ばんえい除外"])

    # ── 2. データ品質チェック ─────────────────────
    if "speed_index_mean" in entries_df.columns:
        missing_rate = entries_df["speed_index_mean"].isna().mean()
        if missing_rate > config.max_missing_rate:
            reasons.append(f"欠損率高({missing_rate:.0%})")
            score *= 0.5
            passed = False

    # ── 3. 新馬・転入初戦比率 ────────────────────
    if "style_hist_n" in entries_df.columns:
        debut_ratio = (entries_df["style_hist_n"] == 0).mean()
        if debut_ratio > config.max_debut_ratio:
            reasons.append(f"初出走多({debut_ratio:.0%})")
            score *= 0.6

    # ── 4. モデル信頼度（top3集中度）─────────────
    if "p_top3" in entries_df.columns:
        top3_sorted = entries_df["p_top3"].nlargest(3).sum()
        if top3_sorted < config.min_top3_concentration:
            reasons.append(f"予測分散({top3_sorted:.2f}<{config.min_top3_concentration})")
            score *= 0.7
            passed = False

    # ── 5. 正のEV組み合わせ数 ────────────────────
    positive_ev = [b for b in bet_candidates if b.get("ev", -1) > 0]
    n_pos = len(positive_ev)

    if n_pos == 0:
        reasons.append("正EV組み合わせなし")
        score = 0.0
        passed = False
    elif n_pos > config.max_positive_ev_combos:
        reasons.append(f"EV組み合わせ過多({n_pos}点)")
        score *= 0.8

    # ── 6. オッズドリフト ────────────────────────
    if config.max_drift_flag and "is_drifter" in entries_df.columns:
        n_drifters = entries_df["is_drifter"].sum()
        if n_drifters >= 2:
            reasons.append(f"ドリフト馬多数({n_drifters}頭)")
            score *= 0.85   # 別管理（排除はしない）

    if not reasons:
        reasons.append("OK")

    return RaceScore(
        race_id=race_id,
        score=round(score, 3),
        passed=passed,
        reasons=reasons,
        positive_ev_combos=n_pos,
    )


def select_races(
    race_scores: list[RaceScore],
    top_n: int | None = None,
) -> list[RaceScore]:
    """
    スコア順にソートし、passed=True のレースを返す。

    Parameters
    ----------
    race_scores : score_race() の結果リスト
    top_n       : 上位N件まで（None=全件）

    Returns
    -------
    選別されたレースのリスト（スコア降順）
    """
    selected = [r for r in race_scores if r.passed]
    selected.sort(key=lambda x: x.score, reverse=True)
    if top_n is not None:
        selected = selected[:top_n]
    return selected


def print_selection_report(race_scores: list[RaceScore]) -> None:
    """選別結果をコンソールに出力。"""
    print("=== レース選別結果 ===")
    for rs in sorted(race_scores, key=lambda x: x.score, reverse=True):
        status = "✅" if rs.passed else "❌"
        print(f"{status} {rs.race_id} | score={rs.score:.3f} | "
              f"EV組み合わせ={rs.positive_ev_combos}点 | {', '.join(rs.reasons)}")
