"""
全特徴量の統合パイプライン。
特徴量リークチェックを最初に実行する。
"""
from __future__ import annotations

import pandas as pd

from features.feature_specs import ALLOWED_FEATURES, assert_no_forbidden
from features.relative_agari import aggregate_hist_agari
from features.running_style import aggregate_style_features
from features.rest_interval import compute_days_since_last
from features.speed_index import aggregate_speed_index, compute_speed_index
from features.ewma_trend import aggregate_ewma_features
from features.race_content_score import aggregate_rcs
from features.jockey_trainer_bayes import compute_jockey_bayes_rates, compute_trainer_bayes_rates
from features.course_geometry import add_course_geometry_features


def build_features(
    target_entries: pd.DataFrame,
    past_races: pd.DataFrame,
    n_recent: int = 5,
    odds_snapshots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    予測対象エントリに全特徴量を付与して返す。

    Parameters
    ----------
    target_entries  : 予測対象レース × 馬のエントリ
    past_races      : 過去走レコード（当走データは含まない）
    n_recent        : 各集計で使う最近N走
    odds_snapshots  : オッズスナップショット履歴（ドリフト特徴量用、省略可）

    Returns
    -------
    特徴量付きDataFrame
    """
    # ── Step 0: リークチェック ──────────────────────────────
    assert_no_forbidden(list(target_entries.columns))
    assert_no_forbidden(list(past_races.columns))

    df = target_entries.copy()

    # ── Step 1: 上がり3F特徴量 ────────────────────────────
    df = aggregate_hist_agari(past_races, df, n_recent=n_recent)

    # ── Step 2: 脚質特徴量 ───────────────────────────────
    df = aggregate_style_features(past_races, df)

    # ── Step 3: 出走間隔 ──────────────────────────────────
    df = compute_days_since_last(df, past_races)

    # ── Step 4: スピード指数 ──────────────────────────────
    past_with_si = compute_speed_index(past_races) if "speed_index_raw" not in past_races.columns \
                   else past_races
    df = aggregate_speed_index(past_with_si, df, n_recent=n_recent)

    # ── Step 5: EWMAトレンド ──────────────────────────────
    df = aggregate_ewma_features(past_with_si, df, score_col="speed_index_raw", n_recent=n_recent)

    # ── Step 6: レース内容スコア ──────────────────────────
    df = aggregate_rcs(past_with_si, df, n_recent=n_recent)

    # ── Step 7: 騎手・調教師ベイズ収縮 ────────────────────
    if "jockey_id" in df.columns:
        df = compute_jockey_bayes_rates(past_races, df)
    if "trainer_id" in df.columns:
        df = compute_trainer_bayes_rates(past_races, df)

    # ── Step 8: コースジオメトリ ──────────────────────────
    if "track" in df.columns:
        df = add_course_geometry_features(df)

    # ── Step 9: オッズドリフト（直前のみ、省略可）──────────
    if odds_snapshots is not None and len(odds_snapshots) > 0:
        from features.odds_drift import add_drift_features
        df = add_drift_features(df, odds_snapshots)

    # ── Step 10: 最終リークチェック ───────────────────────
    assert_no_forbidden(list(df.columns))

    return df


def select_model_features(df: pd.DataFrame) -> list[str]:
    """
    モデルに渡す特徴量カラム名リストを返す。
    ALLOWED_FEATURES に含まれ、かつ df に存在するカラムのみ。
    """
    return [c for c in df.columns if c in ALLOWED_FEATURES]
