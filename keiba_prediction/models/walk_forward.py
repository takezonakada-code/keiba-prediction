"""
月次 expanding walk-forward + 7日embargo
時系列リークを防ぐため random split は使わない。
"""
from __future__ import annotations

from datetime import timedelta
from typing import Iterator, Tuple

import pandas as pd

from config import WF_EMBARGO_DAYS, WF_MIN_TRAIN_MONTHS, WF_TEST_MONTHS


def monthly_expanding_walk_forward(
    races: pd.DataFrame,
    date_col: str = "race_date",
    min_train_months: int = WF_MIN_TRAIN_MONTHS,
    test_months: int = WF_TEST_MONTHS,
    embargo_days: int = WF_EMBARGO_DAYS,
) -> Iterator[Tuple[pd.Index, pd.Index]]:
    """
    月次 expanding walk-forward を生成するジェネレータ。

    Parameters
    ----------
    races : pd.DataFrame
        race_date 列を持つ DataFrame（1行=1馬のエントリ）
    date_col : str
        日付列名
    min_train_months : int
        最低訓練期間（月数）。デフォルト24ヶ月。
    test_months : int
        1foldのテスト期間（月数）。デフォルト1ヶ月。
    embargo_days : int
        trainとtestの間の空白期間（日）。デフォルト7日。

    Yields
    ------
    (train_idx, test_idx) : Tuple[pd.Index, pd.Index]
    """
    dates = pd.to_datetime(races[date_col])
    min_date = dates.min()
    max_date = dates.max()

    # 最初のテスト開始月を計算
    first_test_start = min_date + pd.DateOffset(months=min_train_months)
    if first_test_start > max_date:
        raise ValueError(
            f"データが不足しています。最低 {min_train_months} ヶ月分必要ですが、"
            f"データ期間は {min_date.date()} 〜 {max_date.date()} です。"
        )

    test_start = first_test_start.replace(day=1)

    while test_start <= max_date:
        test_end = test_start + pd.DateOffset(months=test_months) - timedelta(days=1)
        if test_end > max_date:
            test_end = max_date

        # embargo: test_start から embargo_days 前までをtrainの上限とする
        train_cutoff = test_start - timedelta(days=embargo_days)

        train_mask = dates < train_cutoff
        test_mask  = (dates >= test_start) & (dates <= test_end)

        train_idx = races.index[train_mask]
        test_idx  = races.index[test_mask]

        if len(train_idx) == 0 or len(test_idx) == 0:
            test_start += pd.DateOffset(months=test_months)
            continue

        yield train_idx, test_idx

        test_start += pd.DateOffset(months=test_months)


def run_walk_forward_cv(
    races: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    train_fn,
    predict_fn,
    date_col: str = "race_date",
    min_train_months: int = WF_MIN_TRAIN_MONTHS,
    embargo_days: int = WF_EMBARGO_DAYS,
) -> pd.DataFrame:
    """
    walk-forward CV を実行し、OOF予測値を返す。

    Parameters
    ----------
    train_fn : callable(X_train, y_train, groups_train) -> model
    predict_fn : callable(model, X_test) -> np.ndarray (scores)

    Returns
    -------
    pd.DataFrame with columns: [race_date, race_id, horse_id, oof_score, y_true]
    """
    from features.feature_specs import assert_no_forbidden

    assert_no_forbidden(feature_cols)

    results = []

    for fold_i, (train_idx, test_idx) in enumerate(
        monthly_expanding_walk_forward(races, date_col=date_col,
                                       min_train_months=min_train_months,
                                       embargo_days=embargo_days)
    ):
        X_train = races.loc[train_idx, feature_cols]
        y_train = races.loc[train_idx, target_col]
        g_train = races.loc[train_idx, group_col]

        X_test  = races.loc[test_idx, feature_cols]
        y_test  = races.loc[test_idx, target_col]

        model = train_fn(X_train, y_train, g_train)
        scores = predict_fn(model, X_test)

        fold_df = races.loc[test_idx, ["race_date", "race_id", "horse_id"]].copy()
        fold_df["oof_score"] = scores
        fold_df["y_true"]    = y_test.values
        fold_df["fold"]      = fold_i
        results.append(fold_df)

        n_train = len(train_idx)
        n_test  = len(test_idx)
        test_month = races.loc[test_idx, date_col].iloc[0]
        print(f"Fold {fold_i:02d} | test={str(test_month)[:7]} | "
              f"train={n_train:,} | test={n_test:,}")

    return pd.concat(results, ignore_index=True)
