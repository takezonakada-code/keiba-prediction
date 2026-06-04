"""
直前オッズドリフト特徴量。

NARの公式CSVは2分ごと更新なので、
締切30分前・10分前・2分前・最終の4点を記録して変化率を計算する。

drift_score = (最終オッズ - 30分前オッズ) / 30分前オッズ
  drift > 0   = オッズ上昇（人気薄化）→ 穴馬候補
  drift < -0.3 = オッズ急落（資金流入）→ 情報馬候補
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


def compute_drift_features(
    race_id:    str,
    horse_id:   Optional[str] = None,
    conn        = None,
) -> dict:
    """
    1レース分のオッズスナップショット履歴からドリフト特徴量を計算。

    Parameters
    ----------
    race_id   : 対象レースID
    horse_id  : 対象馬ID（None の場合はレース全馬の集計）
    conn      : DBコネクション

    Returns
    -------
    dict:
        win_odds_final   : 最終単勝オッズ
        win_odds_30m     : 30分前オッズ
        win_odds_10m     : 10分前オッズ
        drift_30m        : 30分前→最終の変化率（正=人気薄化）
        drift_10m        : 10分前→最終の変化率
        drift_accel      : ドリフト加速度（10m→2m の変化 / 30m→10m の変化）
        is_drifter       : True = 30分前比+10%以上オッズ上昇（見捨てられた馬）
        is_steamer       : True = 急激な人気上昇（情報馬候補）
    """
    from db.database import get_conn as _gc

    # odds_snapshots テーブルが存在するかチェック
    def _fetch_snaps():
        with _gc() as c:
            try:
                return c.execute("""
                    SELECT horse_id, snapshot_time, win_odds
                    FROM odds_snapshots
                    WHERE race_id = ?
                    ORDER BY snapshot_time
                """, (race_id,)).fetchall()
            except Exception:
                return []

    snaps = _fetch_snaps() if conn is None else []

    if not snaps:
        # スナップショットがない場合: nar_results から最終オッズのみ取得
        def _fetch_final():
            with _gc() as c:
                q = "SELECT draw_number, win_odds FROM nar_results WHERE race_id = ?"
                if horse_id:
                    q += " AND horse_id = ?"
                    return c.execute(q, (race_id, horse_id)).fetchall()
                return c.execute(q, (race_id,)).fetchall()

        final_rows = _fetch_final() if conn is None else []
        if not final_rows:
            return _null_drift()

        return {
            "win_odds_final":  float(final_rows[0]["win_odds"]) if final_rows else None,
            "win_odds_30m":    None,
            "win_odds_10m":    None,
            "drift_30m":       0.0,
            "drift_10m":       0.0,
            "drift_accel":     0.0,
            "is_drifter":      False,
            "is_steamer":      False,
            "has_drift_data":  False,
        }

    df = pd.DataFrame([dict(r) for r in snaps])
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"])

    if horse_id:
        df = df[df["horse_id"] == horse_id]
    if df.empty:
        return _null_drift()

    t_final = df["snapshot_time"].max()

    def _odds_at(minutes_before: int) -> Optional[float]:
        cutoff = t_final - pd.Timedelta(minutes=minutes_before)
        window = df[df["snapshot_time"] <= cutoff]
        if window.empty:
            return None
        return float(window.sort_values("snapshot_time").iloc[-1]["win_odds"])

    final = float(df.sort_values("snapshot_time").iloc[-1]["win_odds"])
    o30   = _odds_at(30)
    o10   = _odds_at(10)
    o2    = _odds_at(2)

    def _drift(prev, curr):
        if prev is None or curr is None or abs(prev) < 0.1:
            return 0.0
        return (curr - prev) / prev

    d30 = _drift(o30, final)
    d10 = _drift(o10, final)

    d30_10 = _drift(o30, o10) if o30 and o10 else 0.0
    d10_2  = _drift(o10, o2)  if o10 and o2  else 0.0
    accel  = d10_2 / d30_10 if abs(d30_10) > 0.01 else 0.0

    return {
        "win_odds_final":  round(final, 2),
        "win_odds_30m":    round(o30,  2) if o30 else None,
        "win_odds_10m":    round(o10,  2) if o10 else None,
        "drift_30m":       round(d30,  4),
        "drift_10m":       round(d10,  4),
        "drift_accel":     round(accel, 4),
        "is_drifter":      d30 > 0.10,    # 30分前比+10%以上上昇
        "is_steamer":      d30 < -0.30,   # 急激な人気上昇
        "has_drift_data":  True,
    }


def _null_drift() -> dict:
    return {
        "win_odds_final":  None,
        "win_odds_30m":    None,
        "win_odds_10m":    None,
        "drift_30m":       0.0,
        "drift_10m":       0.0,
        "drift_accel":     0.0,
        "is_drifter":      False,
        "is_steamer":      False,
        "has_drift_data":  False,
    }


def add_drift_features(
    target_entries: pd.DataFrame,
    odds_snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """
    target_entries の全馬にオッズドリフト特徴量を追加。

    Parameters
    ----------
    target_entries : horse_id, race_id を持つ DataFrame
    odds_snapshots : race_id, horse_id, snapshot_time, win_odds を持つ DataFrame
    """
    drift_records = []
    for _, row in target_entries.iterrows():
        race_snaps = odds_snapshots[odds_snapshots["race_id"] == row["race_id"]]
        drift = compute_drift_features.__wrapped__ if hasattr(
            compute_drift_features, "__wrapped__") else _compute_from_df(
            race_snaps, row.get("horse_id"))
        drift["horse_id"] = row["horse_id"]
        drift["race_id"]  = row["race_id"]
        drift_records.append(drift)

    drift_df = pd.DataFrame(drift_records)
    merge_keys = [k for k in ["race_id", "horse_id"] if k in drift_df.columns]
    return target_entries.merge(drift_df, on=merge_keys, how="left")


def _compute_from_df(
    snaps_df: pd.DataFrame,
    horse_id: Optional[str],
) -> dict:
    """DataFrame から直接ドリフトを計算（DB接続なし）。"""
    if snaps_df.empty:
        return _null_drift()

    df = snaps_df.copy()
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"])
    if horse_id:
        df = df[df["horse_id"] == horse_id]
    if df.empty:
        return _null_drift()

    t_final = df["snapshot_time"].max()
    final   = float(df.sort_values("snapshot_time").iloc[-1]["win_odds"])

    def _odds_at(m):
        cutoff = t_final - pd.Timedelta(minutes=m)
        w = df[df["snapshot_time"] <= cutoff]
        return float(w.sort_values("snapshot_time").iloc[-1]["win_odds"]) if not w.empty else None

    o30 = _odds_at(30)
    d30 = (final - o30) / o30 if o30 and abs(o30) > 0.1 else 0.0

    return {
        "win_odds_final": round(final, 2),
        "win_odds_30m":   round(o30, 2) if o30 else None,
        "win_odds_10m":   None,
        "drift_30m":      round(d30, 4),
        "drift_10m":      0.0,
        "drift_accel":    0.0,
        "is_drifter":     d30 > 0.10,
        "is_steamer":     d30 < -0.30,
        "has_drift_data": True,
    }
