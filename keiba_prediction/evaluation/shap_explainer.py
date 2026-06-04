"""
SHAP top5特徴量の抽出とDB保存。
モデル説明のためにClaude Opusへ渡すプロンプトも生成する。
"""
from __future__ import annotations

import json
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap


def compute_shap_values(
    ranker: lgb.LGBMRanker,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    全馬のSHAP値を計算して返す。

    Returns
    -------
    np.ndarray of shape (n_horses, n_features)
    """
    explainer = shap.TreeExplainer(ranker.booster_)
    shap_values = explainer.shap_values(X)
    return shap_values   # shape: (n, features)


def top_n_shap(
    shap_values: np.ndarray,
    feature_names: list[str],
    horse_idx: int,
    n: int = 5,
) -> list[dict[str, Any]]:
    """
    1頭分のSHAP top-n特徴量を返す。

    Returns
    -------
    list of {"feature": str, "shap_value": float, "direction": str}
    """
    vals = shap_values[horse_idx]
    top_idx = np.argsort(np.abs(vals))[::-1][:n]
    return [
        {
            "feature":    feature_names[i],
            "shap_value": round(float(vals[i]), 4),
            "direction":  "+" if vals[i] >= 0 else "-",
        }
        for i in top_idx
    ]


def make_explanation_prompt(
    horse_name: str,
    predicted_score: float,
    shap_top5: list[dict],
    race_info: dict | None = None,
) -> str:
    """
    Claude Opus に渡す説明プロンプトを生成する。

    Parameters
    ----------
    horse_name     : 馬名
    predicted_score: 予測スコア
    shap_top5      : top_n_shap() の結果
    race_info      : {"distance": 1600, "surface": "芝", "track": "東京", ...}

    Returns
    -------
    str : プロンプト文字列
    """
    shap_lines = "\n".join(
        f"  {i+1}. {s['feature']}: {'+' if s['shap_value'] >= 0 else ''}{s['shap_value']:.4f}"
        for i, s in enumerate(shap_top5)
    )

    race_str = ""
    if race_info:
        race_str = f"レース: {race_info.get('track', '')} {race_info.get('distance', '')}m "
        race_str += f"{race_info.get('surface', '')} {race_info.get('track_condition', '')}\n"

    return (
        f"{race_str}"
        f"馬名: {horse_name}\n"
        f"予測スコア: {predicted_score:.4f}\n"
        f"予測根拠（SHAP top5）:\n{shap_lines}\n\n"
        "上記の予測根拠をもとに、この馬が3着以内に入る可能性とその理由を500字以内で説明してください。"
        "脚質・末脚・コース適性・間隔などの観点から具体的に述べてください。"
    )


def build_shap_summary(
    ranker: lgb.LGBMRanker,
    X: pd.DataFrame,
    df_entries: pd.DataFrame,
    horse_name_col: str = "horse_name",
    score_col: str = "pred_score",
    n: int = 5,
) -> pd.DataFrame:
    """
    全馬のSHAP top-n特徴量をDataFrameで返す。
    DB保存・UI表示用。
    """
    shap_values = compute_shap_values(ranker, X)
    feature_names = list(X.columns)
    records = []

    for i, (_, row) in enumerate(df_entries.iterrows()):
        top = top_n_shap(shap_values, feature_names, i, n=n)
        records.append({
            "horse_id":          row.get("horse_id"),
            "horse_name":        row.get(horse_name_col, ""),
            "pred_score":        row.get(score_col, 0),
            "shap_top1_feature": top[0]["feature"] if top else None,
            "shap_top1_value":   top[0]["shap_value"] if top else None,
            "shap_top5_json":    json.dumps(top, ensure_ascii=False),
        })

    return pd.DataFrame(records)
