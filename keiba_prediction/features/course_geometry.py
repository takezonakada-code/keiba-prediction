"""
コース形状のジオメトリクラスタリング。
one-hot エンコーディングではなく、コースの物理特性で数値化する。
NAR の場差を吸収するために場×コース特性を構造化する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────
# 全場のコース特性マスタ（NAR公式コースガイドに基づく）
# ──────────────────────────────────────────────────
COURSE_FEATURES: dict[str, dict] = {
    # JRA
    "東京":     {"straight_m": 526, "circumference_m": 2083, "has_slope": True,  "left_turn": True,  "tight_corners": False, "organizer": "JRA"},
    "中山":     {"straight_m": 310, "circumference_m": 1840, "has_slope": True,  "left_turn": False, "tight_corners": True,  "organizer": "JRA"},
    "阪神":     {"straight_m": 473, "circumference_m": 1877, "has_slope": True,  "left_turn": True,  "tight_corners": False, "organizer": "JRA"},
    "京都":     {"straight_m": 404, "circumference_m": 1894, "has_slope": False, "left_turn": True,  "tight_corners": False, "organizer": "JRA"},
    "中京":     {"straight_m": 412, "circumference_m": 1530, "has_slope": True,  "left_turn": True,  "tight_corners": False, "organizer": "JRA"},
    "小倉":     {"straight_m": 293, "circumference_m": 1615, "has_slope": False, "left_turn": True,  "tight_corners": False, "organizer": "JRA"},
    "福島":     {"straight_m": 292, "circumference_m": 1600, "has_slope": False, "left_turn": True,  "tight_corners": True,  "organizer": "JRA"},
    "新潟":     {"straight_m": 658, "circumference_m": 2223, "has_slope": False, "left_turn": True,  "tight_corners": False, "organizer": "JRA"},
    "札幌":     {"straight_m": 264, "circumference_m": 1640, "has_slope": False, "left_turn": True,  "tight_corners": True,  "organizer": "JRA"},
    "函館":     {"straight_m": 262, "circumference_m": 1640, "has_slope": False, "left_turn": True,  "tight_corners": True,  "organizer": "JRA"},
    # NAR
    "帯広":     {"straight_m": 200,  "circumference_m": 0,    "has_slope": True,  "left_turn": False, "tight_corners": False, "organizer": "NAR", "banei": True},
    "門別":     {"straight_m": 400,  "circumference_m": 1600, "has_slope": False, "left_turn": False, "tight_corners": False, "organizer": "NAR"},
    "盛岡":     {"straight_m": 400,  "circumference_m": 1600, "has_slope": True,  "left_turn": True,  "tight_corners": False, "organizer": "NAR"},
    "水沢":     {"straight_m": 317,  "circumference_m": 1200, "has_slope": False, "left_turn": False, "tight_corners": True,  "organizer": "NAR"},
    "浦和":     {"straight_m": 220,  "circumference_m": 1200, "has_slope": False, "left_turn": True,  "tight_corners": True,  "organizer": "NAR"},
    "船橋":     {"straight_m": 362,  "circumference_m": 1400, "has_slope": False, "left_turn": True,  "tight_corners": False, "organizer": "NAR"},
    "大井":     {"straight_m": 486,  "circumference_m": 1600, "has_slope": False, "left_turn": True,  "tight_corners": False, "organizer": "NAR"},
    "川崎":     {"straight_m": 300,  "circumference_m": 1200, "has_slope": False, "left_turn": True,  "tight_corners": True,  "organizer": "NAR"},
    "金沢":     {"straight_m": 236,  "circumference_m": 1200, "has_slope": False, "left_turn": False, "tight_corners": True,  "organizer": "NAR"},
    "笠松":     {"straight_m": 201,  "circumference_m": 1100, "has_slope": False, "left_turn": False, "tight_corners": True,  "organizer": "NAR"},
    "名古屋":   {"straight_m": 240,  "circumference_m": 1180, "has_slope": False, "left_turn": False, "tight_corners": False, "organizer": "NAR"},
    "園田":     {"straight_m": 213,  "circumference_m": 1051, "has_slope": True,  "left_turn": False, "tight_corners": True,  "organizer": "NAR"},
    "姫路":     {"straight_m": 230,  "circumference_m": 1200, "has_slope": False, "left_turn": False, "tight_corners": False, "organizer": "NAR"},
    "高知":     {"straight_m": 200,  "circumference_m": 1100, "has_slope": False, "left_turn": False, "tight_corners": True,  "organizer": "NAR"},
    "佐賀":     {"straight_m": 200,  "circumference_m": 1100, "has_slope": False, "left_turn": False, "tight_corners": True,  "organizer": "NAR"},
}

# コースクラスタ（直線長ベース）
STRAIGHT_CLUSTERS = {
    "ultra_short": (0, 220),      # 極短直線（浦和・帯広・笠松・高知・佐賀）
    "short":       (221, 310),    # 短直線（水沢・小倉・福島・園田・川崎・姫路・名古屋・金沢）
    "medium":      (311, 420),    # 中直線（中山・盛岡・京都・中京・門別・船橋）
    "long":        (421, 9999),   # 長直線（東京・阪神・大井・新潟）
}


def get_course_features(track: str) -> dict:
    """コース名から特性辞書を返す。未知コースはデフォルト値。"""
    return COURSE_FEATURES.get(track, {
        "straight_m": 300,
        "circumference_m": 1400,
        "has_slope": False,
        "left_turn": True,
        "tight_corners": False,
        "organizer": "JRA",
    })


def straight_cluster(straight_m: int) -> str:
    """直線長からクラスタ名を返す。"""
    for name, (lo, hi) in STRAIGHT_CLUSTERS.items():
        if lo <= straight_m <= hi:
            return name
    return "medium"


def add_course_geometry_features(df: pd.DataFrame, track_col: str = "track") -> pd.DataFrame:
    """
    DataFrame のtrack列からコース形状特徴量を追加して返す。

    追加カラム:
      - straight_m        : 直線長
      - circumference_m   : 周回距離
      - has_slope         : 坂の有無
      - left_turn         : 左回りか
      - tight_corners     : タイトコーナーか
      - straight_cluster  : ultra_short/short/medium/long
      - is_banei          : ばんえいか（別モデル対象フラグ）
      - organizer_jra     : JRA=1, NAR=0
      - style_x_straight  : straight_m × style_score_wavg3（交互作用）
    """
    result = df.copy()

    course_cols = {
        "straight_m":       [],
        "circumference_m":  [],
        "has_slope":        [],
        "left_turn":        [],
        "tight_corners":    [],
        "straight_cluster": [],
        "is_banei":         [],
        "organizer_jra":    [],
    }

    for _, row in df.iterrows():
        track = row.get(track_col, "")
        cf = get_course_features(track)
        sm = cf["straight_m"]
        course_cols["straight_m"].append(sm)
        course_cols["circumference_m"].append(cf["circumference_m"])
        course_cols["has_slope"].append(int(cf["has_slope"]))
        course_cols["left_turn"].append(int(cf["left_turn"]))
        course_cols["tight_corners"].append(int(cf["tight_corners"]))
        course_cols["straight_cluster"].append(straight_cluster(sm))
        course_cols["is_banei"].append(int(cf.get("banei", False)))
        course_cols["organizer_jra"].append(int(cf["organizer"] == "JRA"))

    for col, vals in course_cols.items():
        result[col] = vals

    # 脚質 × 直線長の交互作用（running_style.py と連携）
    if "style_score_wavg3" in result.columns:
        result["style_x_straight"] = result["style_score_wavg3"] * result["straight_m"]

    return result
