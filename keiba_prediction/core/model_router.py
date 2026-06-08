"""
競馬場・レース種別に応じて正しいモデルを返すルーター。
絶対に間違ったモデルを使わない設計。
"""
from __future__ import annotations
import logging
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)

# ────────────────────────────────────────────────
# 競馬場グループ定義
# ────────────────────────────────────────────────
TRACK_GROUPS = {
    "banei":    {"帯広"},
    "kanto":    {"大井", "川崎", "船橋", "浦和"},
    "nar_local":{"名古屋", "笠松", "園田", "姫路",
                 "金沢", "高知", "佐賀", "盛岡",
                 "水沢", "門別"},
    "jra":      {"東京", "阪神", "中山", "京都",
                 "中京", "小倉", "函館", "札幌",
                 "福島", "新潟"},
}

# ────────────────────────────────────────────────
# 払戻率（場別）
# ────────────────────────────────────────────────
PAYBACK_RATES: dict[str, float] = {
    "大井": 0.725, "川崎": 0.725, "船橋": 0.725, "浦和": 0.725,
    "帯広": 0.70,
    **{t: 0.70 for t in TRACK_GROUPS["nar_local"]},
    **{t: 0.75 for t in TRACK_GROUPS["jra"]},
}


def get_track_group(track: str, race_type: str = "flat") -> str:
    """
    競馬場名とレース種別からグループを返す。
    未知の競馬場は "nar_local" として扱う（安全側に倒す）。
    """
    if race_type == "banei" or track == "帯広":
        return "banei"
    for group, tracks in TRACK_GROUPS.items():
        if track in tracks:
            return group
    log.warning(f"未知の競馬場: '{track}' → nar_local として処理")
    return "nar_local"


def get_payback(track: str) -> float:
    """場別の払戻率を返す（デフォルト 0.70）。"""
    return PAYBACK_RATES.get(track, 0.70)


def predict_race(race: dict, date_str: str) -> dict | None:
    """
    競馬場に応じて正しい予測関数を呼び出す唯一の入口。

    Parameters
    ----------
    race     : nar_races テーブルの1行（dict）
    date_str : 予測日

    Returns
    -------
    predict_dual._predict_one_race() と同じフォーマットの dict、
    またはデータ不足時は None。
    """
    track     = race.get("track", "")
    race_type = race.get("race_type", "flat")
    group     = get_track_group(track, race_type)

    if group == "banei":
        return _predict_banei(race, date_str)
    else:
        return _predict_flat(race, date_str)


# ────────────────────────────────────────────────
# 内部実装
# ────────────────────────────────────────────────
def _predict_flat(race: dict, date_str: str) -> dict | None:
    """平地競馬の予測（既存エンジンを呼ぶ）。"""
    from pipeline.predict_dual import _predict_one_race
    return _predict_one_race(race, date_str)


def _predict_banei(race: dict, date_str: str) -> dict | None:
    """ばんえい競馬の予測（専用エンジンを呼ぶ）。"""
    from data.banei_model import predict_banei_race
    from db.database import get_conn
    from core.validators import validate_race_data, validate_ev
    from features.race_chaos import compute_chaos_score
    import numpy as np

    rid = race.get("race_id", "")
    rno = race.get("race_no", 0)

    with get_conn() as conn:
        horses_db = conn.execute("""
            SELECT DISTINCT draw_number, horse_name, win_odds,
                   horse_weight, weight_carried as burden_weight
            FROM nar_results WHERE race_id=? AND win_odds > 0
            ORDER BY win_odds
        """, (rid,)).fetchall()
    horses_db = [dict(h) for h in horses_db]

    # バリデーション
    is_valid, errors = validate_race_data(horses_db, race_id=rid)
    if len(horses_db) < 3:
        return None   # データ不足 → プレースホルダー表示

    dns   = [h["draw_number"] for h in horses_db]
    names = [h["horse_name"]  for h in horses_db]
    odds  = [float(h["win_odds"]) for h in horses_db]
    hws   = [h.get("horse_weight") for h in horses_db]
    bws   = [h.get("burden_weight") for h in horses_db]

    result = predict_banei_race(dns, names, odds, hws, bws)

    # EV バリデーション
    for t in result["tickets_a"] + result["tickets_b"]:
        t["ev"] = validate_ev(t.get("ev"), label=f"{rid} {t.get('combo')}")
        if t["ev"] is None:
            t["ev"] = 0.0

    # chaos
    odds_arr = np.array(odds)
    chaos = compute_chaos_score(
        race.get("race_class",""), race.get("track_condition","良"),
        len(horses_db), odds_arr, rno, 12, 200, "帯広"
    )

    # 予測上位4頭
    horse_list = [
        {"num":h["num"],"name":h["name"][:8],"score":h["score"],
         "gap":0.0,"pace":0.5,"jvScore":0.5,"weightZ":0.0,
         "rankPct":0.0,"shap":0.0}
        for h in result["horses"][:4]
    ]

    # tickets 整形
    tickets = []
    for i, t in enumerate(result["tickets_a"][:5]):
        tickets.append({
            "combo":t["combo"],"p":t["p_model"],
            "ev":t["ev"],"kelly":"¥100","mode":"A",
            **({"topEv":True} if i==0 else {})
        })
    for i, t in enumerate(result["tickets_b"][:3]):
        tickets.append({
            "combo":t["combo"],"p_model":t["p_model"],
            "estOdds":t["est_odds"],"gap":t["ev"],
            "kelly":"—","mode":"B",
            **({"topGap":True} if i==0 else {})
        })

    return {
        "race_id":    rid,
        "race_name":  race.get("race_name",""),
        "track":      "帯広",
        "race_no":    rno,
        "surface":    "ばんえい",
        "distance":   race.get("distance", 200),
        "going":      race.get("track_condition","良"),
        "field_size": len(horses_db),
        "payback":    get_payback("帯広"),
        "race_class": race.get("race_class",""),
        "post_time":  race.get("post_time","—"),
        "chaos":      chaos,
        "is_high_odds_target": chaos["chaos_score"] >= 40,
        "hot_reasons": [result["top_note"][:50]],
        "system_a":   result["tickets_a"][:5],
        "system_b":   result["tickets_b"][:3],
        "horses":     horse_list,
        "no_data":    False,
        "is_banei":   True,
    }
