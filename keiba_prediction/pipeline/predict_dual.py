"""
システムA（本命・的中率重視）＋ システムB（高配当狙い）の
デュアルモード予測エンジン。

出力:
  - system_a: EV順 5点（本命モード）
  - system_b: gap順 3点（高配当モード、50倍以上限定）
  - race_chaos: 荒れやすさスコア
  - is_high_odds_target: 高配当チャンスレースか
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import get_conn, init_db
from probability.plackett_luce import softmax_worth, all_trifecta_box_probs
from probability.expected_value import score_bet
from features.race_chaos import compute_chaos_score, is_high_odds_target_race
from features.odds_gap import (
    compute_odds_gap, find_high_odds_candidates, market_win_probs
)
from features.pace_position import pace_position_score, get_front_advantage
from features.jockey_venue import jockey_venue_score
from features.weight_relative import batch_relative_weight
from pipeline.daily_update import generate_html, _git_push, _load_roi_stats


def run_dual_predict(
    target_date: Optional[date] = None,
    push: bool = True,
) -> list[dict]:
    """
    今日の全NAR対象レースでシステムA/Bの予測を生成してindex.htmlを更新。

    Returns
    -------
    list[dict] : レースごとの予測結果
    """
    if target_date is None:
        target_date = date.today()
    date_str = target_date.isoformat()

    print(f"=== デュアルモード予測: {date_str} ===")
    init_db()

    # ── STEP 1: 前日結果取得・評価 ───────────────────
    yesterday = target_date - timedelta(days=1)
    try:
        from pipeline.fetch_results import fetch_results_for_date
        from pipeline.evaluate_daily import evaluate_date, init_performance_tables
        init_performance_tables()
        fetch_results_for_date(yesterday)
        perf = evaluate_date(yesterday)
        if perf:
            print(f"  前日評価: A {perf.get('a_hits',0)}/{perf.get('a_races',0)}R "
                  f"= {perf.get('a_hit_rate',0)*100:.0f}% ROI{perf.get('a_roi',0):.0f}%")
    except Exception as e:
        print(f"  前日評価スキップ: {e}")

    # ── STEP 2: 本日NAR全場データ取得 ────────────────
    try:
        from data.nar_scraper import NARScraper
        nar_scraper = NARScraper(sleep_sec=2.0)
        from data.nar_scraper import TRACK_MAP as _TM
        # ばんえい(帯広)はkg_IDを生成するためスクレイプをスキップ
        flat_codes = [k for k, v in _TM.items() if not v.get("banei")]
        nar_stats = nar_scraper.run_today(target_date, track_filter=flat_codes)
        print(f"  NAR取得: {nar_stats}")
    except Exception as e:
        print(f"  NAR取得スキップ: {e}")

    # kg_不正ID修復 & entries→results コピー
    try:
        from pipeline.auto_retry import fix_invalid_ids, copy_entries_to_results
        fix_invalid_ids(date_str)
        copy_entries_to_results(date_str)
    except Exception:
        pass

    with get_conn() as conn:
        races = conn.execute("""
            SELECT * FROM nar_races
            WHERE race_date = ? AND race_id NOT LIKE 'kg_%'
            ORDER BY CASE WHEN race_type='banei' THEN 1 ELSE 0 END, track, race_no
        """, (date_str,)).fetchall()

    if not races:
        print("本日の対象レースなし")
        generate_html(date_str, [], _load_roi_stats())
        if push:
            _git_push(date_str)
        return []

    all_race_data = []
    from core.model_router import predict_race as _route_predict

    for race in races:
        race_dict = dict(race)
        # ばんえいはmodel_router経由で専用モデルを使用
        if race_dict.get("race_type") == "banei":
            race_data = _route_predict(race_dict, date_str)
        else:
            race_data = _predict_one_race(race_dict, date_str)
        if race_data:
            all_race_data.append(race_data)

    # index.html 生成
    html_races = _to_html_format(all_race_data, date_str)
    roi_data   = _load_roi_stats()
    generate_html(date_str, html_races, roi_data)

    if push:
        _git_push(date_str)

    n_high = sum(1 for r in all_race_data if r.get("is_high_odds_target"))
    print(f"完了: {len(all_race_data)}レース（高配当チャンス: {n_high}件）")
    return all_race_data


# ──────────────────────────────────────────────────
# 1レース分の予測
# ──────────────────────────────────────────────────
def _predict_one_race(race: dict, date_str: str) -> Optional[dict]:
    race_id    = race["race_id"]
    track      = race["track"] or ""
    race_no    = race["race_no"]
    surface    = race["surface"] or "ダート"
    distance   = race["distance"] or 0
    going      = race["track_condition"] or "良"
    field_size = race["field_size"] or 0
    payback    = race["payback_rate"] or 0.70
    race_class = race["race_class"] or ""
    post_time  = race.get("post_time") or "—"
    race_name  = race.get("race_name") or f"{track}{race_no}R"

    # 馬データ取得
    # ⚠️ リーク防止: finish_position / agari3f_seconds は予測計算に一切使用しない。
    #    これらはレース後確定値であり、win_odds のみがモデルスコアの入力になる。
    with get_conn() as conn:
        horses_db = conn.execute("""
            SELECT horse_id, horse_name, draw_number, win_odds, popular_rank,
                   NULL AS finish_position,   -- 予測時は着順を参照しない（リーク防止）
                   NULL AS agari3f_seconds,   -- 予測時は上がりを参照しない（リーク防止）
                   horse_weight, horse_weight_diff,
                   weight_carried, jockey_id, jockey_name, corner4_pos
            FROM nar_results WHERE race_id = ?
            ORDER BY draw_number
        """, (race_id,)).fetchall()

    if not horses_db:
        with get_conn() as conn:
            horses_db = conn.execute("""
                SELECT horse_id, horse_name, draw_number, win_odds, popular_rank,
                       NULL as finish_position, NULL as agari3f_seconds,
                       horse_weight, horse_weight_diff, NULL as weight_carried,
                       jockey_id, jockey_name, NULL as corner4_pos
                FROM nar_entries WHERE race_id = ?
                ORDER BY draw_number
            """, (race_id,)).fetchall()

    # データ不足の場合はプレースホルダーカードを返す
    if not horses_db or len(horses_db) < 3:
        return {
            "race_id":    race_id, "race_name": race_name, "track": track,
            "race_no":    race_no, "surface": surface, "distance": distance,
            "going":      going, "field_size": field_size or 12, "payback": payback,
            "race_class": race_class, "post_time": post_time,
            "chaos":      {"chaos_score": 0, "is_high_chaos": False,
                           "is_mid_chaos": False, "level": "取得中"},
            "is_high_odds_target": False, "hot_reasons": [],
            "system_a": [], "system_b": [], "horses": [],
            "no_data":   True,  # プレースホルダーフラグ
        }

    hdf = pd.DataFrame([dict(h) for h in horses_db])
    hdf["win_odds"] = pd.to_numeric(hdf["win_odds"], errors="coerce")
    hdf = hdf[hdf["win_odds"].notna() & (hdf["win_odds"] > 0)].copy()

    if len(hdf) < 3:
        return {
            "race_id":    race_id, "race_name": race_name, "track": track,
            "race_no":    race_no, "surface": surface, "distance": distance,
            "going":      going, "field_size": field_size or 12, "payback": payback,
            "race_class": race_class, "post_time": post_time,
            "chaos":      {"chaos_score": 0, "is_high_chaos": False,
                           "is_mid_chaos": False, "level": "取得中"},
            "is_high_odds_target": False, "hot_reasons": [],
            "system_a": [], "system_b": [], "horses": [],
            "no_data":   True,
        }

    win_odds_arr = hdf["win_odds"].values
    n_horses     = len(hdf)

    # ── 各馬の追加特徴量 ──────────────────────────
    for i, row in hdf.iterrows():
        hid   = row["horse_id"] or ""
        jid   = row["jockey_id"] or ""

        # 先行力スコア
        pace = pace_position_score(hid, date_str, track)
        hdf.loc[i, "pace_score"]     = pace["pace_score"]
        hdf.loc[i, "pace_score_adj"] = pace["pace_score_adj"]

        # 騎手×競馬場
        hdf.loc[i, "jockey_venue_score"] = jockey_venue_score(jid, track, date_str)

    # 相対斤量
    wdf = batch_relative_weight([race_id])
    if not wdf.empty:
        wdf = wdf.rename(columns={"draw_number": "draw_number"})
        hdf = hdf.merge(
            wdf[["draw_number", "weight_z", "weight_rank_pct"]],
            on="draw_number", how="left"
        )
    else:
        hdf["weight_z"]        = 0.0
        hdf["weight_rank_pct"] = 0.5

    # ── モデルスコア（市場確率 + 追加特徴量の簡易合成） ──
    inv_odds = 1.0 / win_odds_arr
    p_market = inv_odds / inv_odds.sum()

    # 先行力・騎手スコアをブースト
    boost = np.ones(n_horses)
    front_adv = get_front_advantage(track)

    for i, row in hdf.reset_index(drop=True).iterrows():
        # 先行馬ブースト（小回りほど強い）
        ps = row.get("pace_score")
        if ps is not None and not np.isnan(float(ps)):
            front_boost = (1.0 - float(ps)) * front_adv   # 先行ほど高値
            boost[i] *= (1.0 + front_boost * 0.3)

        # 騎手×コース相性
        jvs = row.get("jockey_venue_score")
        if jvs is not None and not np.isnan(float(jvs)):
            boost[i] *= (1.0 + float(jvs) * 0.2)

        # 斤量軽さ
        wz = row.get("weight_z")
        if wz is not None and not np.isnan(float(wz)):
            boost[i] *= (1.0 + float(wz) * 0.1)

    model_scores = p_market * boost
    model_scores /= model_scores.sum()

    # ── オッズ乖離 ───────────────────────────────
    gaps = compute_odds_gap(model_scores, win_odds_arr)

    # ── 荒れやすさ ───────────────────────────────
    fav_odds  = float(win_odds_arr.min())
    top3_inv  = np.sort(1.0 / win_odds_arr)[-3:][::-1]
    total_inv = (1.0 / win_odds_arr).sum()
    top3_share = top3_inv.sum() / total_inv if total_inv > 0 else 1.0

    chaos = compute_chaos_score(
        race_class=race_class,
        track_condition=going,
        field_size=n_horses,
        win_odds=win_odds_arr,
        race_no=race_no,
        total_races=12,
        distance=distance,
        track=track,
    )
    is_hot, hot_reasons = is_high_odds_target_race(
        race_class, n_horses, fav_odds, top3_share, chaos["chaos_score"]
    )

    # ── 3連単15点選択 ────────────────────────────
    from probability.harville import market_win_probs as mwp
    from probability.sanrentan import select_sanrentan_15
    from core.validators import validate_ev, validate_scores

    horse_ids    = hdf["horse_id"].values
    draw_numbers = hdf["draw_number"].values

    # モデルスコアをバリデート
    model_scores_list = validate_scores(model_scores.tolist(), label=race_id)
    import numpy as _np
    model_scores = _np.array(model_scores_list)

    p_market_norm = mwp(win_odds_arr)

    tickets_15 = select_sanrentan_15(
        model_scores = model_scores,
        market_probs = p_market_norm,
        win_odds     = win_odds_arr,
        draw_numbers = draw_numbers,
        payback      = payback,
        n_hon = 8,
        n_ana = 4,
        n_are = 3,
    )

    # system_a / system_b に分割（後方互換）
    system_a = []
    system_b = []
    for i, t in enumerate(tickets_15):
        ev = validate_ev(t["ev"], label=f"{race_id} {t['combo']}") or 0.0
        if t["label"] in ("本命",):
            system_a.append({
                "combo":    t["combo"],
                "p_hit":    round(t["p_model"], 5),
                "est_odds": round(t["est_odds"], 1),
                "ev":       round(ev, 4),
                "kelly":    "¥100",
                "mode":     "A",
                "label":    t["label"],
            })
        else:
            system_b.append({
                "combo":    t["combo"],
                "p_model":  round(t["p_model"], 5),
                "est_odds": round(t["est_odds"], 1),
                "gap":      round(t["gap"], 3),
                "ev":       round(ev, 4),
                "mode":     "B",
                "label":    t["label"],
                "topGap":   len(system_b) == 0,
            })
    if system_a:
        system_a[0]["topEv"] = True

    # ── 馬リスト（model_scores降順） ─────────────
    hdf_sorted = hdf.copy()
    hdf_sorted["pred_score"] = model_scores
    hdf_sorted = hdf_sorted.sort_values("pred_score", ascending=False)

    horse_list = []
    for _, row in hdf_sorted.head(4).iterrows():
        gval = float(gaps[hdf.index.get_loc(row.name)] if row.name in hdf.index else 0)
        horse_list.append({
            "num":        int(row.get("draw_number", 0)),
            "name":       str(row.get("horse_name", "")),
            "score":      round(float(row["pred_score"]), 3),
            "gap":        round(gval, 3),
            "pace":       round(float(row.get("pace_score") or 0.5), 3),
            "jvScore":    round(float(row.get("jockey_venue_score") or 0.5), 3),
            "weightZ":    round(float(row.get("weight_z") or 0), 3),
            "rankPct":    0.0,
            "shap":       0.0,
        })

    return {
        "race_id":           race_id,
        "race_name":         race_name,
        "track":             track,
        "race_no":           race_no,
        "surface":           surface,
        "distance":          distance,
        "going":             going,
        "field_size":        n_horses,
        "payback":           payback,
        "race_class":        race_class,
        "post_time":         post_time,
        "chaos":             chaos,
        "is_high_odds_target": is_hot,
        "hot_reasons":       hot_reasons,
        "system_a":          system_a,
        "system_b":          system_b,
        "horses":            horse_list,
    }


# ──────────────────────────────────────────────────
# index.html 用データ変換
# ──────────────────────────────────────────────────
def _to_html_format(race_data: list[dict], date_str: str) -> list[dict]:
    """予測結果をindex.htmlのRACES配列形式に変換。"""
    from core.validators import validate_ev
    races = []
    for rd in race_data:
        # ばんえい専用フォーマット（model_routerから返ってくる場合）
        if rd.get("is_banei"):
            races.append(_format_banei_race(rd))
            continue

        # 買い目リスト（3連単15点 A+B統合）
        tickets = []
        for bet in rd["system_a"]:
            t = {
                "combo":   bet["combo"],
                "p":       bet["p_hit"],
                "ev":      bet["ev"],
                "kelly":   "¥100",
                "mode":    "A",
                "label":   bet.get("label", "本命"),
                "estOdds": bet.get("est_odds", 0),
            }
            if bet.get("topEv"):
                t["topEv"] = True
            tickets.append(t)

        for bet in rd["system_b"]:
            t = {
                "combo":    bet["combo"],
                "p":        bet.get("p_model", 0),
                "estOdds":  bet.get("est_odds", 0),
                "gap":      bet.get("gap", 0),
                "ev":       bet.get("ev", 0),
                "kelly":    "¥100",
                "mode":     "B",
                "label":    bet.get("label", "穴馬"),
                "highOdds": True,
            }
            if bet.get("topGap"):
                t["topGap"] = True
            tickets.append(t)

        chaos = rd["chaos"]
        badge = "nar"
        if rd["track"] in ("大井", "川崎", "船橋", "浦和"):
            badge = "nar-nanka"

        top = rd["horses"][0] if rd["horses"] else {}
        exp = _make_dual_explanation(rd, top)
        no_data = rd.get("no_data", False)

        races.append({
            "id":           rd["race_id"],
            "name":         f"{rd['track']}{rd['race_no']}R {rd['race_name']}",
            "meta":         f"{rd['track']} / {rd['surface']}{rd['distance']}m / {rd['going']}",
            "badge":        badge,
            "time":         rd.get("post_time") or "—",
            "field":        rd["field_size"],
            "raceClass":    rd.get("race_class", ""),
            "track":        rd["track"],
            "raceNo":       rd["race_no"],
            "chaosScore":   int(chaos["chaos_score"]),
            "isHighChaos":  bool(chaos["is_high_chaos"]),
            "chaosLevel":   chaos.get("level", "普通"),
            "isHighOddsTarget": rd["is_high_odds_target"],
            "hotReasons":   rd["hot_reasons"],
            "horses":       rd["horses"],
            "tickets":      tickets,
            "explanation":  exp,
            "shapFeats":    [],
            "noData":       no_data,
        })

    return races


def _format_banei_race(rd: dict) -> dict:
    """ばんえい専用データをRACES配列形式に変換する。"""
    from core.validators import validate_ev

    chaos = rd.get("chaos", {"chaos_score": 0, "is_high_chaos": False,
                              "is_mid_chaos": False, "level": "普通"})
    # tickets（A+B統合）
    tickets = []
    for i, t in enumerate(rd.get("system_a", [])[:5]):
        ev = validate_ev(t.get("ev"), label=rd.get("race_id","")) or 0.0
        tickets.append({
            "combo": t["combo"], "p": t.get("p_model", 0),
            "ev": round(ev, 4), "kelly": "¥100", "mode": "A",
            **({"topEv": True} if i == 0 else {}),
        })
    for i, t in enumerate(rd.get("system_b", [])[:3]):
        ev = validate_ev(t.get("ev"), label=rd.get("race_id","")) or 0.0
        tickets.append({
            "combo": t["combo"], "p_model": t.get("p_model", 0),
            "estOdds": t.get("est_odds", 0), "gap": round(ev, 4),
            "kelly": "—", "mode": "B",
            **({"topGap": True} if i == 0 else {}),
        })

    return {
        "id":           rd.get("race_id", ""),
        "name":         f"帯広{rd.get('race_no','')}R {rd.get('race_name','')[:15]}",
        "meta":         f"帯広 / ばんえい200m / {rd.get('going','良')}",
        "badge":        "banei",
        "time":         rd.get("post_time", "—"),
        "field":        rd.get("field_size", 0),
        "raceClass":    rd.get("race_class", ""),
        "track":        "帯広",
        "raceNo":       rd.get("race_no", 0),
        "chaosScore":   int(chaos.get("chaos_score", 0)),
        "isHighChaos":  bool(chaos.get("is_high_chaos", False)),
        "chaosLevel":   chaos.get("level", "普通"),
        "isHighOddsTarget": rd.get("is_high_odds_target", False),
        "hotReasons":   rd.get("hot_reasons", []),
        "horses":       rd.get("horses", []),
        "tickets":      tickets,
        "explanation":  rd.get("hot_reasons", [""])[0] if rd.get("hot_reasons") else "",
        "shapFeats":    [],
        "noData":       False,
        "isBanei":      True,
    }


def _make_dual_explanation(rd: dict, top_horse: dict) -> str:
    chaos = rd["chaos"]
    lines = []

    if rd["is_high_odds_target"]:
        lines.append(f"【高配当チャンス】荒れスコア{chaos['chaos_score']:.0%} "
                     f"({', '.join(rd['hot_reasons'])})。")

    if top_horse:
        lines.append(f"軸候補: {top_horse.get('name','')}（{top_horse.get('num','')}番）"
                     f" スコア{top_horse.get('score',0):.3f}")
        if top_horse.get("gap", 0) > 0.3:
            lines.append(f" gap+{top_horse['gap']:.2f}（市場過小評価）")
        if top_horse.get("pace", 0.5) < 0.3:
            lines.append(f" 先行力スコア{top_horse['pace']:.2f}（先行馬）")

    if rd["system_b"]:
        sb = rd["system_b"][0]
        lines.append(f"高配当狙い: {sb['combo']} 推定{sb['est_odds']:.0f}倍")

    lines.append(f"{rd['track']} {rd['surface']}{rd['distance']}m {rd['going']}。"
                 f"払戻率{rd['payback']*100:.1f}%。")
    return "".join(lines)


if __name__ == "__main__":
    run_dual_predict()
