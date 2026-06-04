"""
今日の予測をDBに保存してindex.htmlを更新する。
MLモデルが未学習の場合はオッズ逆数（市場確率）をスコアとして使用し、
Plackett-Luce で3連複確率を計算する。
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.database import get_conn, init_db
from probability.plackett_luce import all_trifecta_box_probs, softmax_worth
from probability.expected_value import score_bet
from pipeline.daily_update import generate_html, _git_push, _load_roi_stats


def run_predict_today(target_date: date | None = None) -> None:
    """
    今日のNARレースデータをDBから読み込み、
    オッズベースの予測スコアを生成してindex.htmlを更新する。
    """
    if target_date is None:
        target_date = date.today()
    date_str = target_date.isoformat()

    print(f"=== 今日の予測生成: {date_str} ===")
    init_db()

    # ── 今日のレース取得 ─────────────────────────
    with get_conn() as conn:
        races = conn.execute(
            """SELECT * FROM nar_races WHERE race_date = ?
               AND race_type != 'banei'
               ORDER BY track, race_no""",
            (date_str,),
        ).fetchall()

    if not races:
        print("今日のレースデータなし。")
        generate_html(date_str, [], _load_roi_stats())
        _git_push(date_str)
        return

    print(f"  {len(races)}レース検出")

    races_data = []
    all_bets   = []

    for race in races:
        race_id   = race["race_id"]
        track     = race["track"]
        race_no   = race["race_no"]
        surface   = race["surface"] or "ダート"
        distance  = race["distance"] or 0
        going     = race["track_condition"] or "良"
        field     = race["field_size"] or 0
        payback   = race["payback_rate"] or 0.70
        org       = (race["organizer"] or "NAR").lower()
        race_name = race["race_name"] or f"{track}{race_no}R"

        # ── 馬データ取得（結果あり優先、なければentries）──────
        with get_conn() as conn:
            horses_db = conn.execute(
                """SELECT horse_id, horse_name, draw_number, win_odds, popular_rank,
                          finish_position, agari3f_seconds, horse_weight, horse_weight_diff
                   FROM nar_results WHERE race_id = ?
                   ORDER BY draw_number""",
                (race_id,),
            ).fetchall()
            if not horses_db:
                horses_db = conn.execute(
                    """SELECT horse_id, horse_name, draw_number, win_odds, popular_rank,
                              NULL as finish_position, NULL as agari3f_seconds,
                              horse_weight, horse_weight_diff
                       FROM nar_entries WHERE race_id = ?
                       ORDER BY draw_number""",
                    (race_id,),
                ).fetchall()

        if not horses_db or len(horses_db) < 3:
            continue

        horses_df = pd.DataFrame([dict(h) for h in horses_db])

        # ── スコア計算（オッズ逆数 = 市場確率）───────────────
        horses_df["win_odds"] = pd.to_numeric(horses_df["win_odds"], errors="coerce")
        horses_df = horses_df.dropna(subset=["win_odds"])
        horses_df = horses_df[horses_df["win_odds"] > 0].copy()

        if len(horses_df) < 3:
            continue

        # 市場確率をスコアとして使う
        inv_odds  = 1.0 / horses_df["win_odds"].values
        scores    = inv_odds / inv_odds.sum()   # 正規化勝率

        # ── PL確率で3連複候補を生成 ──────────────────────────
        worth     = softmax_worth(np.log(scores + 1e-9))
        probs     = all_trifecta_box_probs(worth)
        top_combos = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:10]

        horse_ids   = horses_df["horse_id"].values
        draw_numbers = horses_df["draw_number"].values

        # ── bets生成 ─────────────────────────────────────────
        race_bets = []
        for rank, (combo_idx, p_hit) in enumerate(top_combos[:5]):
            nums = sorted(int(draw_numbers[i]) for i in combo_idx)
            combo_str = "-".join(str(n) for n in nums)

            # 理論オッズ（払戻率 ÷ PL確率）
            theory_odds = (payback / p_hit) if p_hit > 0 else 999.0
            # 推奨賭け金: Half-Kelly簡易版（上位ほど厚く）
            kelly_stake = max(100, int(10_000 * p_hit / 2 / 100) * 100)
            kelly_stake = min(kelly_stake, 1000)

            race_bets.append({
                "race_id":    race_id,
                "combo":      combo_str,
                "p_hit":      p_hit,
                "ev":         payback - 1.0,   # 均等買いの期待値（参考）
                "kelly":      f"¥{kelly_stake}",
                "kelly_frac": p_hit / 2,
                "stake":      float(kelly_stake),
                "theory_odds": round(theory_odds, 1),
            })

        # ── 馬リスト（スコア降順top4）────────────────────────
        horses_df["pred_score"] = scores
        horses_df_sorted = horses_df.sort_values("pred_score", ascending=False).head(4)

        horse_list = []
        for _, row in horses_df_sorted.iterrows():
            horse_list.append({
                "num":     int(row.get("draw_number", 0)),
                "name":    str(row.get("horse_name", "")),
                "score":   round(float(row.get("pred_score", 0)), 3),
                "rankPct": 0.0,
                "shap":    0.0,
            })

        # ── 推奨buy目リスト ──────────────────────────────────
        ticket_list = []
        for i, b in enumerate(race_bets[:5]):
            t = {"combo": b["combo"], "p": round(b["p_hit"], 4), "ev": round(b["ev"], 4), "kelly": b["kelly"]}
            if i == 0:
                t["topEv"] = True
            ticket_list.append(t)

        # ── 説明文 ───────────────────────────────────────────
        exp = _make_explanation(horse_list, ticket_list, track, surface, distance, going, payback)

        badge = "nar"
        if track in ("大井", "川崎", "船橋", "浦和"):
            badge = "nar-nanka"

        races_data.append({
            "id":          race_id,
            "name":        f"{track}{race_no}R {race_name}",
            "meta":        f"{track} / {surface}{distance}m / {going}",
            "badge":       badge,
            "time":        dict(race).get("post_time") or "—",
            "field":       field,
            "horses":      horse_list,
            "tickets":     ticket_list,
            "explanation": exp,
            "shapFeats":   [],
        })
        all_bets.extend(race_bets)

        # DBに予測を保存
        _save_predictions_to_db(horses_df_sorted, race_id, date_str)

    print(f"  予測生成: {len(races_data)}レース、推奨買い目 {len(all_bets)}点")

    # ── HTML生成 & push ──────────────────────────────────────
    roi_data = _load_roi_stats()
    generate_html(date_str, races_data, roi_data)
    _git_push(date_str)


def _make_explanation(
    horses: list[dict],
    tickets: list[dict],
    track: str,
    surface: str,
    distance: int,
    going: str,
    payback: float,
) -> str:
    if not horses:
        return "予測データなし。"
    top = horses[0]
    second = horses[1] if len(horses) > 1 else None
    top_t = tickets[0] if tickets else None

    lines = [
        f"【市場確率ベース予測（モデル学習前）】",
        f"軸候補: {top['name']}（{top['num']}番） 市場スコア {top['score']:.3f}。",
    ]
    if second:
        lines.append(f"対抗: {second['name']}（{second['num']}番）。")
    if top_t:
        lines.append(
            f"推奨買い目 {top_t['combo']}: "
            f"PL確率 {top_t['p']:.1%}、理論EV {top_t['ev']:.3f}、"
            f"Half-Kelly {top_t['kelly']}。"
        )
    lines.append(
        f"{track} {surface}{distance}m {going}。"
        f"払戻率 {payback*100:.1f}%。"
        f"※MLモデル学習後に精度が向上します。"
    )
    return "".join(lines)


def _save_predictions_to_db(df: pd.DataFrame, race_id: str, date_str: str) -> None:
    with get_conn() as conn:
        # races テーブルに存在しない NAR レースは先にダミー登録する
        exists = conn.execute(
            "SELECT 1 FROM races WHERE race_id = ?", (race_id,)
        ).fetchone()
        if not exists:
            nr = conn.execute(
                "SELECT * FROM nar_races WHERE race_id = ?", (race_id,)
            ).fetchone()
            if nr:
                d = int(nr["distance"] or 0)
                band = "short" if d <= 1400 else "mile_middle" if d <= 2000 else "long"
                conn.execute("""
                    INSERT OR IGNORE INTO races
                      (race_id, race_date, track, race_no, surface, distance,
                       distance_band, track_condition, field_size, organizer)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (race_id, nr["race_date"], nr["track"], nr["race_no"],
                      nr["surface"], d, band,
                      nr["track_condition"], nr["field_size"], "NAR"))

        for _, row in df.iterrows():
            conn.execute("""
                INSERT OR REPLACE INTO predictions
                  (race_id, horse_id, horse_name, pred_score)
                VALUES (?, ?, ?, ?)
            """, (race_id, row.get("horse_id"), row.get("horse_name"),
                  float(row.get("pred_score", 0))))


if __name__ == "__main__":
    run_predict_today()
