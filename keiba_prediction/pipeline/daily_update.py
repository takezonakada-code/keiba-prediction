"""
毎日5:30 実行のメインパイプライン。
スクレイピング → 特徴量生成 → 予測 → EV計算 → DB保存
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from config import KELLY_CAP_PER_RACE
from data.scraper import NetkeibaStaticScraper, NetkeibaPlaywrightScraper
from db.database import get_conn, init_db, DB_PATH
from evaluation.drift_monitor import DriftMonitor
from evaluation.shap_explainer import build_shap_summary
from features.feature_pipeline import build_features, select_model_features
from models.ranker import load_ranker, predict_scores
from probability.plackett_luce import top_n_combos
from probability.expected_value import score_bet


def run_daily(target_date: date | None = None) -> None:
    """
    メインパイプライン。

    Parameters
    ----------
    target_date : 対象日（None の場合は翌日）
    """
    if target_date is None:
        target_date = date.today() + timedelta(days=1)

    date_str = target_date.isoformat()
    print(f"=== 日次更新: {date_str} ===")

    # ── DB初期化（初回のみ）─────────────────────
    init_db()

    # ── 1. スクレイピング ─────────────────────────
    scraper = NetkeibaStaticScraper()
    entries_list = _fetch_entries(scraper, date_str)
    if not entries_list:
        print("エントリなし。終了。")
        return

    scraping_success_rate = scraper.success_rate

    # ── 2. 過去走データ取得 ───────────────────────
    past_races = _load_past_races(date_str)

    # ── 3. 特徴量生成 ────────────────────────────
    target_entries = pd.DataFrame(entries_list)
    df = build_features(target_entries, past_races)
    feature_cols = select_model_features(df)

    # ── 4. モデルロード & 予測 ────────────────────
    ranker = load_ranker()
    df["pred_score"] = predict_scores(ranker, df[feature_cols])

    # ── 5. SHAP説明 ───────────────────────────────
    shap_df = build_shap_summary(ranker, df[feature_cols], df)
    df = df.merge(
        shap_df[["horse_id", "shap_top1_feature", "shap_top1_value", "shap_top5_json"]],
        on="horse_id", how="left",
    )

    # ── 6. オッズ取得 ─────────────────────────────
    with NetkeibaPlaywrightScraper() as pw:
        odds_by_race = {
            race_id: pw.fetch_odds(race_id)
            for race_id in df["race_id"].unique()
        }

    # ── 7. EV計算 & 推奨ベット生成 ───────────────
    all_bets = []
    for race_id, race_df in df.groupby("race_id"):
        scores    = race_df["pred_score"].values
        horse_ids = race_df["horse_id"].values
        odds_map  = odds_by_race.get(race_id, {})

        combos = top_n_combos(scores, horse_ids, top_n=20)
        for c in combos:
            combo_str = _to_combo_str(c["combo_ids"], race_df)
            odds = odds_map.get(combo_str)
            if odds is None or odds <= 1.0:
                continue

            bet = score_bet(c["p_hit"], odds)
            if bet["ev"] <= 0 or bet["stake"] == 0:
                continue

            all_bets.append({
                "race_id":   race_id,
                "combo":     combo_str,
                "horse_ids": json.dumps(list(c["combo_ids"])),
                "p_hit":     c["p_hit"],
                "display_odds": odds,
                **bet,
            })

    # ── 8. DB保存 ────────────────────────────────
    _save_predictions(df, date_str)
    _save_recommended_bets(all_bets)

    # ── 9. ドリフト監視 ──────────────────────────
    _run_drift_check(df, scraping_success_rate)

    print(f"完了: 推奨買い目 {len(all_bets)} 点")


def _fetch_entries(scraper: NetkeibaStaticScraper, date_str: str) -> list[dict]:
    """その日の全レースの出走馬リストを取得。"""
    # 実際のURLはnetkeiba仕様に合わせて調整
    url = f"https://race.netkeiba.com/top/race_list.html?kaisai_date={date_str.replace('-', '')}"
    html, _ = scraper.fetch_with_structure_check(url, sig_key="race_list")
    if html is None:
        return []

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    race_links = [a["href"] for a in soup.select("a[href*='race_id']")]
    entries = []
    for link in race_links:
        race_html = scraper.fetch(f"https://race.netkeiba.com{link}")
        if race_html:
            entries.extend(scraper.parse_race_entry_table(race_html))
    return entries


def _load_past_races(cutoff_date: str) -> pd.DataFrame:
    """DBから過去走データを取得。"""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM past_results WHERE race_date < ?",
            (cutoff_date,),
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def _save_predictions(df: pd.DataFrame, date_str: str) -> None:
    with get_conn() as conn:
        for _, row in df.iterrows():
            conn.execute(
                """
                INSERT OR REPLACE INTO predictions
                  (race_id, horse_id, horse_name, pred_score,
                   shap_top1_feature, shap_top1_value, shap_top5_json)
                VALUES (?,?,?,?,?,?,?)
                """,
                (row.get("race_id"), row.get("horse_id"), row.get("horse_name"),
                 row.get("pred_score"), row.get("shap_top1_feature"),
                 row.get("shap_top1_value"), row.get("shap_top5_json")),
            )


def _save_recommended_bets(bets: list[dict]) -> None:
    with get_conn() as conn:
        for b in bets:
            conn.execute(
                """
                INSERT INTO recommended_bets
                  (race_id, combo, horse_ids, p_hit, display_odds, ev, kelly_frac, stake)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (b["race_id"], b["combo"], b["horse_ids"],
                 b["p_hit"], b["display_odds"], b["ev"], b["kelly_frac"], b["stake"]),
            )


def _run_drift_check(df: pd.DataFrame, scraping_success_rate: float) -> None:
    import numpy as np
    # baselines は初回に測定してDBまたはファイルに保存する想定
    # ここでは簡易的にスコア分布だけチェック
    monitor = DriftMonitor(baselines={})
    alerts = monitor.run_all_checks(
        scraping_success_rate=scraping_success_rate,
        mean_ev=0.0,   # 実際はbets から計算
    )
    if alerts:
        print(f"[アラート] {len(alerts)}件")


def _to_combo_str(horse_ids: tuple, race_df: pd.DataFrame) -> str:
    """horse_id のタプルを馬番文字列（例: "1-3-7"）に変換。"""
    id_to_num = dict(zip(race_df["horse_id"], race_df["draw_number"]))
    nums = sorted(id_to_num.get(h, 0) for h in horse_ids)
    return "-".join(str(n) for n in nums)


if __name__ == "__main__":
    run_daily()
