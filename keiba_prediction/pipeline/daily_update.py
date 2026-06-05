"""
毎日5:30 実行のメインパイプライン（新フロー）

① 前日の全レース結果を取得・照合
② 的中率・ROIを計算してDBに保存
③ 本日のNARデータを取得
④ 本日の予測を生成
⑤ サイトを更新してGitHubにpush
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from config import KELLY_CAP_PER_RACE
from data.scraper import NetkeibaStaticScraper, NetkeibaPlaywrightScraper
from data.nar_scraper import NARScraper
from db.database import get_conn, init_db, DB_PATH
from evaluation.drift_monitor import DriftMonitor
from evaluation.shap_explainer import build_shap_summary
from features.feature_pipeline import build_features, select_model_features
from models.ranker import load_ranker, predict_scores
from probability.plackett_luce import top_n_combos
from probability.expected_value import score_bet

# index.html の場所（リポジトリルート）
REPO_ROOT   = Path(__file__).parent.parent.parent
INDEX_HTML  = REPO_ROOT / "index.html"
GITHUB_TOKEN = ""   # git credential helper で管理するため空欄


# ──────────────────────────────────────────────────
# メインパイプライン
# ──────────────────────────────────────────────────
def run_daily(target_date: date | None = None) -> None:
    if target_date is None:
        target_date = date.today()

    today_str = target_date.isoformat()
    yesterday = target_date - timedelta(days=1)
    print(f"=== 日次更新: {today_str} ===")

    init_db()

    # ── STEP 1: 前日の結果照合・評価 ─────────────
    print("--- STEP1: 前日結果照合 ---")
    try:
        from pipeline.fetch_results import fetch_results_for_date
        from pipeline.evaluate_daily import evaluate_date, init_performance_tables
        init_performance_tables()
        fetch_results_for_date(yesterday)
        perf = evaluate_date(yesterday)
        if perf:
            print(f"  前日評価完了: A的中率{perf.get('a_hit_rate',0)*100:.0f}% ROI{perf.get('a_roi',0):.0f}%")
    except Exception as e:
        print(f"  前日評価スキップ（{e}）")

    # ── STEP 2: 本日NAR全場データ取得 + 自動リトライ ──
    print("--- STEP2: 本日NAR取得 ---")
    _run_nar_scraping(target_date)

    # kg_不正IDの修復 & entries→results コピー
    try:
        from pipeline.auto_retry import fix_invalid_ids, copy_entries_to_results
        fix_invalid_ids(today_str)
        n = copy_entries_to_results(today_str)
        print(f"  entries→results コピー: {n}頭")
    except Exception as e:
        print(f"  修復スキップ: {e}")

    date_str = today_str

    # ── STEP 3: JRAスクレイピング ─────────────────
    scraper = NetkeibaStaticScraper()
    entries_list = _fetch_entries(scraper, date_str)
    scraping_success_rate = scraper.success_rate

    if not entries_list:
        print("JRAエントリなし（非開催日またはNARのみ）。")

    if not entries_list:
        # NAR予測のみ実行
        print("--- STEP3: NAR予測のみ ---")
        try:
            from pipeline.predict_dual import run_dual_predict
            run_dual_predict(target_date=target_date, push=False)
        except Exception as e:
            print(f"  NAR予測エラー（{e}）")
        roi_data = _load_roi_stats()
        generate_html(date_str, races_data=[], roi_data=roi_data)
        _git_push(date_str)
        return

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
                "race_id":      race_id,
                "combo":        combo_str,
                "horse_ids":    json.dumps(list(c["combo_ids"])),
                "p_hit":        c["p_hit"],
                "display_odds": odds,
                **bet,
            })

    # ── 8. DB保存 ────────────────────────────────
    _save_predictions(df, date_str)
    _save_recommended_bets(all_bets)

    # ── 9. ドリフト監視 ──────────────────────────
    _run_drift_check(df, scraping_success_rate)

    # ── 9. ドリフト監視 ──────────────────────────
    _run_drift_check(df, scraping_success_rate)

    # ── 10. HTML生成（JRA+NAR統合） ──────────────
    try:
        from pipeline.predict_dual import run_dual_predict
        run_dual_predict(target_date=target_date, push=False)
    except Exception as e:
        print(f"  NAR予測エラー（処理継続）: {e}")
        races_data = _build_races_json(df, all_bets, date_str)
        roi_data   = _load_roi_stats()
        generate_html(date_str, races_data, roi_data)

    # ── 11. GitHub push ───────────────────────────
    _git_push(date_str)

    print(f"完了: 推奨買い目 {len(all_bets)} 点")


# ──────────────────────────────────────────────────
# HTML生成
# ──────────────────────────────────────────────────
def generate_html(
    date_str: str,
    races_data: list[dict],
    roi_data: dict | None = None,
) -> None:
    """
    index.html の RACES / SLICE_DATA / WEEKLY_ROI を実データで置き換える。

    Parameters
    ----------
    date_str   : "YYYY-MM-DD"
    races_data : _build_races_json() の出力
    roi_data   : {"slice": [...], "weekly_roi": [...]} または None
    """
    if not INDEX_HTML.exists():
        print(f"[HTML] {INDEX_HTML} が見つかりません。スキップ。")
        return

    html = INDEX_HTML.read_text(encoding="utf-8")

    # ── RACES ────────────────────────────────────
    if races_data:
        races_js = json.dumps(races_data, ensure_ascii=False, indent=2)
    else:
        # 非開催日
        races_js = json.dumps([{
            "id": "NO_RACE",
            "name": "本日は開催なし",
            "meta": "—",
            "badge": "jra",
            "time": "—",
            "field": 0,
            "horses": [],
            "tickets": [],
            "explanation": f"{date_str} は競馬の開催がありません。",
            "shapFeats": [],
        }], ensure_ascii=False, indent=2)

    html = _replace_js_const(html, "RACES", races_js)

    # ── SLICE_DATA ────────────────────────────────
    if roi_data and "slice" in roi_data:
        slice_js = json.dumps(roi_data["slice"], ensure_ascii=False, indent=2)
        html = _replace_js_const(html, "SLICE_DATA", slice_js)

    # ── WEEKLY_ROI ───────────────────────────────
    if roi_data and "weekly_roi" in roi_data:
        weekly_js = json.dumps(roi_data["weekly_roi"], ensure_ascii=False)
        html = _replace_js_const(html, "WEEKLY_ROI", weekly_js)

    # ── HIT_RECORDS (的中事例) ────────────────────
    # 問題4対応: 確認済みの事実のみ。推測データは含めない。
    # 「的中した」事例は今後実際に的中した時だけ追加する。
    # 名古屋12R は「高配当チャンス選別済み・ただし買い目に1-5-6なし」の参照事例として記録。
    VERIFIED_FACTS = [
        {
            "date":       "2026-06-04",
            "race":       "名古屋12R 金シャチ最終戦(C)",
            "result":     "1-5-6",
            "payout":     23360,
            "chaosScore": 76,
            "notes":      "chaos76点で高配当チャンス選別✅ / 買い目に1-5-6は未含❌ / 参照事例",
            "detected":   False,
        },
    ]
    html = _replace_js_const(html, "HIT_RECORDS", json.dumps(VERIFIED_FACTS, ensure_ascii=False))

    INDEX_HTML.write_text(html, encoding="utf-8")
    print(f"[HTML] index.html を更新しました ({date_str}, レース{len(races_data)}件)")


def _replace_js_const(html: str, const_name: str, new_value_js: str) -> str:
    """
    `const NAME = <旧値>;` を `const NAME = <新値>;` に置き換える。
    値は配列 [...] またはオブジェクト {...} を想定。
    """
    # [ ... ] または数値配列を含む1行パターンと複数行パターン両方に対応
    pattern = rf"(const {re.escape(const_name)}\s*=\s*)(\[[\s\S]*?\]|\"[\s\S]*?\"|[0-9.]+)(;)"
    replacement = rf"\g<1>{new_value_js}\3"
    new_html, count = re.subn(pattern, replacement, html)
    if count == 0:
        print(f"[HTML] 警告: const {const_name} が見つかりませんでした。")
    return new_html


# ──────────────────────────────────────────────────
# DBからレース・予測・買い目データを組み立てる
# ──────────────────────────────────────────────────
def _build_races_json(
    pred_df: pd.DataFrame,
    all_bets: list[dict],
    date_str: str,
) -> list[dict]:
    """
    predictions / races / recommended_bets から
    index.html の RACES 配列用 JSON を組み立てる。
    """
    races = []

    with get_conn() as conn:
        race_rows = conn.execute(
            """
            SELECT race_id, track, race_no, surface, distance,
                   distance_band, track_condition, field_size,
                   weather, organizer
            FROM races
            WHERE race_date = ?
            ORDER BY race_no
            """,
            (date_str,),
        ).fetchall()

    for race_row in race_rows:
        race_id  = race_row["race_id"]
        track    = race_row["track"] or ""
        race_no  = race_row["race_no"] or 0
        surface  = race_row["surface"] or ""
        distance = race_row["distance"] or 0
        going    = race_row["track_condition"] or "良"
        field    = race_row["field_size"] or 0
        org      = (race_row["organizer"] or "JRA").lower()

        meta  = f"{track} / {surface}{distance}m / {going}"
        badge = "nar" if org != "jra" else "jra"
        name  = f"{track}{race_no}R"

        # ── 馬リスト（スコア降順 top4）──────────
        race_preds = pred_df[pred_df["race_id"] == race_id].sort_values(
            "pred_score", ascending=False
        ).head(4)

        horses = []
        for _, row in race_preds.iterrows():
            shap5 = []
            if row.get("shap_top5_json"):
                try:
                    shap5 = json.loads(row["shap_top5_json"])
                except Exception:
                    pass
            top_shap_val = shap5[0]["shap_value"] if shap5 else 0.0

            horses.append({
                "num":      int(row.get("draw_number", 0)),
                "name":     str(row.get("horse_name", "")),
                "score":    round(float(row.get("pred_score", 0)), 3),
                "rankPct":  round(float(row.get("agari3f_rank_pct_hist_mean", 0) or 0), 2),
                "shap":     round(float(top_shap_val), 3),
            })

        # ── 推奨買い目（このレース分・EV降順 top5）──
        race_bets = [b for b in all_bets if b["race_id"] == race_id]
        race_bets.sort(key=lambda x: x.get("ev", 0), reverse=True)

        tickets = []
        for i, b in enumerate(race_bets[:5]):
            tickets.append({
                "combo":  b["combo"],
                "p":      round(float(b["p_hit"]), 4),
                "ev":     round(float(b["ev"]), 4),
                "kelly":  f"¥{int(b['stake'])}",
                **({"topEv": True} if i == 0 else {}),
            })

        # ── SHAP説明（軸馬のトップ特徴量）──────────
        shap_feats = []
        if len(race_preds) > 0:
            top_row = race_preds.iloc[0]
            if top_row.get("shap_top5_json"):
                try:
                    shap5 = json.loads(top_row["shap_top5_json"])
                    shap_feats = [
                        {
                            "name": s["feature"],
                            "val":  round(float(s["shap_value"]), 3),
                            "dir":  "pos" if s["shap_value"] >= 0 else "neg",
                        }
                        for s in shap5
                    ]
                except Exception:
                    pass

        explanation = _make_explanation(horses, tickets, track, surface, distance, going)

        races.append({
            "id":          race_id,
            "name":        name,
            "meta":        meta,
            "badge":       badge,
            "time":        "—",   # 発走時刻はDBに追加後に反映
            "field":       field,
            "horses":      horses,
            "tickets":     tickets,
            "explanation": explanation,
            "shapFeats":   shap_feats,
        })

    return races


def _make_explanation(
    horses: list[dict],
    tickets: list[dict],
    track: str,
    surface: str,
    distance: int,
    going: str,
) -> str:
    """シンプルな予測根拠テキストを生成する。"""
    if not horses:
        return "予測データなし。"
    top = horses[0]
    second = horses[1] if len(horses) > 1 else None
    top_ticket = tickets[0] if tickets else None

    lines = [
        f"軸は{top['name']}（{top['num']}番）。",
        f"予測スコア {top['score']:.3f}、SHAP寄与 +{top['shap']:.3f}。",
    ]
    if second:
        lines.append(f"対抗は{second['name']}（{second['num']}番）。")
    if top_ticket:
        lines.append(
            f"最高EV組み合わせ {top_ticket['combo']}: "
            f"的中確率 {top_ticket['p']:.1%}、EV +{top_ticket['ev']:.3f}、"
            f"Half-Kelly推奨 {top_ticket['kelly']}。"
        )
    lines.append(f"コース: {track} {surface}{distance}m {going}。")
    return "".join(lines)


# ──────────────────────────────────────────────────
# ROI統計の読み込み
# ──────────────────────────────────────────────────
def _load_roi_stats() -> dict:
    """
    daily_performance / race_results から集計済みROI統計を返す。
    データが不足している場合はダミー値を返す（HTML表示を壊さない）。
    """
    try:
        from pipeline.evaluate_daily import get_dashboard_data
        dash = get_dashboard_data(days=30)

        # 日次ROI → weekly_roi 形式に変換
        daily = dash.get("daily", [])
        weekly_roi = [round(float(d.get("a_roi") or 0), 1) for d in reversed(daily)][-8:] or [0.0]

        # slice_data → track_roi から生成
        track_roi = dash.get("track_roi", [])
        slice_data = [
            {
                "surface": r.get("track", ""),
                "dist":    "",
                "cnt":     r.get("n", 0),
                "roi":     float(r.get("roi") or 0),
                "hit":     float(r.get("a_hits", 0)) / max(r.get("n", 1), 1) * 100,
                "ev":      0.0,
            }
            for r in track_roi
        ]

        if not slice_data:
            slice_data = _default_slice_data()

        # 累計統計
        cumul = dash.get("cumul", {})
        hit_records = dash.get("hits", [])

        return {
            "slice":       slice_data,
            "weekly_roi":  weekly_roi,
            "cumul":       cumul,
            "hit_records": hit_records,
        }

    except Exception as e:
        print(f"[HTML] ROI統計の取得に失敗（ダミー値を使用）: {e}")
        return {"slice": _default_slice_data(), "weekly_roi": [0.0], "cumul": {}, "hit_records": []}


def _default_slice_data() -> list[dict]:
    return [
        {"surface": "芝",      "dist": "短距離",       "cnt": 0, "roi": 0.0, "hit": 0.0, "ev": 0.0},
        {"surface": "芝",      "dist": "マイル・中距離", "cnt": 0, "roi": 0.0, "hit": 0.0, "ev": 0.0},
        {"surface": "ダート",  "dist": "短距離",        "cnt": 0, "roi": 0.0, "hit": 0.0, "ev": 0.0},
        {"surface": "ダート",  "dist": "マイル・中距離", "cnt": 0, "roi": 0.0, "hit": 0.0, "ev": 0.0},
    ]


# ──────────────────────────────────────────────────
# GitHub push
# ──────────────────────────────────────────────────
def _git_push(date_str: str) -> None:
    """
    index.html を git add → commit → push する。
    失敗してもパイプライン全体はエラーにしない。
    """
    try:
        subprocess.run(
            ["git", "add", "index.html"],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
        )

        # 差分がない場合は commit をスキップ
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(REPO_ROOT),
            capture_output=True,
        )
        if diff.returncode == 0:
            print("[Git] index.html に変更なし。pushをスキップ。")
            return

        subprocess.run(
            ["git", "commit", "-m", f"update {date_str}"],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
        )
        print(f"[Git] push 完了: update {date_str}")

    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr else ""
        print(f"[Git] push 失敗（処理は継続）: {stderr.strip()}")
    except Exception as e:
        print(f"[Git] 予期しないエラー（処理は継続）: {e}")


# ──────────────────────────────────────────────────
# 内部ヘルパー
# ──────────────────────────────────────────────────
def _fetch_entries(scraper: NetkeibaStaticScraper, date_str: str) -> list[dict]:
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


def _run_nar_scraping(target_date: date) -> None:
    """NAR全場の今日のレースを取得してDBに保存。失敗しても継続。"""
    try:
        scraper = NARScraper(sleep_sec=2.5, max_retry=3)
        stats = scraper.run_today(target_date)
        print(f"[NAR] 取得完了: {stats}")
    except Exception as e:
        print(f"[NAR] 取得失敗（処理は継続）: {e}")


def _run_drift_check(df: pd.DataFrame, scraping_success_rate: float) -> None:
    import numpy as np
    monitor = DriftMonitor(baselines={})
    monitor.run_all_checks(scraping_success_rate=scraping_success_rate, mean_ev=0.0)


def _to_combo_str(horse_ids: tuple, race_df: pd.DataFrame) -> str:
    id_to_num = dict(zip(race_df["horse_id"], race_df["draw_number"]))
    nums = sorted(id_to_num.get(h, 0) for h in horse_ids)
    return "-".join(str(n) for n in nums)


if __name__ == "__main__":
    run_daily()
