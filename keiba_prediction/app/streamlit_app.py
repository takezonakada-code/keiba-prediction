"""
Streamlit フロントエンド UI。
Tab1: 本日の予測
Tab2: ROIダッシュボード
Tab3: モデル状態
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="競馬3連複予測システム",
    page_icon="🏇",
    layout="wide",
)


# ────────────────────────────────────────────────
# キャッシュ
# ────────────────────────────────────────────────
@st.cache_resource
def load_model_and_db():
    """モデル・DB接続（起動時1回）。"""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from db.database import init_db
    init_db()
    try:
        from models.ranker import load_ranker
        ranker = load_ranker()
    except Exception:
        ranker = None
    return ranker


@st.cache_data(ttl=60)
def load_predictions(race_date: str, race_id: str) -> pd.DataFrame:
    """予測結果（1分キャッシュ）。"""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from db.database import get_conn
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions WHERE race_id = ?", (race_id,)
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


@st.cache_data(ttl=3600)
def load_roi_dashboard() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """ROI集計（1時間キャッシュ）。"""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from db.database import fetch_bet_results, fetch_roi_by_slice
    from evaluation.metrics import weekly_roi_series
    from pipeline.evaluate_predictions import evaluate_cumulative

    bet_df = pd.DataFrame(fetch_bet_results())
    roi_slice = pd.DataFrame(fetch_roi_by_slice())
    cumulative = evaluate_cumulative() if len(bet_df) > 0 else {}

    weekly = weekly_roi_series(bet_df) if len(bet_df) > 0 else pd.DataFrame()
    return roi_slice, weekly, cumulative


@st.cache_data(ttl=3600)
def load_model_eval_logs() -> pd.DataFrame:
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from db.database import get_conn
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM model_eval_logs ORDER BY created_at").fetchall()
    return pd.DataFrame([dict(r) for r in rows])


# ────────────────────────────────────────────────
# サイドバー
# ────────────────────────────────────────────────
load_model_and_db()

with st.sidebar:
    st.title("🏇 競馬3連複予測")
    selected_date = st.date_input("日付選択", value=date.today())

    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    from db.database import get_conn
    with get_conn() as conn:
        race_rows = conn.execute(
            "SELECT race_id, track, race_no FROM races WHERE race_date = ? ORDER BY race_no",
            (selected_date.isoformat(),),
        ).fetchall()

    race_options = {f"{r['track']} {r['race_no']}R": r["race_id"] for r in race_rows}
    if race_options:
        selected_label = st.selectbox("レース選択", list(race_options.keys()))
        selected_race_id = race_options[selected_label]
    else:
        st.info("該当レースなし")
        selected_race_id = None


# ────────────────────────────────────────────────
# メインタブ
# ────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["本日の予測", "ROIダッシュボード", "モデル状態"])


# ── Tab1: 本日の予測 ───────────────────────────
with tab1:
    st.header("本日の予測")

    if selected_race_id is None:
        st.warning("レースを選択してください。")
    else:
        pred_df = load_predictions(selected_date.isoformat(), selected_race_id)

        if pred_df.empty:
            st.info("予測データがまだありません。")
        else:
            # レース概要
            with get_conn() as conn:
                race_info = conn.execute(
                    "SELECT * FROM races WHERE race_id = ?", (selected_race_id,)
                ).fetchone()

            if race_info:
                ri = dict(race_info)
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("頭数", ri.get("field_size", "-"))
                col2.metric("距離", f"{ri.get('distance', '-')}m")
                col3.metric("馬場", ri.get("track_condition", "-"))
                col4.metric("天気", ri.get("weather", "-"))

            st.subheader("予測スコア一覧")
            display_cols = ["horse_name", "pred_score", "p_top3",
                            "shap_top1_feature", "shap_top1_value"]
            show_df = pred_df[[c for c in display_cols if c in pred_df.columns]]
            show_df = show_df.sort_values("pred_score", ascending=False).reset_index(drop=True)
            st.dataframe(show_df, use_container_width=True)

            # 推奨買い目
            st.subheader("推奨買い目（EV順）")
            with get_conn() as conn:
                bet_rows = conn.execute(
                    """
                    SELECT combo, p_hit, display_odds, ev, kelly_frac, stake
                    FROM recommended_bets
                    WHERE race_id = ?
                    ORDER BY ev DESC
                    LIMIT 10
                    """,
                    (selected_race_id,),
                ).fetchall()

            if bet_rows:
                bet_df_show = pd.DataFrame([dict(r) for r in bet_rows])
                bet_df_show.columns = ["組み合わせ", "的中確率", "表示オッズ", "EV", "Kelly比率", "賭け金(円)"]
                st.dataframe(bet_df_show, use_container_width=True)
            else:
                st.info("推奨買い目なし（EV > 0 の組み合わせが見つかりませんでした）。")

            # SHAP説明
            if "shap_top5_json" in pred_df.columns:
                st.subheader("予測根拠（SHAP）")
                top_horse = show_df.iloc[0]
                horse_row = pred_df[pred_df["horse_name"] == top_horse.get("horse_name", "")]
                if not horse_row.empty and horse_row.iloc[0].get("shap_top5_json"):
                    shap_data = json.loads(horse_row.iloc[0]["shap_top5_json"])
                    shap_display = pd.DataFrame(shap_data)
                    st.write(f"**{top_horse.get('horse_name', '')}** の予測根拠 top5")
                    st.dataframe(shap_display, use_container_width=True)


# ── Tab2: ROIダッシュボード ───────────────────
with tab2:
    st.header("ROIダッシュボード")
    roi_slice, weekly_df, cumulative = load_roi_dashboard()

    if cumulative:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("累計ROI", f"{cumulative.get('roi_pct', 0):.1f}%")
        c2.metric("Sharpe", f"{cumulative.get('sharpe', 0):.3f}")
        c3.metric("最大DD", f"¥{cumulative.get('max_drawdown_yen', 0):,.0f}")
        c4.metric("的中率", f"{100*cumulative.get('hits',0)/max(cumulative.get('tickets',1),1):.1f}%")
    else:
        st.info("ベット実績がまだありません。")

    if not roi_slice.empty:
        st.subheader("スライス別ROI（芝/ダート × 距離帯）")
        st.dataframe(roi_slice, use_container_width=True)

    if not weekly_df.empty:
        st.subheader("週次ROI推移")
        chart_df = weekly_df.set_index("week")[["roi_pct"]].copy()
        chart_df.index = chart_df.index.astype(str)
        st.line_chart(chart_df)


# ── Tab3: モデル状態 ──────────────────────────
with tab3:
    st.header("モデル状態")
    eval_df = load_model_eval_logs()

    if not eval_df.empty:
        st.subheader("walk-forward 評価推移")
        for metric in ["logloss", "brier", "roi_pct"]:
            if metric in eval_df.columns:
                chart = eval_df[["test_month", metric]].dropna().set_index("test_month")
                st.line_chart(chart, height=200)
    else:
        st.info("モデル評価ログがまだありません。")

    st.subheader("禁止特徴量リスト（リーク防止）")
    from features.feature_specs import FORBIDDEN_FEATURES
    st.code("\n".join(FORBIDDEN_FEATURES))
