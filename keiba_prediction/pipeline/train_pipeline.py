"""
学習パイプライン: Step 2〜6を順番に実行する。
Step 1（データ確認）は monitor_and_train.sh から呼ばれる。

実行方法:
  python -m pipeline.train_pipeline          # Step 2-6 全実行
  python -m pipeline.train_pipeline --step 3 # 指定ステップのみ
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ロガー設定
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "training_pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────
# Step 2: 特徴量生成
# ──────────────────────────────────────────────────
def step2_build_features() -> int:
    """全nar_resultsの特徴量を計算してtraining_featuresテーブルに保存。"""
    log.info("=== Step 2: 特徴量生成 ===")
    from db.database import get_conn

    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS training_features (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id         TEXT NOT NULL,
                race_date       TEXT NOT NULL,
                horse_id        TEXT,
                draw_number     INTEGER,
                -- ターゲット
                finish_position INTEGER,
                relevance       INTEGER,    -- max(0, 4-finish_position)
                -- 能力系
                agari3f_z       REAL,       -- 同レース内z-score（過去走平均）
                agari3f_rank_pct REAL,      -- 同レース内順位%（過去走平均）
                speed_index_mean REAL,      -- スピード指数（過去走平均）
                speed_index_trend REAL,     -- 直近傾き
                sf_ewma         REAL,       -- EWMAスコア
                -- 状態系
                rest_rbf_7      REAL,
                rest_rbf_14     REAL,
                rest_rbf_21     REAL,
                rest_rbf_35     REAL,
                rest_rbf_56     REAL,
                rest_rbf_84     REAL,
                rest_rbf_140    REAL,
                rest_rbf_210    REAL,
                layoff_150plus  INTEGER,
                -- 騎手・コース
                jockey_course_winrate REAL,
                jockey_win_rate_3m    REAL,
                jockey_track_winrate  REAL,   -- [NEW] 騎手×競馬場勝率（地元ボーナス）
                -- 斤量
                weight_z              REAL,   -- [NEW] 相対軽さスコア（z-score）
                weight_rank_pct       REAL,   -- [NEW] 斤量軽さ順位%
                -- 先行力
                style_score_c4        REAL,   -- [NEW] 先行力スコア（4角加重平均）
                style_vol_c4          REAL,   -- [NEW] 先行スタイル安定度
                front_pct             REAL,   -- [NEW] 先行率（3番手以内%）
                -- クラス別荒れ率
                class_upset_rate      REAL,   -- [NEW] 条件クラス別荒れ率
                -- 馬場・コース
                surface_enc     INTEGER,    -- 0=ダート,1=芝,2=ばんえい
                distance        INTEGER,
                distance_band_enc INTEGER, -- 0=short,1=mile_middle,2=long
                straight_m      INTEGER,
                -- 馬体重
                horse_weight    INTEGER,
                horse_weight_diff INTEGER,
                -- オッズ
                win_odds        REAL,
                popular_rank    INTEGER,
                field_size      INTEGER,
                -- メタ
                track           TEXT,
                race_type       TEXT,
                as_of_date      TEXT,
                UNIQUE(race_id, horse_id)
            )
        """)

    # 全レースを日付順に処理
    with get_conn() as conn:
        races = conn.execute("""
            SELECT r.race_id, r.race_date, r.track, r.surface, r.distance,
                   r.field_size, r.race_type, r.straight_m
            FROM nar_races r
            WHERE r.race_type != 'banei'
            ORDER BY r.race_date, r.race_id
        """).fetchall()

    log.info(f"  対象レース数: {len(races)}")

    processed = 0
    skipped   = 0

    for race in races:
        race_id   = race["race_id"]
        race_date = race["race_date"]

        # 取得済みスキップ
        with get_conn() as conn:
            exists = conn.execute(
                "SELECT 1 FROM training_features WHERE race_id = ? LIMIT 1", (race_id,)
            ).fetchone()
        if exists:
            skipped += 1
            continue

        rows = _compute_race_features(race_id, race_date, dict(race))
        if not rows:
            continue

        with get_conn() as conn:
            for row in rows:
                # 42列: id(NULL) + race_id〜as_of_date の41列
                conn.execute("""
                    INSERT OR REPLACE INTO training_features
                    VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    row["race_id"], row["race_date"], row["horse_id"],
                    row["draw_number"], row["finish_position"], row["relevance"],
                    row["agari3f_z"], row["agari3f_rank_pct"],
                    row["speed_index_mean"], row["speed_index_trend"], row["sf_ewma"],
                    row["rest_rbf_7"], row["rest_rbf_14"], row["rest_rbf_21"],
                    row["rest_rbf_35"], row["rest_rbf_56"], row["rest_rbf_84"],
                    row["rest_rbf_140"], row["rest_rbf_210"], row["layoff_150plus"],
                    row["jockey_course_winrate"], row["jockey_win_rate_3m"],
                    row.get("jockey_track_winrate"),
                    row.get("weight_z"), row.get("weight_rank_pct"),
                    row.get("style_score_c4"), row.get("style_vol_c4"), row.get("front_pct"),
                    row.get("class_upset_rate"),
                    row["surface_enc"], row["distance"],
                    row["distance_band_enc"], row["straight_m"],
                    row["horse_weight"], row["horse_weight_diff"],
                    row["win_odds"], row["popular_rank"], row["field_size"],
                    row["track"], row["race_type"], row["as_of_date"],
                ))
        processed += 1

    log.info(f"  特徴量生成完了: {processed}レース処理 / {skipped}スキップ")
    return processed


def _compute_race_features(race_id: str, race_date: str, race_info: dict) -> list[dict]:
    """1レース分の特徴量を計算して返す。過去走のみ使用。"""
    from db.database import get_conn
    from features.rest_interval import compute_rest_features
    from features.nar_features import (
        jockey_track_winrate,
        batch_relative_weight,
        frontrun_score,
        class_upset_rates,
        get_class_upset_rate,
    )

    with get_conn() as conn:
        # このレースの出走馬
        entries = conn.execute("""
            SELECT horse_id, draw_number, finish_position,
                   agari3f_seconds, race_time_seconds,
                   win_odds, popular_rank,
                   horse_weight, horse_weight_diff,
                   jockey_id
            FROM nar_results WHERE race_id = ?
        """, (race_id,)).fetchall()

        if not entries:
            return []

        entries = [dict(e) for e in entries]  # sqlite3.Row → dict に変換
        # 同レースの上がり3F（グループ内z-score計算用）
        agari_vals = [e["agari3f_seconds"] for e in entries if e["agari3f_seconds"]]
        agari_mean = np.mean(agari_vals) if agari_vals else None
        agari_std  = np.std(agari_vals)  if len(agari_vals) > 1 else None

    surface_map = {"芝": 1, "ダート": 0, "ばんえい": 2, "障害": 3}
    surface_enc = surface_map.get(race_info.get("surface", "ダート"), 0)
    dist        = race_info.get("distance") or 0
    dist_band   = 0 if dist <= 1400 else 1 if dist <= 2000 else 2
    straight_m  = race_info.get("straight_m") or 0
    field_size  = race_info.get("field_size") or len(entries)

    rows = []
    for entry in entries:
        horse_id = entry["horse_id"]
        if not horse_id:
            continue

        finish_pos = entry["finish_position"]
        relevance  = max(0, 4 - finish_pos) if finish_pos else 0

        # ── 過去走データ取得（当走は含まない）──────────
        with get_conn() as conn:
            past = conn.execute("""
                SELECT nr.race_date, nr.agari3f_seconds, nr.race_time_seconds,
                       nr.finish_position, nr.jockey_id,
                       rc.track, rc.surface, rc.distance, rc.field_size
                FROM nar_results nr
                JOIN nar_races rc ON nr.race_id = rc.race_id
                WHERE nr.horse_id = ? AND nr.race_date < ?
                ORDER BY nr.race_date DESC
                LIMIT 20
            """, (horse_id, race_date)).fetchall()

        past = [dict(p) for p in past]

        # ── 上がり3F z-score（当走） ──────────────────
        agari_cur = entry["agari3f_seconds"]
        agari_z   = None
        agari_pct = None
        if agari_cur and agari_mean and agari_std and agari_std > 0:
            agari_z   = -(agari_cur - agari_mean) / agari_std   # 速いほど正
            agari_pct = sum(1 for a in agari_vals if a < agari_cur) / len(agari_vals)
        elif agari_cur and agari_mean:
            agari_z   = 0.0
            agari_pct = 0.5

        # ── スピード指数（タイム偏差値）──────────────
        past_times  = [p["race_time_seconds"] for p in past if p["race_time_seconds"]]
        speed_mean  = float(np.mean(past_times))  if past_times else None
        speed_trend = 0.0
        sf_ewma     = speed_mean or 0.0
        if len(past_times) >= 3:
            x = np.arange(len(past_times[:5]))
            speed_trend = float(np.polyfit(x, past_times[:5], 1)[0])
            alpha = 0.5
            ewma = past_times[0]
            for t in past_times[1:3]:
                ewma = alpha * t + (1 - alpha) * ewma
            sf_ewma = ewma

        # ── 出走間隔RBF ────────────────────────────
        if past:
            last_date = past[0]["race_date"]
            days = (date.fromisoformat(race_date) - date.fromisoformat(last_date)).days
        else:
            days = 999

        rbf_feats = compute_rest_features(float(days))

        # ── 騎手×コース勝率 ──────────────────────────
        jockey_id = entry.get("jockey_id")
        jk_course_wr   = _jockey_course_winrate(jockey_id, race_info.get("track"), race_date)
        jk_3m_wr       = _jockey_recent_winrate(jockey_id, race_date, months=3)

        # ── [NEW 1] 騎手×競馬場勝率（地元ボーナス）────
        jk_track_wr = jockey_track_winrate(
            jockey_id or "", race_info.get("track", ""), race_date
        )

        # ── [NEW 3] 先行力スコア ─────────────────────
        front_feats = frontrun_score(horse_id, race_date)

        rows.append({
            "race_id":        race_id,
            "race_date":      race_date,
            "horse_id":       horse_id,
            "draw_number":    entry["draw_number"],
            "finish_position": finish_pos,
            "relevance":      relevance,
            "agari3f_z":      agari_z,
            "agari3f_rank_pct": agari_pct,
            "speed_index_mean": speed_mean,
            "speed_index_trend": speed_trend,
            "sf_ewma":        sf_ewma,
            "rest_rbf_7":     rbf_feats.get("rest_rbf_7", 0),
            "rest_rbf_14":    rbf_feats.get("rest_rbf_14", 0),
            "rest_rbf_21":    rbf_feats.get("rest_rbf_21", 0),
            "rest_rbf_35":    rbf_feats.get("rest_rbf_35", 0),
            "rest_rbf_56":    rbf_feats.get("rest_rbf_56", 0),
            "rest_rbf_84":    rbf_feats.get("rest_rbf_84", 0),
            "rest_rbf_140":   rbf_feats.get("rest_rbf_140", 0),
            "rest_rbf_210":   rbf_feats.get("rest_rbf_210", 0),
            "layoff_150plus": rbf_feats.get("layoff_150plus", 0),
            "jockey_course_winrate": jk_course_wr,
            "jockey_win_rate_3m":    jk_3m_wr,
            "jockey_track_winrate":  jk_track_wr,       # [NEW 1]
            # [NEW 2] 相対斤量 → レース単位で後から追加（後処理）
            "weight_z":        None,
            "weight_rank_pct": None,
            # [NEW 3] 先行力
            "style_score_c4":  front_feats.get("style_score_c4"),
            "style_vol_c4":    front_feats.get("style_vol_c4"),
            "front_pct":       front_feats.get("front_pct"),
            # [NEW 4] クラス別荒れ率 → レース単位で後から追加
            "class_upset_rate": None,
            "surface_enc":    surface_enc,
            "distance":       dist,
            "distance_band_enc": dist_band,
            "straight_m":     straight_m,
            "horse_weight":   entry["horse_weight"],
            "horse_weight_diff": entry["horse_weight_diff"],
            "win_odds":       entry["win_odds"],
            "popular_rank":   entry["popular_rank"],
            "field_size":     field_size,
            "track":          race_info.get("track", ""),
            "race_type":      race_info.get("race_type", "flat"),
            "as_of_date":     race_date,
        })

    # ── [NEW 2] 相対斤量スコアをレース単位で計算 ────
    weight_df = batch_relative_weight([race_id])
    if not weight_df.empty:
        wmap = {
            int(r["draw_number"]): (r["weight_z"], r["weight_rank_pct"])
            for _, r in weight_df.iterrows()
        }
        for row in rows:
            dn = row.get("draw_number")
            if dn and dn in wmap:
                row["weight_z"]        = wmap[dn][0]
                row["weight_rank_pct"] = wmap[dn][1]

    # ── [NEW 4] クラス別荒れ率を全行に設定 ──────────
    race_class = race_info.get("race_class", "")
    upset_rates = class_upset_rates(race_date)
    class_upset = get_class_upset_rate(race_class, upset_rates)
    for row in rows:
        row["class_upset_rate"] = class_upset

    return rows


def _jockey_course_winrate(jockey_id, track, as_of_date, m=50.0):
    if not jockey_id:
        return None
    from db.database import get_conn
    with get_conn() as conn:
        row = conn.execute("""
            SELECT COUNT(*) as cnt,
                   SUM(CASE WHEN nr.finish_position = 1 THEN 1 ELSE 0 END) as wins
            FROM nar_results nr
            JOIN nar_races rc ON nr.race_id = rc.race_id
            WHERE nr.jockey_id = ? AND rc.track = ? AND nr.race_date < ?
        """, (jockey_id, track, as_of_date)).fetchone()
        global_row = conn.execute("""
            SELECT COUNT(*) as cnt,
                   SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) as wins
            FROM nar_results WHERE jockey_id = ? AND race_date < ?
        """, (jockey_id, as_of_date)).fetchone()

    cnt     = row["cnt"] or 0
    wins    = row["wins"] or 0
    g_cnt   = global_row["cnt"] or 1
    g_wins  = global_row["wins"] or 0
    g_rate  = g_wins / g_cnt
    return (wins + m * g_rate) / (cnt + m)


def _jockey_recent_winrate(jockey_id, as_of_date, months=3, m=30.0):
    if not jockey_id:
        return None
    from db.database import get_conn
    from dateutil.relativedelta import relativedelta
    cutoff = (date.fromisoformat(as_of_date) - relativedelta(months=months)).isoformat()
    with get_conn() as conn:
        row = conn.execute("""
            SELECT COUNT(*) as cnt,
                   SUM(CASE WHEN finish_position = 1 THEN 1 ELSE 0 END) as wins
            FROM nar_results
            WHERE jockey_id = ? AND race_date >= ? AND race_date < ?
        """, (jockey_id, cutoff, as_of_date)).fetchone()
    cnt  = row["cnt"] or 0
    wins = row["wins"] or 0
    return (wins + m * 0.1) / (cnt + m)


# ──────────────────────────────────────────────────
# Step 3: LGBMRanker 学習
# ──────────────────────────────────────────────────
FEATURE_COLS = [
    # 能力系
    "agari3f_z", "agari3f_rank_pct",
    "speed_index_mean", "speed_index_trend", "sf_ewma",
    # 状態系
    "rest_rbf_7", "rest_rbf_14", "rest_rbf_21", "rest_rbf_35",
    "rest_rbf_56", "rest_rbf_84", "rest_rbf_140", "rest_rbf_210",
    "layoff_150plus",
    # 騎手
    "jockey_course_winrate", "jockey_win_rate_3m",
    "jockey_track_winrate",    # [NEW 1] 騎手×競馬場勝率（地元ボーナス）
    # 斤量
    "weight_z", "weight_rank_pct",  # [NEW 2] 相対軽さスコア
    # 先行力
    "style_score_c4", "style_vol_c4", "front_pct",  # [NEW 3] 先行力スコア
    # クラス
    "class_upset_rate",        # [NEW 4] 条件クラス別荒れ率
    # コース・馬場
    "surface_enc", "distance", "distance_band_enc", "straight_m",
    # 馬体
    "horse_weight", "horse_weight_diff",
    # オッズ・人気
    "win_odds", "popular_rank", "field_size",
]

MODEL_DIR = Path(__file__).parent.parent / "models" / "trained"


def step3_train_ranker() -> dict:
    log.info("=== Step 3: LGBMRanker 学習 ===")
    import lightgbm as lgb
    from db.database import get_conn

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with get_conn() as conn:
        df = pd.read_sql_query("""
            SELECT * FROM training_features
            WHERE finish_position IS NOT NULL
            AND relevance IS NOT NULL
            ORDER BY race_date, race_id, draw_number
        """, conn)

    log.info(f"  学習データ: {len(df)}行, {df['race_id'].nunique()}レース")

    df["race_date"] = pd.to_datetime(df["race_date"])
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Walk-Forward CV ───────────────────────────
    from models.walk_forward import monthly_expanding_walk_forward

    oof_scores  = np.zeros(len(df))
    fold_metrics = []
    best_model   = None

    for fold_i, (train_idx, test_idx) in enumerate(
        monthly_expanding_walk_forward(df, date_col="race_date",
                                       min_train_months=12, embargo_days=7)
    ):
        X_train = df.loc[train_idx, FEATURE_COLS].fillna(0)
        y_train = df.loc[train_idx, "relevance"].astype(int)
        g_train = df.loc[train_idx].groupby("race_id", sort=False).size().values

        X_test  = df.loc[test_idx, FEATURE_COLS].fillna(0)
        y_test  = df.loc[test_idx, "relevance"].astype(int)
        g_test  = df.loc[test_idx].groupby("race_id", sort=False).size().values

        ranker = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg", ndcg_eval_at=[3],
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, importance_type="gain", verbose=-1,
        )
        ranker.fit(
            X_train, y_train, group=g_train,
            eval_set=[(X_test, y_test)], eval_group=[g_test],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )

        scores = ranker.predict(X_test)
        oof_scores[test_idx] = scores

        # NDCG@3
        ndcg = _ndcg_at_k(y_test.values, scores, g_test, k=3)
        fold_metrics.append({"fold": fold_i, "ndcg3": ndcg,
                              "n_test": len(test_idx)})
        log.info(f"  fold {fold_i:02d} | test={df.loc[test_idx,'race_date'].iloc[0].strftime('%Y-%m')} "
                 f"| NDCG@3={ndcg:.4f} | n={len(test_idx)}")
        best_model = ranker

    # 全データで最終モデルを学習
    log.info("  全データで最終モデルを学習...")
    X_all = df[FEATURE_COLS].fillna(0)
    y_all = df["relevance"].astype(int)
    g_all = df.groupby("race_id", sort=False).size().values

    final_ranker = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", ndcg_eval_at=[3],
        n_estimators=best_model.best_iteration_ if hasattr(best_model, "best_iteration_") else 500,
        learning_rate=0.05, num_leaves=31,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, importance_type="gain", verbose=-1,
    )
    final_ranker.fit(X_all, y_all, group=g_all,
                     callbacks=[lgb.log_evaluation(period=-1)])

    model_path = MODEL_DIR / "ranker.txt"
    final_ranker.booster_.save_model(str(model_path))
    log.info(f"  モデル保存: {model_path}")

    metrics = {
        "mean_ndcg3":   float(np.mean([m["ndcg3"] for m in fold_metrics])),
        "fold_metrics": fold_metrics,
        "n_folds":      len(fold_metrics),
        "model_path":   str(model_path),
    }
    _record_metrics("step3", metrics)
    log.info(f"  平均NDCG@3: {metrics['mean_ndcg3']:.4f}")
    return metrics


def _ndcg_at_k(y_true, y_score, groups, k=3):
    """グループごとのNDCG@kを計算して平均を返す。"""
    ndcgs = []
    start = 0
    for size in groups:
        end = start + size
        yt = y_true[start:end]
        ys = y_score[start:end]
        order = np.argsort(-ys)[:k]
        dcg  = sum(yt[i] / np.log2(r + 2) for r, i in enumerate(order))
        ideal = sorted(yt, reverse=True)[:k]
        idcg = sum(v / np.log2(r + 2) for r, v in enumerate(ideal))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        start = end
    return float(np.mean(ndcgs))


# ──────────────────────────────────────────────────
# Step 4: 確率校正
# ──────────────────────────────────────────────────
def step4_calibrate() -> dict:
    log.info("=== Step 4: 確率校正 ===")
    import lightgbm as lgb
    from sklearn.metrics import brier_score_loss, log_loss
    from models.calibrator import IsotonicCalibrator
    from db.database import get_conn
    import pickle

    model_path = MODEL_DIR / "ranker.txt"
    booster = lgb.Booster(model_file=str(model_path))

    with get_conn() as conn:
        df = pd.read_sql_query("""
            SELECT * FROM training_features
            WHERE finish_position IS NOT NULL
            ORDER BY race_date, race_id, draw_number
        """, conn)

    df["race_date"] = pd.to_datetime(df["race_date"])
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df[FEATURE_COLS].fillna(0)
    scores = booster.predict(X)

    # top3 二値ラベル
    y_top3 = (df["finish_position"] <= 3).astype(int).values

    # Isotonic校正（後半50%のデータで学習）
    n = len(scores)
    cal_start = n // 2
    calibrator = IsotonicCalibrator()
    calibrator.fit(scores[cal_start:], y_top3[cal_start:])

    cal_probs = calibrator.predict(scores)
    cal_probs = np.clip(cal_probs, 1e-6, 1 - 1e-6)

    # 指標計算（後半50%のみ）
    ll    = log_loss(y_top3[cal_start:], cal_probs[cal_start:])
    brier = brier_score_loss(y_top3[cal_start:], cal_probs[cal_start:])

    cal_path = MODEL_DIR / "calibrator.pkl"
    with open(cal_path, "wb") as f:
        pickle.dump(calibrator, f)

    metrics = {"logloss": round(ll, 4), "brier": round(brier, 4),
               "cal_path": str(cal_path)}
    _record_metrics("step4", metrics)
    log.info(f"  校正完了: LogLoss={ll:.4f}, Brier={brier:.4f}")
    return metrics


# ──────────────────────────────────────────────────
# Step 5: daily_update を本番モデルに切り替え
# ──────────────────────────────────────────────────
def step5_switch_to_model() -> None:
    log.info("=== Step 5: 本番モデルに切り替え ===")
    # models/trained/ranker.txt の存在確認
    model_path = MODEL_DIR / "ranker.txt"
    cal_path   = MODEL_DIR / "calibrator.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"モデルが見つかりません: {model_path}")
    if not cal_path.exists():
        raise FileNotFoundError(f"校正器が見つかりません: {cal_path}")

    # predict_today.py のフラグファイルを作成
    flag_path = MODEL_DIR / "model_ready.flag"
    flag_path.write_text(f"ready:{datetime.now().isoformat()}\n")
    log.info(f"  本番モデル切り替え完了: {flag_path}")


# ──────────────────────────────────────────────────
# Step 6: 本番モデルで今日の予測を再生成
# ──────────────────────────────────────────────────
def step6_predict_and_push() -> None:
    log.info("=== Step 6: 本番予測・サイト更新 ===")
    from pipeline.predict_today import run_predict_today
    run_predict_today(use_ml_model=True)
    log.info("  index.html 更新・push 完了")


# ──────────────────────────────────────────────────
# メトリクス記録
# ──────────────────────────────────────────────────
def _record_metrics(step: str, metrics: dict) -> None:
    import json
    metrics_path = MODEL_DIR / "metrics.json"
    all_metrics = {}
    if metrics_path.exists():
        all_metrics = json.loads(metrics_path.read_text())
    all_metrics[step] = {"timestamp": datetime.now().isoformat(), **metrics}
    metrics_path.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2))


# ──────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────
def run_all(start_step: int = 2) -> None:
    log.info(f"====== 学習パイプライン開始 (Step {start_step}〜) ======")
    t0 = time.time()

    steps = {
        2: step2_build_features,
        3: step3_train_ranker,
        4: step4_calibrate,
        5: step5_switch_to_model,
        6: step6_predict_and_push,
    }

    for step_no, fn in steps.items():
        if step_no < start_step:
            continue
        try:
            log.info(f"--- Step {step_no} 開始 ---")
            fn()
            log.info(f"--- Step {step_no} 完了 ---")
        except Exception as e:
            log.error(f"Step {step_no} 失敗: {e}", exc_info=True)
            raise

    elapsed = (time.time() - t0) / 60
    log.info(f"====== 全ステップ完了 ({elapsed:.1f}分) ======")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=2, help="開始ステップ番号")
    args = parser.parse_args()
    run_all(start_step=args.step)
