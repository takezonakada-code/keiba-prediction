-- ==================================================
-- 競馬3連複予測システム SQLiteスキーマ
-- ==================================================
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ──────────────────────────────────────────────────
-- レース基本情報
-- ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS races (
    race_id         TEXT PRIMARY KEY,
    race_date       TEXT NOT NULL,
    track           TEXT NOT NULL,     -- 競馬場
    race_no         INTEGER NOT NULL,
    surface         TEXT NOT NULL,     -- 芝/ダート
    distance        INTEGER NOT NULL,
    distance_band   TEXT NOT NULL,     -- short/mile_middle/long
    course_dir      TEXT,              -- 右/左
    straight_length INTEGER,
    race_class      TEXT,
    field_size      INTEGER,
    track_condition TEXT,              -- 良/稍重/重/不良
    weather         TEXT,
    organizer       TEXT DEFAULT 'JRA', -- JRA/TCK/NAR等
    created_at      TEXT DEFAULT (datetime('now'))
);

-- ──────────────────────────────────────────────────
-- 馬マスタ
-- ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS horses (
    horse_id    TEXT PRIMARY KEY,
    horse_name  TEXT NOT NULL,
    birth_year  INTEGER,
    sex         TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

-- ──────────────────────────────────────────────────
-- 出走エントリ（レース × 馬）
-- ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id         TEXT NOT NULL REFERENCES races(race_id),
    horse_id        TEXT NOT NULL REFERENCES horses(horse_id),
    horse_name      TEXT,
    draw_number     INTEGER,           -- 馬番
    jockey_id       TEXT,
    trainer_id      TEXT,
    horse_age       INTEGER,
    horse_weight    INTEGER,
    horse_weight_diff INTEGER,
    win_odds_live   REAL,
    live_popular_rank INTEGER,
    finish_position INTEGER,           -- レース後確定（予測では使わない）
    created_at      TEXT DEFAULT (datetime('now'))
);

-- ──────────────────────────────────────────────────
-- 過去走記録（特徴量計算の元データ）
-- ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS past_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id             TEXT NOT NULL,
    race_date           TEXT NOT NULL,
    horse_id            TEXT NOT NULL,
    finish_position     INTEGER,
    race_time_seconds   REAL,
    agari3f_seconds     REAL,          -- 過去走の上がり3F（特徴量計算に使う）
    corner4_pos         INTEGER,       -- 4角通過順
    field_size          INTEGER,
    distance            INTEGER,
    surface             TEXT,
    track               TEXT,
    course_dir          TEXT,
    straight_length     INTEGER,
    track_condition     TEXT,
    race_class          TEXT,
    created_at          TEXT DEFAULT (datetime('now'))
);

-- ──────────────────────────────────────────────────
-- 予測結果
-- ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id         TEXT NOT NULL REFERENCES races(race_id),
    horse_id        TEXT NOT NULL,
    horse_name      TEXT,
    pred_score      REAL,
    p_top3          REAL,              -- Plackett-Luce 3着以内確率
    shap_top1_feature TEXT,
    shap_top1_value   REAL,
    shap_top5_json    TEXT,            -- JSON
    model_version   TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- ──────────────────────────────────────────────────
-- 推奨買い目
-- ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS recommended_bets (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id         TEXT NOT NULL REFERENCES races(race_id),
    combo           TEXT NOT NULL,     -- "1-3-7" 形式（馬番）
    horse_ids       TEXT,              -- JSON ["id1","id2","id3"]
    p_hit           REAL,
    display_odds    REAL,
    ev              REAL,
    kelly_frac      REAL,
    stake           REAL,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- ──────────────────────────────────────────────────
-- ベット実績 + 結果
-- ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS bet_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    race_date       TEXT NOT NULL,
    race_id         TEXT NOT NULL,
    combo           TEXT NOT NULL,     -- "1-3-7" 形式
    p_hit           REAL,
    ev              REAL,
    kelly_frac      REAL,
    stake           REAL DEFAULT 100,
    payout          REAL DEFAULT 0,
    is_hit          INTEGER DEFAULT 0, -- 0/1
    surface         TEXT,
    distance        INTEGER,
    distance_band   TEXT,
    track           TEXT,
    field_size      INTEGER,
    odds_band       TEXT,              -- low(<10)/mid(10-50)/high(50+)
    shap_top1_feature TEXT,
    shap_top1_value   REAL,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- ──────────────────────────────────────────────────
-- スクレイピング監視ログ
-- ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS scraping_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date        TEXT NOT NULL,
    target_url      TEXT,
    success         INTEGER NOT NULL,  -- 0/1
    error_message   TEXT,
    response_time_ms INTEGER,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- ──────────────────────────────────────────────────
-- モデル評価ログ（walk-forward foldごと）
-- ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_eval_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fold_no     INTEGER,
    test_month  TEXT,                  -- "2025-03"
    logloss     REAL,
    brier       REAL,
    auc         REAL,
    ndcg3       REAL,
    roi_pct     REAL,
    sharpe      REAL,
    n_test      INTEGER,
    model_version TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

-- ──────────────────────────────────────────────────
-- ROI集計ビュー
-- ──────────────────────────────────────────────────
CREATE VIEW IF NOT EXISTS roi_by_slice AS
SELECT
    surface,
    distance_band,
    COUNT(*)                                              AS tickets,
    SUM(stake)                                            AS total_stake,
    SUM(payout)                                           AS total_payout,
    ROUND(100.0 * SUM(payout) / NULLIF(SUM(stake), 0), 1) AS roi_pct,
    SUM(is_hit)                                           AS hits,
    ROUND(100.0 * SUM(is_hit) / COUNT(*), 1)              AS hit_rate_pct
FROM bet_results
GROUP BY surface, distance_band;

CREATE VIEW IF NOT EXISTS roi_by_track AS
SELECT
    track,
    surface,
    COUNT(*)                                              AS tickets,
    SUM(stake)                                            AS total_stake,
    SUM(payout)                                           AS total_payout,
    ROUND(100.0 * SUM(payout) / NULLIF(SUM(stake), 0), 1) AS roi_pct,
    SUM(is_hit)                                           AS hits
FROM bet_results
GROUP BY track, surface;

CREATE VIEW IF NOT EXISTS roi_monthly AS
SELECT
    substr(race_date, 1, 7)                               AS month,
    COUNT(*)                                              AS tickets,
    SUM(stake)                                            AS total_stake,
    SUM(payout)                                           AS total_payout,
    ROUND(100.0 * SUM(payout) / NULLIF(SUM(stake), 0), 1) AS roi_pct
FROM bet_results
GROUP BY month
ORDER BY month;

-- ──────────────────────────────────────────────────
-- インデックス
-- ──────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_entries_race      ON entries(race_id);
CREATE INDEX IF NOT EXISTS idx_entries_horse     ON entries(horse_id);
CREATE INDEX IF NOT EXISTS idx_past_horse_date   ON past_results(horse_id, race_date);
CREATE INDEX IF NOT EXISTS idx_bet_results_date  ON bet_results(race_date);
CREATE INDEX IF NOT EXISTS idx_predictions_race  ON predictions(race_id);
