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
-- NAR専用テーブル
-- ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS nar_races (
    race_id         TEXT PRIMARY KEY,
    race_date       TEXT NOT NULL,
    track           TEXT NOT NULL,
    track_code      TEXT NOT NULL,
    race_no         INTEGER NOT NULL,
    race_name       TEXT,
    surface         TEXT,              -- 芝/ダート/ばんえい/障害
    distance        INTEGER,
    course_dir      TEXT,              -- 右/左/直
    track_condition TEXT,              -- 良/稍重/重/不良
    weather         TEXT,
    field_size      INTEGER,
    race_class      TEXT,
    post_time       TEXT,              -- "14:15"
    race_type       TEXT DEFAULT 'flat', -- flat/banei
    payback_rate    REAL,              -- 場別控除後払戻率
    organizer       TEXT DEFAULT 'NAR',
    as_of_date      TEXT DEFAULT (date('now')), -- データリーク防止
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS nar_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id             TEXT NOT NULL,
    race_date           TEXT NOT NULL,
    horse_id            TEXT,
    horse_name          TEXT,
    draw_number         INTEGER,
    frame_number        INTEGER,
    jockey_id           TEXT,
    jockey_name         TEXT,
    trainer_id          TEXT,
    trainer_name        TEXT,
    sex                 TEXT,
    age                 INTEGER,
    weight_carried      REAL,          -- 斤量
    horse_weight        INTEGER,
    horse_weight_diff   INTEGER,
    finish_position     INTEGER,       -- NULL=取消/除外
    race_time_seconds   REAL,
    agari3f_seconds     REAL,
    corner1_pos         INTEGER,
    corner2_pos         INTEGER,
    corner3_pos         INTEGER,
    corner4_pos         INTEGER,
    win_odds            REAL,
    popular_rank        INTEGER,
    race_type           TEXT DEFAULT 'flat',
    as_of_date          TEXT DEFAULT (date('now')),
    created_at          TEXT DEFAULT (datetime('now')),
    UNIQUE(race_id, draw_number)
);

CREATE TABLE IF NOT EXISTS nar_entries (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id             TEXT NOT NULL,
    race_date           TEXT NOT NULL,
    horse_id            TEXT,
    horse_name          TEXT,
    draw_number         INTEGER,
    frame_number        INTEGER,
    jockey_id           TEXT,
    jockey_name         TEXT,
    trainer_id          TEXT,
    horse_weight        INTEGER,
    horse_weight_diff   INTEGER,
    win_odds            REAL,
    popular_rank        INTEGER,
    race_type           TEXT DEFAULT 'flat',
    as_of_date          TEXT DEFAULT (date('now')),
    created_at          TEXT DEFAULT (datetime('now')),
    UNIQUE(race_id, draw_number)
);

CREATE TABLE IF NOT EXISTS nar_payouts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id         TEXT NOT NULL,
    race_date       TEXT NOT NULL,
    bet_type        TEXT NOT NULL,     -- 'trio'
    combo           TEXT NOT NULL,     -- "1-3-7"
    payout          INTEGER,           -- 払戻金額（円・100円単位）
    popular_info    TEXT,
    as_of_date      TEXT DEFAULT (date('now')),
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(race_id, bet_type, combo)
);

-- ばんえい専用（帯広のみ）
CREATE TABLE IF NOT EXISTS banei_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id         TEXT NOT NULL,
    race_date       TEXT NOT NULL,
    horse_id        TEXT,
    horse_name      TEXT,
    draw_number     INTEGER,
    finish_position INTEGER,
    race_time_seconds REAL,
    win_odds        REAL,
    popular_rank    INTEGER,
    as_of_date      TEXT DEFAULT (date('now')),
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(race_id, draw_number)
);

-- ──────────────────────────────────────────────────
-- スクレイピング進捗管理（途中再開用）
-- ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS scraping_progress (
    race_id     TEXT PRIMARY KEY,
    status      TEXT NOT NULL,         -- done/error/in_progress
    updated_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS historical_progress (
    day_key       TEXT PRIMARY KEY,    -- "YYYY-MM-DD|ALL" or "YYYY-MM-DD|43_46"
    race_date     TEXT NOT NULL,
    status        TEXT NOT NULL,       -- done/error
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    error_message TEXT,
    updated_at    TEXT DEFAULT (datetime('now'))
);

-- ──────────────────────────────────────────────────
-- NARデータ用ビュー
-- ──────────────────────────────────────────────────
CREATE VIEW IF NOT EXISTS nar_trio_ev AS
SELECT
    p.race_id,
    p.race_date,
    r.track,
    r.surface,
    r.distance,
    r.track_condition,
    r.race_class,
    p.combo,
    p.payout,
    r.payback_rate,
    ROUND(1.0 / (p.payout / 100.0), 4) AS implied_prob
FROM nar_payouts p
JOIN nar_races r ON p.race_id = r.race_id
WHERE p.bet_type = 'trio'
  AND p.payout > 0;

-- ──────────────────────────────────────────────────
-- インデックス
-- ──────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_entries_race      ON entries(race_id);
CREATE INDEX IF NOT EXISTS idx_entries_horse     ON entries(horse_id);
CREATE INDEX IF NOT EXISTS idx_past_horse_date   ON past_results(horse_id, race_date);
CREATE INDEX IF NOT EXISTS idx_bet_results_date  ON bet_results(race_date);
CREATE INDEX IF NOT EXISTS idx_predictions_race  ON predictions(race_id);
CREATE INDEX IF NOT EXISTS idx_nar_races_date    ON nar_races(race_date);
CREATE INDEX IF NOT EXISTS idx_nar_results_horse ON nar_results(horse_id, race_date);
CREATE INDEX IF NOT EXISTS idx_nar_payouts_race  ON nar_payouts(race_id);
CREATE INDEX IF NOT EXISTS idx_hist_progress     ON historical_progress(race_date);
