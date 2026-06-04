-- ==================================================
-- 拡張スキーマ: JRA・血統・騎手・マスタテーブル
-- ==================================================
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ──────────────────────────────────────────────────
-- 騎手マスタ
-- ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS jockeys (
    jockey_id       TEXT PRIMARY KEY,
    jockey_name     TEXT NOT NULL,
    jockey_kana     TEXT,
    affiliation     TEXT,    -- 所属（JRA/NAR競馬場名）
    license_year    INTEGER,
    birth_year      INTEGER,
    as_of_date      TEXT DEFAULT (date('now')),
    scraped_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS jockey_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    jockey_id       TEXT NOT NULL,
    as_of_date      TEXT NOT NULL,   -- この統計の基準日（リーク防止）
    period          TEXT NOT NULL,   -- '3m' / '1y' / 'all'
    track           TEXT,            -- NULLなら全場
    surface         TEXT,            -- 芝/ダート/NULL
    distance_band   TEXT,            -- short/mile_middle/long/NULL
    rides           INTEGER DEFAULT 0,
    wins            INTEGER DEFAULT 0,
    top3            INTEGER DEFAULT 0,
    win_rate        REAL,
    top3_rate       REAL,
    UNIQUE(jockey_id, as_of_date, period, track, surface, distance_band)
);

-- ──────────────────────────────────────────────────
-- 調教師マスタ
-- ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trainers (
    trainer_id      TEXT PRIMARY KEY,
    trainer_name    TEXT NOT NULL,
    trainer_kana    TEXT,
    stable_location TEXT,   -- 美浦/栗東/地方
    as_of_date      TEXT DEFAULT (date('now')),
    scraped_at      TEXT DEFAULT (datetime('now'))
);

-- ──────────────────────────────────────────────────
-- 血統情報
-- ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS bloodlines (
    horse_id        TEXT PRIMARY KEY REFERENCES horses(horse_id),
    sire_id         TEXT,    -- 父
    sire_name       TEXT,
    dam_id          TEXT,    -- 母
    dam_name        TEXT,
    sire_sire_id    TEXT,    -- 父の父
    sire_sire_name  TEXT,
    dam_sire_id     TEXT,    -- 母の父
    dam_sire_name   TEXT,
    bloodline_type  TEXT,    -- サンデー/ノーザン/ミスプロ等
    as_of_date      TEXT DEFAULT (date('now')),
    scraped_at      TEXT DEFAULT (datetime('now'))
);

-- ──────────────────────────────────────────────────
-- NARオッズ時系列（締切前スナップショット）
-- ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS nar_odds_snapshot (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id         TEXT NOT NULL,
    race_date       TEXT NOT NULL,
    draw_number     INTEGER NOT NULL,
    snapshot_type   TEXT NOT NULL,   -- '30m' / '10m' / '2m' / 'final'
    win_odds        REAL,
    place_odds_min  REAL,
    place_odds_max  REAL,
    popular_rank    INTEGER,
    scraped_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(race_id, draw_number, snapshot_type)
);

-- ──────────────────────────────────────────────────
-- JRA レース情報
-- ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS jra_races (
    race_id         TEXT PRIMARY KEY,
    race_date       TEXT NOT NULL,
    track           TEXT NOT NULL,
    track_code      TEXT,
    race_no         INTEGER NOT NULL,
    race_name       TEXT,
    surface         TEXT,            -- 芝/ダート/障害
    distance        INTEGER,
    distance_band   TEXT,
    course_dir      TEXT,            -- 右/左/直
    straight_length INTEGER,
    track_condition TEXT,
    weather         TEXT,
    field_size      INTEGER,
    race_class      TEXT,            -- G1/G2/G3/OP/3勝/2勝/1勝/未勝利/新馬
    prize_1st       INTEGER,         -- 1着賞金（万円）
    post_time       TEXT,
    organizer       TEXT DEFAULT 'JRA',
    as_of_date      TEXT DEFAULT (date('now')),
    scraped_at      TEXT DEFAULT (datetime('now'))
);

-- ──────────────────────────────────────────────────
-- JRA 着順・馬ごと成績
-- ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS jra_results (
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
    owner_name          TEXT,
    sex                 TEXT,
    age                 INTEGER,
    weight_carried      REAL,
    horse_weight        INTEGER,
    horse_weight_diff   INTEGER,
    finish_position     INTEGER,
    finish_position_raw TEXT,        -- "取消"等も保存
    race_time_seconds   REAL,
    margin_text         TEXT,        -- "クビ" "ハナ" "1/2"等
    corner1_pos         INTEGER,
    corner2_pos         INTEGER,
    corner3_pos         INTEGER,
    corner4_pos         INTEGER,
    agari3f_seconds     REAL,
    win_odds            REAL,
    popular_rank        INTEGER,
    days_since_last     INTEGER,     -- 前走からの日数
    as_of_date          TEXT DEFAULT (date('now')),
    scraped_at          TEXT DEFAULT (datetime('now')),
    UNIQUE(race_id, draw_number)
);

-- ──────────────────────────────────────────────────
-- JRA 払戻
-- ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS jra_payouts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id         TEXT NOT NULL,
    race_date       TEXT NOT NULL,
    bet_type        TEXT NOT NULL,   -- win/place/quinella/exacta/trio/trifecta等
    combo           TEXT NOT NULL,
    payout          INTEGER,
    popular_info    TEXT,
    as_of_date      TEXT DEFAULT (date('now')),
    scraped_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(race_id, bet_type, combo)
);

-- ──────────────────────────────────────────────────
-- nar_results にコーナー通過順カラムを追加（既存テーブル拡張）
-- ──────────────────────────────────────────────────
-- ※ SQLiteはALTER TABLE ADD COLUMN のみサポート
-- 既にある場合はエラーになるが無視する
-- ──────────────────────────────────────────────────

-- ──────────────────────────────────────────────────
-- 取得進捗管理（拡張版）
-- ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS scrape_progress (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL,   -- 'NAR' / 'JRA'
    race_date       TEXT NOT NULL,
    track           TEXT,
    status          TEXT NOT NULL,   -- done/error/skip
    races_fetched   INTEGER DEFAULT 0,
    results_fetched INTEGER DEFAULT 0,
    payouts_fetched INTEGER DEFAULT 0,
    error_message   TEXT,
    duration_sec    REAL,
    scraped_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(source, race_date, track)
);

-- ──────────────────────────────────────────────────
-- インデックス
-- ──────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_jra_races_date      ON jra_races(race_date);
CREATE INDEX IF NOT EXISTS idx_jra_results_horse   ON jra_results(horse_id, race_date);
CREATE INDEX IF NOT EXISTS idx_jra_results_jockey  ON jra_results(jockey_id, race_date);
CREATE INDEX IF NOT EXISTS idx_jra_payouts_race    ON jra_payouts(race_id);
CREATE INDEX IF NOT EXISTS idx_nar_odds_race       ON nar_odds_snapshot(race_id);
CREATE INDEX IF NOT EXISTS idx_scrape_progress     ON scrape_progress(source, race_date);
CREATE INDEX IF NOT EXISTS idx_jockey_stats        ON jockey_stats(jockey_id, as_of_date);
