"""
SQLite接続管理とDB初期化。
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from config import DB_PATH


def init_db(db_path: Path = DB_PATH) -> None:
    """スキーマを適用してDBを初期化する。"""
    schema_path = Path(__file__).parent / "schema.sql"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.executescript(schema_path.read_text(encoding="utf-8"))
        conn.commit()
    print(f"DB初期化完了: {db_path}")


@contextmanager
def get_conn(db_path: Path = DB_PATH):
    """SQLite接続のコンテキストマネージャ。"""
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def insert_bet_result(
    race_date: str,
    race_id: str,
    combo: str,
    p_hit: float,
    ev: float,
    kelly_frac: float,
    stake: float,
    payout: float,
    is_hit: int,
    surface: str = "",
    distance: int = 0,
    distance_band: str = "",
    track: str = "",
    field_size: int = 0,
    odds_band: str = "",
    shap_top1_feature: str = "",
    shap_top1_value: float = 0.0,
    db_path: Path = DB_PATH,
) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO bet_results
              (race_date, race_id, combo, p_hit, ev, kelly_frac,
               stake, payout, is_hit,
               surface, distance, distance_band, track, field_size,
               odds_band, shap_top1_feature, shap_top1_value)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (race_date, race_id, combo, p_hit, ev, kelly_frac,
             stake, payout, is_hit,
             surface, distance, distance_band, track, field_size,
             odds_band, shap_top1_feature, shap_top1_value),
        )


def fetch_bet_results(db_path: Path = DB_PATH) -> list[dict]:
    """bet_results 全件取得。"""
    with get_conn(db_path) as conn:
        rows = conn.execute("SELECT * FROM bet_results ORDER BY race_date").fetchall()
    return [dict(r) for r in rows]


def fetch_roi_by_slice(db_path: Path = DB_PATH) -> list[dict]:
    """roi_by_slice ビュー取得。"""
    with get_conn(db_path) as conn:
        rows = conn.execute("SELECT * FROM roi_by_slice").fetchall()
    return [dict(r) for r in rows]
