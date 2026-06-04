"""
NAR公式CSVダウンロード・解析モジュール。
地方競馬情報サイトの公式CSVを取得・DBに取り込む。

- 月次レース情報: 1998年1月以降
- 月次オッズ情報: 2026年3月以降
- 当日ファイル: 約2分ごとに更新
- 月次ファイル: 毎日午前2時ごろ更新
"""
from __future__ import annotations

import io
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from config import SCRAPER_BACKOFF, SCRAPER_RETRY_TOTAL, SCRAPER_STATUS_FORCELIST
from data.scraper import make_session


# NAR公式CSVのベースURL（地方競馬情報サイト）
NAR_BASE_URL = "https://www.keiba.go.jp/KeibaWeb/DataFile"


# CSVファイルレイアウト（NAR公式マニュアルに基づく主要カラム）
RACE_INFO_COLS = [
    "race_date", "track_code", "race_no", "race_name", "distance",
    "surface", "track_condition", "field_size", "weather",
    "course_dir", "race_class",
]

HORSE_LIST_COLS = [
    "race_date", "track_code", "race_no", "horse_no", "horse_name",
    "horse_id", "jockey_code", "trainer_code", "horse_weight",
    "horse_weight_diff", "win_odds", "popular_rank",
    "corner1_pos", "corner2_pos", "corner3_pos", "corner4_pos",
    "finish_position", "finish_time_sec", "agari3f_sec",
    "prize_money",
]

ODDS_COLS = [
    "race_date", "track_code", "race_no", "bet_type",
    "num1", "num2", "num3", "odds", "popular_rank",
]


class NARCsvClient:
    """NAR公式CSVを取得・解析するクライアント。"""

    def __init__(self):
        self.session = make_session()
        self._success = 0
        self._failure = 0

    @property
    def success_rate(self) -> float:
        total = self._success + self._failure
        return self._success / total if total > 0 else 1.0

    def _download_zip(self, url: str, sleep: float = 2.0) -> Optional[zipfile.ZipFile]:
        """ZIPファイルをダウンロードして ZipFile オブジェクトを返す。"""
        time.sleep(sleep)
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            self._success += 1
            return zipfile.ZipFile(io.BytesIO(resp.content))
        except Exception as e:
            self._failure += 1
            print(f"[NAR] ダウンロード失敗: {url} → {e}")
            return None

    def fetch_today_race_info(self, target_date: Optional[date] = None) -> pd.DataFrame:
        """
        当日レース情報CSVを取得して DataFrame で返す。
        約2分ごとに更新されるため、当日分のリアルタイム取得に使う。

        Parameters
        ----------
        target_date : 取得日（None = 今日）
        """
        if target_date is None:
            target_date = date.today()

        date_str = target_date.strftime("%Y%m%d")
        url = f"{NAR_BASE_URL}/TodayRaceInfo/{date_str}/race_info.zip"
        zf = self._download_zip(url)
        if zf is None:
            return pd.DataFrame()

        return self._parse_race_info_zip(zf)

    def fetch_today_odds(self, target_date: Optional[date] = None) -> pd.DataFrame:
        """
        当日オッズCSVを取得（約2分更新）。
        三連複（bet_type=4）のみフィルタして返す。
        """
        if target_date is None:
            target_date = date.today()

        date_str = target_date.strftime("%Y%m%d")
        url = f"{NAR_BASE_URL}/TodayOdds/{date_str}/odds.zip"
        zf = self._download_zip(url)
        if zf is None:
            return pd.DataFrame()

        df = self._parse_odds_zip(zf)
        # 三連複のbet_typeは4
        return df[df["bet_type"] == 4] if len(df) > 0 else df

    def fetch_monthly_race_results(
        self,
        year: int,
        month: int,
    ) -> pd.DataFrame:
        """
        月次レース結果CSVを取得（1998年1月以降）。
        着順・上がり・コーナー通過順などの過去走データを含む。
        """
        ym = f"{year}{month:02d}"
        url = f"{NAR_BASE_URL}/MonthlyRaceInfo/{ym}/race_result.zip"
        zf = self._download_zip(url)
        if zf is None:
            return pd.DataFrame()

        return self._parse_horse_list_zip(zf)

    def fetch_monthly_odds(
        self,
        year: int,
        month: int,
    ) -> pd.DataFrame:
        """
        月次オッズCSVを取得（2026年3月以降が公式公開済み）。
        """
        ym = f"{year}{month:02d}"
        url = f"{NAR_BASE_URL}/MonthlyOdds/{ym}/odds.zip"
        zf = self._download_zip(url)
        if zf is None:
            return pd.DataFrame()

        return self._parse_odds_zip(zf)

    def fetch_odds_snapshots(
        self,
        target_date: date,
        interval_minutes: int = 2,
    ) -> pd.DataFrame:
        """
        当日オッズを複数回取得してスナップショット履歴を作る。
        odds_drift.py の drift特徴量計算に使う。

        Parameters
        ----------
        target_date      : 取得日
        interval_minutes : 取得間隔（分）。約2分が更新頻度。

        Returns
        -------
        DataFrame: race_id, horse_id, snapshot_time, win_odds
        """
        import datetime
        snapshots = []
        df = self.fetch_today_odds(target_date)
        if len(df) > 0:
            df["snapshot_time"] = datetime.datetime.now()
            snapshots.append(df)
        return pd.concat(snapshots, ignore_index=True) if snapshots else pd.DataFrame()

    # ──────────────────────────────────────────────
    # パーサー
    # ──────────────────────────────────────────────
    def _parse_race_info_zip(self, zf: zipfile.ZipFile) -> pd.DataFrame:
        """レース情報ZIPをDataFrameに変換。"""
        dfs = []
        for name in zf.namelist():
            if not name.endswith(".csv"):
                continue
            try:
                df = pd.read_csv(
                    zf.open(name),
                    encoding="shift-jis",
                    header=0,
                    dtype=str,
                )
                dfs.append(df)
            except Exception as e:
                print(f"[NAR] CSVパース失敗: {name} → {e}")
        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)
        return self._normalize_race_info(result)

    def _parse_horse_list_zip(self, zf: zipfile.ZipFile) -> pd.DataFrame:
        """馬ごと結果ZIPをDataFrameに変換。"""
        dfs = []
        for name in zf.namelist():
            if not name.endswith(".csv"):
                continue
            try:
                df = pd.read_csv(zf.open(name), encoding="shift-jis", dtype=str)
                dfs.append(df)
            except Exception:
                pass
        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)
        return self._normalize_horse_list(result)

    def _parse_odds_zip(self, zf: zipfile.ZipFile) -> pd.DataFrame:
        """オッズZIPをDataFrameに変換。"""
        dfs = []
        for name in zf.namelist():
            if not name.endswith(".csv"):
                continue
            try:
                df = pd.read_csv(zf.open(name), encoding="shift-jis", dtype=str)
                dfs.append(df)
            except Exception:
                pass
        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def _normalize_race_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """カラム名を内部スキーマに合わせる。"""
        col_map = {
            "開催日":   "race_date",
            "競馬場コード": "track_code",
            "レース番号":  "race_no",
            "距離":     "distance",
            "馬場状態":  "track_condition",
            "頭数":     "field_size",
            "天候":     "weather",
        }
        return df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    def _normalize_horse_list(self, df: pd.DataFrame) -> pd.DataFrame:
        col_map = {
            "開催日":   "race_date",
            "競馬場コード": "track_code",
            "馬番":    "horse_no",
            "馬名":    "horse_name",
            "馬ID":   "horse_id",
            "着順":    "finish_position",
            "上がり3F": "agari3f_seconds",
            "走破タイム": "race_time_seconds",
            "4角順位":  "corner4_pos",
        }
        return df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})


def ingest_nar_monthly(
    year: int,
    month: int,
    db_path: Path | None = None,
) -> None:
    """
    月次NAR CSVをDBに取り込む。

    Parameters
    ----------
    year, month : 対象年月
    db_path     : DBパス（None の場合は config.DB_PATH）
    """
    from db.database import get_conn
    from config import DB_PATH

    if db_path is None:
        db_path = DB_PATH

    client = NARCsvClient()
    results_df = client.fetch_monthly_race_results(year, month)

    if results_df.empty:
        print(f"[NAR] {year}/{month:02d} のデータなし")
        return

    with get_conn(db_path) as conn:
        for _, row in results_df.iterrows():
            conn.execute(
                """
                INSERT OR IGNORE INTO past_results
                  (race_id, race_date, horse_id, finish_position,
                   agari3f_seconds, corner4_pos, field_size, distance)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    row.get("race_id", ""),
                    row.get("race_date", ""),
                    row.get("horse_id", ""),
                    row.get("finish_position"),
                    row.get("agari3f_seconds"),
                    row.get("corner4_pos"),
                    row.get("field_size"),
                    row.get("distance"),
                ),
            )

    print(f"[NAR] {year}/{month:02d}: {len(results_df)}件インポート完了")
