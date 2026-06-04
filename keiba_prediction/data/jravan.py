"""
JRA-VAN Data Lab. JV-Link コネクタ。
Windows + pywin32 が必要。
月額2,090円の契約キー（17文字）が必要。

取得優先順位:
  O5  : 三連複確定オッズ（2002年6月以降）
  O6  : 三連単確定オッズ（2004年8月以降）
  0B35: 速報三連複オッズ（随時）
  0B11: 速報単勝・複勝オッズ
  RACE: レース詳細
  SE  : 馬毎レース情報
  HR  : 払戻
  SLOP: 坂路調教（美浦・栗東）
  WOOD: ウッドチップ調教（美浦2021/7〜、栗東2021/12〜）
  WH  : 馬体重
"""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any

import pandas as pd


JRAVAN_KEY = os.environ.get("JRAVAN_KEY", "")   # 17文字の利用キー

# JV-Data レコード種別（主要なもの）
JV_RECORD_TYPES = {
    "RACE": "2101",   # レース詳細
    "SE":   "2B11",   # 馬毎レース情報（着順・着差・上がり・コーナー通過）
    "HR":   "2H11",   # 払戻
    "O5":   "O5",     # 三連複確定オッズ
    "O6":   "O6",     # 三連単確定オッズ
    "0B35": "0B35",   # 速報三連複
    "0B11": "0B11",   # 速報単勝複勝
    "WH":   "WH",     # 馬体重
    "SLOP": "SLOP",   # 坂路調教
    "WOOD": "WOOD",   # ウッドチップ調教
    "DM":   "DM",     # 公式マイニング予想
    "TM":   "TM",     # 公式マイニング予想（別形式）
}


class JRAVANConnector:
    """
    JV-Link COM オブジェクトの Python ラッパー。
    Windows 環境 + pywin32 が必要。
    非Windows環境ではスタブとして動作する。
    """

    def __init__(self, key: str = JRAVAN_KEY):
        self.key = key
        self._jv = None
        self._available = False
        self._try_init()

    def _try_init(self) -> None:
        try:
            import win32com.client
            self._jv = win32com.client.Dispatch("JVDTLab.JVLink")
            ret = self._jv.JVInit(self.key)
            if ret == 0:
                self._available = True
                print(f"[JRA-VAN] JVLink初期化成功")
            else:
                print(f"[JRA-VAN] JVInit失敗: code={ret}。利用キーを確認してください。")
        except ImportError:
            print("[JRA-VAN] pywin32が見つかりません。Windows環境でのみ利用可能です。")
        except Exception as e:
            print(f"[JRA-VAN] 初期化エラー: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def fetch_records(
        self,
        record_type: str,
        from_date: str,
        to_date: str,
        option: int = 1,
    ) -> list[bytes]:
        """
        JVOpenとJVReadで生レコードを取得してリストで返す。

        Parameters
        ----------
        record_type : JV_RECORD_TYPES のキー（例: "SE", "O5"）
        from_date   : "YYYYMMDD"
        to_date     : "YYYYMMDD"
        option      : 1=通常, 2=差分

        Returns
        -------
        list[bytes] : 生レコードのリスト
        """
        if not self._available:
            print("[JRA-VAN] 利用不可。空リストを返します。")
            return []

        jv_type = JV_RECORD_TYPES.get(record_type, record_type)
        ret, count, _ = self._jv.JVOpen(jv_type, from_date + "000000", option, 0, "", "")
        if ret < 0:
            print(f"[JRA-VAN] JVOpen失敗: code={ret}")
            return []

        records = []
        while True:
            ret, buf, filename = self._jv.JVRead("", 0, "")
            if ret == 0:
                break
            if ret < 0:
                print(f"[JRA-VAN] JVRead失敗: code={ret}")
                break
            records.append(buf)

        self._jv.JVClose()
        return records

    def fetch_realtime_odds(self, race_id: str) -> dict[str, float]:
        """
        速報三連複オッズ（0B35）を取得して {combo_str: odds} で返す。

        Parameters
        ----------
        race_id : "YYYYRRDDHH" 形式のレースID

        Returns
        -------
        dict : {"1-2-3": 12.5, ...}
        """
        if not self._available:
            return {}

        ret, count, _ = self._jv.JVRTOpen("0B35", "")
        if ret < 0:
            return {}

        odds_map: dict[str, float] = {}
        while True:
            ret, buf, _ = self._jv.JVRead("", 0, "")
            if ret <= 0:
                break
            parsed = self._parse_trio_odds_record(buf)
            if parsed:
                odds_map.update(parsed)

        self._jv.JVClose()
        return odds_map

    def _parse_trio_odds_record(self, buf: bytes) -> dict[str, float]:
        """
        0B35レコードをパースして {combo_str: odds} に変換。
        実際のレコード仕様はJRA-VAN JVData仕様書を参照。
        """
        # 実装はJVData仕様書の0B35レコードレイアウトに従う
        # ここでは骨格のみ記述
        return {}

    def fetch_training_data(
        self,
        from_date: str,
        to_date: str,
        surface: str = "SLOP",
    ) -> pd.DataFrame:
        """
        坂路/ウッドチップ調教データをDataFrameで返す。

        Parameters
        ----------
        surface : "SLOP"=坂路, "WOOD"=ウッドチップ
        """
        records = self.fetch_records(surface, from_date, to_date)
        # パース処理（JVData仕様書のSLOP/WOODレコードレイアウトに従う）
        # 実装時に仕様書を参照して具体的なフィールド抽出を行う
        return pd.DataFrame()   # スタブ

    def fetch_mining_prediction(
        self,
        race_date: str,
    ) -> pd.DataFrame:
        """
        公式マイニング予想（DM/TM）を取得。
        stacking の外生特徴量として使う。
        """
        records = self.fetch_records("DM", race_date, race_date)
        return pd.DataFrame()   # スタブ


def fetch_jravan_trio_odds(race_id: str) -> dict[str, float]:
    """スタンドアロン関数: 速報三連複オッズを取得。"""
    connector = JRAVANConnector()
    return connector.fetch_realtime_odds(race_id)
