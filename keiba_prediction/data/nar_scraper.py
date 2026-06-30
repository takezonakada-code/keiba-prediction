"""
NAR（地方競馬）全場スクレイパー（netkeiba nar.netkeiba.com）

対象15場:
  南関東: 大井(33)・川崎(34)・船橋(35)・浦和(36)
  東海:   名古屋(43)・笠松(44)
  近畿:   園田(46)・姫路(47)
  九州:   高知(48)・佐賀(49)
  北陸:   金沢(42)
  東北:   盛岡(50)・水沢(51)
  北海道: 門別(30)
  ばんえい: 帯広(54) → race_type='banei' で別テーブル管理
"""
from __future__ import annotations

import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup

from data.scraper import make_session
from db.database import get_conn


# ──────────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────────
NAR_BASE = "https://nar.netkeiba.com"

# 場コード → 場名・主催・払戻率
# ※ netkeiba NARの実際の場コードを確認済み（kaisai_idから逆引き）
TRACK_MAP: dict[str, dict] = {
    "30": {"name": "門別",   "region": "北海道",  "payback": 0.70,  "banei": False},
    "35": {"name": "盛岡",   "region": "東北",    "payback": 0.70,  "banei": False},
    "36": {"name": "水沢",   "region": "東北",    "payback": 0.70,  "banei": False},
    "42": {"name": "浦和",   "region": "南関東",  "payback": 0.725, "banei": False},
    "43": {"name": "船橋",   "region": "南関東",  "payback": 0.725, "banei": False},
    "44": {"name": "大井",   "region": "南関東",  "payback": 0.725, "banei": False},
    "45": {"name": "川崎",   "region": "南関東",  "payback": 0.725, "banei": False},
    "46": {"name": "金沢",   "region": "北陸",    "payback": 0.70,  "banei": False},
    "47": {"name": "笠松",   "region": "東海",    "payback": 0.70,  "banei": False},
    "48": {"name": "名古屋", "region": "東海",    "payback": 0.70,  "banei": False},
    "50": {"name": "園田",   "region": "近畿",    "payback": 0.70,  "banei": False},
    "51": {"name": "姫路",   "region": "近畿",    "payback": 0.70,  "banei": False},
    "54": {"name": "高知",   "region": "四国",    "payback": 0.70,  "banei": False},
    "55": {"name": "佐賀",   "region": "九州",    "payback": 0.70,  "banei": False},
    "65": {"name": "帯広",   "region": "北海道",  "payback": 0.70,  "banei": True},
}


def track_code_from_race_id(race_id: str) -> str:
    """race_id から場コード(2桁)を取得。例: '202633060412' → '33'"""
    return race_id[4:6]


def race_type_from_race_id(race_id: str) -> str:
    code = track_code_from_race_id(race_id)
    return "banei" if TRACK_MAP.get(code, {}).get("banei") else "flat"


# ──────────────────────────────────────────────────
# NARスクレイパー本体
# ──────────────────────────────────────────────────
class NARScraper:
    """
    NAR全場の今日・過去レースデータを取得してDBに保存する。
    1リクエストごとに2〜3秒待機、エラー時は3回リトライ。
    """

    def __init__(self, sleep_sec: float = 2.5, max_retry: int = 3):
        self.session    = make_session()
        self.sleep_sec  = sleep_sec
        self.max_retry  = max_retry
        self._success   = 0
        self._failure   = 0

    # ────────────────────────────────────────────
    # リクエスト
    # ────────────────────────────────────────────
    def _get(self, url: str) -> Optional[str]:
        """GETリクエスト。失敗したらリトライ。"""
        for attempt in range(self.max_retry):
            try:
                time.sleep(self.sleep_sec + attempt * 1.0)
                resp = self.session.get(url, timeout=20)
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding or "utf-8"
                self._success += 1
                return resp.text
            except Exception as e:
                self._failure += 1
                print(f"  [retry {attempt+1}/{self.max_retry}] {url}: {e}")
        return None

    @property
    def success_rate(self) -> float:
        total = self._success + self._failure
        return self._success / total if total > 0 else 1.0

    # ────────────────────────────────────────────
    # 開催レース一覧の取得
    # ────────────────────────────────────────────
    def fetch_race_ids(self, target_date: date) -> list[str]:
        """
        指定日の全NAR race_id リストを返す。
        """
        date_str = target_date.strftime("%Y%m%d")
        url = f"{NAR_BASE}/top/race_list_sub.html?kaisai_date={date_str}"
        html = self._get(url)
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        race_ids = []
        seen = set()
        for a in soup.select("a[href*='race_id=']"):
            m = re.search(r"race_id=(\d{12})", a["href"])
            if m and m.group(1) not in seen:
                race_ids.append(m.group(1))
                seen.add(m.group(1))

        print(f"  [NAR] {date_str}: {len(race_ids)}レース検出")
        return race_ids

    # ────────────────────────────────────────────
    # 1レース分の取得・保存
    # ────────────────────────────────────────────
    def scrape_race(
        self,
        race_id: str,
        scrape_result: bool = True,
        skip_if_exists: bool = True,
    ) -> bool:
        """
        result.html（確定済み）または shutuba.html（出馬表）を取得してDBに保存。

        Parameters
        ----------
        race_id       : 12桁のレースID
        scrape_result : True=result.html、False=shutuba.html
        skip_if_exists: DBに既にあればスキップ

        Returns
        -------
        bool : 成功したか
        """
        # ── 取得済みスキップ ────────────────────
        if skip_if_exists and self._race_exists(race_id):
            return True

        page = "result" if scrape_result else "shutuba"
        url  = f"{NAR_BASE}/race/{page}.html?race_id={race_id}"
        html = self._get(url)
        if not html:
            self._log_scraping(race_id, url, False, "HTML取得失敗")
            return False

        soup = BeautifulSoup(html, "html.parser")

        try:
            race_info    = self._parse_race_info(soup, race_id)
            horses       = self._parse_horses(soup, race_id, scrape_result)
            payouts      = self._parse_payouts(soup, race_id) if scrape_result else []
            corner_map   = self._parse_corner_passing(soup) if scrape_result else {}

            # コーナー通過順を horses に注入（馬番でマッチ）
            if corner_map:
                for h in horses:
                    dn = h.get("draw_number")
                    if dn and dn in corner_map:
                        c = corner_map[dn]
                        h["corner1_pos"] = c[0] if c[0] else None
                        h["corner2_pos"] = c[1] if c[1] else None
                        h["corner3_pos"] = c[2] if c[2] else None
                        h["corner4_pos"] = c[3] if c[3] else None

            self._save_race(race_info)
            self._save_horses(horses)
            if payouts:
                self._save_payouts(payouts)

            self._update_progress(race_id, "done")
            self._log_scraping(race_id, url, True, None)
            return True

        except Exception as e:
            self._log_scraping(race_id, url, False, str(e))
            print(f"  [NAR] {race_id} パースエラー: {e}")
            return False

    # ────────────────────────────────────────────
    # パーサー
    # ────────────────────────────────────────────
    def _parse_race_info(self, soup: BeautifulSoup, race_id: str) -> dict:
        """レース基本情報をパース。"""
        # race_id 構造: YYYY(4) + jyo(2) + MM(2) + DD(2) + RR(2) = 12桁
        # 例: 202630060401 → 2026年 門別(30) 06月04日 1R
        year       = race_id[0:4]
        track_code = race_id[4:6]
        month      = race_id[6:8]
        day        = race_id[8:10]
        race_no    = int(race_id[10:12])
        race_date  = f"{year}-{month}-{day}"

        track_info = TRACK_MAP.get(track_code, {})
        track_name = track_info.get("name", "不明")
        race_type  = "banei" if track_info.get("banei") else "flat"
        payback    = track_info.get("payback", 0.70)

        # レース名
        rname_el = soup.select_one(".RaceName")
        race_name = rname_el.get_text(strip=True) if rname_el else ""

        # RaceData01: 発走時刻・距離・コース方向
        rd1 = soup.select_one(".RaceData01")
        rd1_text = rd1.get_text(strip=True) if rd1 else ""

        post_time    = ""
        distance     = 0
        surface      = "ダート"
        course_dir   = ""
        going        = "良"
        weather      = ""

        # 発走時刻: "14:15発走"
        m_time = re.search(r"(\d{2}:\d{2})発走", rd1_text)
        if m_time:
            post_time = m_time.group(1)

        # 距離・芝/ダ・コース: "/ダ1000m(右)" or "/芝1200m(左)"
        m_dist = re.search(r"([芝ダ障])(\d+)m\(?(右|左|直)?\)?", rd1_text)
        if m_dist:
            surf_char  = m_dist.group(1)
            distance   = int(m_dist.group(2))
            course_dir = m_dist.group(3) or ""
            if surf_char == "芝":
                surface = "芝"
            elif surf_char == "障":
                surface = "障害"

        # ばんえいは特殊
        if race_type == "banei":
            surface = "ばんえい"

        # 天候: "天候:晴"
        m_weather = re.search(r"天候:([^\s/]+)", rd1_text)
        if m_weather:
            weather = m_weather.group(1)

        # 馬場状態: "馬場:良"
        m_going = re.search(r"馬場:([^\s/]+)", rd1_text)
        if m_going:
            going = m_going.group(1)

        # RaceData02: 頭数・クラス
        rd2 = soup.select_one(".RaceData02")
        rd2_text = rd2.get_text(strip=True) if rd2 else ""

        field_size  = 0
        race_class  = ""
        m_field = re.search(r"(\d+)頭", rd2_text)
        if m_field:
            field_size = int(m_field.group(1))

        # クラス名（「C4」「A」「重賞」等）
        m_class = re.search(r"(重賞|[A-Z][0-9A-Z\-ー]+|オープン|特別)", rd2_text)
        if m_class:
            race_class = m_class.group(1)

        return {
            "race_id":    race_id,
            "race_date":  race_date,
            "track":      track_name,
            "track_code": track_code,
            "race_no":    race_no,
            "race_name":  race_name,
            "surface":    surface,
            "distance":   distance,
            "course_dir": course_dir,
            "track_condition": going,
            "weather":    weather,
            "field_size": field_size,
            "race_class": race_class,
            "post_time":  post_time,
            "race_type":  race_type,
            "payback_rate": payback,
            "organizer":  "NAR",
        }

    def _parse_horses(
        self, soup: BeautifulSoup, race_id: str, is_result: bool
    ) -> list[dict]:
        """馬ごとの成績/出馬情報をパース。"""
        table = soup.select_one("table.RaceTable01, table.Shutuba_Table")
        if not table:
            return []

        year       = race_id[0:4]
        track_code = race_id[4:6]
        month      = race_id[6:8]
        day        = race_id[8:10]
        race_date  = f"{year}-{month}-{day}"
        race_type  = "banei" if TRACK_MAP.get(track_code, {}).get("banei") else "flat"

        horses = []
        # NARのresult/shutuba tableは tr.HorseList を持たないケースがある
        # クラスなし・またはHorseList・MiddleListの行を全て対象にする
        rows = table.select("tr.HorseList, tr.MiddleList")
        if not rows:
            # ヘッダー行(Header class or th要素を持つ)を除いた全tr
            rows = [r for r in table.select("tr")
                    if "Header" not in (r.get("class") or [])
                    and not r.select("th")
                    and len(r.select("td")) >= 5]

        # ヘッダー行からカラム位置を特定
        headers = [th.get_text(strip=True) for th in table.select("tr:first-child th")]

        def col_idx(names: list[str]) -> int:
            for n in names:
                for i, h in enumerate(headers):
                    if n in h:
                        return i
            return -1

        idx_finish  = col_idx(["着順"])
        idx_waku    = col_idx(["枠"])
        idx_num     = col_idx(["馬番"])
        idx_name    = col_idx(["馬名"])
        idx_sex_age = col_idx(["性齢"])
        idx_weight_load = col_idx(["斤量"])
        idx_jockey  = col_idx(["騎手"])
        idx_time    = col_idx(["タイム"])
        idx_popular = col_idx(["人気"])
        idx_odds    = col_idx(["単勝オッズ", "オッズ"])
        idx_agari   = col_idx(["後3F"])
        idx_trainer = col_idx(["厩舎"])
        idx_hw      = col_idx(["馬体重"])

        for row in rows:
            cells = [td.get_text(strip=True) for td in row.select("td")]
            if len(cells) < 4:
                continue

            def _cell(idx: int) -> str:
                return cells[idx].strip() if 0 <= idx < len(cells) else ""

            # 馬ID取得
            hl = row.select_one("td.HorseName a, td a[href*='horse']")
            horse_id = ""
            if hl:
                m = re.search(r"horse/(\d+)", hl.get("href", ""))
                if m:
                    horse_id = m.group(1)

            # 騎手ID
            # NAR URL例: ../jockey/result/recent/01083/ → ID=01083
            #            ../jockey/?jockey_id=01083      → ID=01083
            jl = row.select_one("td a[href*='jockey']")
            jockey_id = ""
            if jl:
                href = jl.get("href", "")
                m = re.search(r"jockey/(?:result/\w+/|result/)(\d+)", href) \
                    or re.search(r"jockey_id=(\d+)", href) \
                    or re.search(r"/jockey/(\d{5,})", href)
                if m:
                    jockey_id = m.group(1)

            # 調教師ID
            trl = row.select_one("td a[href*='trainer']")
            trainer_id = ""
            if trl:
                href = trl.get("href", "")
                m = re.search(r"trainer/(?:result/\w+/|result/)(\d+)", href) \
                    or re.search(r"trainer_id=(\d+)", href) \
                    or re.search(r"/trainer/(\d{5,})", href)
                if m:
                    trainer_id = m.group(1)

            # 着順（"1"〜"18"、"取"=取消、"除"=除外、"失"=失格）
            finish_str = _cell(idx_finish)
            finish_pos = None
            if re.match(r"^\d+$", finish_str):
                finish_pos = int(finish_str)

            # 馬体重・増減: "480(+4)" or "480(+4)202.2" など
            hw_text = _cell(idx_hw)
            hw_match = re.search(r"(\d{3,4})\(([+\-]\d+)\)", hw_text)
            horse_weight      = int(hw_match.group(1)) if hw_match else None
            horse_weight_diff = int(hw_match.group(2)) if hw_match else None

            # タイム（秒変換: "1:02.3" → 62.3）
            time_str = _cell(idx_time)
            race_time_sec = _parse_time(time_str)

            # 上がり3F（秒）
            agari_str = _cell(idx_agari)
            agari_sec = float(agari_str) if re.match(r"^\d+\.\d+$", agari_str) else None

            # 単勝オッズ
            odds_str = _cell(idx_odds)
            odds_str_clean = odds_str.replace(",", "").split("\n")[0].strip()
            win_odds = None
            try:
                win_odds = float(odds_str_clean)
            except ValueError:
                pass

            # 人気
            pop_str = _cell(idx_popular)
            popular_rank = int(pop_str) if re.match(r"^\d+$", pop_str) else None

            # 性齢: "牡3" → sex="牡", age=3
            sex_age = _cell(idx_sex_age)
            sex = sex_age[0] if sex_age else ""
            age_m = re.search(r"(\d+)$", sex_age)
            age   = int(age_m.group(1)) if age_m else None

            horses.append({
                "race_id":          race_id,
                "race_date":        race_date,
                "race_type":        race_type,
                "horse_id":         horse_id,
                "horse_name":       _cell(idx_name).split("\n")[0].strip(),
                "draw_number":      _safe_int(_cell(idx_num)),
                "frame_number":     _safe_int(_cell(idx_waku)),
                "jockey_id":        jockey_id,
                "jockey_name":      _cell(idx_jockey),
                "trainer_id":       trainer_id,
                "trainer_name":     _cell(idx_trainer).replace("北海道", "").replace("南関東", "").strip(),
                "sex":              sex,
                "age":              age,
                "weight_carried":   _safe_float(_cell(idx_weight_load)),
                "horse_weight":     horse_weight,
                "horse_weight_diff": horse_weight_diff,
                "finish_position":  finish_pos,
                "race_time_seconds": race_time_sec,
                "agari3f_seconds":  agari_sec,
                "win_odds":         win_odds,
                "popular_rank":     popular_rank,
                "is_result":        int(is_result),
            })

        return horses

    def _parse_payouts(self, soup: BeautifulSoup, race_id: str) -> list[dict]:
        """払戻テーブルをパース。3連複のみを返す。"""
        payouts = []
        year  = race_id[0:4]
        month = race_id[6:8]   # race_id[4:6]=場コード, [6:8]=月
        day   = race_id[8:10]
        race_date = f"{year}-{month}-{day}"

        for pt in soup.select(".Payout_Detail_Table"):
            th = pt.select_one("th")
            if not th:
                continue
            bet_type_name = th.get_text(strip=True)

            # 3連複のみ対象
            if "3連複" not in bet_type_name:
                continue

            tds = [td.get_text(strip=True) for td in pt.select("td")]
            # tds構造: [combo, payout_str, popular, ...] が繰り返す
            # 例: ['1-3-7', '12,340円', '5人気', '2-4-8', '...]
            i = 0
            while i + 2 < len(tds):
                combo_str   = tds[i].strip()
                payout_str  = tds[i + 1].strip()
                popular_str = tds[i + 2].strip() if i + 2 < len(tds) else ""

                # comboのバリデーション（数字-数字-数字）
                if not re.match(r"^\d{1,2}[\-－]\d{1,2}[\-－]\d{1,2}$", combo_str):
                    i += 1
                    continue

                # 払戻金額を整数に
                payout_amount = _parse_payout(payout_str)
                if payout_amount is None:
                    i += 3
                    continue

                combo_normalized = re.sub(r"[－]", "-", combo_str)
                payouts.append({
                    "race_id":      race_id,
                    "race_date":    race_date,
                    "bet_type":     "trio",
                    "combo":        combo_normalized,
                    "payout":       payout_amount,
                    "popular_info": popular_str,
                })
                i += 3

        return payouts

    def _parse_corner_passing(self, soup: BeautifulSoup) -> dict[int, list[int]]:
        """
        コーナー通過順を解析して {horse_num: [c1, c2, c3, c4]} を返す。
        ナビゲーション情報が取れない場合は空辞書。
        """
        corner_el = soup.select_one(".Corner_Num")
        if not corner_el:
            return {}

        text = corner_el.get_text(strip=True)
        result: dict[int, dict[int, int]] = {}

        for corner_match in re.finditer(r"(\d)コーナー(.+?)(?=\d+コーナー|$)", text):
            corner_no = int(corner_match.group(1))
            order_text = corner_match.group(2)
            pos = 1
            for part in re.split(r",", order_text):
                part = part.strip("()（） ")
                nums = re.findall(r"\d+", part)
                for n in nums:
                    hn = int(n)
                    if hn not in result:
                        result[hn] = {}
                    result[hn][corner_no] = pos
                pos += len(nums)

        return {hn: [
            result[hn].get(1, 0),
            result[hn].get(2, 0),
            result[hn].get(3, 0),
            result[hn].get(4, 0),
        ] for hn in result}

    # ────────────────────────────────────────────
    # DB操作
    # ────────────────────────────────────────────
    def _race_exists(self, race_id: str) -> bool:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM nar_races WHERE race_id = ?", (race_id,)
            ).fetchone()
        return row is not None

    def _has_horses(self, race_id: str) -> bool:
        """nar_results または nar_entries に馬データが存在するか確認。"""
        with get_conn() as conn:
            r = conn.execute(
                "SELECT COUNT(*) FROM nar_results WHERE race_id = ?", (race_id,)
            ).fetchone()
            if r and r[0] > 0:
                return True
            e = conn.execute(
                "SELECT COUNT(*) FROM nar_entries WHERE race_id = ?", (race_id,)
            ).fetchone()
            return (e and e[0] > 0)

    def _save_race(self, info: dict) -> None:
        with get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO nar_races
                  (race_id, race_date, track, track_code, race_no, race_name,
                   surface, distance, course_dir, track_condition, weather,
                   field_size, race_class, post_time, race_type, payback_rate, organizer)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                info["race_id"], info["race_date"], info["track"], info["track_code"],
                info["race_no"], info["race_name"], info["surface"], info["distance"],
                info["course_dir"], info["track_condition"], info["weather"],
                info["field_size"], info["race_class"], info["post_time"],
                info["race_type"], info["payback_rate"], info["organizer"],
            ))

    def _save_horses(self, horses: list[dict]) -> None:
        with get_conn() as conn:
            for h in horses:
                # horsesマスタ
                if h["horse_id"]:
                    conn.execute("""
                        INSERT OR IGNORE INTO horses (horse_id, horse_name)
                        VALUES (?, ?)
                    """, (h["horse_id"], h["horse_name"]))

                # nar_results（着順確定分）または nar_entries（出馬表分）
                if h["is_result"]:
                    conn.execute("""
                        INSERT OR REPLACE INTO nar_results
                          (race_id, race_date, horse_id, horse_name, draw_number, frame_number,
                           jockey_id, jockey_name, trainer_id, trainer_name,
                           sex, age, weight_carried, horse_weight, horse_weight_diff,
                           finish_position, race_time_seconds, agari3f_seconds,
                           corner1_pos, corner2_pos, corner3_pos, corner4_pos,
                           win_odds, popular_rank, race_type)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        h["race_id"], h["race_date"], h["horse_id"], h["horse_name"],
                        h["draw_number"], h["frame_number"],
                        h["jockey_id"], h["jockey_name"], h["trainer_id"], h["trainer_name"],
                        h["sex"], h["age"], h["weight_carried"],
                        h["horse_weight"], h["horse_weight_diff"],
                        h["finish_position"], h["race_time_seconds"], h["agari3f_seconds"],
                        h.get("corner1_pos"), h.get("corner2_pos"),
                        h.get("corner3_pos"), h.get("corner4_pos"),
                        h["win_odds"], h["popular_rank"], h["race_type"],
                    ))
                    # past_results にも反映（特徴量計算用）
                    if h["finish_position"] and h["race_id"]:
                        conn.execute("""
                            INSERT OR IGNORE INTO past_results
                              (race_id, race_date, horse_id, finish_position,
                               race_time_seconds, agari3f_seconds, field_size,
                               distance, surface, track, track_condition)
                            SELECT ?, ?, ?, ?, ?, ?, field_size, distance, surface, track, track_condition
                            FROM nar_races WHERE race_id = ?
                        """, (
                            h["race_id"], h["race_date"], h["horse_id"],
                            h["finish_position"], h["race_time_seconds"],
                            h["agari3f_seconds"], h["race_id"],
                        ))
                else:
                    conn.execute("""
                        INSERT OR REPLACE INTO nar_entries
                          (race_id, race_date, horse_id, horse_name, draw_number, frame_number,
                           jockey_id, jockey_name, trainer_id, horse_weight, horse_weight_diff,
                           win_odds, popular_rank, race_type)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        h["race_id"], h["race_date"], h["horse_id"], h["horse_name"],
                        h["draw_number"], h["frame_number"],
                        h["jockey_id"], h["jockey_name"], h["trainer_id"],
                        h["horse_weight"], h["horse_weight_diff"],
                        h["win_odds"], h["popular_rank"], h["race_type"],
                    ))

    def _save_payouts(self, payouts: list[dict]) -> None:
        with get_conn() as conn:
            for p in payouts:
                conn.execute("""
                    INSERT OR REPLACE INTO nar_payouts
                      (race_id, race_date, bet_type, combo, payout, popular_info)
                    VALUES (?,?,?,?,?,?)
                """, (p["race_id"], p["race_date"], p["bet_type"],
                      p["combo"], p["payout"], p["popular_info"]))

    def _log_scraping(
        self, race_id: str, url: str, success: bool, error: Optional[str]
    ) -> None:
        with get_conn() as conn:
            conn.execute("""
                INSERT INTO scraping_logs (run_date, target_url, success, error_message)
                VALUES (date('now'), ?, ?, ?)
            """, (url, int(success), error))

    def _update_progress(self, race_id: str, status: str) -> None:
        with get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO scraping_progress
                  (race_id, status, updated_at)
                VALUES (?, ?, datetime('now'))
            """, (race_id, status))

    # ────────────────────────────────────────────
    # 日次実行
    # ────────────────────────────────────────────
    def run_today(
        self,
        target_date: Optional[date] = None,
        track_filter: Optional[list[str]] = None,
    ) -> dict:
        """
        今日の全NAR開催レースを取得してDBに保存。

        Parameters
        ----------
        target_date  : 対象日（None = 今日）
        track_filter : 場コードのリスト（例: ["46", "43"]）。None = 全場。

        Returns
        -------
        dict: {total, success, failure, skipped}
        """
        if target_date is None:
            target_date = date.today()

        race_ids = self.fetch_race_ids(target_date)

        if track_filter:
            race_ids = [r for r in race_ids if track_code_from_race_id(r) in track_filter]

        stats = {"total": len(race_ids), "success": 0, "failure": 0, "skipped": 0}

        for race_id in race_ids:
            if self._race_exists(race_id) and self._has_horses(race_id):
                stats["skipped"] += 1
                continue

            # result.html → 馬データなし → shutuba.html にフォールバック
            ok = self.scrape_race(race_id, scrape_result=True, skip_if_exists=False)
            if ok and not self._has_horses(race_id):
                # まだ未発走 → 出馬表（shutuba）から取得
                ok = self.scrape_race(race_id, scrape_result=False, skip_if_exists=False)
            if ok and self._has_horses(race_id):
                stats["success"] += 1
            elif ok:
                # 出馬表も馬データなし（時刻前など）
                stats["skipped"] += 1
            else:
                stats["failure"] += 1

        print(f"[NAR] {target_date}: "
              f"取得{stats['success']} / スキップ{stats['skipped']} / 失敗{stats['failure']}")
        return stats


# ──────────────────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────────────────
def _parse_time(time_str: str) -> Optional[float]:
    """タイム文字列を秒に変換。"1:02.3" → 62.3"""
    if not time_str:
        return None
    m = re.match(r"(\d+):(\d+)\.(\d+)", time_str)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2)) + float(f"0.{m.group(3)}")
    m2 = re.match(r"(\d+)\.(\d+)", time_str)
    if m2:
        return float(f"{m2.group(1)}.{m2.group(2)}")
    return None


def _parse_payout(payout_str: str) -> Optional[int]:
    """払戻金額文字列を整数に変換。"12,340円" → 12340"""
    cleaned = re.sub(r"[円,，\s]", "", payout_str)
    try:
        return int(cleaned)
    except ValueError:
        return None


def _safe_int(s: str) -> Optional[int]:
    try:
        return int(re.sub(r"[^\d]", "", s))
    except (ValueError, TypeError):
        return None


def _safe_float(s: str) -> Optional[float]:
    try:
        return float(re.sub(r"[^\d.]", "", s))
    except (ValueError, TypeError):
        return None
