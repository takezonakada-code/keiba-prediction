"""
JRA スクレイパー（race.netkeiba.com）

取得対象:
  - レース基本情報（距離・馬場・クラス・賞金）
  - 着順・タイム・上がり・コーナー通過順
  - 払戻（単複馬連ワイド馬単3連複3連単）
  - 馬体重・騎手・調教師
"""
from __future__ import annotations

import re
import time
from datetime import date
from typing import Optional

from bs4 import BeautifulSoup

from data.scraper import make_session
from db.database import get_conn

JRA_BASE = "https://race.netkeiba.com"

# JRA競馬場コード → 競馬場名
JRA_TRACK_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}


class JRAScraper:
    """JRA race.netkeiba.com スクレイパー。"""

    def __init__(self, sleep_sec: float = 2.0, max_retry: int = 5):
        self.session   = make_session()
        self.sleep_sec = sleep_sec
        self.max_retry = max_retry
        self._success  = 0
        self._failure  = 0

    @property
    def success_rate(self) -> float:
        total = self._success + self._failure
        return self._success / total if total > 0 else 1.0

    def _get(self, url: str) -> Optional[str]:
        for attempt in range(self.max_retry):
            try:
                time.sleep(self.sleep_sec * (1.0 + attempt * 0.5))
                resp = self.session.get(url, timeout=20)
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding or "utf-8"
                self._success += 1
                return resp.text
            except Exception as e:
                self._failure += 1
                print(f"  [JRA retry {attempt+1}/{self.max_retry}] {url[:80]}: {e}")
        return None

    # ── レース一覧取得 ─────────────────────────────
    def fetch_race_ids(self, target_date: date) -> list[str]:
        """指定日の全JRA race_id を返す。"""
        date_str = target_date.strftime("%Y%m%d")
        url = f"{JRA_BASE}/top/race_list_sub.html?kaisai_date={date_str}"
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
        return race_ids

    # ── 1レース取得 ────────────────────────────────
    def scrape_race(self, race_id: str, skip_if_exists: bool = True) -> bool:
        if skip_if_exists and self._race_exists(race_id):
            return True

        url = f"{JRA_BASE}/race/result.html?race_id={race_id}"
        html = self._get(url)
        if not html:
            return False

        soup = BeautifulSoup(html, "html.parser")
        try:
            race_info = self._parse_race_info(soup, race_id)
            horses    = self._parse_horses(soup, race_id)
            payouts   = self._parse_payouts(soup, race_id)

            self._save_race(race_info)
            self._save_horses(horses)
            self._save_payouts(payouts)
            self._save_past_result(horses, race_info)  # past_resultsにも反映
            return True
        except Exception as e:
            print(f"  [JRA] {race_id} パースエラー: {e}")
            return False

    # ── パーサー ────────────────────────────────────
    def _parse_race_info(self, soup: BeautifulSoup, race_id: str) -> dict:
        year  = race_id[0:4]
        tc    = race_id[4:6]
        month = race_id[8:10]
        day   = race_id[10:12]
        race_no = int(race_id[11:12] or "1")
        # race_id: YYYY + KJ(2) + KAI(2) + MMDD(4) + RR(2)
        # 例: 202305010101 → 2023年 札幌(01) 第5回(01) 01月01日 1R
        year    = race_id[0:4]
        kj      = race_id[4:6]
        kai     = race_id[6:8]
        mmdd    = race_id[8:12]
        race_no = int(race_id[10:12])
        month   = mmdd[0:2]
        day     = mmdd[2:4]
        race_date = f"{year}-{month}-{day}"
        track_name = JRA_TRACK_MAP.get(kj, f"場{kj}")

        rname_el = soup.select_one(".RaceName")
        race_name = rname_el.get_text(strip=True) if rname_el else ""

        rd1 = soup.select_one(".RaceData01")
        rd1_text = rd1.get_text(strip=True) if rd1 else ""

        # 距離・芝ダ・回り
        m_dist = re.search(r"([芝ダ障])(\d+)m[\s(（]?(右|左|直)?", rd1_text)
        surface  = "芝" if m_dist and m_dist.group(1)=="芝" else "ダート"
        distance = int(m_dist.group(2)) if m_dist else 0
        course_dir = m_dist.group(3) if m_dist else ""

        # 馬場状態
        m_going = re.search(r"馬場:?[\s]*([^\s/\n]+)", rd1_text)
        going = m_going.group(1).strip() if m_going else "良"

        # 天気
        m_weather = re.search(r"天候:?[\s]*([^\s/\n]+)", rd1_text)
        weather = m_weather.group(1).strip() if m_weather else ""

        # 頭数・クラス
        rd2 = soup.select_one(".RaceData02")
        rd2_text = rd2.get_text(strip=True) if rd2 else ""
        m_field = re.search(r"(\d+)頭", rd2_text)
        field_size = int(m_field.group(1)) if m_field else 0
        m_class = re.search(r"(G[1-3]|オープン|3勝|2勝|1勝|未勝利|新馬|障害)", rd2_text)
        race_class = m_class.group(1) if m_class else ""

        # 距離帯
        dist_band = "short" if distance <= 1400 else "mile_middle" if distance <= 2000 else "long"

        return {
            "race_id": race_id, "race_date": race_date,
            "track": track_name, "track_code": kj,
            "race_no": race_no, "race_name": race_name,
            "surface": surface, "distance": distance, "distance_band": dist_band,
            "course_dir": course_dir, "track_condition": going,
            "weather": weather, "field_size": field_size,
            "race_class": race_class,
        }

    def _parse_horses(self, soup: BeautifulSoup, race_id: str) -> list[dict]:
        year  = race_id[0:4]
        month = race_id[8:10]
        day   = race_id[10:12]
        race_date = f"{year}-{month}-{day}"

        table = soup.select_one("table.RaceTable01")
        if not table:
            return []

        headers = [th.get_text(strip=True) for th in table.select("tr:first-child th")]
        def ci(names):
            for n in names:
                for i, h in enumerate(headers):
                    if n in h: return i
            return -1

        i_fin   = ci(["着順"])
        i_waku  = ci(["枠"])
        i_num   = ci(["馬番"])
        i_name  = ci(["馬名"])
        i_sex   = ci(["性齢"])
        i_kg    = ci(["斤量"])
        i_jock  = ci(["騎手"])
        i_time  = ci(["タイム"])
        i_diff  = ci(["着差"])
        i_pop   = ci(["人気"])
        i_odds  = ci(["オッズ","単勝"])
        i_agari = ci(["上がり","後3F"])
        i_train = ci(["厩舎","調教師"])
        i_hw    = ci(["馬体重"])

        rows = table.select("tr.HorseList, tr.MiddleList")
        if not rows:
            rows = [r for r in table.select("tr")
                    if "Header" not in str(r.get("class",""))
                    and not r.select("th") and len(r.select("td")) >= 5]

        horses = []
        for row in rows:
            cells = [td.get_text(strip=True) for td in row.select("td")]
            def c(idx): return cells[idx].strip() if 0 <= idx < len(cells) else ""

            # 馬ID
            hl = row.select_one("a[href*='horse']")
            horse_id = ""
            if hl:
                m = re.search(r"horse/(\d+)", hl.get("href",""))
                if m: horse_id = m.group(1)

            # 騎手ID
            jl = row.select_one("a[href*='jockey']")
            jockey_id = ""
            if jl:
                m = re.search(r"jockey/result/\w+/(\d+)", jl.get("href","")) \
                    or re.search(r"jockey/(\d{5})", jl.get("href",""))
                if m: jockey_id = m.group(1)

            # 調教師ID
            trl = row.select_one("a[href*='trainer']")
            trainer_id = ""
            if trl:
                m = re.search(r"trainer/result/\w+/(\d+)", trl.get("href","")) \
                    or re.search(r"trainer/(\d{5})", trl.get("href",""))
                if m: trainer_id = m.group(1)

            # 着順
            fin_str = c(i_fin)
            fin_pos = int(fin_str) if re.match(r"^\d+$", fin_str) else None

            # 馬体重
            hw_text = c(i_hw)
            hw_m = re.search(r"(\d{3,4})\(([+\-]\d+)\)", hw_text)
            horse_weight      = int(hw_m.group(1)) if hw_m else None
            horse_weight_diff = int(hw_m.group(2)) if hw_m else None

            # タイム
            race_time_sec = _parse_time(c(i_time))

            # 上がり3F
            agari_str = c(i_agari)
            agari_sec = float(agari_str) if re.match(r"^\d+\.\d+$", agari_str) else None

            # オッズ
            odds_str = c(i_odds).replace(",","").split("\n")[0].strip()
            try: win_odds = float(odds_str)
            except: win_odds = None

            # 人気
            pop_str = c(i_pop)
            popular_rank = int(pop_str) if re.match(r"^\d+$", pop_str) else None

            # 性齢
            sa = c(i_sex)
            sex = sa[0] if sa else ""
            age_m = re.search(r"(\d+)$", sa)
            age = int(age_m.group(1)) if age_m else None

            horses.append({
                "race_id": race_id, "race_date": race_date,
                "horse_id": horse_id,
                "horse_name": c(i_name).split("\n")[0].strip(),
                "draw_number": _safe_int(c(i_num)),
                "frame_number": _safe_int(c(i_waku)),
                "jockey_id": jockey_id, "jockey_name": c(i_jock),
                "trainer_id": trainer_id, "trainer_name": c(i_train),
                "sex": sex, "age": age,
                "weight_carried": _safe_float(c(i_kg)),
                "horse_weight": horse_weight, "horse_weight_diff": horse_weight_diff,
                "finish_position": fin_pos, "finish_position_raw": fin_str,
                "race_time_seconds": race_time_sec,
                "margin_text": c(i_diff),
                "agari3f_seconds": agari_sec,
                "win_odds": win_odds, "popular_rank": popular_rank,
            })

        # コーナー通過順
        corner_map = self._parse_corners(soup)
        for h in horses:
            dn = h.get("draw_number")
            if dn and dn in corner_map:
                cp = corner_map[dn]
                h["corner1_pos"] = cp.get(1)
                h["corner2_pos"] = cp.get(2)
                h["corner3_pos"] = cp.get(3)
                h["corner4_pos"] = cp.get(4)

        return horses

    def _parse_corners(self, soup: BeautifulSoup) -> dict:
        """コーナー通過順を解析。"""
        corner_el = soup.select_one(".Corner_Num")
        if not corner_el:
            return {}
        text = corner_el.get_text(strip=True)
        result = {}
        for cm in re.finditer(r"(\d)コーナー(.+?)(?=\d+コーナー|$)", text):
            cn = int(cm.group(1))
            pos = 1
            for part in re.split(r",", cm.group(2)):
                part = part.strip("()（） ")
                nums = re.findall(r"\d+", part)
                for n in nums:
                    hn = int(n)
                    if hn not in result: result[hn] = {}
                    result[hn][cn] = pos
                pos += len(nums)
        return result

    def _parse_payouts(self, soup: BeautifulSoup, race_id: str) -> list[dict]:
        year  = race_id[0:4]
        month = race_id[8:10]
        day   = race_id[10:12]
        race_date = f"{year}-{month}-{day}"

        payouts = []
        BET_TYPES = {
            "単勝": "win", "複勝": "place", "枠連": "bracket_quinella",
            "馬連": "quinella", "馬単": "exacta",
            "ワイド": "wide", "3連複": "trio", "3連単": "trifecta",
        }
        for pt in soup.select(".Payout_Detail_Table"):
            th = pt.select_one("th")
            if not th: continue
            bet_name = th.get_text(strip=True)
            bet_type = BET_TYPES.get(bet_name, bet_name)
            tds = [td.get_text(strip=True) for td in pt.select("td")]

            i = 0
            while i + 1 < len(tds):
                combo_str = tds[i].strip()
                pay_str   = tds[i+1].strip() if i+1 < len(tds) else ""
                pop_str   = tds[i+2].strip() if i+2 < len(tds) else ""

                if not combo_str or not re.search(r"\d", combo_str):
                    i += 1
                    continue

                payout = _parse_payout(pay_str)
                if payout is None:
                    i += 3
                    continue

                combo = re.sub(r"[－]", "-", combo_str)
                payouts.append({
                    "race_id": race_id, "race_date": race_date,
                    "bet_type": bet_type, "combo": combo,
                    "payout": payout, "popular_info": pop_str,
                })
                i += 3
        return payouts

    # ── DB保存 ──────────────────────────────────────
    def _race_exists(self, race_id: str) -> bool:
        with get_conn() as conn:
            return conn.execute(
                "SELECT 1 FROM jra_races WHERE race_id=?", (race_id,)
            ).fetchone() is not None

    def _save_race(self, info: dict) -> None:
        with get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO jra_races
                  (race_id, race_date, track, track_code, race_no, race_name,
                   surface, distance, distance_band, course_dir,
                   track_condition, weather, field_size, race_class)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (info["race_id"], info["race_date"], info["track"], info["track_code"],
                  info["race_no"], info["race_name"], info["surface"], info["distance"],
                  info["distance_band"], info["course_dir"],
                  info["track_condition"], info["weather"],
                  info["field_size"], info["race_class"]))

    def _save_horses(self, horses: list[dict]) -> None:
        with get_conn() as conn:
            for h in horses:
                if h["horse_id"]:
                    conn.execute("INSERT OR IGNORE INTO horses(horse_id,horse_name) VALUES(?,?)",
                                 (h["horse_id"], h["horse_name"]))
                conn.execute("""
                    INSERT OR REPLACE INTO jra_results
                      (race_id, race_date, horse_id, horse_name,
                       draw_number, frame_number,
                       jockey_id, jockey_name, trainer_id, trainer_name,
                       sex, age, weight_carried, horse_weight, horse_weight_diff,
                       finish_position, finish_position_raw,
                       race_time_seconds, margin_text,
                       corner1_pos, corner2_pos, corner3_pos, corner4_pos,
                       agari3f_seconds, win_odds, popular_rank)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    h["race_id"], h["race_date"], h["horse_id"], h["horse_name"],
                    h["draw_number"], h["frame_number"],
                    h["jockey_id"], h["jockey_name"], h["trainer_id"], h["trainer_name"],
                    h["sex"], h["age"], h["weight_carried"],
                    h["horse_weight"], h["horse_weight_diff"],
                    h["finish_position"], h["finish_position_raw"],
                    h["race_time_seconds"], h["margin_text"],
                    h.get("corner1_pos"), h.get("corner2_pos"),
                    h.get("corner3_pos"), h.get("corner4_pos"),
                    h["agari3f_seconds"], h["win_odds"], h["popular_rank"],
                ))

    def _save_past_result(self, horses: list[dict], race_info: dict) -> None:
        """past_results テーブルにも反映（特徴量計算用）。"""
        with get_conn() as conn:
            for h in horses:
                if h["finish_position"] and h["horse_id"]:
                    conn.execute("""
                        INSERT OR IGNORE INTO past_results
                          (race_id, race_date, horse_id, finish_position,
                           race_time_seconds, agari3f_seconds,
                           field_size, distance, surface, track, track_condition)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        h["race_id"], h["race_date"], h["horse_id"],
                        h["finish_position"],
                        h["race_time_seconds"], h["agari3f_seconds"],
                        race_info["field_size"], race_info["distance"],
                        race_info["surface"], race_info["track"],
                        race_info["track_condition"],
                    ))

    def _save_payouts(self, payouts: list[dict]) -> None:
        with get_conn() as conn:
            for p in payouts:
                conn.execute("""
                    INSERT OR REPLACE INTO jra_payouts
                      (race_id, race_date, bet_type, combo, payout, popular_info)
                    VALUES (?,?,?,?,?,?)
                """, (p["race_id"], p["race_date"], p["bet_type"],
                      p["combo"], p["payout"], p["popular_info"]))

    # ── 日次実行 ────────────────────────────────────
    def run_today(self, target_date: date) -> dict:
        """指定日の全JRAレースを取得。"""
        race_ids = self.fetch_race_ids(target_date)
        stats = {"total": len(race_ids), "success": 0, "failure": 0, "skipped": 0}
        for rid in race_ids:
            if self._race_exists(rid):
                stats["skipped"] += 1
                continue
            ok = self.scrape_race(rid)
            if ok: stats["success"] += 1
            else:  stats["failure"] += 1
        return stats


# ── ユーティリティ ──────────────────────────────────
def _parse_time(s: str) -> Optional[float]:
    m = re.match(r"(\d+):(\d+)\.(\d+)", s)
    if m: return int(m.group(1))*60 + int(m.group(2)) + float(f"0.{m.group(3)}")
    m2 = re.match(r"(\d+)\.(\d+)", s)
    if m2: return float(f"{m2.group(1)}.{m2.group(2)}")
    return None

def _parse_payout(s: str) -> Optional[int]:
    c = re.sub(r"[円,，\s]","",s)
    try: return int(c)
    except: return None

def _safe_int(s: str) -> Optional[int]:
    try: return int(re.sub(r"[^\d]","",s))
    except: return None

def _safe_float(s: str) -> Optional[float]:
    try: return float(re.sub(r"[^\d.]","",s))
    except: return None
