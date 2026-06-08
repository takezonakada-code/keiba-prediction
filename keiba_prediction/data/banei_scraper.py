"""
帯広ばんえい競馬 専用スクレイパー

keiba.go.jp / banei-keiba.or.jp からデータ取得
- 出馬表（馬体重・負担重量・騎手）
- リアルタイムオッズ（単複・3連単・3連複）
- 前5走成績
- 馬番バイアス

ばんえい競馬の特徴:
  直線200m / 障害2個 / 重そり引き
  馬体重800〜1000kg / 専用ルール
"""
from __future__ import annotations

import re
import time
from datetime import date
from typing import Optional

import requests
from bs4 import BeautifulSoup

from data.scraper import make_session
from db.database import get_conn

BASE = "https://www.keiba.go.jp"
BABA_CODE = "3"   # 帯広

# 帯広公式サイト（オッズ詳細）
BANEI_BASE = "https://banei-keiba.or.jp"

# 馬番バイアス（公式統計）
BANEI_POST_BIAS = {
    1: {"win": 9.4, "place": 19.6, "show": 30.5},
    2: {"win": 12.4,"place": 21.9, "show": 32.6},
    3: {"win": 11.6,"place": 23.3, "show": 34.2},
    4: {"win": 10.4,"place": 21.2, "show": 31.4},
    5: {"win": 11.1,"place": 21.1, "show": 32.9},
    6: {"win": 9.8, "place": 20.4, "show": 30.6},
    7: {"win": 10.7,"place": 23.2, "show": 34.8},
    8: {"win": 10.8,"place": 22.7, "show": 33.5},
    9: {"win": 10.3,"place": 20.1, "show": 29.9},
    10:{"win": 10.1,"place": 19.8, "show": 28.6},
}

BANEI_POP_BIAS = {
    1: {"win": 35.6, "show": 68.8},
    2: {"win": 19.4, "show": 53.8},
    3: {"win": 12.6, "show": 45.0},
    4: {"win": 8.2,  "show": 37.2},
    5: {"win": 5.8,  "show": 30.1},
    6: {"win": 4.1,  "show": 24.5},
    7: {"win": 3.0,  "show": 20.2},
    8: {"win": 2.1,  "show": 16.8},
    9: {"win": 1.5,  "show": 13.4},
    10:{"win": 1.0,  "show": 10.7},
}


class BaneiScraper:
    """帯広ばんえい競馬スクレイパー。"""

    def __init__(self, sleep_sec: float = 1.5):
        self.session   = make_session()
        self.sleep_sec = sleep_sec

    def _get(self, url: str) -> Optional[str]:
        try:
            time.sleep(self.sleep_sec)
            r = self.session.get(url, timeout=15)
            r.raise_for_status()
            return r.text
        except Exception as e:
            print(f"  [banei] エラー: {url[:60]}: {e}")
            return None

    # ── 出馬表取得 ─────────────────────────────────
    def fetch_race_data(self, target_date: date, race_no: int) -> dict:
        """1レース分の出馬表・オッズを取得。"""
        date_enc = target_date.strftime("%Y%%2F%m%%2F%d")
        url = f"{BASE}/KeibaWeb/TodayRaceInfo/DebaTable?k_raceDate={date_enc}&k_raceNo={race_no}&k_babaCode={BABA_CODE}"
        html = self._get(url)
        if not html:
            return {}

        soup = BeautifulSoup(html, "html.parser")
        horses = self._parse_horses(soup, race_no)
        race_info = self._parse_race_info(soup, target_date, race_no)

        return {"race_info": race_info, "horses": horses}

    def _parse_race_info(self, soup: BeautifulSoup, d: date, rno: int) -> dict:
        title = soup.find("title")
        race_name = title.get_text(strip=True) if title else ""
        # 馬場状態
        going = "良"
        for el in soup.find_all(string=re.compile("馬場")):
            m = re.search(r"馬場[:\s]*([良稍重不]+)", str(el))
            if m:
                going = m.group(1)
                break
        return {
            "race_date":  d.isoformat(),
            "race_no":    rno,
            "track":      "帯広",
            "race_name":  race_name,
            "track_condition": going,
            "race_type":  "banei",
        }

    def _parse_horses(self, soup: BeautifulSoup, race_no: int) -> list[dict]:
        horses = []
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue
            headers = [th.get_text(strip=True) for th in rows[0].find_all(["th","td"])]
            if "馬番" not in " ".join(headers) and "馬名" not in " ".join(headers):
                continue

            def ci(names):
                for n in names:
                    for i, h in enumerate(headers):
                        if n in h: return i
                return -1

            i_no   = ci(["馬番"])
            i_name = ci(["馬名"])
            i_jock = ci(["騎手"])
            i_kg   = ci(["負担重量","斤量"])
            i_hw   = ci(["馬体重"])
            i_odds = ci(["オッズ","単勝"])
            i_pop  = ci(["人気"])

            seen_no = set()
            for row in rows[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all(["td","th"])]
                if len(cells) < 2:
                    continue
                try:
                    no_str = cells[i_no] if i_no >= 0 and i_no < len(cells) else ""
                    no = int(re.sub(r"[^\d]","",no_str))
                    if no <= 0 or no in seen_no:
                        continue
                    seen_no.add(no)
                except:
                    continue

                def c(idx): return cells[idx] if 0 <= idx < len(cells) else ""

                # 馬体重（ばんえいは800〜1000kg）
                hw_text = c(i_hw)
                hw_m = re.search(r"(\d{3,4})", hw_text)
                horse_weight = int(hw_m.group(1)) if hw_m else None

                # 負担重量（ソリの重量）
                kg_text = c(i_kg)
                kg_m = re.search(r"(\d+)", kg_text)
                burden_weight = float(kg_m.group(1)) if kg_m else None

                # オッズ
                try: odds = float(c(i_odds).replace(",",""))
                except: odds = None

                # 人気
                try: pop = int(c(i_pop))
                except: pop = None

                # 馬番バイアス適用
                pp_bias = BANEI_POST_BIAS.get(no, {})
                pop_bias = BANEI_POP_BIAS.get(pop, {}) if pop else {}

                horses.append({
                    "draw_number":      no,
                    "horse_name":       c(i_name)[:20],
                    "jockey_name":      c(i_jock),
                    "horse_weight":     horse_weight,
                    "burden_weight":    burden_weight,
                    "win_odds":         odds,
                    "popular_rank":     pop,
                    "post_position_win_bias": pp_bias.get("win"),
                    "post_position_show_bias": pp_bias.get("show"),
                    "popularity_show_bias": pop_bias.get("show"),
                })
            if horses:
                break
        return horses

    # ── 3連単オッズ取得 ───────────────────────────
    def fetch_trifecta_odds(self, target_date: date, race_no: int) -> list[dict]:
        """3連単オッズTop20を取得して market_prior を計算。"""
        date_enc = target_date.strftime("%Y%%2F%m%%2F%d")
        url = f"{BASE}/KeibaWeb/TodayRaceInfo/OddsTrifecta?k_raceDate={date_enc}&k_raceNo={race_no}&k_babaCode={BABA_CODE}"
        html = self._get(url)
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        results = []
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            for row in rows[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all(["td","th"])]
                if len(cells) < 2:
                    continue
                try:
                    combo = re.findall(r"\d+", cells[0])
                    odds  = float(cells[1].replace(",",""))
                    if len(combo) == 3 and odds > 0:
                        results.append({
                            "order": [int(c) for c in combo],
                            "odds":  odds,
                        })
                except:
                    pass
        return results[:20]

    # ── 日次実行 ────────────────────────────────────
    def run_today(self, target_date: Optional[date] = None) -> dict:
        """今日の全ばんえいレースを取得してDBに保存。"""
        if target_date is None:
            target_date = date.today()

        stats = {"success": 0, "failure": 0}

        # 帯広の race_id を取得
        with get_conn() as conn:
            races = conn.execute("""
                SELECT race_id, race_no FROM nar_races
                WHERE race_date=? AND track='帯広' ORDER BY race_no
            """, (target_date.isoformat(),)).fetchall()

        for race_row in races:
            rno = race_row["race_no"]
            rid = race_row["race_id"]

            data = self.fetch_race_data(target_date, rno)
            if not data or not data.get("horses"):
                stats["failure"] += 1
                continue

            # オッズをDBに保存
            with get_conn() as conn:
                for h in data["horses"]:
                    if h.get("win_odds"):
                        conn.execute("""
                            UPDATE nar_results SET win_odds=?, popular_rank=?,
                              horse_weight=?
                            WHERE race_id=? AND draw_number=?
                        """, (h["win_odds"], h["popular_rank"], h["horse_weight"],
                              rid, h["draw_number"]))
                        conn.execute("""
                            UPDATE nar_entries SET win_odds=?, popular_rank=?,
                              horse_weight=?
                            WHERE race_id=? AND draw_number=?
                        """, (h["win_odds"], h["popular_rank"], h["horse_weight"],
                              rid, h["draw_number"]))

            stats["success"] += 1

        print(f"[banei] 取得完了: {stats['success']}R成功 / {stats['failure']}R失敗")
        return stats


# ──────────────────────────────────────────────────
# 市場prior計算（3連単オッズから）
# ──────────────────────────────────────────────────
def extract_market_prior(trifecta_list: list[dict]) -> dict[int, float]:
    """
    3連単オッズから各馬の市場確率を逆算する。

    Parameters
    ----------
    trifecta_list : [{"order": [1st,2nd,3rd], "odds": float}, ...]

    Returns
    -------
    {horse_no: probability}
    """
    scores: dict[int, float] = {}
    for entry in trifecta_list:
        order = entry["order"]
        odds  = entry["odds"]
        for pos, horse in enumerate(order):
            weight = 1.0 / (odds * (pos + 1))   # 着順後方ほど重みを下げる
            scores[horse] = scores.get(horse, 0.0) + weight

    total = sum(scores.values())
    if total == 0:
        return {}
    return {h: round(s / total, 4) for h, s in sorted(scores.items(), key=lambda x: -x[1])}


# ──────────────────────────────────────────────────
# ばんえい専用スコア計算
# ──────────────────────────────────────────────────
def banei_horse_score(
    draw_number: int,
    popular_rank: Optional[int],
    horse_weight: Optional[float],
    burden_weight: Optional[float],
    win_odds: Optional[float],
) -> float:
    """
    ばんえい専用の総合スコアを0〜1で返す。

    Components:
    - 馬番バイアス (30%)
    - 人気バイアス (30%)
    - 市場オッズ (40%)
    """
    score = 0.0

    # 馬番バイアス（3着内率を正規化）
    pp_bias = BANEI_POST_BIAS.get(draw_number, {})
    pp_show = pp_bias.get("show", 30.0) / 100.0
    score += pp_show * 0.30

    # 人気バイアス
    if popular_rank:
        pop_bias = BANEI_POP_BIAS.get(min(popular_rank, 10), {})
        pop_show = pop_bias.get("show", 10.0) / 100.0
        score += pop_show * 0.30
    else:
        score += 0.10

    # 市場オッズ（低いほど高スコア）
    if win_odds and win_odds > 0:
        market_score = 1.0 / win_odds
        score += market_score * 0.40
    else:
        score += 0.02

    return round(score, 4)
