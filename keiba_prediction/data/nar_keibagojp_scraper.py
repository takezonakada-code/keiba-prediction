"""
NAR公式サイト（keiba.go.jp）スクレイパー

netkeibaが400ブロックされている場合の代替取得手段。
keiba.go.jp はネットケイバとは別サーバーのため400なし。

取得URL:
  開催日程: /KeibaWeb/MonthlyConveneInfo/MonthlyConveneInfoTop?k_year=YYYY&k_month=M
  レース一覧: /KeibaWeb/TodayRaceInfo/RaceList?k_raceDate=YYYY%2FMM%2FDD&k_babaCode=N
  成績: /KeibaWeb/TodayRaceInfo/RaceMarkTable?k_raceDate=...&k_raceNo=N&k_babaCode=N
  出馬表: /KeibaWeb/TodayRaceInfo/DebaTable?k_raceDate=...&k_raceNo=N&k_babaCode=N
"""
from __future__ import annotations

import re
import time
import queue
import threading
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup

from data.scraper import make_session
from db.database import get_conn

BASE = "https://www.keiba.go.jp"

# keiba.go.jp の場コード → 場名・払戻率
# ※ netkeiba の場コードとは異なる
BABA_CODE_MAP = {
    "3":  {"name": "帯広",   "payback": 0.70, "banei": True},
    "10": {"name": "大井",   "payback": 0.725},
    "11": {"name": "川崎",   "payback": 0.725},
    "18": {"name": "船橋",   "payback": 0.725},
    "19": {"name": "浦和",   "payback": 0.725},
    "20": {"name": "盛岡",   "payback": 0.70},
    "21": {"name": "水沢",   "payback": 0.70},
    "22": {"name": "金沢",   "payback": 0.70},
    "23": {"name": "笠松",   "payback": 0.70},
    "24": {"name": "名古屋", "payback": 0.70},
    "27": {"name": "園田",   "payback": 0.70},
    "28": {"name": "姫路",   "payback": 0.70},
    "31": {"name": "高知",   "payback": 0.70},
    "32": {"name": "佐賀",   "payback": 0.70},
    "36": {"name": "門別",   "payback": 0.70},
}


class NARKeibaGojpScraper:
    """
    keiba.go.jp からNARデータを取得する。
    400ブロックなし・公式サイト・無料。
    """

    def __init__(self, sleep_sec: float = 2.0, max_retry: int = 3):
        self.session   = make_session()
        self.sleep_sec = sleep_sec
        self.max_retry = max_retry
        self._success  = 0
        self._failure  = 0

    @property
    def success_rate(self) -> float:
        total = self._success + self._failure
        return self._success / total if total > 0 else 1.0

    def _get(self, url: str, wait_extra: float = 0) -> Optional[str]:
        """GETリクエスト（リトライ付き）。"""
        for attempt in range(self.max_retry):
            try:
                time.sleep(self.sleep_sec + wait_extra + attempt * 1.0)
                resp = self.session.get(url, timeout=20)
                resp.raise_for_status()
                resp.encoding = "utf-8"
                self._success += 1
                return resp.text
            except Exception as e:
                self._failure += 1
                print(f"  [retry {attempt+1}/{self.max_retry}] {url[:80]}: {e}")
        return None

    # ──────────────────────────────────────────────
    # 月次開催日程の取得
    # ──────────────────────────────────────────────
    def fetch_month_schedule(self, year: int, month: int) -> list[tuple[str, str]]:
        """
        指定月の全開催日×場コードを返す。
        Returns: [(date_str "YYYY/MM/DD", baba_code), ...]
        """
        url = f"{BASE}/KeibaWeb/MonthlyConveneInfo/MonthlyConveneInfoTop?k_year={year}&k_month={month}"
        html = self._get(url)
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        schedule = []
        seen = set()

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "RaceList" not in href:
                continue
            m_date = re.search(r"k_raceDate=([^&]+)", href)
            m_code = re.search(r"k_babaCode=(\d+)", href)
            if not m_date or not m_code:
                continue
            race_date = m_date.group(1).replace("%2F", "/").replace("%20", "")
            baba_code = m_code.group(1)

            if baba_code not in BABA_CODE_MAP:
                continue
            key = (race_date, baba_code)
            if key not in seen:
                schedule.append(key)
                seen.add(key)

        return sorted(schedule)

    # ──────────────────────────────────────────────
    # 1日分の取得
    # ──────────────────────────────────────────────
    def run_date(
        self,
        target_date: date,
        baba_codes: Optional[list[str]] = None,
    ) -> dict:
        """
        指定日の全NAR開催レースを取得してDBに保存。

        Parameters
        ----------
        target_date : 対象日
        baba_codes  : 場コードのリスト（None = 全場）
        """
        date_str_slash = target_date.strftime("%Y/%m/%d")
        date_str_dash  = target_date.strftime("%Y-%m-%d")
        year, month    = target_date.year, target_date.month

        # 開催日程から当日の場コードを取得
        schedule = self.fetch_month_schedule(year, month)
        day_codes = [
            bc for d, bc in schedule
            if d == date_str_slash and (baba_codes is None or bc in baba_codes)
        ]

        if not day_codes:
            return {"total": 0, "success": 0, "failure": 0, "skipped": 0}

        stats = {"total": 0, "success": 0, "failure": 0, "skipped": 0}

        for baba_code in day_codes:
            race_ids = self._get_race_ids(date_str_slash, baba_code)
            stats["total"] += len(race_ids)

            for race_id in race_ids:
                if self._race_exists(race_id):
                    stats["skipped"] += 1
                    continue
                ok = self._scrape_race(date_str_slash, baba_code, race_id)
                if ok:
                    stats["success"] += 1
                else:
                    stats["failure"] += 1

        if stats["success"] > 0:
            self._update_progress(date_str_dash, stats)

        return stats

    def _get_race_ids(self, date_slash: str, baba_code: str) -> list[str]:
        """RaceList から race_id (内部キー) のリストを返す。"""
        date_enc = date_slash.replace("/", "%2F")
        url = f"{BASE}/KeibaWeb/TodayRaceInfo/RaceList?k_raceDate={date_enc}&k_babaCode={baba_code}"
        html = self._get(url)
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        race_nos = []
        seen = set()

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "RaceMarkTable" not in href and "DebaTable" not in href:
                continue
            m_no = re.search(r"k_raceNo=(\d+)", href)
            if m_no:
                rno = int(m_no.group(1))
                if rno not in seen:
                    race_nos.append(rno)
                    seen.add(rno)

        return [
            self._build_race_id(date_slash, baba_code, rno)
            for rno in sorted(race_nos)
        ]

    def _build_race_id(self, date_slash: str, baba_code: str, race_no: int) -> str:
        """内部 race_id を生成。例: 2023-06-10_27_12"""
        date_dash = date_slash.replace("/", "-")
        return f"{date_dash}_{baba_code}_{race_no:02d}"

    def _scrape_race(self, date_slash: str, baba_code: str, race_id: str) -> bool:
        """1レースの成績を取得してDBに保存。"""
        race_no = int(race_id.split("_")[-1])
        date_enc = date_slash.replace("/", "%2F")
        url = f"{BASE}/KeibaWeb/TodayRaceInfo/RaceMarkTable?k_raceDate={date_enc}&k_raceNo={race_no}&k_babaCode={baba_code}"

        html = self._get(url, wait_extra=0.5)
        if not html:
            return False

        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")

        if not tables or "情報がありません" in tables[0].get_text():
            # 成績なし → 出馬表から試みる
            return self._scrape_shutuba(date_slash, baba_code, race_id, race_no)

        try:
            track_info = BABA_CODE_MAP.get(baba_code, {})
            track_name  = track_info.get("name", f"場{baba_code}")
            payback     = track_info.get("payback", 0.70)
            is_banei    = track_info.get("banei", False)
            race_type   = "banei" if is_banei else "flat"
            date_dash   = date_slash.replace("/", "-")

            race_info = self._parse_race_info(soup, date_dash, baba_code, race_no, track_name, race_type, payback)
            horses    = self._parse_horses(soup, race_id, date_dash, race_type)
            payouts   = self._parse_payouts(soup, race_id, date_dash)

            self._save_race(race_info)
            self._save_horses(horses)
            if payouts:
                self._save_payouts(payouts)

            return True
        except Exception as e:
            print(f"  [keibagoJP] {race_id} パースエラー: {e}")
            return False

    def _scrape_shutuba(self, date_slash: str, baba_code: str, race_id: str, race_no: int) -> bool:
        """出馬表ページから取得（成績未確定の場合）。"""
        date_enc = date_slash.replace("/", "%2F")
        url = f"{BASE}/KeibaWeb/TodayRaceInfo/DebaTable?k_raceDate={date_enc}&k_raceNo={race_no}&k_babaCode={baba_code}"
        html = self._get(url)
        if not html:
            return False

        soup = BeautifulSoup(html, "html.parser")
        if "情報がありません" in soup.get_text():
            return False

        track_info = BABA_CODE_MAP.get(baba_code, {})
        track_name = track_info.get("name", f"場{baba_code}")
        payback    = track_info.get("payback", 0.70)
        is_banei   = track_info.get("banei", False)
        race_type  = "banei" if is_banei else "flat"
        date_dash  = date_slash.replace("/", "-")

        try:
            race_info = self._parse_race_info(soup, date_dash, baba_code, race_no, track_name, race_type, payback)
            self._save_race(race_info)
            return True
        except Exception:
            return False

    # ──────────────────────────────────────────────
    # パーサー
    # ──────────────────────────────────────────────
    def _parse_race_info(self, soup, date_dash, baba_code, race_no, track_name, race_type, payback):
        # タイトルからレース名・クラスを取得
        title_el = soup.find("h1") or soup.find(class_="title") or soup.find("caption")
        race_name = title_el.get_text(strip=True) if title_el else ""

        # テーブルのヘッダー行からコース・天気・馬場を取得
        tables = soup.find_all("table")
        surface = "ダート"
        course_dir = ""
        going = "良"
        weather = ""
        field_size = 0
        race_class = ""
        post_time = ""
        distance = 0

        # table[0] の行からレース情報を探す
        for t in tables:
            rows = t.find_all("tr")
            if len(rows) >= 2:
                headers = [th.get_text(strip=True) for th in rows[0].find_all(["th","td"])]
                if "着順" in headers or "競走" in headers:
                    # ヘッダー行でコース・天気を探す
                    for i, h in enumerate(headers):
                        if "コース" in h or "距離" in h:
                            continue
                    # 列値から情報抽出
                    if len(rows) > 1:
                        data_cells = [td.get_text(strip=True) for td in rows[1].find_all(["th","td"])]
                        field_size = len(rows) - 1  # 着順行数
                        break

        # レース一覧テーブルからの情報（ヘッダー）
        for t in tables:
            header_text = t.get_text()
            # 発走時刻
            m_time = re.search(r"(\d{1,2}:\d{2})", header_text)
            if m_time:
                post_time = m_time.group(1)
            # 距離（帯広は "直200m"）
            m_dist = re.search(r"(\d{3,4})m", header_text)
            if m_dist:
                distance = int(m_dist.group(1))
            # 天候
            m_weather = re.search(r"(晴|曇|雨|雪|小雨|霧)", header_text)
            if m_weather:
                weather = m_weather.group(1)
            # 馬場状態（ばんえいは数値）
            m_going = re.search(r"馬場[:\s]*(良|稍重|稍|重|不良)", header_text)
            if m_going:
                going = m_going.group(1)

        dist_band = "short" if distance <= 1400 else "mile_middle" if distance <= 2000 else "long"

        return {
            "race_id":    f"kg_{date_dash.replace('-','')}_{baba_code}_{race_no:02d}",
            "race_date":  date_dash,
            "track":      track_name,
            "track_code": baba_code,
            "race_no":    race_no,
            "race_name":  race_name,
            "surface":    surface,
            "distance":   distance,
            "distance_band": dist_band,
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

    def _parse_horses(self, soup, race_id, date_dash, race_type):
        horses = []
        tables = soup.find_all("table")
        if not tables:
            return horses

        # 着順テーブルを特定（ヘッダーに「着順」がある）
        result_table = None
        for t in tables:
            rows = t.find_all("tr")
            if rows:
                headers = [th.get_text(strip=True) for th in rows[0].find_all(["th","td"])]
                if "着順" in headers:
                    result_table = t
                    break

        if not result_table:
            return horses

        rows = result_table.find_all("tr")
        headers = [th.get_text(strip=True) for th in rows[0].find_all(["th","td"])]

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
        i_kg    = ci(["積載", "斤量"])
        i_jock  = ci(["騎手"])
        i_train = ci(["調教師"])
        i_hw    = ci(["馬体重"])
        i_time  = ci(["タイム"])
        i_agari = ci(["上がり3F", "上り3F"])
        i_pop   = ci(["人気"])
        i_odds  = ci(["単勝オッズ", "オッズ"])

        for row in rows[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all(["td","th"])]
            if len(cells) < 3:
                continue

            def c(idx): return cells[idx] if 0 <= idx < len(cells) else ""

            fin_str = c(i_fin)
            try: fin_pos = int(fin_str)
            except: fin_pos = None

            hw_text = c(i_hw)
            hw_m = re.search(r"(\d{3,4})\(([+\-]\d+)\)", hw_text)
            horse_weight      = int(hw_m.group(1)) if hw_m else None
            horse_weight_diff = int(hw_m.group(2)) if hw_m else None

            agari_str = c(i_agari)
            try: agari_sec = float(agari_str)
            except: agari_sec = None

            race_time_sec = _parse_time(c(i_time))

            try: win_odds = float(c(i_odds).replace(",",""))
            except: win_odds = None

            try: pop_rank = int(c(i_pop))
            except: pop_rank = None

            sa = c(i_sex)
            sex = sa[0] if sa else ""
            age_m = re.search(r"(\d+)$", sa)
            age = int(age_m.group(1)) if age_m else None

            draw_no = _safe_int(c(i_num))
            if draw_no is None:
                continue

            horses.append({
                "race_id":          race_id,
                "race_date":        date_dash,
                "horse_id":         "",
                "horse_name":       c(i_name).split("\n")[0].strip(),
                "draw_number":      draw_no,
                "frame_number":     _safe_int(c(i_waku)),
                "jockey_id":        "",
                "jockey_name":      c(i_jock),
                "trainer_id":       "",
                "trainer_name":     c(i_train),
                "sex":              sex,
                "age":              age,
                "weight_carried":   _safe_float(c(i_kg)),
                "horse_weight":     horse_weight,
                "horse_weight_diff": horse_weight_diff,
                "finish_position":  fin_pos,
                "race_time_seconds": race_time_sec,
                "agari3f_seconds":  agari_sec,
                "win_odds":         win_odds,
                "popular_rank":     pop_rank,
                "race_type":        race_type,
            })

        return horses

    def _parse_payouts(self, soup, race_id, date_dash):
        payouts = []
        BET_MAP = {
            "単勝": "win", "複勝": "place",
            "馬連単": "quinella_exacta", "ワイド": "wide",
            "馬単": "exacta", "3連複": "trio", "3連単": "trifecta",
        }
        for t in soup.find_all("table"):
            rows = t.find_all("tr")
            if not rows:
                continue
            first_text = rows[0].get_text(strip=True)
            bet_type = BET_MAP.get(first_text)
            if not bet_type:
                continue

            for row in rows:
                cells = [td.get_text(strip=True) for td in row.find_all(["td","th"])]
                if len(cells) < 2:
                    continue
                combo = cells[0].strip()
                pay_str = cells[1].strip() if len(cells) > 1 else ""

                if not re.search(r"\d", combo):
                    continue
                payout = _parse_payout(pay_str)
                if payout is None:
                    continue

                combo_normalized = re.sub(r"[－]", "-", combo)
                payouts.append({
                    "race_id":    race_id,
                    "race_date":  date_dash,
                    "bet_type":   bet_type,
                    "combo":      combo_normalized,
                    "payout":     payout,
                    "popular_info": cells[2] if len(cells) > 2 else "",
                })

        return payouts

    # ──────────────────────────────────────────────
    # DB操作
    # ──────────────────────────────────────────────
    def _race_exists(self, race_id: str) -> bool:
        with get_conn() as conn:
            return conn.execute(
                "SELECT 1 FROM nar_races WHERE race_id=?", (race_id,)
            ).fetchone() is not None

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
                conn.execute("""
                    INSERT OR REPLACE INTO nar_results
                      (race_id, race_date, horse_id, horse_name,
                       draw_number, jockey_name, trainer_name,
                       sex, age, weight_carried, horse_weight, horse_weight_diff,
                       finish_position, race_time_seconds, agari3f_seconds,
                       win_odds, popular_rank, race_type)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    h["race_id"], h["race_date"], h["horse_id"], h["horse_name"],
                    h["draw_number"], h["jockey_name"], h["trainer_name"],
                    h["sex"], h["age"], h["weight_carried"],
                    h["horse_weight"], h["horse_weight_diff"],
                    h["finish_position"], h["race_time_seconds"],
                    h["agari3f_seconds"], h["win_odds"], h["popular_rank"],
                    h["race_type"],
                ))
                # past_resultsにも反映
                if h["finish_position"]:
                    conn.execute("""
                        INSERT OR IGNORE INTO past_results
                          (race_id, race_date, horse_id, finish_position,
                           race_time_seconds, agari3f_seconds)
                        VALUES (?,?,?,?,?,?)
                    """, (h["race_id"], h["race_date"], h["horse_id"] or h["horse_name"],
                          h["finish_position"], h["race_time_seconds"], h["agari3f_seconds"]))

    def _save_payouts(self, payouts: list[dict]) -> None:
        with get_conn() as conn:
            for p in payouts:
                conn.execute("""
                    INSERT OR REPLACE INTO nar_payouts
                      (race_id, race_date, bet_type, combo, payout, popular_info)
                    VALUES (?,?,?,?,?,?)
                """, (p["race_id"], p["race_date"], p["bet_type"],
                      p["combo"], p["payout"], p["popular_info"]))

    def _update_progress(self, date_dash: str, stats: dict) -> None:
        with get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO scrape_progress
                  (source, race_date, track, status, races_fetched)
                VALUES ('NAR_KEIBAGOUJP', ?, 'ALL', 'done', ?)
            """, (date_dash, stats.get("success", 0)))


# ────────────────────────────────────────────────
# バッチ取得関数
# ────────────────────────────────────────────────
def fetch_range_keibagoujp(
    start_date: date,
    end_date:   date,
    sleep_sec:  float = 2.0,
    workers:    int   = 2,
) -> None:
    """
    指定期間のNARデータをkeiba.go.jpから取得する。
    """
    scraper = NARKeibaGojpScraper(sleep_sec=sleep_sec)

    # 月次スケジュールをキャッシュ
    schedule_cache: dict[tuple, list] = {}
    cur = date(start_date.year, start_date.month, 1)
    print(f"=== keiba.go.jp NAR取得: {start_date}〜{end_date} ===")
    print(f"    {workers}並列 / {sleep_sec}秒待機")

    # 対象日リストを構築
    all_tasks: list[date] = []
    target = start_date
    while target <= end_date:
        # scrape_progressで取得済みかチェック
        with get_conn() as conn:
            done = conn.execute("""
                SELECT 1 FROM scrape_progress
                WHERE source='NAR_KEIBAGOUJP' AND race_date=? AND status='done'
            """, (target.isoformat(),)).fetchone()
        if not done:
            all_tasks.append(target)
        target += timedelta(days=1)

    print(f"    対象: {len(all_tasks)}日 / スキップ: {(end_date-start_date).days+1-len(all_tasks)}日")

    dq: queue.Queue[date] = queue.Queue()
    for d in all_tasks: dq.put(d)

    lock    = threading.Lock()
    cnt     = {"done": 0, "fail": 0, "total": len(all_tasks)}
    t_start = time.time()

    def worker(wid: int):
        sc = NARKeibaGojpScraper(sleep_sec=sleep_sec)
        while True:
            try: target = dq.get_nowait()
            except queue.Empty: break

            try:
                stats = sc.run_date(target)
                with lock:
                    cnt["done"] += 1
                    done  = cnt["done"]
                    total = cnt["total"]
                    elapsed = time.time() - t_start
                    rate  = done / elapsed * 60 if elapsed > 0 else 0
                    eta   = (total-done)/(done/elapsed)/60 if done > 0 else 0
                    print(f"  [W{wid}] {target} ok={stats['success']} "
                          f"({done}/{total}) {rate:.1f}日/分 残~{eta:.0f}分")
            except Exception as e:
                with lock: cnt["fail"] += 1
                print(f"  [W{wid}] {target} エラー: {e}")

            dq.task_done()

    threads = []
    for i in range(workers):
        t = threading.Thread(target=worker, args=(i,), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.5)
    for t in threads: t.join()

    elapsed = (time.time() - t_start) / 60
    print(f"\n=== 完了: {cnt['done']}日/{cnt['fail']}日失敗 ({elapsed:.1f}分) ===")
    print(f"    成功率: {scraper.success_rate:.1%}")


# ────────────────────────────────────────────────
# ユーティリティ
# ────────────────────────────────────────────────
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
