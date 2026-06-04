"""
netkeiba スクレイパー。
静的HTML → requests+BeautifulSoup（高速）
JS必要ページ → Playwright（必要最小限）
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    SCRAPER_BACKOFF,
    SCRAPER_RETRY_TOTAL,
    SCRAPER_STATUS_FORCELIST,
)


# ────────────────────────────────────────────────
# セッション設定
# ────────────────────────────────────────────────
def make_session(headers: dict | None = None) -> requests.Session:
    """リトライ付きHTTPセッションを生成。"""
    session = requests.Session()
    retry = Retry(
        total=SCRAPER_RETRY_TOTAL,
        backoff_factor=SCRAPER_BACKOFF,
        status_forcelist=SCRAPER_STATUS_FORCELIST,
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    default_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ja,en-US;q=0.9",
    }
    if headers:
        default_headers.update(headers)
    session.headers.update(default_headers)
    return session


# ────────────────────────────────────────────────
# HTMLシグネチャ（テーブル構造変化検知）
# ────────────────────────────────────────────────
def html_signature(html: str, selector: str = "table") -> str:
    """
    HTMLからテーブル構造のハッシュを計算。
    構造変化検知に使う。
    """
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.select(selector)
    sig = "".join(
        str([th.get_text(strip=True) for th in t.select("th")])
        for t in tables
    )
    return hashlib.md5(sig.encode()).hexdigest()


# ────────────────────────────────────────────────
# 静的HTML取得
# ────────────────────────────────────────────────
class NetkeibaStaticScraper:
    """
    静的HTMLページのスクレイパー。
    レース一覧・出馬表など構造が安定したページに使う。
    """

    BASE = "https://race.netkeiba.com"

    def __init__(self):
        self.session = make_session()
        self._sig_cache: dict[str, str] = {}
        self._success = 0
        self._failure = 0

    @property
    def success_rate(self) -> float:
        total = self._success + self._failure
        return self._success / total if total > 0 else 1.0

    def fetch(self, url: str, sleep: float = 1.0) -> str | None:
        """HTMLを取得して返す。失敗時は None。"""
        time.sleep(sleep)
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding
            self._success += 1
            return resp.text
        except Exception as e:
            self._failure += 1
            print(f"[Scraper] 取得失敗: {url} → {e}")
            return None

    def fetch_with_structure_check(
        self, url: str, sig_key: str, sleep: float = 1.0
    ) -> tuple[str | None, bool]:
        """
        HTML取得 + テーブル構造変化検知。

        Returns
        -------
        (html, structure_changed)
        """
        html = self.fetch(url, sleep=sleep)
        if html is None:
            return None, False

        sig = html_signature(html)
        changed = self._sig_cache.get(sig_key) not in (None, sig)
        if changed:
            print(f"[構造変化] {sig_key}: {self._sig_cache.get(sig_key)} → {sig}")
        self._sig_cache[sig_key] = sig
        return html, changed

    def parse_race_entry_table(self, html: str) -> list[dict[str, Any]]:
        """
        出馬表テーブルをパースして出走馬リストを返す。

        Returns
        -------
        list of {draw_number, horse_name, horse_id, jockey_id, ...}
        """
        soup = BeautifulSoup(html, "html.parser")
        table = soup.select_one("table.Shutuba_Table") or soup.select_one("table")
        if table is None:
            return []

        rows = table.select("tr.HorseList, tr[class*='HorseList']")
        entries = []
        for row in rows:
            cells = row.select("td")
            if len(cells) < 5:
                continue
            entry: dict[str, Any] = {}
            try:
                entry["draw_number"] = int(cells[1].get_text(strip=True))
                entry["horse_name"]  = cells[3].get_text(strip=True)
                horse_link = cells[3].select_one("a")
                if horse_link and horse_link.get("href"):
                    entry["horse_id"] = horse_link["href"].split("/")[-2]
                entry["jockey_id"]   = cells[6].get_text(strip=True)
                entry["horse_weight"] = cells[8].get_text(strip=True)
            except (IndexError, ValueError):
                continue
            entries.append(entry)

        # 取消・除外馬の検知
        scratch_rows = soup.select("tr.Cancel_Row, span.Scratch")
        if scratch_rows:
            scratch_names = {r.get_text(strip=True) for r in scratch_rows}
            entries = [e for e in entries if e.get("horse_name") not in scratch_names]
            if scratch_names:
                print(f"[取消/除外] {scratch_names}")

        return entries


# ────────────────────────────────────────────────
# Playwright（JS必要ページ）
# ────────────────────────────────────────────────
class NetkeibaPlaywrightScraper:
    """
    JSレンダリングが必要なページ（オッズ画面等）のスクレイパー。
    遅いので必要最小限の利用に留める。
    """

    def __init__(self):
        self._playwright = None
        self._browser = None

    def start(self) -> None:
        from playwright.sync_api import sync_playwright
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)

    def stop(self) -> None:
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

    def fetch_odds(self, race_id: str) -> dict[str, float]:
        """
        レースIDから3連複オッズテーブルを取得。
        Returns {combo_str: odds}
        """
        if self._browser is None:
            self.start()

        url = f"https://race.netkeiba.com/odds/index.html?race_id={race_id}&type=b8"
        page = self._browser.new_page()
        try:
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.wait_for_selector("#odds-fukusho-table", timeout=10000)
            html = page.content()
        except Exception as e:
            print(f"[Playwright] オッズ取得失敗: {race_id} → {e}")
            return {}
        finally:
            page.close()

        return self._parse_trio_odds(html)

    def _parse_trio_odds(self, html: str) -> dict[str, float]:
        soup = BeautifulSoup(html, "html.parser")
        odds_map: dict[str, float] = {}
        for row in soup.select("tr[id]"):
            cells = row.select("td")
            if len(cells) < 2:
                continue
            try:
                combo = cells[0].get_text(strip=True)
                odds  = float(cells[1].get_text(strip=True).replace(",", ""))
                odds_map[combo] = odds
            except ValueError:
                continue
        return odds_map

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
