"""
netkeibaの400ブロック解除を監視して自動再開するウォッチャー。

1時間ごとにアクセステストを実行し、
解除されたら historical_scraper を自動起動する。

実行方法:
  python -m data.retry_watcher
  python -m data.retry_watcher --check-only   # 一度だけチェック
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

LOG_DIR  = Path(__file__).parent.parent.parent / "logs"
LOG_FILE = LOG_DIR / "retry_watcher.log"

TEST_URL = "https://nar.netkeiba.com/top/race_list_sub.html?kaisai_date=20230601"
CHECK_INTERVAL_SEC = 3600  # 1時間


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def check_netkeiba() -> bool:
    """netkeiba にアクセスできるか確認。True = OK。"""
    try:
        import requests
        s = requests.Session()
        s.headers["User-Agent"] = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        r = s.get(TEST_URL, timeout=15)
        if r.status_code == 200 and len(r.content) > 500:
            log(f"✅ netkeiba アクセス OK (status={r.status_code} len={len(r.content)})")
            return True
        else:
            log(f"❌ netkeiba ブロック中 (status={r.status_code})")
            return False
    except Exception as e:
        log(f"❌ netkeiba エラー: {e}")
        return False


def start_netkeiba_scraping() -> None:
    """netkeiba が解除されたら、長いインターバルで再起動。"""
    import sys
    proj = Path(__file__).parent.parent
    log("🚀 netkeiba 解除確認 → 高優先取得（NAR 2023-06〜2026-05）を開始")

    # 3秒間隔・3並列で再起動（以前より控えめ）
    cmd = [
        sys.executable, "-u", "-m", "data.historical_scraper",
        "--start", "2023-06-01",
        "--end",   "2026-05-31",
        "--workers", "2",
    ]
    log_path = LOG_DIR / f"scrape_resumed_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
    with open(log_path, "w") as lf:
        subprocess.Popen(cmd, cwd=str(proj), stdout=lf, stderr=lf)
    log(f"  プロセス起動: {' '.join(cmd)}")
    log(f"  ログ: {log_path}")


def check_and_start_keibagoujp() -> None:
    """keiba.go.jp スクレイプを開始（ブロックなし）。"""
    import sys
    proj = Path(__file__).parent.parent
    log("🌐 keiba.go.jp NAR取得を開始（ブロックなし）")

    cmd = [
        sys.executable, "-u", "-c",
        """
import sys; sys.path.insert(0, '.')
from data.nar_keibagojp_scraper import fetch_range_keibagoujp
from datetime import date
fetch_range_keibagoujp(
    start_date=date(2023, 6, 1),
    end_date=date(2026, 5, 31),
    sleep_sec=2.0, workers=2
)
print("keiba.go.jp 取得完了")
""",
    ]
    log_path = LOG_DIR / f"keibagoujp_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
    with open(log_path, "w") as lf:
        subprocess.Popen(cmd, cwd=str(proj), stdout=lf, stderr=lf)
    log(f"  ログ: {log_path}")


def run_watcher(check_only: bool = False) -> None:
    """メインループ。"""
    log("=" * 50)
    log("retry_watcher 起動")
    log(f"チェック間隔: {CHECK_INTERVAL_SEC // 60}分")
    log("=" * 50)

    # keiba.go.jp は常に使える → まず起動
    check_and_start_keibagoujp()

    if check_only:
        result = check_netkeiba()
        if result:
            start_netkeiba_scraping()
        return

    # 定期チェックループ
    while True:
        log(f"--- netkeiba チェック ({datetime.now().strftime('%H:%M')}) ---")
        if check_netkeiba():
            start_netkeiba_scraping()
            log("netkeiba 取得開始。ウォッチャーは継続監視します。")
        else:
            log(f"次回チェック: {CHECK_INTERVAL_SEC // 60}分後")
        time.sleep(CHECK_INTERVAL_SEC)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run_watcher(check_only=args.check_only)
