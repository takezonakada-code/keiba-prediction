"""
設定・定数
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "db" / "keiba.db"

# JRA・地方競馬払戻率（控除後払戻率）
PAYBACK_RATE = {
    "JRA":  0.75,   # JRA 3連複控除率25%
    "TCK":  0.725,  # 大井(TCK)
    "KAWASAKI": 0.725,
    "FUNABASHI": 0.725,
    "URAWA": 0.725,
    "NAGOYA": 0.70,
    "KASAEMATSU": 0.70,
    "SONODA": 0.70,
    "KOCHI": 0.70,
    "SAGA": 0.70,
    "KANAZAWA": 0.70,
    "MORIOKA": 0.70,
    "MIZUSAWA": 0.70,
    "OBIHIRO": 0.70,  # ばんえい
}

# 距離帯定義
DISTANCE_BANDS = {
    "short":        (0, 1400),
    "mile_middle":  (1401, 2000),
    "long":         (2001, 9999),
}

# Ranker目的変数
RELEVANCE_MAP = {1: 3, 2: 2, 3: 1}  # 4着以下=0

# Half-Kelly上限
KELLY_CAP_PER_RACE = 0.01   # 1レース1%

# walk-forward設定
WF_MIN_TRAIN_MONTHS = 24
WF_TEST_MONTHS = 1
WF_EMBARGO_DAYS = 7

# スクレイピング設定
SCRAPER_RETRY_TOTAL = 3
SCRAPER_BACKOFF = 1.0
SCRAPER_STATUS_FORCELIST = [429, 500, 502, 503, 504]

# Slack通知
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")

# PSIアラート
PSI_ALERT_THRESHOLD = 0.25   # ベースライン測定後に調整
LOGLOSS_DEGRADATION_RATIO = 1.10   # 8週平均比+10%
EV_MIN_THRESHOLD = 0.02
SCRAPING_SUCCESS_MIN = 0.95

# RBF centers（出走間隔）
REST_RBF_CENTERS = [7, 14, 21, 35, 56, 84, 140, 210]
REST_RBF_SIGMA = 14.0

# 脚質加重
STYLE_WEIGHTS = [0.5, 0.3, 0.2]  # 直近→古い順

# モデル保存ディレクトリ
MODEL_DIR = BASE_DIR / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
