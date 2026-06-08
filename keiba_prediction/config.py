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

# ──────────────────────────────────────────────────
# 競馬場コース特性マスタ（全15場）
# ──────────────────────────────────────────────────
COURSE_FEATURES = {
    '大井':  {'straight':486,'front_bias':0.45,'circumference':1600,
              'direction':'left','slope':False,'spiral':False,
              'notes':'地方最長直線・差し届く'},
    '川崎':  {'straight':300,'front_bias':0.79,'circumference':1200,
              'direction':'left','slope':False,'spiral':False,
              'notes':'タイトコーナー・先行有利'},
    '船橋':  {'straight':362,'front_bias':0.55,'circumference':1400,
              'direction':'left','slope':False,'spiral':True,
              'notes':'スパイラルカーブ・中間的'},
    '浦和':  {'straight':220,'front_bias':0.82,'circumference':1200,
              'direction':'left','slope':False,'spiral':False,
              'notes':'極小回り・逃げ天国'},
    '名古屋':{'straight':240,'front_bias':0.72,'circumference':1180,
              'direction':'right','slope':False,'spiral':True,
              'notes':'スパイラル・先行有利'},
    '笠松':  {'straight':201,'front_bias':0.85,'circumference':1100,
              'direction':'right','slope':False,'spiral':False,
              'notes':'小回り・追込困難・内枠有利'},
    '園田':  {'straight':213,'front_bias':0.88,'circumference':1051,
              'direction':'right','slope':True,'spiral':False,
              'notes':'超小回り・逃げ先行天国・上り坂あり'},
    '姫路':  {'straight':230,'front_bias':0.75,'circumference':1200,
              'direction':'right','slope':False,'spiral':False,
              'notes':'園田よりフラット・やや差し届く'},
    '金沢':  {'straight':236,'front_bias':0.76,'circumference':1200,
              'direction':'right','slope':False,'spiral':False,
              'notes':'ポケット付き4コーナー・内回り器用さ重要'},
    '高知':  {'straight':200,'front_bias':0.80,'circumference':1100,
              'direction':'right','slope':False,'spiral':False,
              'notes':'内砂深い・先行残り・枠順重要'},
    '佐賀':  {'straight':200,'front_bias':0.78,'circumference':1100,
              'direction':'right','slope':False,'spiral':False,
              'notes':'砂粒大・パワー型有利・先行粘り'},
    '盛岡':  {'straight':400,'front_bias':0.48,'circumference':1600,
              'direction':'left','slope':True,'spiral':False,
              'notes':'地方唯一芝併設・高低差大・差し有利'},
    '水沢':  {'straight':317,'front_bias':0.65,'circumference':1200,
              'direction':'right','slope':False,'spiral':False,
              'notes':'盛岡と対照的な平坦小回り'},
    '門別':  {'straight':400,'front_bias':0.50,'circumference':1600,
              'direction':'right','slope':False,'spiral':False,
              'notes':'直線長く差し届く・外回り適性重要'},
    '帯広':  {'straight':200,'front_bias':0.60,'circumference':200,
              'direction':'straight','slope':True,'spiral':False,
              'notes':'別競技・障害2つ・重量・パワー・再加速力'},
}
