#!/bin/zsh
# 30分ごとにデータ取得進捗を確認し、
# nar_races が 50,000件以上になったら学習パイプラインを自動実行する。
# 完了フラグが立ったら終了する。

export PATH="/Users/takezo/miniforge3/bin:/usr/local/bin:/usr/bin:/bin"
export HOME="/Users/takezo"
export LANG="ja_JP.UTF-8"

LOG_DIR="$HOME/競馬_3連複/logs"
LOG="$LOG_DIR/training_pipeline.log"
PROJ="$HOME/競馬_3連複/keiba_prediction"
PYTHON="/Users/takezo/miniforge3/bin/python"
FLAG="$PROJ/models/trained/model_ready.flag"
INTERVAL=1800  # 30分

mkdir -p "$LOG_DIR" "$PROJ/models/trained"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "$(ts) [monitor] 監視開始 (確認間隔: ${INTERVAL}秒)" | tee -a "$LOG"

while true; do
    # ── 既に完了済みなら終了 ──────────────────────────
    if [ -f "$FLAG" ]; then
        echo "$(ts) [monitor] model_ready.flag 検出 → 監視終了" | tee -a "$LOG"
        exit 0
    fi

    # ── nar_races 件数確認 ───────────────────────────
    COUNT=$("$PYTHON" -c "
import sys; sys.path.insert(0,'$PROJ')
from db.database import get_conn
with get_conn() as conn:
    r = conn.execute('SELECT COUNT(*) as c FROM nar_races').fetchone()
    print(r['c'])
" 2>/dev/null)

    DONE=$("$PYTHON" -c "
import sys; sys.path.insert(0,'$PROJ')
from db.database import get_conn
with get_conn() as conn:
    r = conn.execute(\"SELECT COUNT(*) as c FROM historical_progress WHERE status='done'\").fetchone()
    print(r['c'])
" 2>/dev/null)

    LATEST=$("$PYTHON" -c "
import sys; sys.path.insert(0,'$PROJ')
from db.database import get_conn
with get_conn() as conn:
    r = conn.execute('SELECT race_date FROM historical_progress ORDER BY race_date DESC LIMIT 1').fetchone()
    print(r['race_date'] if r else '-')
" 2>/dev/null)

    echo "$(ts) [monitor] nar_races=${COUNT}件 | 完了日数=${DONE} | 最終日=${LATEST}" | tee -a "$LOG"

    # ── 50,000件以上 → 学習パイプライン起動 ──────────
    if [ "${COUNT:-0}" -ge 50000 ]; then
        echo "$(ts) [monitor] 50,000件達成 → 学習パイプラインを開始します" | tee -a "$LOG"
        cd "$PROJ"
        "$PYTHON" -u -m pipeline.train_pipeline >> "$LOG" 2>&1
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 0 ]; then
            echo "$(ts) [monitor] 学習パイプライン完了 (exit:0)" | tee -a "$LOG"
        else
            echo "$(ts) [monitor] 学習パイプライン失敗 (exit:$EXIT_CODE)" | tee -a "$LOG"
        fi
        exit $EXIT_CODE
    fi

    # ── まだ足りない → 待機 ─────────────────────────
    NEEDED=$((50000 - ${COUNT:-0}))
    echo "$(ts) [monitor] あと約${NEEDED}件 → ${INTERVAL}秒後に再確認" | tee -a "$LOG"
    sleep $INTERVAL
done
