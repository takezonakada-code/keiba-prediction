#!/bin/zsh
# 毎晩23:00 launchdから呼ばれる – 当日の全レース結果を取得する

export PATH="/Users/takezo/miniforge3/bin:/usr/local/bin:/usr/bin:/bin"
export HOME="/Users/takezo"
export LANG="ja_JP.UTF-8"

LOG_DIR="$HOME/競馬_3連複/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/fetch_results_$(date +%Y%m%d).log"

echo "=== $(date '+%Y-%m-%d %H:%M:%S') 結果取得開始 ===" >> "$LOG"
cd "$HOME/競馬_3連複/keiba_prediction" || exit 1

/Users/takezo/miniforge3/bin/python -u -m pipeline.fetch_results >> "$LOG" 2>&1

echo "=== $(date '+%Y-%m-%d %H:%M:%S') 完了 ===" >> "$LOG"
