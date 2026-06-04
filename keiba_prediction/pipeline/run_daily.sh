#!/bin/zsh
# 毎朝5:30 launchdから呼ばれるエントリポイント
# PATH を明示的に設定（launchdは通常のPATHを引き継がない）

export PATH="/Users/takezo/miniforge3/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/Library/Developer/CommandLineTools/usr/bin"
export HOME="/Users/takezo"
export LANG="ja_JP.UTF-8"

# ログ出力先
LOG_DIR="$HOME/競馬_3連複/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily_$(date +%Y%m%d).log"

echo "=== $(date '+%Y-%m-%d %H:%M:%S') 開始 ===" >> "$LOG_FILE"

# プロジェクトディレクトリへ移動
cd "$HOME/競馬_3連複/keiba_prediction" || exit 1

# Python実行（miniforge3 の Python を使用）
/Users/takezo/miniforge3/bin/python -m pipeline.daily_update >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
echo "=== $(date '+%Y-%m-%d %H:%M:%S') 終了 (exit: $EXIT_CODE) ===" >> "$LOG_FILE"

# 古いログを30日分だけ保持
find "$LOG_DIR" -name "daily_*.log" -mtime +30 -delete

exit $EXIT_CODE
