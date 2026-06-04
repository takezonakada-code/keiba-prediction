"""
特徴量ごとに「発走何秒前に確定するか」を定義。
available_before_post_sec < 0 = レース後確定 = 使用禁止
"""

FEATURE_SPECS: dict[str, dict] = {
    # ===== 使用可能（レース前確定） =====
    "horse_weight":              {"available_sec": 3600,  "ok": True,  "note": "馬体重（発表は1時間前）"},
    "horse_weight_diff":         {"available_sec": 3600,  "ok": True,  "note": "前走比増減"},
    "weather":                   {"available_sec": 120,   "ok": True,  "note": "天気（直前確定）"},
    "track_condition":           {"available_sec": 120,   "ok": True,  "note": "馬場状態（直前確定）"},
    "live_popular_rank":         {"available_sec": 120,   "ok": True,  "note": "ライブ人気順位"},
    "win_odds_live":             {"available_sec": 120,   "ok": True,  "note": "ライブ単勝オッズ"},
    "draw_number":               {"available_sec": 86400, "ok": True,  "note": "枠番・馬番"},
    "field_size":                {"available_sec": 86400, "ok": True,  "note": "出走頭数"},
    "jockey_id":                 {"available_sec": 86400, "ok": True,  "note": "騎手ID"},
    "trainer_id":                {"available_sec": 86400, "ok": True,  "note": "調教師ID"},
    "horse_age":                 {"available_sec": 86400, "ok": True,  "note": "馬齢"},
    "sex":                       {"available_sec": 86400, "ok": True,  "note": "性別"},
    "distance":                  {"available_sec": 86400, "ok": True,  "note": "距離"},
    "surface":                   {"available_sec": 86400, "ok": True,  "note": "芝/ダート"},
    "course_dir":                {"available_sec": 86400, "ok": True,  "note": "右回り/左回り"},
    "straight_length":           {"available_sec": 86400, "ok": True,  "note": "直線長（m）"},
    "track":                     {"available_sec": 86400, "ok": True,  "note": "競馬場"},
    "race_class":                {"available_sec": 86400, "ok": True,  "note": "クラス"},

    # ===== 過去走から集計（as_of_dateで管理）=====
    "agari3f_rank_pct_hist":     {"available_sec": 86400, "ok": True,  "note": "過去走の上がり3F順位%（当走は使わない）"},
    "agari3f_z_hist":            {"available_sec": 86400, "ok": True,  "note": "過去走の上がり3F z-score"},
    "style_score_wavg3":         {"available_sec": 86400, "ok": True,  "note": "脚質スコア3走加重平均"},
    "style_vol3":                {"available_sec": 86400, "ok": True,  "note": "脚質安定度（3走std）"},
    "speed_index_dist_norm":     {"available_sec": 86400, "ok": True,  "note": "距離帯別スピード指数"},
    "jockey_course_winrate":     {"available_sec": 86400, "ok": True,  "note": "騎手×コース勝率（過去）"},
    "jockey_win_rate_3m":        {"available_sec": 86400, "ok": True,  "note": "騎手直近3ヶ月勝率"},
    "trainer_win_rate_3m":       {"available_sec": 86400, "ok": True,  "note": "調教師直近3ヶ月勝率"},
    "rest_rbf_7":                {"available_sec": 86400, "ok": True,  "note": "出走間隔RBF center=7日"},
    "rest_rbf_14":               {"available_sec": 86400, "ok": True,  "note": "出走間隔RBF center=14日"},
    "rest_rbf_21":               {"available_sec": 86400, "ok": True,  "note": "出走間隔RBF center=21日"},
    "rest_rbf_35":               {"available_sec": 86400, "ok": True,  "note": "出走間隔RBF center=35日"},
    "rest_rbf_56":               {"available_sec": 86400, "ok": True,  "note": "出走間隔RBF center=56日"},
    "rest_rbf_84":               {"available_sec": 86400, "ok": True,  "note": "出走間隔RBF center=84日"},
    "rest_rbf_140":              {"available_sec": 86400, "ok": True,  "note": "出走間隔RBF center=140日"},
    "rest_rbf_210":              {"available_sec": 86400, "ok": True,  "note": "出走間隔RBF center=210日"},
    "layoff_150plus":            {"available_sec": 86400, "ok": True,  "note": "150日以上休養ダミー"},
    "prev_finish_pos":           {"available_sec": 86400, "ok": True,  "note": "前走着順"},
    "prev_distance_diff":        {"available_sec": 86400, "ok": True,  "note": "前走との距離差"},
    "prev_class_diff":           {"available_sec": 86400, "ok": True,  "note": "前走クラス差（昇級/降級）"},
    "wins_same_distance_band":   {"available_sec": 86400, "ok": True,  "note": "同距離帯勝利数（過去）"},
    "wins_same_surface":         {"available_sec": 86400, "ok": True,  "note": "同surface勝利数（過去）"},
    "style_x_straight":          {"available_sec": 86400, "ok": True,  "note": "脚質×直線長の交互作用"},

    # ===== 新規追加: EWMAトレンド =====
    "sf_ewma":                   {"available_sec": 86400, "ok": True,  "note": "スピード指数EWMA(span=3)"},
    "sf_trend_slope":            {"available_sec": 86400, "ok": True,  "note": "スピード指数トレンド傾き"},
    "sf_latest_vs_mean":         {"available_sec": 86400, "ok": True,  "note": "最新走 - 前3走平均"},

    # ===== 新規追加: レース内容スコア =====
    "rcs_mean":                  {"available_sec": 86400, "ok": True,  "note": "レース内容スコア平均"},
    "rcs_max":                   {"available_sec": 86400, "ok": True,  "note": "レース内容スコア最高値"},
    "rcs_trend":                 {"available_sec": 86400, "ok": True,  "note": "レース内容スコアトレンド"},

    # ===== 新規追加: 騎手・調教師ベイズ収縮 =====
    "jockey_bayes_top3_rate":    {"available_sec": 86400, "ok": True,  "note": "騎手m-estimate 3着内率"},
    "jockey_cond_count":         {"available_sec": 86400, "ok": True,  "note": "騎手条件別出走数"},
    "trainer_bayes_top3_rate":   {"available_sec": 86400, "ok": True,  "note": "調教師m-estimate 3着内率"},

    # ===== 新規追加: コースジオメトリ =====
    "straight_m":                {"available_sec": 86400, "ok": True,  "note": "直線長(m)"},
    "circumference_m":           {"available_sec": 86400, "ok": True,  "note": "周回距離(m)"},
    "has_slope":                 {"available_sec": 86400, "ok": True,  "note": "坂の有無"},
    "left_turn":                 {"available_sec": 86400, "ok": True,  "note": "左回りか"},
    "tight_corners":             {"available_sec": 86400, "ok": True,  "note": "タイトコーナーか"},
    "is_banei":                  {"available_sec": 86400, "ok": True,  "note": "ばんえいフラグ"},
    "organizer_jra":             {"available_sec": 86400, "ok": True,  "note": "JRA=1/NAR=0"},

    # ===== 新規追加: オッズドリフト（直前確定） =====
    "drift_30m_to_final":        {"available_sec": 120,  "ok": True,  "note": "30分前→最終オッズ変化率"},
    "drift_10m_to_final":        {"available_sec": 120,  "ok": True,  "note": "10分前→最終オッズ変化率"},
    "drift_2m_to_final":         {"available_sec": 120,  "ok": True,  "note": "2分前→最終オッズ変化率"},
    "drift_acceleration":        {"available_sec": 120,  "ok": True,  "note": "ドリフト加速度"},
    "is_drifter":                {"available_sec": 120,  "ok": True,  "note": "人気薄化フラグ"},

    # ===== 新規追加: 市場乖離 =====
    "model_market_gap":          {"available_sec": 120,  "ok": True,  "note": "モデルvs市場ロジット差"},

    # ===== 絶対使用禁止（レース後確定） =====
    "closing_odds":              {"available_sec": -1, "ok": False, "note": "最終オッズ（レース後確定）"},
    "finish_position":           {"available_sec": -1, "ok": False, "note": "着順"},
    "passing_order_1c":          {"available_sec": -1, "ok": False, "note": "1角通過順"},
    "passing_order_2c":          {"available_sec": -1, "ok": False, "note": "2角通過順"},
    "passing_order_3c":          {"available_sec": -1, "ok": False, "note": "3角通過順"},
    "passing_order_4c":          {"available_sec": -1, "ok": False, "note": "4角通過順"},
    "agari3f_seconds":           {"available_sec": -1, "ok": False, "note": "当走の上がり3F秒数"},
    "agari3f_rank":              {"available_sec": -1, "ok": False, "note": "当走の上がり3F着内順位"},
    "race_time":                 {"available_sec": -1, "ok": False, "note": "当走タイム"},
    "prize_money_won":           {"available_sec": -1, "ok": False, "note": "当走獲得賞金"},
}

FORBIDDEN_FEATURES: list[str] = [k for k, v in FEATURE_SPECS.items() if not v["ok"]]
ALLOWED_FEATURES: list[str]   = [k for k, v in FEATURE_SPECS.items() if v["ok"]]


def assert_no_forbidden(columns: list[str]) -> None:
    """DataFrameのカラムリストに禁止特徴量が含まれていないか検証。"""
    violations = [c for c in columns if c in FORBIDDEN_FEATURES]
    if violations:
        raise ValueError(
            f"データリーク: 禁止特徴量が含まれています → {violations}\n"
            "これらはレース後に確定する値です。学習・予測データから除外してください。"
        )
