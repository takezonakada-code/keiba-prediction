"""
PSI監視 + Slack通知。
監視対象: win_odds分布 / agari3f_z分布 / モデルスコア分布 / LogLoss劣化 / スクレイピング成功率
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import requests

from config import (
    EV_MIN_THRESHOLD,
    LOGLOSS_DEGRADATION_RATIO,
    PSI_ALERT_THRESHOLD,
    SCRAPING_SUCCESS_MIN,
    SLACK_WEBHOOK_URL,
)


# ────────────────────────────────────────────────
# PSI計算
# ────────────────────────────────────────────────
def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Population Stability Index（PSI）を計算する。

    Parameters
    ----------
    reference : ベースライン分布（最初の1ヶ月で測定）
    current   : 監視対象の最新分布
    n_bins    : ビン数
    epsilon   : ゼロ除算防止

    Returns
    -------
    float: PSI値（0=変化なし、>0.25=大きな変化）
    """
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins[0]  = -np.inf
    bins[-1] =  np.inf

    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bins)

    ref_pct = (ref_counts + epsilon) / (ref_counts.sum() + epsilon * n_bins)
    cur_pct = (cur_counts + epsilon) / (cur_counts.sum() + epsilon * n_bins)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


# ────────────────────────────────────────────────
# Slack通知
# ────────────────────────────────────────────────
def send_slack(message: str, webhook_url: str = SLACK_WEBHOOK_URL) -> bool:
    """Slack Incoming Webhook でメッセージを送信。"""
    if not webhook_url:
        print(f"[SLACK未設定] {message}")
        return False
    try:
        resp = requests.post(
            webhook_url,
            json={"text": message},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as e:
        print(f"Slack送信エラー: {e}")
        return False


# ────────────────────────────────────────────────
# 監視ロジック
# ────────────────────────────────────────────────
class DriftMonitor:
    def __init__(self, baselines: dict[str, np.ndarray]):
        """
        Parameters
        ----------
        baselines : {"win_odds": array, "agari3f_z": array, "model_score": array}
                    最初の1ヶ月で測定したベースライン分布
        """
        self.baselines = baselines
        self.alerts: list[dict[str, Any]] = []

    def check_psi(self, current_distributions: dict[str, np.ndarray]) -> list[dict]:
        """PSI監視。閾値超過でアラート生成。"""
        alerts = []
        for name, current in current_distributions.items():
            if name not in self.baselines:
                continue
            psi = compute_psi(self.baselines[name], current)
            if psi > PSI_ALERT_THRESHOLD:
                alerts.append({
                    "type":   "PSI",
                    "name":   name,
                    "psi":    round(psi, 4),
                    "threshold": PSI_ALERT_THRESHOLD,
                    "message": f"[PSIアラート] {name}: PSI={psi:.4f} > {PSI_ALERT_THRESHOLD}",
                })
        return alerts

    def check_logloss(
        self,
        recent_logloss: float,
        historical_logloss: list[float],
        n_weeks: int = 8,
    ) -> list[dict]:
        """直近LogLossが過去8週平均比+10%を超えたらアラート。"""
        if len(historical_logloss) < n_weeks:
            return []
        baseline_mean = np.mean(historical_logloss[-n_weeks:])
        if recent_logloss > baseline_mean * LOGLOSS_DEGRADATION_RATIO:
            return [{
                "type":    "LogLoss",
                "value":   round(recent_logloss, 4),
                "baseline": round(baseline_mean, 4),
                "message": (
                    f"[LogLossアラート] 直近={recent_logloss:.4f}, "
                    f"8週平均={baseline_mean:.4f}, "
                    f"劣化率={recent_logloss/baseline_mean:.2f}×"
                ),
            }]
        return []

    def check_ev(self, mean_ev: float) -> list[dict]:
        """平均候補EV < 閾値でアラート。"""
        if mean_ev < EV_MIN_THRESHOLD:
            return [{
                "type":    "EV",
                "value":   round(mean_ev, 4),
                "threshold": EV_MIN_THRESHOLD,
                "message": f"[EVアラート] 平均候補EV={mean_ev:.4f} < {EV_MIN_THRESHOLD}",
            }]
        return []

    def check_scraping(self, success_rate: float) -> list[dict]:
        """スクレイピング成功率 < 95% でアラート。"""
        if success_rate < SCRAPING_SUCCESS_MIN:
            return [{
                "type":    "Scraping",
                "value":   round(success_rate, 4),
                "threshold": SCRAPING_SUCCESS_MIN,
                "message": f"[スクレイピングアラート] 成功率={success_rate:.1%} < {SCRAPING_SUCCESS_MIN:.0%}",
            }]
        return []

    def run_all_checks(
        self,
        current_distributions: dict[str, np.ndarray] | None = None,
        recent_logloss: float | None = None,
        historical_logloss: list[float] | None = None,
        mean_ev: float | None = None,
        scraping_success_rate: float | None = None,
    ) -> list[dict]:
        """全チェックを実行してアラートリストを返し、Slackに通知する。"""
        all_alerts: list[dict] = []

        if current_distributions:
            all_alerts.extend(self.check_psi(current_distributions))
        if recent_logloss is not None and historical_logloss is not None:
            all_alerts.extend(self.check_logloss(recent_logloss, historical_logloss))
        if mean_ev is not None:
            all_alerts.extend(self.check_ev(mean_ev))
        if scraping_success_rate is not None:
            all_alerts.extend(self.check_scraping(scraping_success_rate))

        for alert in all_alerts:
            send_slack(alert["message"])

        self.alerts = all_alerts
        return all_alerts
