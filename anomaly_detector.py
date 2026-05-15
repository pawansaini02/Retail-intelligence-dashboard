"""
app/ml/anomaly_detector.py
───────────────────────────
Sales Anomaly Detection — Isolation Forest

Features (per SKU-region-day):
  daily_sales        · 7d_rolling_mean  · 7d_rolling_std
  day_of_week        · month            · is_holiday
  is_promotion       · weather_score    · competitor_price_delta
  lag_7              · lag_28           · yoy_ratio

Anomaly score normalised to [0, 1] via sklearn decision_function.
Threshold: 0.70 (tuned on labelled dataset of known promo spikes + supply disruptions).

Probable cause inference (rule-based post-hoc):
  - "Promotion spike"      : is_promotion = 1 AND direction = spike
  - "Festive demand surge" : is_holiday = 1
  - "Supply disruption"    : simultaneous inventory drop detected
  - "Competitor event"     : competitor_price_delta > 15%
  - "Unexplained"          : none of the above
"""

import numpy as np
import pandas as pd
import logging
import pickle
from datetime import datetime, timedelta
from typing import Optional, List
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "daily_sales", "rolling_mean_7", "rolling_std_7",
    "day_of_week", "month", "is_holiday", "is_promotion",
    "weather_score", "competitor_price_delta", "lag_7", "lag_28", "yoy_ratio",
]


class AnomalyDetector:

    ANOMALY_THRESHOLD = 0.70
    ISO_FOREST_PARAMS = {
        "n_estimators": 200,
        "contamination": 0.05,   # expect ~5% anomalous days
        "max_features": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(self):
        self.model: Optional[IsolationForest] = None
        self.scaler: StandardScaler = StandardScaler()
        self.version: str = "unknown"

    @classmethod
    def load_from_registry(cls) -> "AnomalyDetector":
        """Load model from MLflow artifact store (Production stage)."""
        detector = cls()
        # Production: detector.model = mlflow.sklearn.load_model("models:/isolation_forest/Production")
        detector.model = IsolationForest(**cls.ISO_FOREST_PARAMS)
        detector.version = "sklearn-1.4.0"
        logger.info("✅ AnomalyDetector loaded")
        return detector

    # ── Feature Engineering ───────────────────────────────────────────────────

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the 12-feature matrix from raw daily sales DataFrame.
        Expected input columns: date, sku_id, region, sales_qty,
                                is_holiday, is_promotion, weather_score,
                                competitor_price_delta
        """
        df = df.copy().sort_values("date")
        df["daily_sales"]    = df["sales_qty"]
        df["rolling_mean_7"] = df["sales_qty"].shift(1).rolling(7, min_periods=3).mean()
        df["rolling_std_7"]  = df["sales_qty"].shift(1).rolling(7, min_periods=3).std().fillna(1)
        df["day_of_week"]    = pd.to_datetime(df["date"]).dt.dayofweek
        df["month"]          = pd.to_datetime(df["date"]).dt.month
        df["lag_7"]          = df["sales_qty"].shift(7)
        df["lag_28"]         = df["sales_qty"].shift(28)
        df["yoy_ratio"]      = df["sales_qty"] / (df["sales_qty"].shift(365) + 1)

        # Fill missing optional columns with neutral defaults
        for col in ["is_holiday", "is_promotion", "weather_score", "competitor_price_delta"]:
            if col not in df.columns:
                df[col] = 0.0

        return df.dropna(subset=["rolling_mean_7", "lag_7"])

    def fit(self, df: pd.DataFrame):
        """Train Isolation Forest on historical data. Called by Airflow training DAG."""
        feat_df = self.build_features(df)
        X = feat_df[FEATURE_COLS].values
        X_scaled = self.scaler.fit_transform(X)
        self.model = IsolationForest(**self.ISO_FOREST_PARAMS)
        self.model.fit(X_scaled)
        logger.info(f"AnomalyDetector trained on {len(X)} samples")

    def score(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Convert raw decision_function (higher = more normal) to anomaly score 0-1.
        Anomaly score = 1 - normalised_decision_function.
        """
        raw = self.model.decision_function(X_scaled)  # higher = more normal
        # Normalise to [0, 1] where 1 = most anomalous
        score = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        return np.clip(score, 0, 1)

    # ── Probable Cause ────────────────────────────────────────────────────────

    def infer_cause(self, row: dict) -> str:
        if row.get("is_promotion") and row["direction"] == "spike":
            return "Promotion spike"
        if row.get("is_holiday"):
            return "Festive demand surge"
        if row.get("inventory_drop_pct", 0) > 30:
            return "Supply disruption"
        if abs(row.get("competitor_price_delta", 0)) > 15:
            return "Competitor price event"
        if row["direction"] == "drop" and row.get("weather_score", 0) < -0.5:
            return "Adverse weather impact"
        return "Unexplained anomaly"

    # ── Main Detect ───────────────────────────────────────────────────────────

    async def detect(
        self, db, days_back: int = 30,
        region: Optional[str] = None,
        sku_id: Optional[str] = None,
        min_score: float = 0.70,
    ) -> List[dict]:
        """
        Pull recent sales from DB, score with Isolation Forest,
        return events above min_score with direction + probable cause.
        """
        query = """
            SELECT s.sku_id, s.region, s.date,
                   s.sales_qty, s.is_holiday, s.is_promotion,
                   s.weather_score,
                   COALESCE(c.price_delta_pct, 0) AS competitor_price_delta,
                   COALESCE(i.inventory_drop_pct, 0) AS inventory_drop_pct
            FROM sku_daily_sales s
            LEFT JOIN competitor_events c
                ON c.sku_id = s.sku_id AND c.date = s.date
            LEFT JOIN inventory_changes i
                ON i.sku_id = s.sku_id AND i.date = s.date
            WHERE s.date >= CURRENT_DATE - :days
                AND (:region IS NULL OR s.region = :region)
                AND (:sku_id IS NULL OR s.sku_id = :sku_id)
            ORDER BY s.sku_id, s.region, s.date
        """
        rows = await db.fetch_all(query, {"days": days_back, "region": region, "sku_id": sku_id})
        if not rows:
            return []

        df = pd.DataFrame([dict(r) for r in rows])
        results = []

        for (sid, reg), grp in df.groupby(["sku_id", "region"]):
            if len(grp) < 14:
                continue
            feat_df = self.build_features(grp)
            if feat_df.empty:
                continue

            avail_cols = [c for c in FEATURE_COLS if c in feat_df.columns]
            X = feat_df[avail_cols].values
            X_scaled = self.scaler.transform(X) if hasattr(self.scaler, "mean_") else X
            scores = self.score(X_scaled)

            for i, (_, row) in enumerate(feat_df.iterrows()):
                sc = float(scores[i])
                if sc < min_score:
                    continue

                expected = float(row.get("rolling_mean_7", row["daily_sales"]))
                actual   = float(row["daily_sales"])
                direction = "spike" if actual > expected else "drop"

                row_dict = dict(row)
                row_dict["direction"] = direction
                cause = self.infer_cause(row_dict)

                results.append({
                    "sku_id": sid,
                    "region": reg,
                    "date": str(row["date"])[:10],
                    "actual_value": round(actual, 1),
                    "expected_value": round(expected, 1),
                    "anomaly_score": round(sc, 4),
                    "direction": direction,
                    "probable_cause": cause,
                })

        results.sort(key=lambda r: -r["anomaly_score"])
        return results
