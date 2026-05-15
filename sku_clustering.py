"""
app/ml/sku_clustering.py
─────────────────────────
SKU Clustering Engine — KMeans on PCA-reduced Feature Matrix

Feature matrix (per SKU, trailing 90 days):
  sales_velocity      avg daily units sold
  gross_margin        (revenue - COGS) / revenue * 100
  total_volume        cumulative units sold
  seasonality_idx     peak_month_sales / avg_monthly_sales
  return_rate         units_returned / units_sold * 100
  promo_sensitivity   avg_lift_during_promos (ratio)
  avg_order_value     mean revenue per transaction

Pipeline:
  1. StandardScaler  → zero-mean, unit-variance
  2. PCA             → retain 95% explained variance
  3. KMeans          → optimal k via elbow + silhouette
  4. Label mapping   → business strategy per cluster

Cluster Labels (k=4 default):
  Stars        : high velocity, high margin  → premium shelf, tight safety stock
  Cash Cows    : low velocity, high margin   → long tail, lower reorder frequency
  Volume Movers: high velocity, low margin   → bulk ordering, efficiency focus
  Tail SKUs    : low velocity, low margin    → clearance / discontinue review
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

CLUSTER_STRATEGIES = {
    "Stars":         "Premium placement · high service level · tighter safety stock",
    "Cash Cows":     "Long tail management · lower reorder frequency · premium pricing",
    "Volume Movers": "Bulk ordering · warehouse efficiency · thin margin protection",
    "Tail SKUs":     "Review for clearance · discontinue if margin negative",
}

FEATURE_COLS = [
    "sales_velocity", "gross_margin", "total_volume",
    "seasonality_idx", "return_rate", "promo_sensitivity", "avg_order_value",
]


class SKUClusteringEngine:

    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.model: Optional[KMeans] = None
        self.scaler = StandardScaler()
        self.pca: Optional[PCA] = None
        self._label_map: dict = {}

    @classmethod
    def load_from_registry(cls) -> "SKUClusteringEngine":
        engine = cls()
        engine.model = KMeans(n_clusters=4, random_state=42, n_init="auto")
        logger.info("✅ SKUClusteringEngine loaded")
        return engine

    # ── Optimal k selection ───────────────────────────────────────────────────

    def find_optimal_k(self, X_scaled: np.ndarray, k_range=(2, 10)) -> int:
        """
        Elbow method + silhouette score to pick k.
        Returns k with highest silhouette coefficient.
        """
        best_k, best_sil = 4, -1
        for k in range(k_range[0], min(k_range[1] + 1, len(X_scaled))):
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = km.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels)
            if sil > best_sil:
                best_sil, best_k = sil, k
        logger.info(f"Optimal k={best_k} (silhouette={best_sil:.3f})")
        return best_k

    # ── Cluster labeling ──────────────────────────────────────────────────────

    def label_clusters(self, df: pd.DataFrame, labels: np.ndarray) -> dict:
        """
        Assign business labels to cluster IDs based on centroid characteristics.
        High velocity + high margin → Stars
        Low velocity  + high margin → Cash Cows
        High velocity + low margin  → Volume Movers
        Low velocity  + low margin  → Tail SKUs
        """
        df = df.copy()
        df["cluster"] = labels
        centroids = df.groupby("cluster")[["sales_velocity", "gross_margin"]].mean()
        med_vel = centroids["sales_velocity"].median()
        med_mar = centroids["gross_margin"].median()

        label_map = {}
        for cid, row in centroids.iterrows():
            high_vel = row["sales_velocity"] >= med_vel
            high_mar = row["gross_margin"] >= med_mar
            if high_vel and high_mar:
                label_map[cid] = "Stars"
            elif not high_vel and high_mar:
                label_map[cid] = "Cash Cows"
            elif high_vel and not high_mar:
                label_map[cid] = "Volume Movers"
            else:
                label_map[cid] = "Tail SKUs"
        return label_map

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, n_clusters: Optional[int] = None):
        """Train clustering pipeline. Called by Airflow training DAG weekly."""
        avail = [c for c in FEATURE_COLS if c in df.columns]
        X = df[avail].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)

        # PCA: retain 95% variance
        self.pca = PCA(n_components=0.95, random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)
        logger.info(f"PCA: {X_pca.shape[1]} components retain 95% variance (from {X_scaled.shape[1]})")

        k = n_clusters or self.find_optimal_k(X_pca)
        self.n_clusters = k
        self.model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = self.model.fit_predict(X_pca)
        self._label_map = self.label_clusters(df, labels)
        logger.info(f"KMeans fitted: k={k}, labels={self._label_map}")

    # ── Predict ───────────────────────────────────────────────────────────────

    async def compute(self, db, n_clusters: int, features: List[str]) -> List[dict]:
        """
        Pull 90-day SKU metrics from DB, run clustering, return labelled results.
        """
        query = """
            SELECT
                s.sku_id,
                AVG(s.sales_qty)                         AS sales_velocity,
                AVG((s.revenue_inr - p.unit_cost_inr * s.sales_qty)
                    / NULLIF(s.revenue_inr, 0) * 100)    AS gross_margin,
                SUM(s.sales_qty)                         AS total_volume,
                MAX(monthly.peak) / NULLIF(AVG(s.sales_qty) * 30, 0) AS seasonality_idx,
                COALESCE(AVG(r.return_rate_pct), 0)      AS return_rate,
                COALESCE(AVG(pr.lift_ratio), 1.0)        AS promo_sensitivity,
                AVG(s.revenue_inr / NULLIF(s.sales_qty, 0)) AS avg_order_value
            FROM sku_daily_sales s
            JOIN products p ON s.sku_id = p.sku_id
            LEFT JOIN (
                SELECT sku_id, MAX(monthly_sales) AS peak
                FROM (SELECT sku_id, DATE_TRUNC('month', date) AS m, SUM(sales_qty) AS monthly_sales
                      FROM sku_daily_sales WHERE date >= CURRENT_DATE - 365
                      GROUP BY sku_id, m) t
                GROUP BY sku_id
            ) monthly ON monthly.sku_id = s.sku_id
            LEFT JOIN sku_return_rates r ON r.sku_id = s.sku_id
            LEFT JOIN promo_lift_summary pr ON pr.sku_id = s.sku_id
            WHERE s.date >= CURRENT_DATE - 90
            GROUP BY s.sku_id
            HAVING COUNT(*) >= 30
        """
        rows = await db.fetch_all(query)
        if not rows:
            return []

        df = pd.DataFrame([dict(r) for r in rows])
        df = df.fillna({"gross_margin": 20.0, "seasonality_idx": 1.0,
                        "return_rate": 0.0, "promo_sensitivity": 1.0})

        avail_cols = [c for c in features if c in df.columns]
        if not avail_cols:
            avail_cols = [c for c in FEATURE_COLS if c in df.columns]

        X = df[avail_cols].values
        X_scaled = self.scaler.fit_transform(X)

        if self.pca is None:
            self.pca = PCA(n_components=min(0.95, X_scaled.shape[1]), random_state=42)
        X_pca = self.pca.fit_transform(X_scaled)

        if self.model is None or self.n_clusters != n_clusters:
            self.n_clusters = n_clusters
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = self.model.fit_predict(X_pca)
        label_map = self.label_clusters(df, labels)

        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            cid = int(labels[i])
            clabel = label_map.get(cid, f"Cluster {cid}")
            results.append({
                "sku_id": row["sku_id"],
                "cluster_id": cid,
                "cluster_label": clabel,
                "sales_velocity": round(float(row.get("sales_velocity", 0)), 2),
                "gross_margin": round(float(row.get("gross_margin", 0)), 2),
                "total_volume": int(row.get("total_volume", 0)),
                "recommended_strategy": CLUSTER_STRATEGIES.get(clabel, "Review manually"),
            })

        results.sort(key=lambda r: (r["cluster_label"], -r["sales_velocity"]))
        return results
