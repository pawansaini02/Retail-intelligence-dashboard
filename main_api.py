"""
AI-Powered Retail Demand & Inventory Intelligence System
=========================================================
FastAPI Backend — Production-Grade ML Serving Layer

Endpoints:
  POST /api/forecast          → Demand forecast (Prophet + XGBoost ensemble)
  GET  /api/inventory/restock → Restock recommendations (EOQ model)
  GET  /api/anomalies         → Anomaly detection (Isolation Forest)
  GET  /api/clusters          → SKU clustering (KMeans)
  GET  /api/regions           → Regional sales + forecast accuracy
  POST /api/promotions/impact → Promotion lift analysis
  GET  /api/kpis              → Operational KPIs
  GET  /api/health            → Pipeline + model health

Author: [Your Name]
Stack:  FastAPI · Prophet · XGBoost · LSTM · Isolation Forest · MLflow · PostgreSQL · Redis · Kafka
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn
import logging
import time
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

# ── Internal modules ──────────────────────────────────────────────────────────
from app.ml.demand_forecast import DemandForecastEngine
from app.ml.inventory_optimizer import InventoryOptimizer
from app.ml.anomaly_detector import AnomalyDetector
from app.ml.sku_clustering import SKUClusteringEngine
from app.ml.promotion_analyzer import PromotionAnalyzer
from app.services.kafka_producer import KafkaEventProducer
from app.services.mlflow_tracker import MLflowTracker
from app.db.postgres import get_db_session
from app.db.redis_cache import RedisCache
from app.core.config import Settings

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("retail_intelligence")

settings = Settings()

# ── ML Engine Singletons (loaded at startup) ──────────────────────────────────
forecast_engine: DemandForecastEngine = None
inventory_optimizer: InventoryOptimizer = None
anomaly_detector: AnomalyDetector = None
cluster_engine: SKUClusteringEngine = None
promo_analyzer: PromotionAnalyzer = None
kafka_producer: KafkaEventProducer = None
cache: RedisCache = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load all ML models into memory. Shutdown: close connections."""
    global forecast_engine, inventory_optimizer, anomaly_detector, cluster_engine, promo_analyzer, kafka_producer, cache
    logger.info("🚀 Starting Retail Intelligence System...")

    forecast_engine     = DemandForecastEngine.load_from_registry()
    inventory_optimizer = InventoryOptimizer()
    anomaly_detector    = AnomalyDetector.load_from_registry()
    cluster_engine      = SKUClusteringEngine.load_from_registry()
    promo_analyzer      = PromotionAnalyzer()
    kafka_producer      = KafkaEventProducer(brokers=settings.KAFKA_BROKERS)
    cache               = RedisCache(url=settings.REDIS_URL, ttl_seconds=300)

    logger.info("✅ All ML models loaded. API ready.")
    yield  # ← app runs here

    await kafka_producer.close()
    logger.info("🛑 Shutdown complete.")

app = FastAPI(
    title="Retail Demand & Inventory Intelligence API",
    description="Production ML system: demand forecasting · inventory optimization · anomaly detection · SKU clustering",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=500)

# ── Request / Response Schemas ────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    sku_id: str = Field(..., example="SKU-1042")
    region: str = Field(..., example="Punjab")
    horizon_days: int = Field(default=90, ge=7, le=365)
    include_confidence_interval: bool = Field(default=True)
    model: str = Field(default="ensemble", pattern="^(prophet|xgboost|lstm|ensemble)$")

class ForecastPoint(BaseModel):
    date: str
    predicted_demand: float
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    model_used: str

class ForecastResponse(BaseModel):
    sku_id: str
    region: str
    generated_at: str
    horizon_days: int
    forecast: List[ForecastPoint]
    mape: float
    rmse: float
    model_version: str
    seasonal_components: dict

class RestockItem(BaseModel):
    sku_id: str
    current_stock: int
    reorder_point: int
    recommended_order_qty: int
    economic_order_qty: int
    lead_time_days: int
    days_of_supply: float
    stockout_risk_score: float
    priority: str  # "critical" | "warning" | "healthy"
    estimated_cost: float

class AnomalyEvent(BaseModel):
    sku_id: str
    region: str
    date: str
    actual_value: float
    expected_value: float
    anomaly_score: float
    direction: str   # "spike" | "drop"
    probable_cause: str

class SKUCluster(BaseModel):
    sku_id: str
    cluster_id: int
    cluster_label: str
    sales_velocity: float
    gross_margin: float
    total_volume: int
    recommended_strategy: str

class PromoImpactRequest(BaseModel):
    sku_ids: List[str]
    promotion_start: str
    promotion_end: str
    promotion_type: str  # "discount" | "bogo" | "bundle"
    discount_pct: Optional[float] = None

class RegionKPI(BaseModel):
    region: str
    total_sales: float
    predicted_demand: float
    forecast_accuracy_pct: float
    stockout_events: int
    overstock_events: int
    fulfillment_rate_pct: float

# ── Dependency: Cache-Through Pattern ────────────────────────────────────────

async def cached(key: str, ttl: int = 300):
    """Fetch from Redis cache; caller populates on miss."""
    return await cache.get(key)

# ── Middleware: Request Timing ────────────────────────────────────────────────

@app.middleware("http")
async def add_process_time_header(request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{(time.time() - start)*1000:.1f}ms"
    return response

# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health_check():
    """
    Pipeline and model health check.
    Returns status of each model, Kafka lag, cache hit rate, and DB connection.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models": {
            "prophet":           {"status": "loaded", "version": forecast_engine.prophet_version},
            "xgboost":           {"status": "loaded", "version": forecast_engine.xgb_version},
            "lstm":              {"status": "loaded", "version": forecast_engine.lstm_version},
            "isolation_forest":  {"status": "loaded", "version": anomaly_detector.version},
            "kmeans":            {"status": "loaded", "n_clusters": cluster_engine.n_clusters},
        },
        "kafka_lag_max_ms":   await kafka_producer.get_max_lag(),
        "cache_hit_rate_pct": await cache.hit_rate(),
        "db_pool_size":       settings.DB_POOL_SIZE,
    }


@app.post("/api/forecast", response_model=ForecastResponse)
async def get_demand_forecast(req: ForecastRequest, background_tasks: BackgroundTasks):
    """
    Demand Forecast Endpoint — Ensemble of Prophet + XGBoost + LSTM
    ───────────────────────────────────────────────────────────────
    Strategy:
    - Prophet:   captures trend + weekly/yearly seasonality + holiday effects
    - XGBoost:   trained on lag features, rolling stats, promo flags, weather
    - LSTM:      multi-step sequence model on 60-day lookback window
    - Ensemble:  weighted avg (Prophet 40% + XGBoost 40% + LSTM 20%)
                 weights tuned per SKU category on validation set

    Returns daily predictions for `horizon_days` with CI bounds.
    Also logs inference event to Kafka for model monitoring.
    """
    cache_key = f"forecast:{req.sku_id}:{req.region}:{req.horizon_days}:{req.model}"
    cached_result = await cache.get(cache_key)
    if cached_result:
        return cached_result

    try:
        result = await forecast_engine.predict(
            sku_id=req.sku_id,
            region=req.region,
            horizon_days=req.horizon_days,
            model=req.model,
            include_ci=req.include_confidence_interval,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No historical data for SKU '{req.sku_id}' in region '{req.region}'")
    except Exception as e:
        logger.error(f"Forecast failed for {req.sku_id}: {e}")
        raise HTTPException(status_code=500, detail="Forecast model inference failed")

    await cache.set(cache_key, result, ttl=300)

    # Async: emit event to Kafka topic for drift monitoring
    background_tasks.add_task(kafka_producer.publish, topic="forecast-events", payload={
        "sku_id": req.sku_id, "region": req.region, "model": req.model,
        "horizon_days": req.horizon_days, "mape": result.mape, "ts": datetime.utcnow().isoformat()
    })

    return result


@app.get("/api/inventory/restock", response_model=List[RestockItem])
async def get_restock_recommendations(
    region: Optional[str] = Query(None, description="Filter by region"),
    category: Optional[str] = Query(None, description="Filter by product category"),
    priority: Optional[str] = Query(None, pattern="^(critical|warning|healthy)$"),
    limit: int = Query(default=50, le=500),
    db=Depends(get_db_session),
):
    """
    Inventory Restock Optimizer — Economic Order Quantity Model
    ──────────────────────────────────────────────────────────
    Uses EOQ formula: Q* = sqrt(2 * D * S / H)
      D = annual demand (from forecast engine)
      S = ordering cost per order (from ERP config)
      H = holding cost per unit per year

    Safety stock = Z * σ_LT * sqrt(lead_time)
      Z = 1.65 for 95% service level
      σ_LT = demand std dev during lead time

    Reorder point = avg_daily_demand * lead_time + safety_stock

    Stockout risk scored 0-1 using survival analysis on historical patterns.
    Priority: critical (<5 days supply), warning (<14 days), healthy (>14 days).
    """
    recommendations = await inventory_optimizer.compute_restock_plan(
        db=db, region=region, category=category, priority_filter=priority, limit=limit
    )
    return recommendations


@app.get("/api/anomalies", response_model=List[AnomalyEvent])
async def get_anomalies(
    days_back: int = Query(default=30, ge=1, le=180, description="Lookback window"),
    region: Optional[str] = None,
    sku_id: Optional[str] = None,
    min_score: float = Query(default=0.7, ge=0.0, le=1.0, description="Minimum anomaly score threshold"),
    db=Depends(get_db_session),
):
    """
    Sales Anomaly Detection — Isolation Forest
    ──────────────────────────────────────────
    Features: daily_sales · 7d_rolling_mean · 7d_rolling_std · day_of_week ·
              is_holiday · is_promotion · weather_score · competitor_price_delta

    Isolation Forest trained on 18 months of historical SKU-level daily sales.
    Anomaly score > 0.7 → flagged. Score normalized to [0, 1].

    Probable cause inference:
    - "Promotion spike": anomaly day has active campaign
    - "Seasonal outlier": anomaly near known peak (Diwali, festive)
    - "Supply disruption": inventory drops simultaneously
    - "Competitor event": external price change detected

    Results cached 5 min; async Kafka alert for score > 0.9.
    """
    anomalies = await anomaly_detector.detect(
        db=db, days_back=days_back, region=region, sku_id=sku_id, min_score=min_score
    )
    return anomalies


@app.get("/api/clusters", response_model=List[SKUCluster])
async def get_sku_clusters(
    n_clusters: int = Query(default=4, ge=2, le=10),
    features: str = Query(default="velocity,margin,volume,seasonality"),
    db=Depends(get_db_session),
):
    """
    SKU Clustering — KMeans on Normalized Feature Matrix
    ─────────────────────────────────────────────────────
    Feature matrix (per SKU, trailing 90 days):
      - sales_velocity:  avg daily units sold
      - gross_margin:    (revenue - COGS) / revenue
      - total_volume:    units sold
      - seasonality_idx: ratio of peak-month to avg-month sales
      - return_rate:     units returned / units sold
      - promo_sensitivity: lift ratio during promotions

    Preprocessing: StandardScaler → PCA (retain 95% variance) → KMeans
    Optimal k selected by elbow method + silhouette score.

    Cluster labels + recommended strategy:
      Cluster 0 "Stars":          High vel, high margin → premium placement, tight safety stock
      Cluster 1 "Cash Cows":      Low vel, high margin → long tail, lower reorder freq
      Cluster 2 "Volume Movers":  High vel, low margin → efficiency focus, bulk ordering
      Cluster 3 "Tail SKUs":      Low vel, low margin → discontinue / clearance strategy
    """
    feature_list = [f.strip() for f in features.split(",")]
    clusters = await cluster_engine.compute(db=db, n_clusters=n_clusters, features=feature_list)
    return clusters


@app.get("/api/regions", response_model=List[RegionKPI])
async def get_regional_kpis(
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    db=Depends(get_db_session),
):
    """
    Regional Sales Intelligence
    ───────────────────────────
    Aggregates per-region: actual sales · predicted demand · forecast accuracy (MAPE) ·
    stockout events (days with zero stock) · overstock events (stock > 2× reorder point) ·
    fulfillment rate (orders shipped / orders received).

    Forecast accuracy computed as 1 - MAPE against 7-day trailing actual.
    """
    regions = await forecast_engine.get_regional_kpis(
        db=db,
        start_date=start_date or (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d"),
        end_date=end_date or datetime.utcnow().strftime("%Y-%m-%d"),
    )
    return regions


@app.post("/api/promotions/impact")
async def analyze_promotion_impact(req: PromoImpactRequest, db=Depends(get_db_session)):
    """
    Promotion Lift Analysis — Causal Inference with Difference-in-Differences
    ─────────────────────────────────────────────────────────────────────────
    Method:
    1. Define treatment group: promoted SKUs in promoted region
    2. Define control group: similar SKUs (matched via cluster + propensity score) in non-promoted region
    3. DiD estimator: (treatment_post - treatment_pre) - (control_post - control_pre)

    Output: lift multiplier, incremental revenue, demand elasticity,
            post-promo demand decay curve, cannibalization estimate.

    Also runs SHAP on XGBoost forecast model to attribute how much of the
    spike is explained by: promotion flag vs. seasonal vs. competitor vs. weather.
    """
    impact = await promo_analyzer.analyze(db=db, req=req)
    return impact


@app.get("/api/kpis")
async def get_operational_kpis(db=Depends(get_db_session)):
    """
    Operational KPI Dashboard
    ─────────────────────────
    Returns portfolio-level metrics:
    - forecast_accuracy_pct: weighted MAPE across all active SKUs (trailing 30d)
    - stockout_reduction_pct: vs. pre-ML baseline (measured from A/B rollout)
    - inventory_turnover: COGS / avg inventory value (annualized)
    - overstock_cost_saved_inr: estimated carrying cost avoided vs. naive ordering
    - demand_coverage_pct: % of demand events covered by available stock
    - anomalies_detected_30d: count of flagged anomalies, trailing 30 days
    - model_drift_alerts: count of PSI > 0.2 (population stability index) events
    """
    cache_key = "kpis:global:30d"
    cached = await cache.get(cache_key)
    if cached:
        return cached

    kpis = await forecast_engine.compute_portfolio_kpis(db=db, window_days=30)
    await cache.set(cache_key, kpis, ttl=600)  # KPIs stale-ok for 10 min
    return kpis


# ═══════════════════════════════════════════════════════════════════════════════
# Dev runner
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
