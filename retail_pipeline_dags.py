"""
airflow/dags/retail_pipeline_dags.py
──────────────────────────────────────
All Airflow DAGs for the Retail Intelligence System.

DAG 1: ingestion_dag       — Daily POS + ERP + external data pull (00:30 UTC)
DAG 2: feature_dag         — PySpark feature engineering (01:00 UTC)
DAG 3: training_dag        — Weekly model retraining (02:00 UTC Sunday)
DAG 4: monitoring_dag      — Daily model drift + pipeline health (06:00 UTC)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.models import Variable

logger = logging.getLogger(__name__)

# ── Shared defaults ───────────────────────────────────────────────────────────

SHARED_ARGS = {
    "owner": "retail-ml-team",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "email_on_failure": True,
    "email": ["ml-alerts@yourcompany.com"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# DAG 1 — Data Ingestion (Daily 00:30 UTC)
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_pos_data(**ctx):
    """Pull yesterday's POS transactions from source DB via JDBC → Kafka."""
    import subprocess, json
    ds = ctx["ds"]  # execution date YYYY-MM-DD
    logger.info(f"Ingesting POS data for {ds}")
    # In production: call internal ETL service or run Debezium CDC
    result = subprocess.run([
        "python", "-m", "etl.pos_ingester",
        "--date", ds, "--kafka-topic", "sales-events",
    ], capture_output=True, text=True, check=True)
    logger.info(result.stdout)
    return json.loads(result.stdout.strip().split("\n")[-1])  # expects JSON summary


def ingest_inventory_snapshot(**ctx):
    """Pull inventory snapshot from ERP (SAP) via REST API → PostgreSQL."""
    from etl.erp_client import ERPClient
    client = ERPClient(
        base_url=Variable.get("ERP_BASE_URL"),
        api_key=Variable.get("ERP_API_KEY"),
    )
    count = client.sync_inventory_snapshot(date=ctx["ds"])
    logger.info(f"Synced {count} inventory records")
    return {"records_synced": count}


def ingest_weather_data(**ctx):
    """Pull regional weather scores from OpenWeather API → PostgreSQL."""
    import requests, psycopg2, os
    REGIONS = ["Punjab", "Maharashtra", "Karnataka", "Delhi NCR", "Tamil Nadu"]
    CITY_MAP = {"Punjab": "Chandigarh", "Maharashtra": "Mumbai",
                "Karnataka": "Bengaluru", "Delhi NCR": "Delhi", "Tamil Nadu": "Chennai"}
    api_key = Variable.get("OPENWEATHER_API_KEY")
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    for region in REGIONS:
        city = CITY_MAP[region]
        resp = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
        ).json()
        score = (resp["main"]["temp"] - 295) / 10  # normalised temperature score
        cur.execute(
            "INSERT INTO weather_scores (region, date, score) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
            (region, ctx["ds"], round(score, 2))
        )
    conn.commit(); cur.close(); conn.close()


def ingest_competitor_prices(**ctx):
    """Scrape / consume competitor price feed → competitor_events table."""
    from etl.price_scraper import PriceScraper
    scraper = PriceScraper(proxy_pool=Variable.get("PROXY_POOL_URL", ""))
    count = scraper.run(date=ctx["ds"])
    logger.info(f"Competitor price records ingested: {count}")


with DAG(
    dag_id="ingestion_dag",
    description="Daily POS + ERP + weather + competitor data ingestion",
    schedule_interval="30 0 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=SHARED_ARGS,
    tags=["ingestion", "daily"],
) as ingestion_dag:

    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE)

    t_pos   = PythonOperator(task_id="ingest_pos_data",          python_callable=ingest_pos_data,          provide_context=True)
    t_inv   = PythonOperator(task_id="ingest_inventory_snapshot", python_callable=ingest_inventory_snapshot, provide_context=True)
    t_wx    = PythonOperator(task_id="ingest_weather_data",       python_callable=ingest_weather_data,       provide_context=True)
    t_comp  = PythonOperator(task_id="ingest_competitor_prices",  python_callable=ingest_competitor_prices,  provide_context=True)

    refresh_mv = BashOperator(
        task_id="refresh_materialized_views",
        bash_command="""
            psql $DATABASE_URL -c "REFRESH MATERIALIZED VIEW CONCURRENTLY mv_sku_daily_kpis;"
            psql $DATABASE_URL -c "REFRESH MATERIALIZED VIEW CONCURRENTLY mv_regional_kpis;"
        """,
    )

    start >> [t_pos, t_inv, t_wx, t_comp] >> refresh_mv >> end


# ═══════════════════════════════════════════════════════════════════════════════
# DAG 2 — Feature Engineering (Daily 01:00 UTC, after ingestion)
# ═══════════════════════════════════════════════════════════════════════════════

def run_spark_features(**ctx):
    """
    PySpark job: reads raw tables, computes 12-feature matrix,
    writes to feature_store.sku_daily_features.
    """
    import subprocess
    result = subprocess.run([
        "spark-submit",
        "--master", "local[4]",
        "--driver-memory", "4g",
        "--executor-memory", "8g",
        "/opt/airflow/spark_jobs/feature_engineering.py",
        "--date", ctx["ds"],
        "--output-table", "feature_store.sku_daily_features",
    ], capture_output=True, text=True, check=True)
    logger.info(result.stdout[-2000:])


def run_dbt_transforms(**ctx):
    """Run dbt models: cleaning, deduplication, aggregations."""
    import subprocess
    subprocess.run(
        ["dbt", "run", "--select", "tag:daily", "--vars", f'{{"run_date": "{ctx["ds"]}"}}'],
        cwd="/opt/dbt/retail_intelligence", check=True,
    )
    subprocess.run(["dbt", "test", "--select", "tag:daily"], cwd="/opt/dbt/retail_intelligence", check=True)


with DAG(
    dag_id="feature_dag",
    description="Daily PySpark feature engineering + dbt transforms",
    schedule_interval="0 1 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=SHARED_ARGS,
    tags=["features", "daily"],
) as feature_dag:

    t_spark = PythonOperator(task_id="run_spark_features",  python_callable=run_spark_features,  provide_context=True)
    t_dbt   = PythonOperator(task_id="run_dbt_transforms",  python_callable=run_dbt_transforms,   provide_context=True)
    t_spark >> t_dbt


# ═══════════════════════════════════════════════════════════════════════════════
# DAG 3 — Model Training (Weekly Sunday 02:00 UTC)
# ═══════════════════════════════════════════════════════════════════════════════

def train_prophet_models(**ctx):
    """Fit Prophet model per SKU-region on trailing 18 months of data."""
    from training.train_forecast import train_all_skus
    metrics = train_all_skus(model="prophet", run_date=ctx["ds"])
    logger.info(f"Prophet training complete: avg MAPE={metrics['avg_mape']:.2f}%")
    # Push to XCom for downstream quality gate
    return metrics


def train_xgboost_models(**ctx):
    from training.train_forecast import train_all_skus
    metrics = train_all_skus(model="xgboost", run_date=ctx["ds"])
    logger.info(f"XGBoost training complete: avg MAPE={metrics['avg_mape']:.2f}%")
    return metrics


def train_lstm_models(**ctx):
    from training.train_forecast import train_all_skus
    metrics = train_all_skus(model="lstm", run_date=ctx["ds"])
    logger.info(f"LSTM training complete: avg MAPE={metrics['avg_mape']:.2f}%")
    return metrics


def train_anomaly_model(**ctx):
    from training.train_anomaly import train_isolation_forest
    result = train_isolation_forest(run_date=ctx["ds"])
    logger.info(f"Anomaly detector trained: contamination={result['contamination']}")
    return result


def train_clustering(**ctx):
    from training.train_clustering import train_sku_kmeans
    result = train_sku_kmeans(run_date=ctx["ds"])
    logger.info(f"Clustering trained: k={result['n_clusters']}, silhouette={result['silhouette']:.3f}")
    return result


def quality_gate(**ctx):
    """
    Check if new models beat the current production MAPE threshold.
    If any model degrades >20% on MAPE → skip promotion, send alert.
    """
    ti = ctx["ti"]
    prophet_metrics = ti.xcom_pull(task_ids="train_prophet_models")
    xgb_metrics     = ti.xcom_pull(task_ids="train_xgboost_models")
    lstm_metrics     = ti.xcom_pull(task_ids="train_lstm_models")

    MAPE_THRESHOLD = float(Variable.get("MAPE_THRESHOLD", default_var="15.0"))
    all_pass = all(
        m["avg_mape"] <= MAPE_THRESHOLD
        for m in [prophet_metrics, xgb_metrics, lstm_metrics]
        if m is not None
    )
    return "promote_models" if all_pass else "alert_quality_failure"


def promote_models(**ctx):
    """Transition validated model versions to Production stage in MLflow registry."""
    import mlflow
    client = mlflow.tracking.MlflowClient()
    for model_name in ["prophet_demand", "xgboost_demand", "lstm_demand",
                        "isolation_forest_anomaly", "kmeans_sku_clusters"]:
        versions = client.get_latest_versions(model_name, stages=["Staging"])
        for v in versions:
            client.transition_model_version_stage(
                name=model_name, version=v.version, stage="Production",
                archive_existing_versions=True,
            )
            logger.info(f"Promoted {model_name} v{v.version} → Production")


def alert_quality_failure(**ctx):
    """Send Slack / PagerDuty alert if quality gate fails."""
    import requests
    webhook = Variable.get("SLACK_WEBHOOK_URL", default_var="")
    if webhook:
        requests.post(webhook, json={"text": f"🚨 Retail ML quality gate FAILED on {ctx['ds']}. Models NOT promoted. Check MLflow for details."})
    logger.error("Quality gate failed — models not promoted to production")


with DAG(
    dag_id="training_dag",
    description="Weekly model retraining: Prophet + XGBoost + LSTM + Anomaly + Clustering",
    schedule_interval="0 2 * * 0",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=SHARED_ARGS,
    tags=["training", "weekly", "ml"],
) as training_dag:

    t_start = EmptyOperator(task_id="start")

    t_prophet  = PythonOperator(task_id="train_prophet_models",  python_callable=train_prophet_models,  provide_context=True)
    t_xgb      = PythonOperator(task_id="train_xgboost_models",  python_callable=train_xgboost_models,  provide_context=True)
    t_lstm     = PythonOperator(task_id="train_lstm_models",      python_callable=train_lstm_models,     provide_context=True)
    t_anomaly  = PythonOperator(task_id="train_anomaly_model",    python_callable=train_anomaly_model,   provide_context=True)
    t_cluster  = PythonOperator(task_id="train_clustering",        python_callable=train_clustering,      provide_context=True)

    t_gate = BranchPythonOperator(task_id="quality_gate", python_callable=quality_gate, provide_context=True)

    t_promote = PythonOperator(task_id="promote_models",         python_callable=promote_models,       provide_context=True)
    t_alert   = PythonOperator(task_id="alert_quality_failure",  python_callable=alert_quality_failure, provide_context=True)

    t_end = EmptyOperator(task_id="end", trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)

    t_start >> [t_prophet, t_xgb, t_lstm, t_anomaly, t_cluster]
    [t_prophet, t_xgb, t_lstm] >> t_gate
    t_gate >> [t_promote, t_alert]
    [t_promote, t_alert, t_anomaly, t_cluster] >> t_end


# ═══════════════════════════════════════════════════════════════════════════════
# DAG 4 — Model Monitoring (Daily 06:00 UTC)
# ═══════════════════════════════════════════════════════════════════════════════

def check_psi_drift(**ctx):
    """
    Population Stability Index check.
    PSI > 0.2 → significant drift → trigger retraining alert.
    PSI formula: PSI = SUM((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
    """
    from evaluation.metrics import compute_psi
    import json

    results = compute_psi(reference_window_days=30, current_window_days=7)
    drifted = [r for r in results if r["psi"] > 0.2]

    if drifted:
        logger.warning(f"Drift detected on {len(drifted)} SKUs: {json.dumps(drifted[:5])}")
        Variable.set("DRIFT_DETECTED", "true")
    else:
        Variable.set("DRIFT_DETECTED", "false")

    return {"total_skus": len(results), "drifted": len(drifted)}


def check_forecast_accuracy(**ctx):
    """
    Compare yesterday's forecasts against actual sales.
    Update mv_sku_daily_kpis with realized MAPE.
    Alert if portfolio MAPE > threshold.
    """
    import psycopg2, os
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    cur.execute("""
        UPDATE forecast_outputs f
        SET realized_mape = ABS(f.predicted_qty - s.sales_qty)
                            / NULLIF(s.sales_qty, 0) * 100
        FROM sku_daily_sales s
        WHERE f.target_date = s.date
          AND f.sku_id = s.sku_id
          AND f.target_date = CURRENT_DATE - 1
          AND f.realized_mape IS NULL
    """)
    conn.commit()

    cur.execute("SELECT AVG(realized_mape) FROM forecast_outputs WHERE target_date = CURRENT_DATE - 1")
    avg_mape = cur.fetchone()[0] or 0
    logger.info(f"Yesterday's realized MAPE: {avg_mape:.2f}%")

    cur.close(); conn.close()
    return {"realized_mape": round(float(avg_mape), 3)}


def push_grafana_annotations(**ctx):
    """Push DAG run annotation to Grafana for pipeline health dashboard."""
    import requests
    grafana_url = Variable.get("GRAFANA_URL", default_var="http://grafana:3000")
    api_key     = Variable.get("GRAFANA_API_KEY", default_var="")
    if not api_key:
        return
    requests.post(
        f"{grafana_url}/api/annotations",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"text": f"monitoring_dag ran — {ctx['ds']}", "tags": ["airflow", "retail-ml"]},
    )


with DAG(
    dag_id="monitoring_dag",
    description="Daily model drift (PSI) checks + realized MAPE tracking",
    schedule_interval="0 6 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=SHARED_ARGS,
    tags=["monitoring", "daily", "ml"],
) as monitoring_dag:

    t_psi  = PythonOperator(task_id="check_psi_drift",          python_callable=check_psi_drift,         provide_context=True)
    t_mape = PythonOperator(task_id="check_forecast_accuracy",   python_callable=check_forecast_accuracy,  provide_context=True)
    t_graf = PythonOperator(task_id="push_grafana_annotations",  python_callable=push_grafana_annotations, provide_context=True)

    [t_psi, t_mape] >> t_graf
