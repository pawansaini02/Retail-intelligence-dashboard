"""
spark_jobs/feature_engineering.py
───────────────────────────────────
PySpark Feature Engineering Pipeline

Reads:  raw tables  (sku_daily_sales, inventory_snapshots, promotions, weather_scores, competitor_events)
Writes: feature_store.sku_daily_features  (12-feature matrix used by all ML models)

Run:
  spark-submit --master local[4] spark_jobs/feature_engineering.py --date 2024-11-15

Features computed (per SKU × region × date):
  lag_7, lag_14, lag_21, lag_28
  rolling_mean_7, rolling_mean_28
  rolling_std_7
  day_of_week, month
  is_holiday, is_promotion
  weather_score
  yoy_ratio               (sales / same_day_last_year)
  competitor_price_delta  (% change vs last week)
"""

import argparse
from datetime import datetime, timedelta

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--date",          required=True, help="Processing date YYYY-MM-DD")
parser.add_argument("--output-table",  default="feature_store.sku_daily_features")
parser.add_argument("--lookback-days", type=int, default=400)
args = parser.parse_args()

RUN_DATE    = args.date
LOOKBACK    = args.lookback_days
OUT_TABLE   = args.output_table
DB_URL      = "jdbc:postgresql://localhost:5432/retail_intelligence"
DB_PROPS    = {"user": "retail", "password": "secret", "driver": "org.postgresql.Driver"}

# ── Spark Session ─────────────────────────────────────────────────────────────

spark = (
    SparkSession.builder
    .appName(f"RetailFeatureEngineering_{RUN_DATE}")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

start_date = (datetime.strptime(RUN_DATE, "%Y-%m-%d") - timedelta(days=LOOKBACK)).strftime("%Y-%m-%d")

print(f"[Feature Engineering] date={RUN_DATE}, lookback={LOOKBACK}d, output={OUT_TABLE}")

# ── Load Raw Tables ───────────────────────────────────────────────────────────

sales = (
    spark.read.jdbc(DB_URL, "sku_daily_sales", properties=DB_PROPS)
    .filter((F.col("date") >= start_date) & (F.col("date") <= RUN_DATE))
    .select("sku_id", "region", "date", "sales_qty", "is_holiday", "is_promotion")
)

weather = (
    spark.read.jdbc(DB_URL, "weather_scores", properties=DB_PROPS)
    .filter((F.col("date") >= start_date) & (F.col("date") <= RUN_DATE))
    .select("region", "date", "score")
    .withColumnRenamed("score", "weather_score")
)

competitor = (
    spark.read.jdbc(DB_URL, "competitor_events", properties=DB_PROPS)
    .filter((F.col("date") >= start_date) & (F.col("date") <= RUN_DATE))
    .select("sku_id", "date", "price_delta_pct")
    .withColumnRenamed("price_delta_pct", "competitor_price_delta")
)

# ── Join External Sources ─────────────────────────────────────────────────────

df = (
    sales
    .join(weather,     on=["region", "date"], how="left")
    .join(competitor,  on=["sku_id", "date"], how="left")
    .fillna({"weather_score": 0.0, "competitor_price_delta": 0.0})
)

# ── Window Specs ──────────────────────────────────────────────────────────────

sku_region_w     = Window.partitionBy("sku_id", "region").orderBy("date")
sku_region_7d_w  = sku_region_w.rowsBetween(-7,  -1)
sku_region_28d_w = sku_region_w.rowsBetween(-28, -1)

# ── Lag Features ──────────────────────────────────────────────────────────────

df = df \
    .withColumn("lag_7",  F.lag("sales_qty",  7).over(sku_region_w)) \
    .withColumn("lag_14", F.lag("sales_qty", 14).over(sku_region_w)) \
    .withColumn("lag_21", F.lag("sales_qty", 21).over(sku_region_w)) \
    .withColumn("lag_28", F.lag("sales_qty", 28).over(sku_region_w))

# ── Rolling Aggregates ────────────────────────────────────────────────────────

df = df \
    .withColumn("rolling_mean_7",  F.mean("sales_qty").over(sku_region_7d_w).cast(DoubleType())) \
    .withColumn("rolling_mean_28", F.mean("sales_qty").over(sku_region_28d_w).cast(DoubleType())) \
    .withColumn("rolling_std_7",   F.stddev("sales_qty").over(sku_region_7d_w).cast(DoubleType()))

# ── Calendar Features ─────────────────────────────────────────────────────────

df = df \
    .withColumn("day_of_week", F.dayofweek(F.col("date")).cast(IntegerType())) \
    .withColumn("month",       F.month(F.col("date")).cast(IntegerType()))

# ── Year-over-Year Ratio ──────────────────────────────────────────────────────

yoy_alias = (
    sales
    .withColumn("date_plus_365", F.date_add(F.col("date"), 365))
    .withColumnRenamed("sales_qty", "sales_qty_yoy")
    .select("sku_id", "region", F.col("date_plus_365").alias("date"), "sales_qty_yoy")
)

df = df.join(yoy_alias, on=["sku_id", "region", "date"], how="left")
df = df.withColumn(
    "yoy_ratio",
    (F.col("sales_qty") / (F.col("sales_qty_yoy") + 1)).cast(DoubleType())
)

# ── Drop nulls from lag window ────────────────────────────────────────────────

df = df.dropna(subset=["lag_7", "rolling_mean_7"])

# ── Final Feature Schema ──────────────────────────────────────────────────────

feature_cols = [
    "sku_id", "region", "date",
    "sales_qty",               # target
    "lag_7", "lag_14", "lag_21", "lag_28",
    "rolling_mean_7", "rolling_mean_28", "rolling_std_7",
    "day_of_week", "month",
    "is_holiday", "is_promotion",
    "weather_score", "competitor_price_delta",
    "yoy_ratio",
]
features_df = df.select(*feature_cols)

# ── Write to Feature Store ─────────────────────────────────────────────────────

(
    features_df
    .write
    .mode("overwrite")
    .partitionBy("date")
    .jdbc(DB_URL, OUT_TABLE, properties=DB_PROPS)
)

count = features_df.count()
print(f"[Feature Engineering] ✅ Written {count:,} feature rows to {OUT_TABLE}")

spark.stop()
