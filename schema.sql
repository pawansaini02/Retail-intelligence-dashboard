-- =============================================================================
-- Retail Demand & Inventory Intelligence System — PostgreSQL Schema
-- =============================================================================
-- Run order: 1. extensions → 2. tables → 3. indexes → 4. views → 5. functions
-- DB: retail_intelligence
-- =============================================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";       -- fuzzy search on sku names

-- =============================================================================
-- DIMENSION TABLES
-- =============================================================================

CREATE TABLE regions (
    region_id   SERIAL PRIMARY KEY,
    name        VARCHAR(80) UNIQUE NOT NULL,
    state       VARCHAR(50),
    country     CHAR(2) DEFAULT 'IN',
    lat         NUMERIC(9,6),
    lng         NUMERIC(9,6),
    tier        SMALLINT DEFAULT 1   -- 1=metro, 2=tier2, 3=tier3
);

CREATE TABLE products (
    sku_id          VARCHAR(20) PRIMARY KEY,
    product_name    VARCHAR(200) NOT NULL,
    category        VARCHAR(50)  NOT NULL,
    subcategory     VARCHAR(50),
    brand           VARCHAR(100),
    unit_cost_inr   NUMERIC(10,2) NOT NULL DEFAULT 0,
    unit_price_inr  NUMERIC(10,2) NOT NULL DEFAULT 0,
    weight_kg       NUMERIC(6,3),
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_products_category ON products (category);
CREATE INDEX idx_products_name_trgm ON products USING gin (product_name gin_trgm_ops);

CREATE TABLE warehouses (
    warehouse_id    VARCHAR(20) PRIMARY KEY,
    name            VARCHAR(100),
    region          VARCHAR(80) REFERENCES regions (name),
    capacity_units  INTEGER,
    lat             NUMERIC(9,6),
    lng             NUMERIC(9,6)
);

CREATE TABLE promotions (
    promo_id        VARCHAR(30) PRIMARY KEY,
    name            VARCHAR(150) NOT NULL,
    promo_type      VARCHAR(30),    -- discount | bogo | bundle | loyalty
    discount_pct    NUMERIC(5,2),
    start_date      DATE NOT NULL,
    end_date        DATE NOT NULL,
    applicable_skus TEXT[],         -- NULL means all SKUs
    applicable_regions TEXT[],      -- NULL means all regions
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- FACT TABLES
-- =============================================================================

CREATE TABLE sku_daily_sales (
    id              BIGSERIAL PRIMARY KEY,
    sku_id          VARCHAR(20)   NOT NULL REFERENCES products (sku_id),
    region          VARCHAR(80)   NOT NULL,
    date            DATE          NOT NULL,
    sales_qty       INTEGER       NOT NULL DEFAULT 0,
    revenue_inr     NUMERIC(14,2) NOT NULL DEFAULT 0,
    returns_qty     INTEGER                DEFAULT 0,
    is_holiday      BOOLEAN                DEFAULT FALSE,
    is_promotion    BOOLEAN                DEFAULT FALSE,
    promo_id        VARCHAR(30)   REFERENCES promotions (promo_id),
    weather_score   NUMERIC(6,3),
    created_at      TIMESTAMPTZ   DEFAULT NOW(),
    UNIQUE (sku_id, region, date)
);

CREATE INDEX idx_sales_sku_region_date ON sku_daily_sales (sku_id, region, date DESC);
CREATE INDEX idx_sales_date            ON sku_daily_sales (date DESC);
CREATE INDEX idx_sales_region          ON sku_daily_sales (region, date DESC);

CREATE TABLE inventory_snapshots (
    id                  BIGSERIAL PRIMARY KEY,
    sku_id              VARCHAR(20)   NOT NULL REFERENCES products (sku_id),
    warehouse_id        VARCHAR(20)   REFERENCES warehouses (warehouse_id),
    region              VARCHAR(80),
    snapshot_date       DATE          NOT NULL,
    quantity_on_hand    INTEGER       NOT NULL DEFAULT 0,
    quantity_on_order   INTEGER                DEFAULT 0,
    quantity_reserved   INTEGER                DEFAULT 0,
    reorder_point       INTEGER,
    lead_time_days      SMALLINT,
    unit_cost_inr       NUMERIC(10,2),
    created_at          TIMESTAMPTZ   DEFAULT NOW(),
    UNIQUE (sku_id, warehouse_id, snapshot_date)
);

CREATE INDEX idx_inv_sku_date     ON inventory_snapshots (sku_id, snapshot_date DESC);
CREATE INDEX idx_inv_region_date  ON inventory_snapshots (region, snapshot_date DESC);

CREATE TABLE forecast_outputs (
    id              BIGSERIAL PRIMARY KEY,
    sku_id          VARCHAR(20)   NOT NULL REFERENCES products (sku_id),
    region          VARCHAR(80)   NOT NULL,
    forecast_date   DATE          NOT NULL,   -- when forecast was generated
    target_date     DATE          NOT NULL,   -- date being forecasted
    predicted_qty   NUMERIC(12,1) NOT NULL,
    lower_bound     NUMERIC(12,1),
    upper_bound     NUMERIC(12,1),
    model_name      VARCHAR(30)   DEFAULT 'ensemble',
    model_version   VARCHAR(60),
    mape            NUMERIC(6,3),             -- error against actual (filled post-hoc)
    realized_mape   NUMERIC(6,3),             -- filled when actual arrives
    created_at      TIMESTAMPTZ   DEFAULT NOW(),
    UNIQUE (sku_id, region, forecast_date, target_date, model_name)
);

CREATE INDEX idx_fc_sku_region_target ON forecast_outputs (sku_id, region, target_date DESC);
CREATE INDEX idx_fc_forecast_date     ON forecast_outputs (forecast_date DESC);

CREATE TABLE anomaly_events (
    id              BIGSERIAL PRIMARY KEY,
    sku_id          VARCHAR(20)   NOT NULL REFERENCES products (sku_id),
    region          VARCHAR(80)   NOT NULL,
    event_date      DATE          NOT NULL,
    actual_value    NUMERIC(12,1),
    expected_value  NUMERIC(12,1),
    anomaly_score   NUMERIC(6,4)  NOT NULL,
    direction       VARCHAR(10)   CHECK (direction IN ('spike','drop')),
    probable_cause  VARCHAR(150),
    resolved        BOOLEAN       DEFAULT FALSE,
    resolved_by     VARCHAR(60),
    resolved_at     TIMESTAMPTZ,
    notes           TEXT,
    created_at      TIMESTAMPTZ   DEFAULT NOW()
);

CREATE INDEX idx_anomaly_date  ON anomaly_events (event_date DESC);
CREATE INDEX idx_anomaly_score ON anomaly_events (anomaly_score DESC);
CREATE INDEX idx_anomaly_sku   ON anomaly_events (sku_id, event_date DESC);

CREATE TABLE weather_scores (
    id          BIGSERIAL PRIMARY KEY,
    region      VARCHAR(80) NOT NULL,
    date        DATE        NOT NULL,
    score       NUMERIC(5,3),
    temp_c      NUMERIC(5,2),
    condition   VARCHAR(50),
    UNIQUE (region, date)
);

CREATE TABLE competitor_events (
    id              BIGSERIAL PRIMARY KEY,
    sku_id          VARCHAR(20)   REFERENCES products (sku_id),
    date            DATE          NOT NULL,
    competitor_name VARCHAR(80),
    price_delta_pct NUMERIC(7,3),
    source_url      TEXT,
    created_at      TIMESTAMPTZ   DEFAULT NOW()
);

CREATE TABLE sku_return_rates (
    sku_id          VARCHAR(20) PRIMARY KEY REFERENCES products (sku_id),
    return_rate_pct NUMERIC(6,3),
    computed_date   DATE DEFAULT CURRENT_DATE
);

CREATE TABLE promo_lift_summary (
    sku_id      VARCHAR(20) REFERENCES products (sku_id),
    promo_id    VARCHAR(30) REFERENCES promotions (promo_id),
    lift_ratio  NUMERIC(7,4),
    PRIMARY KEY (sku_id, promo_id)
);

-- MLflow tracking DB is separate; this table stores summary references
CREATE TABLE model_versions (
    id              SERIAL PRIMARY KEY,
    model_name      VARCHAR(80)   NOT NULL,
    version         INTEGER       NOT NULL,
    stage           VARCHAR(20)   DEFAULT 'Staging',
    mlflow_run_id   VARCHAR(36),
    avg_mape        NUMERIC(6,3),
    avg_rmse        NUMERIC(10,2),
    promoted_at     TIMESTAMPTZ,
    promoted_by     VARCHAR(60),
    created_at      TIMESTAMPTZ   DEFAULT NOW(),
    UNIQUE (model_name, version)
);

-- Feature store (written by PySpark, read by training scripts)
CREATE TABLE feature_store (
    sku_id                  VARCHAR(20) NOT NULL,
    region                  VARCHAR(80) NOT NULL,
    date                    DATE        NOT NULL,
    sales_qty               INTEGER,
    lag_7                   NUMERIC(12,2),
    lag_14                  NUMERIC(12,2),
    lag_21                  NUMERIC(12,2),
    lag_28                  NUMERIC(12,2),
    rolling_mean_7          NUMERIC(12,4),
    rolling_mean_28         NUMERIC(12,4),
    rolling_std_7           NUMERIC(12,4),
    day_of_week             SMALLINT,
    month                   SMALLINT,
    is_holiday              BOOLEAN DEFAULT FALSE,
    is_promotion            BOOLEAN DEFAULT FALSE,
    weather_score           NUMERIC(6,3),
    competitor_price_delta  NUMERIC(7,3),
    yoy_ratio               NUMERIC(8,4),
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (sku_id, region, date)
) PARTITION BY RANGE (date);

-- Monthly partitions (extend as needed)
CREATE TABLE feature_store_2024_01 PARTITION OF feature_store FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE feature_store_2024_02 PARTITION OF feature_store FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
-- ... extend to current month

CREATE INDEX idx_feature_store_sku_date ON feature_store (sku_id, region, date DESC);

-- =============================================================================
-- MATERIALIZED VIEWS (refreshed nightly by Airflow)
-- =============================================================================

CREATE MATERIALIZED VIEW mv_sku_daily_kpis AS
SELECT
    s.sku_id,
    s.region,
    s.date,
    s.sales_qty                                           AS actual_sales,
    f.predicted_qty                                       AS predicted_demand,
    ROUND(
        ABS(s.sales_qty - f.predicted_qty)
        / NULLIF(s.sales_qty, 0) * 100, 3
    )                                                     AS mape,
    CASE WHEN i.quantity_on_hand = 0 THEN 1 ELSE 0 END   AS stockout_events,
    CASE WHEN i.quantity_on_hand > i.reorder_point * 2
         THEN 1 ELSE 0 END                                AS overstock_events,
    COALESCE(i.quantity_on_hand, 0)                       AS stock_level,
    COALESCE(i.reorder_point, 0)                          AS reorder_point,
    s.is_promotion,
    s.promo_id
FROM sku_daily_sales s
LEFT JOIN forecast_outputs f
    ON f.sku_id = s.sku_id AND f.region = s.region
    AND f.target_date = s.date AND f.model_name = 'ensemble'
LEFT JOIN inventory_snapshots i
    ON i.sku_id = s.sku_id AND i.snapshot_date = s.date
WITH DATA;

CREATE UNIQUE INDEX ON mv_sku_daily_kpis (sku_id, region, date);

-- Regional rollup view
CREATE MATERIALIZED VIEW mv_regional_kpis AS
SELECT
    region,
    date,
    SUM(actual_sales)                 AS total_sales,
    SUM(predicted_demand)             AS total_predicted_demand,
    AVG(mape)                         AS forecast_accuracy,
    SUM(stockout_events)              AS stockout_events,
    SUM(overstock_events)             AS overstock_events,
    ROUND(
        (1 - SUM(stockout_events)::NUMERIC / NULLIF(COUNT(*), 0)) * 100, 2
    )                                 AS fulfillment_rate
FROM mv_sku_daily_kpis
GROUP BY region, date
WITH DATA;

CREATE UNIQUE INDEX ON mv_regional_kpis (region, date);

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Refresh all materialized views (called by Airflow refresh task)
CREATE OR REPLACE FUNCTION refresh_all_views()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_sku_daily_kpis;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_regional_kpis;
END;
$$ LANGUAGE plpgsql;

-- Get current stockout risk summary per region
CREATE OR REPLACE FUNCTION stockout_risk_summary(p_region VARCHAR DEFAULT NULL)
RETURNS TABLE (region VARCHAR, critical_count INT, warning_count INT, healthy_count INT) AS $$
BEGIN
    RETURN QUERY
    SELECT
        i.region,
        COUNT(*) FILTER (WHERE i.quantity_on_hand / NULLIF(d.avg_daily, 0) < 5)::INT  AS critical_count,
        COUNT(*) FILTER (WHERE i.quantity_on_hand / NULLIF(d.avg_daily, 0) BETWEEN 5 AND 14)::INT AS warning_count,
        COUNT(*) FILTER (WHERE i.quantity_on_hand / NULLIF(d.avg_daily, 0) > 14)::INT  AS healthy_count
    FROM inventory_snapshots i
    JOIN (
        SELECT sku_id, region, AVG(sales_qty) AS avg_daily
        FROM sku_daily_sales
        WHERE date >= CURRENT_DATE - 30
        GROUP BY sku_id, region
    ) d ON d.sku_id = i.sku_id AND d.region = i.region
    WHERE i.snapshot_date = CURRENT_DATE - 1
      AND (p_region IS NULL OR i.region = p_region)
    GROUP BY i.region;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- SEED SAMPLE DATA (remove in production)
-- =============================================================================

INSERT INTO regions (name, state, tier) VALUES
    ('Punjab',      'Punjab',      2),
    ('Maharashtra', 'Maharashtra', 1),
    ('Karnataka',   'Karnataka',   1),
    ('Delhi NCR',   'Delhi',       1),
    ('Tamil Nadu',  'Tamil Nadu',  1);

INSERT INTO products (sku_id, product_name, category, unit_cost_inr, unit_price_inr) VALUES
    ('SKU-1042', 'Wireless Headphones Pro',  'Electronics',    1200, 2499),
    ('SKU-2187', 'Cotton Kurta Set',         'Apparel',          350,  799),
    ('SKU-3301', 'Basmati Rice 5kg',         'Grocery',          280,  420),
    ('SKU-4560', 'Non-stick Cookware Set',   'Home & Kitchen',  1800, 3499),
    ('SKU-5891', 'Yoga Mat Premium',         'Sports',           600, 1299),
    ('SKU-6234', 'Smart LED Bulb Pack',      'Electronics',      180,  549),
    ('SKU-7780', 'Denim Jeans Slim Fit',     'Apparel',          450, 1199),
    ('SKU-8102', 'Organic Pulses Combo',     'Grocery',          320,  599);

COMMENT ON TABLE sku_daily_sales    IS 'Core fact table: one row per SKU × region × date';
COMMENT ON TABLE inventory_snapshots IS 'Daily warehouse inventory levels from ERP sync';
COMMENT ON TABLE forecast_outputs   IS 'Model predictions: filled at forecast time, evaluated post-hoc';
COMMENT ON TABLE anomaly_events     IS 'Isolation Forest flagged sales anomalies';
COMMENT ON TABLE feature_store      IS 'Precomputed ML feature matrix (PySpark output)';
