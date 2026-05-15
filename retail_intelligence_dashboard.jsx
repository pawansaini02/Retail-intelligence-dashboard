import { useState, useEffect, useRef } from "react";
import { LineChart, Line, BarChart, Bar, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Area, AreaChart, RadarChart, Radar, PolarGrid, PolarAngleAxis, Cell, PieChart, Pie } from "recharts";

// ─── Synthetic Data Generators ────────────────────────────────────────────────
const REGIONS = ["Punjab", "Maharashtra", "Karnataka", "Delhi NCR", "Tamil Nadu"];
const CATEGORIES = ["Electronics", "Apparel", "Grocery", "Home & Kitchen", "Sports"];
const SKUS = ["SKU-1042", "SKU-2187", "SKU-3301", "SKU-4560", "SKU-5891", "SKU-6234", "SKU-7780", "SKU-8102"];

function rand(min, max) { return Math.floor(Math.random() * (max - min + 1)) + min; }
function randf(min, max, d = 1) { return parseFloat((Math.random() * (max - min) + min).toFixed(d)); }

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function generateForecastData(region, sku) {
  const base = rand(800, 3000);
  return MONTHS.map((m, i) => {
    const seasonal = Math.sin((i / 11) * Math.PI * 2) * 300;
    const trend = i * 40;
    const actual = i < 8 ? base + seasonal + trend + rand(-150, 150) : null;
    const predicted = base + seasonal + trend + rand(-80, 80);
    const upper = predicted + rand(80, 180);
    const lower = Math.max(0, predicted - rand(60, 160));
    const isAnomaly = actual && Math.random() < 0.08;
    return { month: m, actual: actual ? Math.max(0, Math.round(actual + (isAnomaly ? rand(400, 800) : 0))) : null, predicted: Math.max(0, Math.round(predicted)), upper: Math.round(upper), lower: Math.max(0, Math.round(lower)), anomaly: isAnomaly ? Math.round(actual + rand(400, 800)) : null };
  });
}

function generateInventoryData() {
  return SKUS.map(sku => ({
    sku,
    currentStock: rand(50, 2000),
    reorderPoint: rand(100, 500),
    recommendedOrder: rand(200, 1500),
    leadTime: rand(2, 14),
    daysOfSupply: randf(5, 60),
    status: Math.random() < 0.2 ? "critical" : Math.random() < 0.3 ? "warning" : "healthy",
    category: CATEGORIES[rand(0, 4)],
    turnoverRate: randf(2, 12),
  }));
}

function generateRegionData() {
  return REGIONS.map(r => ({
    region: r,
    sales: rand(120000, 950000),
    demand: rand(100000, 900000),
    fulfillmentRate: randf(78, 99),
    stockouts: rand(2, 45),
    overstock: rand(5, 80),
    forecastAccuracy: randf(82, 97),
  }));
}

function generateSKUClusters() {
  return Array.from({ length: 40 }, (_, i) => ({
    sku: `SKU-${1000 + i}`,
    velocity: randf(0, 100),
    margin: randf(5, 65),
    cluster: rand(0, 3),
    volume: rand(100, 5000),
  }));
}

function generateKPIs() {
  return {
    forecastAccuracy: randf(91, 96),
    stockoutReduction: randf(28, 42),
    inventoryTurnover: randf(6.2, 9.8),
    overstockCostSaved: rand(420000, 980000),
    demandCoverage: randf(94, 99),
    anomaliesDetected: rand(12, 34),
  };
}

function generatePromoImpact() {
  return ["Diwali Sale", "Weekend Flash", "End of Season", "New Launch", "Clearance"].map(promo => ({
    promo,
    baseline: rand(1000, 4000),
    during: rand(3000, 12000),
    lift: randf(1.5, 4.5),
  }));
}

// ─── Color Palette ────────────────────────────────────────────────────────────
const C = {
  blue: "#378ADD", blueLight: "#B5D4F4", blueDark: "#0C447C",
  teal: "#1D9E75", tealLight: "#9FE1CB",
  coral: "#D85A30", coralLight: "#F5C4B3",
  purple: "#7F77DD", purpleLight: "#CECBF6",
  amber: "#BA7517", amberLight: "#FAC775",
  green: "#639922", greenLight: "#C0DD97",
  gray: "#888780", grayLight: "#D3D1C7",
  red: "#E24B4A",
};
const CLUSTER_COLORS = [C.blue, C.coral, C.teal, C.purple];

// ─── Sub-components ────────────────────────────────────────────────────────────

function KPICard({ label, value, unit, sub, color, icon }) {
  return (
    <div style={{ background: "var(--color-background-secondary)", borderRadius: 10, padding: "14px 16px", display: "flex", flexDirection: "column", gap: 4 }}>
      <div style={{ fontSize: 12, color: "var(--color-text-secondary)", fontWeight: 500, letterSpacing: "0.03em" }}>{icon} {label}</div>
      <div style={{ fontSize: 28, fontWeight: 500, color: color || "var(--color-text-primary)", lineHeight: 1.1 }}>
        {value}<span style={{ fontSize: 14, fontWeight: 400, marginLeft: 3 }}>{unit}</span>
      </div>
      {sub && <div style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>{sub}</div>}
    </div>
  );
}

function StatusBadge({ status }) {
  const cfg = {
    healthy: { bg: "var(--color-background-success)", color: "var(--color-text-success)", label: "Healthy" },
    warning: { bg: "var(--color-background-warning)", color: "var(--color-text-warning)", label: "Low Stock" },
    critical: { bg: "var(--color-background-danger)", color: "var(--color-text-danger)", label: "Critical" },
  }[status] || {};
  return <span style={{ background: cfg.bg, color: cfg.color, fontSize: 11, padding: "2px 8px", borderRadius: 6, fontWeight: 500 }}>{cfg.label}</span>;
}

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-secondary)", borderRadius: 8, padding: "10px 14px", fontSize: 12 }}>
      <div style={{ fontWeight: 500, marginBottom: 4 }}>{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color || "var(--color-text-secondary)", marginTop: 2 }}>
          {p.name}: <strong>{typeof p.value === "number" ? p.value.toLocaleString() : p.value}</strong>
        </div>
      ))}
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function RetailIntelligenceDashboard() {
  const [activeTab, setActiveTab] = useState("forecast");
  const [selectedRegion, setSelectedRegion] = useState("Punjab");
  const [selectedSKU, setSelectedSKU] = useState("SKU-1042");
  const [forecastData, setForecastData] = useState([]);
  const [inventoryData, setInventoryData] = useState([]);
  const [regionData, setRegionData] = useState([]);
  const [clusterData, setClusterData] = useState([]);
  const [kpis, setKpis] = useState({});
  const [promoData, setPromoData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [lastRefresh, setLastRefresh] = useState(new Date());

  function refresh() {
    setLoading(true);
    setTimeout(() => {
      setForecastData(generateForecastData(selectedRegion, selectedSKU));
      setInventoryData(generateInventoryData());
      setRegionData(generateRegionData());
      setClusterData(generateSKUClusters());
      setKpis(generateKPIs());
      setPromoData(generatePromoImpact());
      setLastRefresh(new Date());
      setLoading(false);
    }, 400);
  }

  useEffect(() => { refresh(); }, [selectedRegion, selectedSKU]);

  const tabs = [
    { id: "forecast", label: "Demand Forecast" },
    { id: "inventory", label: "Inventory Optimizer" },
    { id: "regions", label: "Regional Sales" },
    { id: "clusters", label: "SKU Clustering" },
    { id: "promotions", label: "Promo Impact" },
  ];

  return (
    <div style={{ fontFamily: "var(--font-sans, system-ui)", color: "var(--color-text-primary)", padding: "0 0 40px" }}>
      {/* Header */}
      <div style={{ borderBottom: "0.5px solid var(--color-border-tertiary)", paddingBottom: 16, marginBottom: 20 }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", flexWrap: "wrap", gap: 10 }}>
          <div>
            <div style={{ fontSize: 11, letterSpacing: "0.08em", color: "var(--color-text-tertiary)", marginBottom: 4, fontWeight: 500 }}>RETAIL AI — DEMAND & INVENTORY INTELLIGENCE</div>
            <h1 style={{ fontSize: 22, fontWeight: 500, margin: 0, lineHeight: 1.2 }}>Operations Command Center</h1>
            <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginTop: 4 }}>
              Live pipeline · Prophet + XGBoost + LSTM · Last refresh: {lastRefresh.toLocaleTimeString()}
            </div>
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <select value={selectedRegion} onChange={e => setSelectedRegion(e.target.value)} style={{ fontSize: 13, padding: "6px 10px" }}>
              {REGIONS.map(r => <option key={r}>{r}</option>)}
            </select>
            <select value={selectedSKU} onChange={e => setSelectedSKU(e.target.value)} style={{ fontSize: 13, padding: "6px 10px" }}>
              {SKUS.map(s => <option key={s}>{s}</option>)}
            </select>
            <button onClick={refresh} style={{ fontSize: 13, padding: "6px 14px", opacity: loading ? 0.5 : 1 }}>
              {loading ? "Refreshing…" : "↻ Refresh"}
            </button>
          </div>
        </div>
      </div>

      {/* KPI Row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 10, marginBottom: 24 }}>
        <KPICard label="Forecast Accuracy" value={kpis.forecastAccuracy?.toFixed(1)} unit="%" sub="MAPE < 8%" color={C.blue} icon="◎" />
        <KPICard label="Stockout Reduction" value={kpis.stockoutReduction?.toFixed(0)} unit="%" sub="vs. baseline" color={C.green} icon="↓" />
        <KPICard label="Inventory Turnover" value={kpis.inventoryTurnover?.toFixed(1)} unit="×" sub="annualized" color={C.teal} icon="⟳" />
        <KPICard label="Overstock Saved" value={"₹" + ((kpis.overstockCostSaved || 0) / 100000).toFixed(1) + "L"} sub="this quarter" color={C.coral} icon="₹" />
        <KPICard label="Demand Coverage" value={kpis.demandCoverage?.toFixed(1)} unit="%" sub="fill rate" color={C.purple} icon="✓" />
        <KPICard label="Anomalies Caught" value={kpis.anomaliesDetected} sub="auto-flagged" color={C.amber} icon="⚡" />
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 2, borderBottom: "0.5px solid var(--color-border-tertiary)", marginBottom: 24, overflowX: "auto" }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setActiveTab(t.id)} style={{
            background: "none", border: "none", padding: "10px 16px", fontSize: 13, fontWeight: activeTab === t.id ? 500 : 400,
            color: activeTab === t.id ? "var(--color-text-primary)" : "var(--color-text-secondary)",
            borderBottom: activeTab === t.id ? `2px solid ${C.blue}` : "2px solid transparent",
            cursor: "pointer", whiteSpace: "nowrap", transition: "all 0.15s"
          }}>{t.label}</button>
        ))}
      </div>

      {/* ── TAB: Demand Forecast ── */}
      {activeTab === "forecast" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          <div>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 4 }}>Demand Forecast — {selectedRegion} · {selectedSKU}</div>
            <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 16 }}>Actual sales vs. Prophet model prediction with 90% confidence interval. Anomalies flagged in amber.</div>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedForecastChart data={forecastData} />
            </ResponsiveContainer>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            <div>
              <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Seasonal Decomposition</div>
              <ResponsiveContainer width="100%" height={180}>
                <AreaChart data={forecastData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" />
                  <XAxis dataKey="month" tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} />
                  <YAxis tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Area type="monotone" dataKey="predicted" stroke={C.blue} fill={C.blueLight} fillOpacity={0.3} name="Predicted" strokeWidth={1.5} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <div>
              <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Model Accuracy by Month</div>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={forecastData.slice(0, 8).map(d => ({
                  month: d.month,
                  accuracy: d.actual && d.predicted ? Math.max(0, 100 - Math.abs(d.actual - d.predicted) / d.actual * 100) : 0,
                }))} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" />
                  <XAxis dataKey="month" tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} />
                  <YAxis domain={[60, 100]} tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="accuracy" name="Accuracy %" fill={C.teal} radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* ── TAB: Inventory Optimizer ── */}
      {activeTab === "inventory" && (
        <div>
          <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 4 }}>Restock Recommendations — All SKUs</div>
          <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 16 }}>EOQ-based reorder quantities with lead time and safety stock adjustments. Critical items flagged for immediate action.</div>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: "0.5px solid var(--color-border-secondary)" }}>
                  {["SKU", "Category", "Stock", "Reorder Point", "Rec. Order Qty", "Lead Time", "Days Supply", "Turnover", "Status"].map(h => (
                    <th key={h} style={{ padding: "8px 10px", textAlign: "left", fontWeight: 500, fontSize: 12, color: "var(--color-text-secondary)" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {inventoryData.map((row, i) => (
                  <tr key={i} style={{ borderBottom: "0.5px solid var(--color-border-tertiary)", background: row.status === "critical" ? "var(--color-background-danger)" : "transparent" }}>
                    <td style={{ padding: "9px 10px", fontWeight: 500 }}>{row.sku}</td>
                    <td style={{ padding: "9px 10px", color: "var(--color-text-secondary)" }}>{row.category}</td>
                    <td style={{ padding: "9px 10px" }}>{row.currentStock.toLocaleString()}</td>
                    <td style={{ padding: "9px 10px" }}>
                      <span style={{ color: row.currentStock < row.reorderPoint ? C.coral : "var(--color-text-primary)" }}>
                        {row.reorderPoint.toLocaleString()}
                        {row.currentStock < row.reorderPoint && " ⚠"}
                      </span>
                    </td>
                    <td style={{ padding: "9px 10px", color: C.blue, fontWeight: 500 }}>{row.recommendedOrder.toLocaleString()}</td>
                    <td style={{ padding: "9px 10px" }}>{row.leadTime}d</td>
                    <td style={{ padding: "9px 10px" }}>
                      <span style={{ color: row.daysOfSupply < 10 ? C.red : "var(--color-text-primary)" }}>{row.daysOfSupply.toFixed(1)}d</span>
                    </td>
                    <td style={{ padding: "9px 10px" }}>{row.turnoverRate.toFixed(1)}×</td>
                    <td style={{ padding: "9px 10px" }}><StatusBadge status={row.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ marginTop: 24 }}>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Stock vs Reorder Point — Visual Overview</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={inventoryData} margin={{ left: -10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" />
                <XAxis dataKey="sku" tick={{ fontSize: 10, fill: "var(--color-text-secondary)" }} />
                <YAxis tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="currentStock" name="Current Stock" fill={C.blue} radius={[3, 3, 0, 0]} />
                <Bar dataKey="reorderPoint" name="Reorder Point" fill={C.coral} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ── TAB: Regional Sales ── */}
      {activeTab === "regions" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          <div>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 4 }}>Sales & Demand by Region</div>
            <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 16 }}>Regional comparison of actual sales vs. predicted demand. Gap indicates potential unmet demand or over-supply.</div>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={regionData} margin={{ left: -5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" />
                <XAxis dataKey="region" tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} />
                <YAxis tickFormatter={v => "₹" + (v / 100000).toFixed(0) + "L"} tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} />
                <Tooltip content={<CustomTooltip />} formatter={v => "₹" + v.toLocaleString()} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="sales" name="Actual Sales" fill={C.blue} radius={[3, 3, 0, 0]} />
                <Bar dataKey="demand" name="Predicted Demand" fill={C.teal} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            <div>
              <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Forecast Accuracy by Region</div>
              <ResponsiveContainer width="100%" height={200}>
                <RadarChart data={regionData}>
                  <PolarGrid stroke="var(--color-border-tertiary)" />
                  <PolarAngleAxis dataKey="region" tick={{ fontSize: 10, fill: "var(--color-text-secondary)" }} />
                  <Radar name="Accuracy %" dataKey="forecastAccuracy" stroke={C.purple} fill={C.purple} fillOpacity={0.2} />
                  <Tooltip content={<CustomTooltip />} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
            <div>
              <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Stockout Events by Region</div>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={regionData} layout="vertical" margin={{ left: 60, right: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" horizontal={false} />
                  <XAxis type="number" tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} />
                  <YAxis type="category" dataKey="region" tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} width={60} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="stockouts" name="Stockouts" fill={C.coral} radius={[0, 3, 3, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Regional Performance Summary</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 10 }}>
              {regionData.map((r, i) => (
                <div key={i} style={{ background: "var(--color-background-secondary)", borderRadius: 10, padding: "12px 14px" }}>
                  <div style={{ fontWeight: 500, marginBottom: 8 }}>{r.region}</div>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
                    <span style={{ color: "var(--color-text-secondary)" }}>Sales</span>
                    <span>₹{(r.sales / 100000).toFixed(1)}L</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
                    <span style={{ color: "var(--color-text-secondary)" }}>Accuracy</span>
                    <span style={{ color: r.forecastAccuracy > 90 ? C.green : C.amber }}>{r.forecastAccuracy.toFixed(1)}%</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
                    <span style={{ color: "var(--color-text-secondary)" }}>Stockouts</span>
                    <span style={{ color: r.stockouts > 30 ? C.red : "var(--color-text-primary)" }}>{r.stockouts}</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                    <span style={{ color: "var(--color-text-secondary)" }}>Fill rate</span>
                    <span>{r.fulfillmentRate.toFixed(1)}%</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── TAB: SKU Clustering ── */}
      {activeTab === "clusters" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          <div>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 4 }}>SKU Clusters — Velocity vs. Margin Analysis</div>
            <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 8 }}>KMeans clustering on sales velocity, gross margin, and volume. Bubble size = order volume. Use to prioritize stocking strategy per segment.</div>
            <div style={{ display: "flex", gap: 16, fontSize: 12, marginBottom: 16, flexWrap: "wrap" }}>
              {["Star SKUs (high vel, high margin)", "Cash Cows (low vel, high margin)", "Volume Movers (high vel, low margin)", "Tail SKUs (low vel, low margin)"].map((l, i) => (
                <span key={i} style={{ display: "flex", alignItems: "center", gap: 5, color: "var(--color-text-secondary)" }}>
                  <span style={{ width: 10, height: 10, borderRadius: 2, background: CLUSTER_COLORS[i], display: "inline-block" }} />{l}
                </span>
              ))}
            </div>
            <ResponsiveContainer width="100%" height={320}>
              <ScatterChart margin={{ top: 10, right: 20, left: -10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" />
                <XAxis dataKey="velocity" name="Sales Velocity" label={{ value: "Sales Velocity", position: "insideBottom", offset: -5, fontSize: 11, fill: "var(--color-text-secondary)" }} tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} />
                <YAxis dataKey="margin" name="Gross Margin %" label={{ value: "Gross Margin %", angle: -90, position: "insideLeft", fontSize: 11, fill: "var(--color-text-secondary)" }} tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const d = payload[0]?.payload;
                  return (
                    <div style={{ background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-secondary)", borderRadius: 8, padding: "10px 14px", fontSize: 12 }}>
                      <div style={{ fontWeight: 500, marginBottom: 4 }}>{d?.sku}</div>
                      <div>Velocity: {d?.velocity?.toFixed(1)}</div>
                      <div>Margin: {d?.margin?.toFixed(1)}%</div>
                      <div>Volume: {d?.volume?.toLocaleString()}</div>
                      <div>Cluster: {["Star", "Cash Cow", "Volume Mover", "Tail"][d?.cluster]}</div>
                    </div>
                  );
                }} />
                {[0, 1, 2, 3].map(c => (
                  <Scatter key={c} name={c} data={clusterData.filter(d => d.cluster === c)} fill={CLUSTER_COLORS[c]} opacity={0.8} />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Cluster Distribution</div>
            <ResponsiveContainer width="100%" height={180}>
              <PieChart>
                <Pie data={[0, 1, 2, 3].map(c => ({ name: ["Star", "Cash Cow", "Volume Mover", "Tail"][c], value: clusterData.filter(d => d.cluster === c).length }))} dataKey="value" cx="50%" cy="50%" outerRadius={75} label={({ name, value }) => `${name}: ${value}`} labelLine={true} fontSize={11}>
                  {[0, 1, 2, 3].map(c => <Cell key={c} fill={CLUSTER_COLORS[c]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ── TAB: Promo Impact ── */}
      {activeTab === "promotions" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          <div>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 4 }}>Promotion Lift Analysis</div>
            <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 16 }}>Baseline vs. actual sales during promotional events. Lift multiplier quantifies campaign effectiveness.</div>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={promoData} margin={{ left: -5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" />
                <XAxis dataKey="promo" tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} />
                <YAxis tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="baseline" name="Baseline Sales" fill={C.grayLight} radius={[3, 3, 0, 0]} />
                <Bar dataKey="during" name="During Promo" fill={C.blue} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Campaign Lift Multiplier</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 10 }}>
              {promoData.map((p, i) => (
                <div key={i} style={{ background: "var(--color-background-secondary)", borderRadius: 10, padding: "14px 16px" }}>
                  <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 4 }}>{p.promo}</div>
                  <div style={{ fontSize: 26, fontWeight: 500, color: p.lift > 3 ? C.green : p.lift > 2 ? C.blue : C.amber }}>{p.lift.toFixed(2)}×</div>
                  <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginTop: 2 }}>
                    +{((p.during - p.baseline) / p.baseline * 100).toFixed(0)}% uplift
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 12 }}>Demand Spike Detection — Anomaly Timeline</div>
            <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 12 }}>
              Isolation Forest flags unexpected spikes outside ±2σ of rolling forecast window.
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={forecastData} margin={{ left: -20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" />
                <XAxis dataKey="month" tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} />
                <YAxis tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="upper" stroke="none" fill={C.blueLight} fillOpacity={0.2} name="Upper bound" />
                <Area type="monotone" dataKey="lower" stroke="none" fill="white" name="Lower bound" />
                <Line type="monotone" dataKey="actual" stroke={C.blue} strokeWidth={2} dot={(p) => p.payload.anomaly ? <circle cx={p.cx} cy={p.cy} r={6} fill={C.coral} stroke="white" strokeWidth={1.5} /> : null} name="Actual" connectNulls={false} />
                <Line type="monotone" dataKey="predicted" stroke={C.gray} strokeWidth={1.5} strokeDasharray="5 3" name="Predicted" />
              </AreaChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginTop: 6 }}>
              ● Amber dots = detected anomalies (Isolation Forest score &gt; threshold). Dashed = model prediction.
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div style={{ borderTop: "0.5px solid var(--color-border-tertiary)", marginTop: 32, paddingTop: 14, display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
        <div style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>
          Stack: Prophet · XGBoost · LSTM · Isolation Forest · FastAPI · PostgreSQL · Kafka · Airflow · MLflow · Docker
        </div>
        <div style={{ fontSize: 11, color: "var(--color-text-tertiary)" }}>
          AI-Powered Retail Intelligence System — Portfolio Project
        </div>
      </div>
    </div>
  );
}

// ─── Composed Forecast Chart (custom — recharts ComposedChart) ─────────────────
import { ComposedChart, ComposedChart as CC } from "recharts";

function ComposedForecastChart({ data }) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" />
        <XAxis dataKey="month" tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} />
        <YAxis tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Area type="monotone" dataKey="upper" stroke="none" fill={C.blueLight} fillOpacity={0.25} name="Confidence band" legendType="none" />
        <Area type="monotone" dataKey="lower" stroke="none" fill="white" fillOpacity={1} legendType="none" />
        <Line type="monotone" dataKey="predicted" stroke={C.blue} strokeWidth={2} strokeDasharray="5 3" dot={false} name="Forecast (Prophet)" />
        <Line type="monotone" dataKey="actual" stroke={C.teal} strokeWidth={2.5} dot={{ r: 3, fill: C.teal }} name="Actual Sales" connectNulls={false} />
        <ReferenceLine x="Aug" stroke={C.coral} strokeDasharray="4 3" label={{ value: "Forecast window →", position: "top", fontSize: 11, fill: C.coral }} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
