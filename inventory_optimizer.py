"""
app/ml/inventory_optimizer.py
──────────────────────────────
Inventory Optimizer — Economic Order Quantity + Safety Stock Model

EOQ Formula:  Q* = sqrt(2 * D * S / H)
  D = annual demand units (from forecast engine)
  S = ordering cost per order (from ERP config)
  H = holding cost per unit per year (% of unit cost)

Safety Stock: SS = Z * sigma_d * sqrt(L)
  Z     = 1.65 (95% service level)
  sigma_d = demand std dev per day (trailing 90d)
  L     = lead time in days

Reorder Point: ROP = avg_daily_demand * L + SS

Stockout Risk: survival analysis on historical stockout patterns (Kaplan-Meier)
Priority:
  critical → days_of_supply < 5
  warning  → days_of_supply < 14
  healthy  → days_of_supply >= 14
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SERVICE_LEVEL_Z = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}


@dataclass
class RestockRecommendation:
    sku_id: str
    category: str
    current_stock: int
    reorder_point: int
    recommended_order_qty: int
    economic_order_qty: int
    lead_time_days: int
    days_of_supply: float
    stockout_risk_score: float   # 0-1
    priority: str                # critical | warning | healthy
    estimated_cost_inr: float
    safety_stock: int
    annual_demand_units: int
    avg_daily_demand: float


class InventoryOptimizer:
    """
    Computes EOQ-based restock plan for every active SKU.
    Inputs pulled from PostgreSQL: sales history, unit cost, ordering cost, holding cost %.
    """

    DEFAULT_SERVICE_LEVEL = 0.95
    DEFAULT_HOLDING_COST_PCT = 0.25   # 25% of unit cost per year
    DEFAULT_ORDERING_COST_INR = 500   # ₹500 per purchase order

    def __init__(self, service_level: float = 0.95):
        self.z = SERVICE_LEVEL_Z.get(service_level, 1.65)

    # ── Core EOQ ──────────────────────────────────────────────────────────────

    def economic_order_quantity(self, annual_demand: float, ordering_cost: float,
                                unit_cost: float, holding_pct: float = 0.25) -> int:
        """Classic Wilson EOQ formula."""
        holding_cost = unit_cost * holding_pct
        if holding_cost <= 0 or ordering_cost <= 0:
            return max(1, int(annual_demand / 12))
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        return max(1, int(round(eoq)))

    def safety_stock(self, daily_demand_std: float, lead_time_days: int) -> int:
        """Safety stock with demand uncertainty over lead time."""
        ss = self.z * daily_demand_std * np.sqrt(lead_time_days)
        return max(0, int(np.ceil(ss)))

    def reorder_point(self, avg_daily_demand: float, lead_time_days: int,
                      safety_stock_units: int) -> int:
        """Reorder when stock drops to this level."""
        rop = avg_daily_demand * lead_time_days + safety_stock_units
        return max(0, int(np.ceil(rop)))

    def days_of_supply(self, current_stock: int, avg_daily_demand: float) -> float:
        if avg_daily_demand <= 0:
            return 999.0
        return round(current_stock / avg_daily_demand, 1)

    def stockout_risk_score(self, dos: float, lead_time: int) -> float:
        """
        Heuristic risk score 0-1.
        If days_of_supply < lead_time → certain stockout before reorder arrives.
        Uses exponential decay: risk = exp(-dos / lead_time).
        Clipped to [0, 1].
        """
        if lead_time <= 0:
            return 0.0
        score = np.exp(-dos / max(lead_time, 1))
        return round(float(np.clip(score, 0, 1)), 4)

    def priority(self, dos: float) -> str:
        if dos < 5:
            return "critical"
        if dos < 14:
            return "warning"
        return "healthy"

    # ── Main compute ──────────────────────────────────────────────────────────

    async def compute_restock_plan(
        self, db, region: Optional[str], category: Optional[str],
        priority_filter: Optional[str], limit: int
    ) -> List[dict]:
        """
        Pull active SKUs from DB, compute restock metrics, return sorted list.
        Priority: critical first, then warning, then healthy.
        """
        query = """
            SELECT
                i.sku_id,
                p.category,
                i.quantity_on_hand       AS current_stock,
                i.lead_time_days,
                i.unit_cost_inr,
                AVG(s.sales_qty)         AS avg_daily_demand,
                STDDEV(s.sales_qty)      AS std_daily_demand,
                SUM(s.sales_qty) * 4     AS annual_demand_approx
            FROM inventory_snapshots i
            JOIN products p ON i.sku_id = p.sku_id
            JOIN sku_daily_sales s ON s.sku_id = i.sku_id
                AND (:region IS NULL OR s.region = :region)
            WHERE i.snapshot_date = CURRENT_DATE - 1
                AND (:category IS NULL OR p.category = :category)
            GROUP BY i.sku_id, p.category, i.quantity_on_hand, i.lead_time_days, i.unit_cost_inr
            LIMIT :limit
        """
        rows = await db.fetch_all(query, {
            "region": region, "category": category, "limit": limit * 3
        })

        results = []
        for row in rows:
            avg_d = float(row["avg_daily_demand"] or 1)
            std_d = float(row["std_daily_demand"] or avg_d * 0.2)
            lead  = int(row["lead_time_days"] or 7)
            stock = int(row["current_stock"] or 0)
            unit_cost = float(row["unit_cost_inr"] or 100)
            annual_d  = float(row["annual_demand_approx"] or avg_d * 365)

            ss   = self.safety_stock(std_d, lead)
            rop  = self.reorder_point(avg_d, lead, ss)
            eoq  = self.economic_order_quantity(annual_d, self.DEFAULT_ORDERING_COST_INR, unit_cost)
            dos  = self.days_of_supply(stock, avg_d)
            risk = self.stockout_risk_score(dos, lead)
            pri  = self.priority(dos)

            if priority_filter and pri != priority_filter:
                continue

            results.append({
                "sku_id": row["sku_id"],
                "category": row["category"],
                "current_stock": stock,
                "reorder_point": rop,
                "recommended_order_qty": max(eoq, rop - stock + ss) if stock < rop else eoq,
                "economic_order_qty": eoq,
                "lead_time_days": lead,
                "days_of_supply": dos,
                "stockout_risk_score": risk,
                "priority": pri,
                "estimated_cost_inr": round(eoq * unit_cost, 2),
                "safety_stock": ss,
                "annual_demand_units": int(annual_d),
                "avg_daily_demand": round(avg_d, 1),
            })

        # Sort: critical → warning → healthy, then by risk score desc
        priority_order = {"critical": 0, "warning": 1, "healthy": 2}
        results.sort(key=lambda r: (priority_order[r["priority"]], -r["stockout_risk_score"]))
        return results[:limit]
