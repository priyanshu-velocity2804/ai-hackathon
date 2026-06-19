"""
Engine 1 — Product weight estimator.

Cascade:
  1. SKU history  (sorter median dead weight)
  2. Name parser  (category × pack-qty × tare)
  3. Declared weight prior / sanity bound

Returns: predicted_dead_weight_kg, confidence, drives_slab, basis, notes
"""

from __future__ import annotations
import statistics
from dataclasses import dataclass, field

import pandas as pd

from .db import ch_query
from .parse import ParseResult, parse_sku_name
from .slab import billable_kg, drives_slab, slab, volumetric_kg


@dataclass
class WeightResult:
    predicted_dead_weight_kg: float
    confidence: float          # 0–1
    drives_slab: bool          # True if dead weight ≥ volumetric
    basis: str                 # "sku_history" | "name_parse" | "declared"
    target_slab: float
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 1 — SKU history from ClickHouse
# ---------------------------------------------------------------------------

def _sku_sorter_history(sku: str, client_id: str) -> list[float]:
    """Return list of sorter dead weights (kg) for this SKU/client."""
    df = ch_query("""
        SELECT min_sorter_weight
        FROM shipfast_weight_discrepancy.shipfast_weight_discrepancy
        WHERE client_id = {cid:String}
          AND sku = {sku:String}
          AND min_sorter_weight > 0
        ORDER BY shipment_created_at DESC
        LIMIT 200
    """, {"cid": client_id, "sku": sku})
    if df.empty or "min_sorter_weight" not in df.columns:
        return []
    return df["min_sorter_weight"].dropna().tolist()


def _sku_sorter_dims(sku: str, client_id: str) -> dict | None:
    """Return median sorter dims for volumetric check."""
    df = ch_query("""
        SELECT
            median(min_sorter_length) AS l,
            median(min_sorter_width)  AS w,
            median(min_sorter_height) AS h,
            median(max_dead_vol_sorter) AS billable
        FROM shipfast_weight_discrepancy.shipfast_weight_discrepancy
        WHERE client_id = {cid:String}
          AND sku = {sku:String}
          AND min_sorter_weight > 0
    """, {"cid": client_id, "sku": sku})
    if df.empty or not {"l", "w", "h", "billable"}.issubset(df.columns):
        return None
    if df["l"].iloc[0] is None:
        return None
    row = df.iloc[0]
    try:
        return {"l": float(row["l"]), "w": float(row["w"]), "h": float(row["h"]),
                "billable": float(row["billable"])}
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Step 2 — Name parse
# ---------------------------------------------------------------------------

def _name_parse_weight_kg(
    sku_name: str,
    quantity: int,
    tare_g: float = 100.0,
) -> tuple[float, ParseResult]:
    result = parse_sku_name(sku_name, quantity=quantity, tare_g=tare_g)
    return result.estimated_total_g / 1000.0, result


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

def estimate_weight(
    sku: str,
    sku_name: str,
    quantity: int,
    client_id: str,
    declared_weight_kg: float | None = None,
    pkg_l: float | None = None,
    pkg_w: float | None = None,
    pkg_h: float | None = None,
    tare_g: float = 100.0,
) -> WeightResult:
    notes: list[str] = []

    # --- Step 1: SKU history ---
    weights = _sku_sorter_history(sku, client_id)
    if len(weights) >= 3:
        median_w = statistics.median(weights)
        dims = _sku_sorter_dims(sku, client_id)
        if dims:
            vol = volumetric_kg(dims["l"], dims["w"], dims["h"])
            bl = billable_kg(median_w, vol)
            ts = slab(bl)
            ds = drives_slab(median_w, vol)
            confidence = min(0.95, 0.70 + 0.005 * len(weights))
            notes.append(f"SKU history: n={len(weights)}, median={median_w:.3f} kg")
            return WeightResult(
                predicted_dead_weight_kg=round(median_w, 3),
                confidence=confidence,
                drives_slab=ds,
                basis="sku_history",
                target_slab=ts,
                notes=notes,
            )

    # --- Step 2: Name parse ---
    parsed_kg, parse_result = _name_parse_weight_kg(sku_name, quantity, tare_g)
    notes.append(
        f"Name parse: {len(parse_result.items)} items, "
        f"content={parse_result.estimated_content_g:.0f}g + tare={tare_g:.0f}g"
    )

    # Declared as sanity check / upper bound
    if declared_weight_kg and declared_weight_kg > 0:
        if parsed_kg < declared_weight_kg * 0.5:
            notes.append(
                f"Parse undershot declared ({parsed_kg:.3f} vs {declared_weight_kg:.3f}); "
                "using declared as floor"
            )
            parsed_kg = max(parsed_kg, declared_weight_kg * 0.8)
        elif parsed_kg > declared_weight_kg * 2.0:
            notes.append(f"Parse overshot declared; clamping down")
            parsed_kg = declared_weight_kg * 1.1

    # Volumetric from package dims (if available)
    if pkg_l and pkg_w and pkg_h:
        vol = volumetric_kg(pkg_l, pkg_w, pkg_h)
        bl = billable_kg(parsed_kg, vol)
        ts = slab(bl)
        ds = drives_slab(parsed_kg, vol)
    else:
        ts = slab(parsed_kg)
        ds = True  # unknown volumetric
        notes.append("No package dims provided; assuming dead weight drives slab")

    confidence = 0.35 if len(weights) == 0 else 0.45  # low without history

    if declared_weight_kg and declared_weight_kg > 0:
        confidence = min(confidence + 0.10, 0.60)
        basis = "name_parse"  # declared used only as guard
    else:
        basis = "name_parse"

    return WeightResult(
        predicted_dead_weight_kg=round(parsed_kg, 3),
        confidence=round(confidence, 2),
        drives_slab=ds,
        basis=basis,
        target_slab=ts,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_on_labeled_set(client_id: str, limit: int = 1000) -> pd.DataFrame:
    """
    Pull labeled shipments (have sorter weight) and compare predicted vs actual.
    Returns a DataFrame with MAE and slab-accuracy metrics.
    """
    df = ch_query("""
        SELECT
            sku, sku_name, quantity,
            dead_weight_shipfast,
            min_sorter_weight,
            min_sorter_length, min_sorter_width, min_sorter_height,
            max_dead_vol_sorter,
            package_id
        FROM shipfast_weight_discrepancy.shipfast_weight_discrepancy
        WHERE client_id = {cid:String}
          AND min_sorter_weight > 0
          AND sku_name != ''
        ORDER BY shipment_created_at DESC
        LIMIT {lim:Int32}
    """, {"cid": client_id, "lim": limit})

    records = []
    for _, row in df.iterrows():
        result = estimate_weight(
            sku=str(row["sku"]),
            sku_name=str(row["sku_name"]),
            quantity=int(row["quantity"] or 1),
            client_id=client_id,
            declared_weight_kg=float(row["dead_weight_shipfast"] or 0),
            pkg_l=float(row["min_sorter_length"] or 0) or None,
            pkg_w=float(row["min_sorter_width"] or 0) or None,
            pkg_h=float(row["min_sorter_height"] or 0) or None,
        )
        actual_kg = float(row["min_sorter_weight"])
        from .slab import slab as slab_fn, volumetric_kg as vkg, billable_kg as bkg
        actual_slab = slab_fn(bkg(actual_kg, vkg(
            float(row["min_sorter_length"] or 0),
            float(row["min_sorter_width"] or 0),
            float(row["min_sorter_height"] or 0),
        )))
        records.append({
            "sku": row["sku"],
            "sku_name": row["sku_name"],
            "predicted_kg": result.predicted_dead_weight_kg,
            "actual_kg": actual_kg,
            "abs_error_kg": abs(result.predicted_dead_weight_kg - actual_kg),
            "predicted_slab": result.target_slab,
            "actual_slab": actual_slab,
            "slab_match": result.target_slab == actual_slab,
            "basis": result.basis,
        })

    out = pd.DataFrame(records)
    if not out.empty:
        print(f"MAE: {out['abs_error_kg'].mean():.3f} kg")
        print(f"Slab accuracy: {out['slab_match'].mean():.1%}")
    return out
