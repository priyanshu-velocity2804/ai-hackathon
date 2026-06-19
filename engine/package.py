"""
Engine 2 — Package recommendation (slab-aware, ClickHouse-only).

All package knowledge is derived from sorter history in ClickHouse:
  - Package dims   → median sorter dims grouped by package_id (sorter = ground truth)
  - Alternatives   → all other package_ids the client has used, same logic
  - Sorter images  → min_sorter_image_link column

No Postgres / Shazam connection required.

Output: KEEP | SWITCH | CREATE  with reason + evidence.
"""

from __future__ import annotations
import statistics
from dataclasses import dataclass, field

import pandas as pd

from .db import ch_query
from .slab import (
    billable_kg,
    find_best_switch_package,
    slab,
    slab_from_dims,
    suggest_new_dims,
    volumetric_kg,
)
from .vision import DimExtraction, extract_dims_from_urls


# ---------------------------------------------------------------------------
# ClickHouse lookups
# ---------------------------------------------------------------------------

def _sorter_history(sku: str, client_id: str) -> pd.DataFrame:
    """Sorter re-weighs for this SKU."""
    return ch_query("""
        SELECT
            min_sorter_weight,
            min_sorter_length, min_sorter_width, min_sorter_height,
            max_dead_vol_sorter,
            min_sorter_image_link,
            package_id
        FROM shipfast_weight_discrepancy.shipfast_weight_discrepancy
        WHERE client_id = {cid:String}
          AND sku = {sku:String}
          AND min_sorter_weight > 0
        ORDER BY shipment_created_at DESC
        LIMIT 300
    """, {"cid": client_id, "sku": sku})


def _client_package_catalogue(client_id: str) -> list[dict]:
    """
    Build a package catalogue entirely from sorter history.
    Each distinct package_id → median sorter dims (the real filled-box size).
    Returns list of dicts compatible with slab.find_best_switch_package().
    """
    df = ch_query("""
        SELECT
            package_id                       AS id,
            any(package_type_seller)         AS name,
            count()                          AS n,
            median(min_sorter_length)        AS length,
            median(min_sorter_width)         AS width,
            median(min_sorter_height)        AS height,
            median(min_sorter_weight)        AS dead_weight,
            median(max_dead_vol_sorter)      AS median_billable
        FROM shipfast_weight_discrepancy.shipfast_weight_discrepancy
        WHERE client_id = {cid:String}
          AND notEmpty(toString(package_id))
          AND min_sorter_weight > 0
        GROUP BY package_id
        HAVING count() >= 3
        ORDER BY count() DESC
    """, {"cid": client_id})
    # Mark all as non-discarded (we can't know from ClickHouse; use n>=3 as proxy for active)
    packages = df.to_dict(orient="records")
    for p in packages:
        p["discarded_at"] = None
    return packages


def _sorter_image_urls(sku: str, client_id: str, limit: int = 3) -> list[str]:
    df = ch_query("""
        SELECT min_sorter_image_link
        FROM shipfast_weight_discrepancy.shipfast_weight_discrepancy
        WHERE client_id = {cid:String}
          AND sku = {sku:String}
          AND min_sorter_image_link != ''
          AND min_sorter_weight > 0
        ORDER BY shipment_created_at DESC
        LIMIT {lim:Int32}
    """, {"cid": client_id, "sku": sku, "lim": limit})
    return df["min_sorter_image_link"].dropna().tolist()


def _applied_package_sorter_profile(package_id: str, client_id: str) -> dict | None:
    """Median sorter profile for this specific package_id across all SKUs."""
    df = ch_query("""
        SELECT
            count()                     AS n,
            median(min_sorter_length)   AS length,
            median(min_sorter_width)    AS width,
            median(min_sorter_height)   AS height,
            median(min_sorter_weight)   AS dead_weight,
            median(max_dead_vol_sorter) AS median_billable
        FROM shipfast_weight_discrepancy.shipfast_weight_discrepancy
        WHERE client_id = {cid:String}
          AND package_id = {pid:String}
          AND min_sorter_weight > 0
    """, {"cid": client_id, "pid": package_id})
    if df.empty or int(df["n"].iloc[0]) < 1:
        return None
    row = df.iloc[0]
    return {k: float(row[k]) if row[k] is not None else 0.0
            for k in ("n", "length", "width", "height", "dead_weight", "median_billable")}


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

@dataclass
class PackageResult:
    decision: str                           # "keep" | "switch" | "create"
    current_package_id: str
    current_slab: float
    target_slab: float
    recommended_package_id: str | None = None
    suggested_new_dims_cm: dict | None = None
    confidence: float = 0.0
    evidence: dict = field(default_factory=lambda: {"sorter_image_urls": [], "package_image_urls": []})
    reason: str = ""


# ---------------------------------------------------------------------------
# Main recommender
# ---------------------------------------------------------------------------

def recommend_package(
    sku: str,
    client_id: str,
    applied_package_id: str,
    billing_mode: str = "box",
    use_vision: bool = True,
    tolerance_cm: float = 1.0,
) -> PackageResult:

    sorter_df = _sorter_history(sku, client_id)
    sorter_image_urls = _sorter_image_urls(sku, client_id)

    # No package images from Shazam — use sorter images as primary evidence
    evidence = {
        "sorter_image_urls": sorter_image_urls,
        "package_image_urls": [],   # not available without Shazam
    }

    if sorter_df.empty:
        return PackageResult(
            decision="keep",
            current_package_id=applied_package_id,
            current_slab=0.0,
            target_slab=0.0,
            confidence=0.0,
            evidence=evidence,
            reason="No sorter history for this SKU; cannot make a recommendation.",
        )

    # ---- 1. Target slab from sorter truth ----
    billables = sorter_df["max_dead_vol_sorter"].dropna()
    if billables.empty or float(billables.median()) == 0:
        billables = sorter_df["min_sorter_weight"].dropna()

    median_billable = float(billables.median())
    target_slab_kg = slab(median_billable)

    # Parcel dims (median sorter dims for this SKU)
    l_vals = sorter_df["min_sorter_length"].dropna()
    w_vals = sorter_df["min_sorter_width"].dropna()
    h_vals = sorter_df["min_sorter_height"].dropna()
    parcel_l = float(l_vals.median()) if len(l_vals) >= 1 else 0.0
    parcel_w = float(w_vals.median()) if len(w_vals) >= 1 else 0.0
    parcel_h = float(h_vals.median()) if len(h_vals) >= 1 else 0.0

    # ---- 2. Applied package profile from sorter history ----
    applied_profile = _applied_package_sorter_profile(applied_package_id, client_id)

    if applied_profile and applied_profile["length"] > 0:
        pkg_l = applied_profile["length"]
        pkg_w = applied_profile["width"]
        pkg_h = applied_profile["height"]
        pkg_dw = applied_profile["dead_weight"]
        applied_slab = slab_from_dims(pkg_dw, pkg_l, pkg_w, pkg_h)
        profile_source = "sorter_history"
    else:
        # No sorter profile for this package; use SKU-level dims as proxy
        pkg_l, pkg_w, pkg_h, pkg_dw = parcel_l, parcel_w, parcel_h, 0.0
        applied_slab = slab(median_billable)
        profile_source = "sku_median_fallback"

    # Optionally refine dims via vision on sorter images
    vision_used = False
    if use_vision and sorter_image_urls:
        dim_ext: DimExtraction = extract_dims_from_urls(sorter_image_urls)
        if dim_ext.confidence > 0.6 and all(
            v is not None for v in [dim_ext.length_cm, dim_ext.width_cm, dim_ext.height_cm]
        ):
            pkg_l = dim_ext.length_cm
            pkg_w = dim_ext.width_cm
            pkg_h = dim_ext.height_cm
            applied_slab = slab_from_dims(pkg_dw, pkg_l, pkg_w, pkg_h)
            vision_used = True

    n_sorter = len(sorter_df)
    confidence = min(0.95, 0.55 + 0.003 * n_sorter)

    reason_parts = [
        f"Sorter n={n_sorter}, median billable={median_billable:.3f} kg → target slab={target_slab_kg} kg.",
        f"Applied package (id={applied_package_id}) sorter-derived slab={applied_slab} kg "
        f"[source={profile_source}].",
    ]
    if vision_used:
        reason_parts.append("Package dims refined via vision on sorter image.")

    # ---- 3. Fits check ----
    fits = True
    if parcel_l > 0 and pkg_l > 0:
        pkg_dims_sorted = sorted([pkg_l, pkg_w, pkg_h], reverse=True)
        par_dims_sorted = sorted([parcel_l, parcel_w, parcel_h], reverse=True)
        fits = all(pkg_dims_sorted[i] + tolerance_cm >= par_dims_sorted[i] for i in range(3))

    slab_match = abs(applied_slab - target_slab_kg) < 0.01

    if slab_match and fits:
        return PackageResult(
            decision="keep",
            current_package_id=applied_package_id,
            current_slab=applied_slab,
            target_slab=target_slab_kg,
            confidence=confidence,
            evidence=evidence,
            reason=" ".join(reason_parts + ["Slab matches and package fits parcel — KEEP."]),
        )

    # ---- 4. Try SWITCH ----
    all_packages = _client_package_catalogue(client_id)
    other_packages = [p for p in all_packages if str(p["id"]) != str(applied_package_id)]

    if parcel_l > 0:
        best = find_best_switch_package(
            target_slab_kg, parcel_l, parcel_w, parcel_h,
            other_packages, tolerance_cm=tolerance_cm,
        )
    else:
        best = None

    if best:
        best_slab = slab_from_dims(
            float(best["dead_weight"]), float(best["length"]),
            float(best["width"]), float(best["height"])
        )
        return PackageResult(
            decision="switch",
            current_package_id=applied_package_id,
            current_slab=applied_slab,
            target_slab=target_slab_kg,
            recommended_package_id=str(best["id"]),
            confidence=confidence,
            evidence=evidence,
            reason=" ".join(reason_parts + [
                f"SWITCH to package_id={best['id']} "
                f"(sorter-derived slab={best_slab} kg, n={int(best.get('n', 0))} shipments)."
            ]),
        )

    # ---- 5. CREATE ----
    new_dims: dict | None = None
    if parcel_l > 0:
        new_dims = suggest_new_dims(target_slab_kg, parcel_l, parcel_w, parcel_h, billing_mode)

    return PackageResult(
        decision="create",
        current_package_id=applied_package_id,
        current_slab=applied_slab,
        target_slab=target_slab_kg,
        suggested_new_dims_cm=new_dims,
        confidence=confidence * 0.85,
        evidence=evidence,
        reason=" ".join(reason_parts + ["No existing package matches target slab — CREATE new."]),
    )


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def backtest(client_id: str, limit: int = 500) -> pd.DataFrame:
    df = ch_query("""
        SELECT
            sku, package_id,
            max_dead_vol_sorter, sorter_slab,
            min_sorter_length, min_sorter_width, min_sorter_height
        FROM shipfast_weight_discrepancy.shipfast_weight_discrepancy
        WHERE client_id = {cid:String}
          AND max_dead_vol_sorter > 0
          AND notEmpty(toString(package_id))
        ORDER BY shipment_created_at DESC
        LIMIT {lim:Int32}
    """, {"cid": client_id, "lim": limit})

    records = []
    for _, row in df.iterrows():
        result = recommend_package(
            sku=str(row["sku"]),
            client_id=client_id,
            applied_package_id=str(row["package_id"]),
            use_vision=False,
        )
        actual_slab = float(row["sorter_slab"] or slab(float(row["max_dead_vol_sorter"])))
        records.append({
            "sku": row["sku"],
            "decision": result.decision,
            "current_slab": result.current_slab,
            "target_slab": result.target_slab,
            "actual_slab": actual_slab,
            "slab_match_recommended": abs(result.target_slab - actual_slab) < 0.01,
            "slab_match_original": abs(result.current_slab - actual_slab) < 0.01,
        })

    out = pd.DataFrame(records)
    if not out.empty:
        print(f"Original slab-match rate:     {out['slab_match_original'].mean():.1%}")
        print(f"Recommended slab-match rate:  {out['slab_match_recommended'].mean():.1%}")
        print(f"Decision distribution:\n{out['decision'].value_counts()}")
    return out
