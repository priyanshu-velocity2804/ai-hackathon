"""Slab utilities — the single source of truth for all billing step logic."""

from __future__ import annotations
import math
import pandas as pd


# ---------------------------------------------------------------------------
# Core slab arithmetic
# ---------------------------------------------------------------------------

DIVISOR = 5000  # cm³ → kg volumetric
SLAB_STEP = 0.5  # kg


def volumetric_kg(length_cm: float, width_cm: float, height_cm: float) -> float:
    """L*W*H / 5000 (all dims in cm)."""
    return (length_cm * width_cm * height_cm) / DIVISOR


def billable_kg(dead_weight_kg: float, vol_kg: float) -> float:
    return max(dead_weight_kg, vol_kg)


def slab(billable: float) -> float:
    """Ceiling to nearest 0.5 kg slab."""
    if billable <= 0:
        return SLAB_STEP
    return math.ceil(billable / SLAB_STEP) * SLAB_STEP


def slab_from_dims(dead_kg: float, l: float, w: float, h: float) -> float:
    return slab(billable_kg(dead_kg, volumetric_kg(l, w, h)))


def drives_slab(dead_kg: float, vol_kg: float) -> bool:
    """True if dead weight is the binding constraint (≥ volumetric)."""
    return dead_kg >= vol_kg


# ---------------------------------------------------------------------------
# Verify rule against real data
# ---------------------------------------------------------------------------

def verify_slab_rule(df: pd.DataFrame) -> dict:
    """
    Cross-check the slab rule against ClickHouse weight_slab_shipfast vs
    applied_weight_shipfast.

    Expects columns: dead_weight_shipfast, volumetric_weight_shipfast,
                     applied_weight_shipfast, weight_slab_shipfast
    """
    df = df.copy()
    df = df.dropna(subset=["dead_weight_shipfast", "volumetric_weight_shipfast",
                            "applied_weight_shipfast", "weight_slab_shipfast"])

    df["computed_billable"] = df.apply(
        lambda r: billable_kg(r["dead_weight_shipfast"], r["volumetric_weight_shipfast"]),
        axis=1,
    )
    df["computed_slab"] = df["computed_billable"].apply(slab)
    df["slab_match"] = df["computed_slab"] == df["weight_slab_shipfast"]

    match_rate = df["slab_match"].mean()
    n = len(df)

    # Check if applied_weight == weight_slab (sometimes mis-labeled)
    df["applied_eq_slab"] = (
        (df["applied_weight_shipfast"] - df["weight_slab_shipfast"]).abs() < 0.01
    )

    mismatches = df[~df["slab_match"]][
        ["dead_weight_shipfast", "volumetric_weight_shipfast",
         "applied_weight_shipfast", "weight_slab_shipfast", "computed_slab"]
    ].head(10)

    return {
        "n": n,
        "slab_rule_match_rate": match_rate,
        "rule_confirmed": match_rate > 0.90,
        "sample_mismatches": mismatches.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Package slab decision helpers
# ---------------------------------------------------------------------------

def find_best_switch_package(
    target_slab_kg: float,
    parcel_l: float,
    parcel_w: float,
    parcel_h: float,
    packages: list[dict],
    tolerance_cm: float = 1.0,
) -> dict | None:
    """
    Return the smallest (by volume) non-discarded package that:
      - lands in target_slab when filled with this parcel, AND
      - physically contains the parcel (with tolerance).
    Returns None if no such package exists.
    """
    candidates = []
    for pkg in packages:
        if pkg.get("discarded_at"):
            continue
        l, w, h = pkg["length"], pkg["width"], pkg["height"]
        dw = pkg.get("dead_weight", 0) or 0
        # Does it contain the parcel (any orientation, with tolerance)?
        dims_pkg = sorted([l, w, h], reverse=True)
        dims_par = sorted([parcel_l, parcel_w, parcel_h], reverse=True)
        fits = all(
            dims_pkg[i] + tolerance_cm >= dims_par[i] for i in range(3)
        )
        if not fits:
            continue
        pkg_slab = slab_from_dims(dw, l, w, h)
        if abs(pkg_slab - target_slab_kg) < 0.01:
            vol = l * w * h
            candidates.append((vol, pkg))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def suggest_new_dims(
    target_slab_kg: float,
    parcel_l: float,
    parcel_w: float,
    parcel_h: float,
    billing_mode: str = "box",  # "box" or "flyer"
) -> dict:
    """
    Suggest minimal package dims that:
      - contain the parcel (5% clearance each side), AND
      - land in target_slab.
    billing_mode='flyer' bills on dead weight only (volumetric irrelevant).
    """
    cl = parcel_l * 1.05
    cw = parcel_w * 1.05
    ch = parcel_h * 1.05

    if billing_mode == "flyer":
        # Dims don't affect slab; just return fitted box
        return {
            "length_cm": round(cl, 1),
            "width_cm": round(cw, 1),
            "height_cm": round(ch, 1),
            "note": "Flyer mode: volumetric not billed; dims sized to contain parcel only.",
        }

    # For box mode: if the clearance box already lands in target_slab, use it
    # Otherwise compress height (most flexible) until we hit target
    for h_try in [ch, ch * 0.9, ch * 0.8, ch * 0.7]:
        computed = slab_from_dims(0, cl, cw, h_try)
        if abs(computed - target_slab_kg) < 0.01:
            return {
                "length_cm": round(cl, 1),
                "width_cm": round(cw, 1),
                "height_cm": round(h_try, 1),
                "estimated_slab": computed,
            }

    # Fallback: return clearance box, note it may exceed target
    return {
        "length_cm": round(cl, 1),
        "width_cm": round(cw, 1),
        "height_cm": round(ch, 1),
        "estimated_slab": slab_from_dims(0, cl, cw, ch),
        "note": "Could not compress to target slab; review manually.",
    }
