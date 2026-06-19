"""FastAPI routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from engine.package import PackageResult, recommend_package
from engine.slab import verify_slab_rule
from engine.weight import WeightResult, estimate_weight, validate_on_labeled_set
from engine.db import ch_query

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class WeightRequest(BaseModel):
    sku: str
    sku_name: str
    quantity: int = 1
    client_id: str
    declared_weight_kg: float | None = None
    package_length_cm: float | None = None
    package_width_cm: float | None = None
    package_height_cm: float | None = None


class WeightResponse(BaseModel):
    predicted_dead_weight_kg: float
    confidence: float
    drives_slab: bool
    basis: str
    target_slab: float
    notes: list[str]


class PackageRequest(BaseModel):
    sku: str
    client_id: str
    applied_package_id: str
    billing_mode: str = "box"
    use_vision: bool = True


class PackageResponse(BaseModel):
    decision: str
    current_package_id: str
    current_slab: float
    target_slab: float
    recommended_package_id: str | None = None
    suggested_new_dims_cm: dict | None = None
    confidence: float
    evidence: dict
    reason: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/weight", response_model=WeightResponse)
def predict_weight(req: WeightRequest):
    result: WeightResult = estimate_weight(
        sku=req.sku,
        sku_name=req.sku_name,
        quantity=req.quantity,
        client_id=req.client_id,
        declared_weight_kg=req.declared_weight_kg,
        pkg_l=req.package_length_cm,
        pkg_w=req.package_width_cm,
        pkg_h=req.package_height_cm,
    )
    return WeightResponse(
        predicted_dead_weight_kg=result.predicted_dead_weight_kg,
        confidence=result.confidence,
        drives_slab=result.drives_slab,
        basis=result.basis,
        target_slab=result.target_slab,
        notes=result.notes,
    )


@router.post("/package", response_model=PackageResponse)
def recommend(req: PackageRequest):
    result: PackageResult = recommend_package(
        sku=req.sku,
        client_id=req.client_id,
        applied_package_id=req.applied_package_id,
        billing_mode=req.billing_mode,
        use_vision=req.use_vision,
    )
    return PackageResponse(
        decision=result.decision,
        current_package_id=result.current_package_id,
        current_slab=result.current_slab,
        target_slab=result.target_slab,
        recommended_package_id=result.recommended_package_id,
        suggested_new_dims_cm=result.suggested_new_dims_cm,
        confidence=result.confidence,
        evidence=result.evidence,
        reason=result.reason,
    )


@router.get("/verify-slab-rule")
def verify_slab(client_id: str, limit: int = 5000):
    """Cross-check slab formula against real ClickHouse data."""
    df = ch_query("""
        SELECT
            dead_weight_shipfast,
            volumetric_weight_shipfast,
            applied_weight_shipfast,
            weight_slab_shipfast
        FROM shipfast_weight_discrepancy.shipfast_weight_discrepancy
        WHERE client_id = {cid:String}
          AND weight_slab_shipfast > 0
        LIMIT {lim:Int32}
    """, {"cid": client_id, "lim": limit})
    return verify_slab_rule(df)


@router.get("/backtest/weight")
def backtest_weight(client_id: str, limit: int = 500):
    df = validate_on_labeled_set(client_id=client_id, limit=limit)
    if df.empty:
        return {"error": "no labeled data"}
    return {
        "n": len(df),
        "mae_kg": round(float(df["abs_error_kg"].mean()), 4),
        "slab_accuracy": round(float(df["slab_match"].mean()), 4),
        "basis_counts": df["basis"].value_counts().to_dict(),
    }


@router.get("/backtest/package")
def backtest_package(client_id: str, limit: int = 500):
    from engine.package import backtest
    df = backtest(client_id=client_id, limit=limit)
    if df.empty:
        return {"error": "no data"}
    return {
        "n": len(df),
        "original_slab_match_rate": round(float(df["slab_match_original"].mean()), 4),
        "recommended_slab_match_rate": round(float(df["slab_match_recommended"].mean()), 4),
        "decision_counts": df["decision"].value_counts().to_dict(),
    }
