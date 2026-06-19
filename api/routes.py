"""FastAPI routes."""

from __future__ import annotations
from fastapi import APIRouter
from pydantic import BaseModel

from engine.package import PackageResult, recommend_package
from engine.slab import verify_slab_rule, slab, volumetric_kg, billable_kg
from engine.weight import WeightResult, estimate_weight, validate_on_labeled_set
from engine.db import ch_query

router = APIRouter()


# ---------------------------------------------------------------------------
# Shared models
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
    use_vision: bool = False


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
# Analyze-order — the main endpoint the frontend calls
# ---------------------------------------------------------------------------

class OrderRequest(BaseModel):
    # Order identity
    sku: str
    sku_name: str
    quantity: int = 1
    client_id: str

    # What the seller declared
    applied_weight_kg: float | None = None   # seller's declared billable
    applied_package_id: str | None = None

    # Package dims (if known)
    package_length_cm: float | None = None
    package_width_cm: float | None = None
    package_height_cm: float | None = None


class WeightIssue(BaseModel):
    severity: str          # "ok" | "warning" | "critical"
    title: str
    detail: str
    suggestion: str


class OrderAnalysis(BaseModel):
    # Weight engine
    predicted_dead_weight_kg: float
    predicted_slab: float
    applied_slab: float | None
    weight_confidence: float
    weight_basis: str
    weight_issues: list[WeightIssue]

    # Package engine (if package_id given)
    package_decision: str | None        # "keep" | "switch" | "create" | None
    package_reason: str | None
    recommended_package_id: str | None
    suggested_new_dims_cm: dict | None
    package_confidence: float | None

    # Summary
    overall_status: str                 # "ok" | "warning" | "critical"
    summary: str


def _build_weight_issues(
    predicted_kg: float,
    predicted_slab: float,
    applied_weight_kg: float | None,
    pkg_l: float | None,
    pkg_w: float | None,
    pkg_h: float | None,
    basis: str,
    notes: list[str],
) -> list[WeightIssue]:
    issues: list[WeightIssue] = []

    if applied_weight_kg is None or applied_weight_kg <= 0:
        issues.append(WeightIssue(
            severity="warning",
            title="No applied weight declared",
            detail="Seller has not declared a billable weight for this order.",
            suggestion=f"Declare {predicted_kg:.3f} kg (predicted dead weight) → expected slab {predicted_slab} kg.",
        ))
        return issues

    applied_slab = slab(applied_weight_kg)

    # Volumetric
    if pkg_l and pkg_w and pkg_h:
        vol = volumetric_kg(pkg_l, pkg_w, pkg_h)
        bill = billable_kg(applied_weight_kg, vol)
        applied_slab = slab(bill)

    slab_diff = predicted_slab - applied_slab

    if abs(slab_diff) < 0.01:
        issues.append(WeightIssue(
            severity="ok",
            title="Weight slab matches prediction",
            detail=f"Applied weight {applied_weight_kg:.3f} kg → slab {applied_slab} kg matches predicted slab {predicted_slab} kg.",
            suggestion="No change needed.",
        ))
        return issues

    # Slab mismatch
    direction = "under-declared" if slab_diff > 0 else "over-declared"
    slabs_off = abs(slab_diff) / 0.5
    severity = "critical" if abs(slab_diff) >= 1.0 else "warning"

    issues.append(WeightIssue(
        severity=severity,
        title=f"Weight {direction} by {slabs_off:.0f} slab{'s' if slabs_off > 1 else ''}",
        detail=(
            f"Applied weight {applied_weight_kg:.3f} kg lands in slab {applied_slab} kg, "
            f"but the engine predicts slab {predicted_slab} kg "
            f"(predicted dead weight: {predicted_kg:.3f} kg, basis: {basis})."
        ),
        suggestion=(
            f"Update declared weight to ≥ {predicted_kg:.3f} kg so the order is correctly "
            f"billed at slab {predicted_slab} kg. "
            f"Current under-declaration risks a carrier weight discrepancy charge."
        ),
    ))

    if basis == "name_parse":
        issues.append(WeightIssue(
            severity="warning",
            title="Prediction based on product name (no sorter history)",
            detail="No past sorter re-weighs found for this SKU. Estimate uses product category weights and pack quantities.",
            suggestion="Once this shipment is scanned at the sorter, confidence will improve for future orders.",
        ))

    return issues


@router.post("/analyze-order", response_model=OrderAnalysis)
def analyze_order(req: OrderRequest):
    # --- Weight engine ---
    w: WeightResult = estimate_weight(
        sku=req.sku,
        sku_name=req.sku_name,
        quantity=req.quantity,
        client_id=req.client_id,
        declared_weight_kg=req.applied_weight_kg,
        pkg_l=req.package_length_cm,
        pkg_w=req.package_width_cm,
        pkg_h=req.package_height_cm,
    )

    # Applied slab from declared weight
    applied_slab: float | None = None
    if req.applied_weight_kg and req.applied_weight_kg > 0:
        if req.package_length_cm and req.package_width_cm and req.package_height_cm:
            vol = volumetric_kg(req.package_length_cm, req.package_width_cm, req.package_height_cm)
            applied_slab = slab(billable_kg(req.applied_weight_kg, vol))
        else:
            applied_slab = slab(req.applied_weight_kg)

    weight_issues = _build_weight_issues(
        predicted_kg=w.predicted_dead_weight_kg,
        predicted_slab=w.target_slab,
        applied_weight_kg=req.applied_weight_kg,
        pkg_l=req.package_length_cm,
        pkg_w=req.package_width_cm,
        pkg_h=req.package_height_cm,
        basis=w.basis,
        notes=w.notes,
    )

    # --- Package engine (optional) ---
    pkg_decision = pkg_reason = rec_pkg_id = None
    suggested_dims = None
    pkg_confidence = None

    if req.applied_package_id:
        p: PackageResult = recommend_package(
            sku=req.sku,
            client_id=req.client_id,
            applied_package_id=req.applied_package_id,
            use_vision=False,
        )
        pkg_decision = p.decision
        pkg_reason = p.reason
        rec_pkg_id = p.recommended_package_id
        suggested_dims = p.suggested_new_dims_cm
        pkg_confidence = p.confidence

    # --- Overall status ---
    severities = [i.severity for i in weight_issues]
    if "critical" in severities or pkg_decision in ("switch", "create"):
        overall_status = "critical"
    elif "warning" in severities:
        overall_status = "warning"
    else:
        overall_status = "ok"

    # --- Summary sentence ---
    if overall_status == "ok":
        summary = f"Order looks correct. Predicted slab {w.target_slab} kg matches applied weight."
    else:
        parts = []
        for issue in weight_issues:
            if issue.severity in ("warning", "critical"):
                parts.append(issue.title)
        if pkg_decision == "switch":
            parts.append(f"Switch to package {rec_pkg_id}")
        elif pkg_decision == "create":
            parts.append("Create a new package to match the correct slab")
        summary = ". ".join(parts) + "." if parts else "Review required."

    return OrderAnalysis(
        predicted_dead_weight_kg=w.predicted_dead_weight_kg,
        predicted_slab=w.target_slab,
        applied_slab=applied_slab,
        weight_confidence=w.confidence,
        weight_basis=w.basis,
        weight_issues=weight_issues,
        package_decision=pkg_decision,
        package_reason=pkg_reason,
        recommended_package_id=rec_pkg_id,
        suggested_new_dims_cm=suggested_dims,
        package_confidence=pkg_confidence,
        overall_status=overall_status,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Existing routes
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
    df = ch_query("""
        SELECT dead_weight_shipfast, volumetric_weight_shipfast,
               applied_weight_shipfast, weight_slab_shipfast
        FROM shipfast_weight_discrepancy.shipfast_weight_discrepancy
        WHERE client_id = {cid:String} AND weight_slab_shipfast > 0
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
