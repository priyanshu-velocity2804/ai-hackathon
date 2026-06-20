"""
WD Dashboard API — lazy-loads the WD Excel on first request, caches in memory.
Pre-analyses top 10 clients by 3PL charge at load time; all others on demand.
"""
from __future__ import annotations
import math, os, sys
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, Query
from pydantic import BaseModel
import pandas as pd

# Engine imports for AI name-parse predictions
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from engine.parse import parse_sku_name
    from engine.slab import slab as _slab
    _AI = True
except ImportError:
    _AI = False
    def _slab(kg): return math.ceil(kg / 0.5) * 0.5 if kg > 0 else 0.5

wd_router = APIRouter()

# ---------------------------------------------------------------------------
# Column aliases
# ---------------------------------------------------------------------------
AWB       = "AWB"
CLIENT_ID = "client_id"
CLIENT_NM = "client_name"
SKU       = "sku"
SKU_NM    = "sku_name"
QTY       = "quantity"
CARRIER   = "carrier_name"
PKG_ID    = "package_id"
DEAD_WT   = "dead_weight_shipfast"
APPL_WT   = "applied_weight_shipfast"
APPL_SL   = "weight_slab_shipfast"
SORT_WT   = "min_sorter_weight"
SORT_SL   = "sorter_slab"
MAX_BILL  = "max_dead_vol_sorter"
CH_3PL    = "additional_cost_on_invoice_slab"           # charged by 3PL
R_SELLER  = "additional_charges_on_seller_upload_slab_invoice"  # raise to seller
R_3PL_AMT = "additional_cost_on_invoice_recon_slab"     # raise to 3PL (don't pay)
R_3PL_FLG = "raise_to_3pl"                              # Yes / No flag
WD_STATUS = "wd_status"
DISC_TYPE = "discrepancy_type_ai"
SL_CHANGE = "slab_change"
IMAGE     = "min_sorter_image_link"
N_IMAGES  = "no_of_sorter_images"
SORT_L    = "min_sorter_length"
SORT_W    = "min_sorter_width"
SORT_H    = "min_sorter_height"

# ---------------------------------------------------------------------------
# Global cache
# ---------------------------------------------------------------------------
_state: dict = {"loaded": False, "loading": False, "df": None, "summary": None, "client_cache": {}}

_WD_DIR  = os.path.dirname(os.path.dirname(__file__))
_WD_CSV  = os.path.join(_WD_DIR, "data_wd.csv")
_WD_FILE = os.path.join(_WD_DIR, "Weight Discrepancies non-Haritu.xlsx")

_NUM_COLS = [DEAD_WT, APPL_WT, APPL_SL, SORT_WT, SORT_SL, MAX_BILL,
             CH_3PL, R_SELLER, R_3PL_AMT, SORT_L, SORT_W, SORT_H, N_IMAGES, QTY]


def _f(v, d=0.0) -> float:
    try:
        x = float(v)
        return x if not math.isnan(x) else d
    except Exception:
        return d


def _load():
    """Load Excel + pre-analyse top 10 clients. Runs in a background thread."""
    if _state["loaded"] or _state["loading"]:
        return
    _state["loading"] = True
    import logging, threading
    def _do_load():
        try:
            if os.path.exists(_WD_CSV):
                logging.info("[WD] Loading CSV cache …")
                df = pd.read_csv(_WD_CSV, dtype=str, low_memory=False)
            else:
                logging.info("[WD] No CSV cache — loading Excel (slow) …")
                df = pd.read_excel(_WD_FILE, dtype=str)
                df.to_csv(_WD_CSV, index=False)
                logging.info("[WD] CSV cache saved for future restarts")
            for col in _NUM_COLS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            _state["df"] = df
            _state["summary"] = _build_summary(df)
            top10 = (
                df.groupby(CLIENT_ID)[CH_3PL].sum()
                .sort_values(ascending=False).head(10).index.tolist()
            )
            for cid in top10:
                cdf = df[df[CLIENT_ID] == cid]
                cname = str(cdf[CLIENT_NM].iloc[0]) if not cdf.empty else cid
                _state["client_cache"][cid] = _analyse_client(cdf, cid, cname)
            _state["loaded"] = True
            _state["loading"] = False
            logging.info(f"[WD] Ready — {len(df):,} rows, {df[CLIENT_ID].nunique()} clients")
        except Exception as e:
            _state["loading"] = False
            logging.error(f"[WD] Load failed: {e}")
    threading.Thread(target=_do_load, daemon=True).start()




# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------

def _build_summary(df: pd.DataFrame) -> dict:
    total = len(df)
    ch3pl   = round(_f(df[CH_3PL].fillna(0).sum()), 2)
    rseller = round(_f(df[R_SELLER].fillna(0).sum()), 2)
    r3pl    = round(_f(df[R_3PL_AMT].fillna(0).sum()), 2)
    r3pl_n  = int((df[R_3PL_FLG].astype(str).str.strip().str.lower() == "yes").sum())

    status_raw = df[WD_STATUS].fillna("unknown").str.strip().value_counts()
    status_map = {
        "raised":              "Raised",
        "auto_approved":       "Auto Approved",
        "approved":            "Approved",
        "challenged":          "Challenged",
        "challenge_approved":  "Challenge Accepted",
        "rejected_closed":     "Rejected / Closed",
        "escalated":           "Escalated",
        "escalation_rejected": "Escalation Rejected",
        "unknown":             "Unknown",
    }
    status_counts = {status_map.get(k, k): int(v) for k, v in status_raw.items()}

    top_clients = (
        df.groupby([CLIENT_ID, CLIENT_NM])
        .agg(orders=(AWB, "count"), ch3pl=(CH_3PL, "sum"),
             rseller=(R_SELLER, "sum"), r3pl=(R_3PL_AMT, "sum"))
        .reset_index().sort_values("ch3pl", ascending=False).head(10)
    )
    top_clients_list = []
    for _, r in top_clients.iterrows():
        top_clients_list.append({
            "client_id":   str(r[CLIENT_ID]),
            "client_name": str(r[CLIENT_NM]),
            "orders":      int(r["orders"]),
            "ch3pl":       round(_f(r["ch3pl"]), 2),
            "rseller":     round(_f(r["rseller"]), 2),
            "r3pl":        round(_f(r["r3pl"]), 2),
        })

    return {
        "total_orders":      total,
        "charge_3pl":        ch3pl,
        "raise_seller":      rseller,
        "raise_3pl":         r3pl,
        "raise_3pl_orders":  r3pl_n,
        "unique_clients":    int(df[CLIENT_ID].nunique()),
        "status_breakdown":  status_counts,
        "top_clients":       top_clients_list,
    }


# ---------------------------------------------------------------------------
# Per-client analyser
# ---------------------------------------------------------------------------

def _analyse_client(cdf: pd.DataFrame, client_id: str, client_name: str) -> dict:
    total   = len(cdf)
    ch3pl   = round(_f(cdf[CH_3PL].fillna(0).sum()), 2)
    rseller = round(_f(cdf[R_SELLER].fillna(0).sum()), 2)
    r3pl    = round(_f(cdf[R_3PL_AMT].fillna(0).sum()), 2)
    recs    = []

    # 1. Zero dead weight
    z_mask = (cdf[DEAD_WT].fillna(0) == 0) | cdf[DEAD_WT].isna()
    z_cnt  = int(z_mask.sum())
    z_chrg = round(_f(cdf.loc[z_mask, CH_3PL].fillna(0).sum()), 2)
    if z_cnt:
        ai_note = ""
        if _AI and z_cnt > 0:
            # Sample 3 SKU names to show parser output
            samples = cdf.loc[z_mask, [SKU_NM, QTY]].dropna().head(3)
            previews = []
            for _, row in samples.iterrows():
                try:
                    r = parse_sku_name(str(row[SKU_NM]), quantity=max(1, int(_f(row[QTY], 1))))
                    previews.append(f"{str(row[SKU_NM])[:30]} → {r.estimated_total_g/1000:.2f} kg")
                except Exception:
                    pass
            if previews:
                ai_note = " AI estimates: " + "; ".join(previews)
        recs.append({
            "id": "zero_dead_weight", "severity": "critical",
            "title": f"Dead weight = 0 on {z_cnt:,} orders ({round(100*z_cnt/total)}%)",
            "savings": z_chrg,
            "action": "Use weight engine to auto-declare dead weight per SKU." + ai_note,
        })

    # 2. Single package dominates
    pkg_vc = cdf[PKG_ID].value_counts()
    if not pkg_vc.empty:
        top_p, top_p_n = str(pkg_vc.index[0]), int(pkg_vc.iloc[0])
        top_p_pct = top_p_n / total
        if top_p_pct > 0.60:
            p_df = cdf[cdf[PKG_ID].astype(str) == top_p]
            med_s = p_df[SORT_SL].median() if SORT_SL in p_df else None
            recs.append({
                "id": "default_package", "severity": "critical" if top_p_pct > 0.80 else "warning",
                "title": f"1 package on {round(top_p_pct*100)}% of orders — not sized per product",
                "savings": round(_f(p_df[CH_3PL].fillna(0).sum()) * 0.55, 2),
                "action": (
                    f"Package {top_p[:12]}… used {top_p_n:,} times. "
                    f"Sorter median slab: {med_s} kg. Create S/M/L variants."
                ),
            })

    # 3. Top SKUs under-declared
    if SKU in cdf.columns and APPL_SL in cdf.columns and SORT_SL in cdf.columns:
        sku_g = (
            cdf.groupby([SKU, SKU_NM])
            .agg(cnt=(AWB, "count"), avg_dec=(APPL_SL, "mean"),
                 avg_srt=(SORT_SL, "mean"), chrg=(CH_3PL, "sum"))
            .reset_index()
        )
        sku_g["diff"] = sku_g["avg_srt"].fillna(0) - sku_g["avg_dec"].fillna(0)
        bad = sku_g[sku_g["diff"] > 0.4].sort_values("chrg", ascending=False)
        if not bad.empty:
            top3 = "; ".join(
                f"{str(r[SKU_NM])[:25]} (+{r['diff']:.1f} kg)"
                for _, r in bad.head(3).iterrows()
            )
            recs.append({
                "id": "sku_underdeclared", "severity": "warning",
                "title": f"{len(bad)} SKUs consistently under-declared by >0.5 kg slab",
                "savings": round(_f(bad["chrg"].sum()) * 0.70, 2),
                "action": f"Set SKU-level floor weights. Top: {top3}",
            })
    else:
        bad = pd.DataFrame()

    # 4. Dispute opportunity
    minor = cdf[DISC_TYPE].astype(str).str.contains("Minor", na=False, case=False)
    no_img = cdf[N_IMAGES].fillna(0) == 0 if N_IMAGES in cdf.columns else pd.Series(False, index=cdf.index)
    disp = minor | no_img
    d_cnt = int(disp.sum())
    d_amt = round(_f(cdf.loc[disp, R_3PL_AMT].fillna(0).sum()) * 0.5, 2)
    if d_cnt:
        recs.append({
            "id": "dispute_opportunity", "severity": "info",
            "title": f"{d_cnt:,} orders disputable (minor discrepancy or no sorter image)",
            "savings": d_amt,
            "action": "Raise dispute for all no-image orders — carrier has no proof.",
        })

    # 5. Carrier concentration
    if CH_3PL in cdf.columns:
        c_g = cdf.groupby(CARRIER)[CH_3PL].sum().sort_values(ascending=False)
        if not c_g.empty and ch3pl > 0:
            top_c = c_g.index[0]
            top_c_pct = round(c_g.iloc[0] / ch3pl * 100)
            if top_c_pct > 50:
                recs.append({
                    "id": "carrier_concentration", "severity": "warning",
                    "title": f"{top_c} drives {top_c_pct}% of WD charges for this client",
                    "savings": round(_f(c_g.iloc[0]) * 0.30, 2),
                    "action": f"Prioritise weight accuracy for {top_c} shipments first.",
                })

    recs.sort(key=lambda r: r["savings"], reverse=True)

    # Top SKUs table
    top_skus = []
    if not bad.empty:
        for _, r in bad.head(10).iterrows():
            top_skus.append({
                "sku": str(r[SKU]), "sku_name": str(r[SKU_NM])[:50],
                "orders": int(r["cnt"]),
                "avg_declared": round(_f(r["avg_dec"]), 2),
                "avg_sorter": round(_f(r["avg_srt"]), 2),
                "avg_diff": round(_f(r["diff"]), 2),
                "charges": round(_f(r["chrg"]), 2),
            })

    status_counts = cdf[WD_STATUS].fillna("unknown").value_counts().to_dict()
    carrier_bkdn  = (
        cdf.groupby(CARRIER)
        .agg(orders=(AWB, "count"), charge=(CH_3PL, "sum"))
        .reset_index().sort_values("charge", ascending=False).head(5)
        .to_dict(orient="records")
    )
    for r in carrier_bkdn:
        r["charge"] = round(_f(r.get("charge")), 2)

    return {
        "client_id": client_id, "client_name": client_name,
        "total_orders": total, "charge_3pl": ch3pl,
        "raise_seller": rseller, "raise_3pl": r3pl,
        "status_breakdown": status_counts,
        "carrier_breakdown": carrier_bkdn,
        "recommendations": recs,
        "top_skus": top_skus,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@wd_router.get("/wd/summary")
def wd_summary():
    _load()  # starts background thread if not already loading
    if not _state["loaded"]:
        return {"loading": True, "message": "Data loading in background — retry in a few seconds."}
    return _state["summary"]


@wd_router.get("/wd/clients")
def wd_clients(
    search: str = Query(default=""),
    page: int   = Query(default=1),
    limit: int  = Query(default=10),
):
    _load()
    df = _state["df"]
    client_totals = (
        df.groupby([CLIENT_ID, CLIENT_NM])
        .agg(orders=(AWB, "count"), ch3pl=(CH_3PL, "sum"),
             rseller=(R_SELLER, "sum"), r3pl=(R_3PL_AMT, "sum"))
        .reset_index().sort_values("ch3pl", ascending=False)
    )
    if search:
        m = client_totals[CLIENT_NM].astype(str).str.contains(search, case=False, na=False)
        client_totals = client_totals[m]

    if not _state["loaded"]:
        return {"clients": [], "total": 0, "page": page, "limit": limit, "loading": True}
    total_count = len(client_totals)
    page_data   = client_totals.iloc[(page - 1) * limit: page * limit]
    results = []
    for _, r in page_data.iterrows():
        cid     = str(r[CLIENT_ID])
        cached  = _state["client_cache"].get(cid)
        top_rec = cached["recommendations"][0]["title"] if cached and cached["recommendations"] else None
        results.append({
            "client_id":          cid,
            "client_name":        str(r[CLIENT_NM]),
            "orders":             int(r["orders"]),
            "ch3pl":              round(_f(r["ch3pl"]), 2),
            "rseller":            round(_f(r["rseller"]), 2),
            "r3pl":               round(_f(r["r3pl"]), 2),
            "top_recommendation": top_rec,
            "pre_analyzed":       cid in _state["client_cache"],
        })
    return {"clients": results, "total": total_count, "page": page, "limit": limit}


@wd_router.get("/wd/client/{client_id}")
def wd_client(client_id: str):
    _load()
    if not _state["loaded"]:
        return {"error": "Data still loading, please retry."}
    if client_id not in _state["client_cache"]:
        df  = _state["df"]
        cdf = df[df[CLIENT_ID] == client_id]
        if cdf.empty:
            return {"error": "Client not found"}
        cname = str(cdf[CLIENT_NM].iloc[0])
        _state["client_cache"][client_id] = _analyse_client(cdf, client_id, cname)
    return _state["client_cache"][client_id]


@wd_router.get("/wd/discrepancies")
def wd_discrepancies(
    client_id:   str = Query(default=""),
    status:      str = Query(default=""),
    raise_to_3pl:str = Query(default=""),
    search:      str = Query(default=""),
    page:        int = Query(default=1),
    limit:       int = Query(default=50),
):
    _load()
    if not _state["loaded"]:
        return {"rows": [], "total": 0, "page": page, "limit": limit, "loading": True}
    df = _state["df"]
    if client_id:
        df = df[df[CLIENT_ID] == client_id]
    if status:
        df = df[df[WD_STATUS].astype(str).str.strip().str.lower() == status.lower()]
    if raise_to_3pl in ("Yes", "No"):
        df = df[df[R_3PL_FLG].astype(str).str.strip() == raise_to_3pl]
    if search:
        m = (
            df[CLIENT_NM].astype(str).str.contains(search, case=False, na=False) |
            df[SKU_NM].astype(str).str.contains(search, case=False, na=False) |
            df[AWB].astype(str).str.contains(search, case=False, na=False)
        )
        df = df[m]

    total = len(df)
    df    = df.sort_values(CH_3PL, ascending=False, na_position="last")
    pg    = df.iloc[(page - 1) * limit: page * limit]

    want_cols = [AWB, CLIENT_NM, SKU_NM, QTY, CARRIER, APPL_WT, APPL_SL,
                 SORT_WT, SORT_SL, MAX_BILL, CH_3PL, R_SELLER, R_3PL_AMT,
                 WD_STATUS, R_3PL_FLG, DISC_TYPE, SL_CHANGE, IMAGE, N_IMAGES]
    cols = [c for c in want_cols if c in pg.columns]

    rows = []
    for _, row in pg[cols].iterrows():
        r = {}
        for c in cols:
            v = row[c]
            r[c] = None if (isinstance(v, float) and math.isnan(v)) else v
        rows.append(r)

    return {"rows": rows, "total": total, "page": page, "limit": limit}


class RaiseBody(BaseModel):
    awbs: List[str]
    action: str   # "toggle_3pl"


@wd_router.post("/wd/raise")
def wd_raise(body: RaiseBody):
    _load()
    df   = _state["df"]
    mask = df[AWB].isin(body.awbs)
    n    = int(mask.sum())
    if body.action == "toggle_3pl" and n:
        cur = df.loc[mask, R_3PL_FLG].astype(str).str.strip()
        _state["df"].loc[mask, R_3PL_FLG] = cur.apply(lambda v: "No" if v.lower() == "yes" else "Yes")
        _state["summary"] = _build_summary(_state["df"])
    return {"updated": n}
