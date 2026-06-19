"""
DB layer — queries ClickHouse via the Metabase REST API.
No direct ClickHouse connection; only METABASE_URL + METABASE_API_KEY required.
"""

from __future__ import annotations
import os
from functools import lru_cache

import httpx
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ClickHouse Cloud database ID inside Metabase
METABASE_CH_DB_ID = int(os.environ.get("METABASE_DATABASE_ID", 35))
CH_TABLE = "shipfast_weight_discrepancy.shipfast_weight_discrepancy"


# ---------------------------------------------------------------------------
# Metabase API client
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _base_url() -> str:
    return os.environ["METABASE_URL"].rstrip("/")


def _headers() -> dict:
    return {
        "x-api-key": os.environ["METABASE_API_KEY"],
        "Content-Type": "application/json",
    }


def ch_query(sql: str, params: dict | None = None, raise_on_error: bool = False) -> pd.DataFrame:
    """
    Run a native SQL query against ClickHouse via Metabase /api/dataset.

    Returns an empty DataFrame on any network/API error (graceful degradation)
    unless raise_on_error=True.
    """
    try:
        sql_mb, tag_params = _rewrite_params(sql, params or {})

        payload: dict = {
            "database": METABASE_CH_DB_ID,
            "type": "native",
            "native": {
                "query": sql_mb,
                "template-tags": _build_template_tags(tag_params),
            },
            "parameters": _build_parameters(tag_params),
        }

        resp = httpx.post(
            f"{_base_url()}/api/dataset",
            headers=_headers(),
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        if "error" in data:
            if raise_on_error:
                raise RuntimeError(f"Metabase query error: {data['error']}")
            import logging
            logging.warning("Metabase query error: %s", data["error"])
            return pd.DataFrame()

        rows = data["data"]["rows"]
        cols = [c["name"] for c in data["data"]["cols"]]
        return pd.DataFrame(rows, columns=cols)

    except Exception as exc:
        if raise_on_error:
            raise
        import logging
        logging.warning("ch_query failed (returning empty): %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Parameter rewriting helpers
# ---------------------------------------------------------------------------

import re

# Match ClickHouse-style {name:Type} placeholders
_CH_PARAM_RE = re.compile(r"\{(\w+):[^}]+\}")


def _rewrite_params(sql: str, params: dict) -> tuple[str, dict]:
    """
    Convert {name:Type} → {{name}} (Metabase template-tag syntax).
    Returns (rewritten_sql, params_dict).
    """
    found: list[str] = _CH_PARAM_RE.findall(sql)
    rewritten = _CH_PARAM_RE.sub(lambda m: "{{" + m.group(1) + "}}", sql)
    # Only keep params that actually appear in the query
    filtered = {k: v for k, v in params.items() if k in found}
    return rewritten, filtered


def _mb_type(value) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "number"
    if isinstance(value, float):
        return "number"
    return "text"


def _build_template_tags(params: dict) -> dict:
    return {
        name: {
            "id": name,
            "name": name,
            "display-name": name,
            "type": "text",         # Metabase treats all native tags as text by default
            "required": False,
        }
        for name in params
    }


def _build_parameters(params: dict) -> list[dict]:
    return [
        {
            "type": f"category",
            "target": ["variable", ["template-tag", name]],
            "value": str(value),
        }
        for name, value in params.items()
    ]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

SEJALIMPEX_CLIENT_ID = "764c0f86-cb84-432f-a77f-b8f6cfaaed4f"


def smoke_test():
    print("=== Metabase → ClickHouse smoke test ===")
    df = ch_query("""
        SELECT
            count()                                AS total_shipments,
            countIf(min_sorter_weight > 0)         AS with_sorter,
            countIf(notEmpty(toString(package_id)))              AS with_package_id,
            countIf(min_sorter_image_link != '')   AS with_sorter_image
        FROM shipfast_weight_discrepancy.shipfast_weight_discrepancy
        WHERE client_id = {client_id:String}
          AND toYYYYMM(shipment_created_at) = 202604
    """, {"client_id": SEJALIMPEX_CLIENT_ID})
    print(df.to_string())

    print("\n=== Packages used by client (sorter-derived) ===")
    pkgs = ch_query("""
        SELECT
            package_id,
            count()                     AS shipments,
            median(min_sorter_length)   AS med_l,
            median(min_sorter_width)    AS med_w,
            median(min_sorter_height)   AS med_h,
            median(max_dead_vol_sorter) AS med_billable
        FROM shipfast_weight_discrepancy.shipfast_weight_discrepancy
        WHERE client_id = {client_id:String}
          AND notEmpty(toString(package_id))
          AND min_sorter_weight > 0
        GROUP BY package_id
        ORDER BY shipments DESC
    """, {"client_id": SEJALIMPEX_CLIENT_ID})
    print(pkgs.to_string())

    print("\n=== Column list ===")
    sample = ch_query("""
        SELECT * FROM shipfast_weight_discrepancy.shipfast_weight_discrepancy
        WHERE client_id = {client_id:String} LIMIT 1
    """, {"client_id": SEJALIMPEX_CLIENT_ID})
    print(list(sample.columns))


if __name__ == "__main__":
    smoke_test()
