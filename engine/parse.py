"""
Name parser for Engine 1.
Splits sku_name (comma-separated basket), classifies product category,
extracts pack-quantity tokens, estimates per-unit weight, sums across items.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Category seed weights (grams per unit).
# These are PRIORS — must be calibrated against sorter data before trusting.
# ---------------------------------------------------------------------------

CATEGORY_WEIGHTS_G: dict[str, float] = {
    "claw_clip": 12,
    "pin": 5,
    "clip": 7,
    "bow": 6,
    "scrunchie": 6,
    "hair_tie": 4,
    "rubber_band": 4,
    "comb": 22,
    "mirror": 40,
    "bottle": 55,
    "sponge": 6,
    "puff": 6,
    "nail": 12,
    "headband": 25,
    "tic_tac": 6,
    "default": 10,
}

# keyword → category key
KEYWORDS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"claw\s*clip", re.I), "claw_clip"),
    (re.compile(r"\bscrunchie\b", re.I), "scrunchie"),
    (re.compile(r"\bheadband\b", re.I), "headband"),
    (re.compile(r"\bhair\s*tie\b|\brubber\s*band\b", re.I), "hair_tie"),
    (re.compile(r"\bcomb\b", re.I), "comb"),
    (re.compile(r"\bmirror\b", re.I), "mirror"),
    (re.compile(r"\bbottle\b|\bcontainer\b|\bjar\b", re.I), "bottle"),
    (re.compile(r"\bsponge\b", re.I), "sponge"),
    (re.compile(r"\bpuff\b", re.I), "puff"),
    (re.compile(r"\bnail\b", re.I), "nail"),
    (re.compile(r"\bbow\b", re.I), "bow"),
    (re.compile(r"\bpin\b", re.I), "pin"),
    (re.compile(r"\bclip\b", re.I), "clip"),
    (re.compile(r"\btic[\s-]?tac\b", re.I), "tic_tac"),
]

# Pack quantity patterns — highest priority first
PACK_QTY_PATTERNS: list[re.Pattern] = [
    re.compile(r"(?:pack|set)\s+of\s+(\d+)\s*pcs", re.I),
    re.compile(r"(?:pack|set)\s+of\s+(\d+)", re.I),
    re.compile(r"(\d+)\s*pcs?\b", re.I),
    re.compile(r"(\d+)\s*pieces?\b", re.I),
    re.compile(r"x\s*(\d+)\b", re.I),
]

# Explicit weight patterns — checked BEFORE category-based estimation.
# If the seller put the actual weight in the product name ("Shampoo 500g", "Dumbbell 2kg"),
# use that directly instead of guessing from category.
_EXPL_KG: list[re.Pattern] = [
    re.compile(r"(\d+(?:\.\d+)?)\s*kgs?\b", re.I),   # 1.2kg, 1.2kgs, 2 KG
]
_EXPL_G: list[re.Pattern] = [
    re.compile(r"(\d+(?:\.\d+)?)\s*grams?\b", re.I),  # 500gram, 500grams
    re.compile(r"(\d+(?:\.\d+)?)\s*gms?\b", re.I),    # 500gm, 500gms
    re.compile(r"(\d+(?:\.\d+)?)\s*g\b", re.I),       # 500g  (lower priority)
]

# Default package tare (grams) for a small corrugated box ~20×15×8
DEFAULT_TARE_G = 100.0


def extract_explicit_weight_g(text: str) -> float | None:
    """
    If the text contains an explicit weight declaration (e.g. '1.2kg', '500g', '250gm'),
    return that weight in grams. Returns None if no explicit weight found.
    KG patterns are checked before gram patterns to avoid false matches.
    """
    for pat in _EXPL_KG:
        m = pat.search(text)
        if m:
            val = float(m.group(1))
            if 0.01 <= val <= 100:   # sanity: 10g–100kg
                return val * 1000.0
    for pat in _EXPL_G:
        m = pat.search(text)
        if m:
            val = float(m.group(1))
            if 1 <= val <= 50000:    # sanity: 1g–50kg
                return val
    return None


@dataclass
class LineItem:
    raw: str
    category: str
    unit_weight_g: float
    quantity: int
    total_weight_g: float


@dataclass
class ParseResult:
    items: list[LineItem] = field(default_factory=list)
    tare_g: float = DEFAULT_TARE_G
    estimated_content_g: float = 0.0
    estimated_total_g: float = 0.0
    notes: list[str] = field(default_factory=list)


def classify_category(text: str) -> str:
    for pattern, cat in KEYWORDS:
        if pattern.search(text):
            return cat
    return "default"


def extract_pack_qty(text: str) -> int:
    """Return the pack multiplier from name tokens, defaulting to 1."""
    for pattern in PACK_QTY_PATTERNS:
        m = pattern.search(text)
        if m:
            qty = int(m.group(1))
            if 1 <= qty <= 500:   # sanity guard
                return qty
    return 1


def parse_sku_name(
    sku_name: str,
    quantity: int = 1,
    tare_g: float = DEFAULT_TARE_G,
    category_weights: dict[str, float] | None = None,
) -> ParseResult:
    """
    Parse a potentially comma-separated basket sku_name.
    `quantity` = order qty (multiplied into each line item).
    """
    weights = category_weights or CATEGORY_WEIGHTS_G
    result = ParseResult(tare_g=tare_g)

    # Split basket into line items
    parts = [p.strip() for p in sku_name.split(",") if p.strip()]

    for part in parts:
        cat = classify_category(part)
        pack_qty = extract_pack_qty(part)
        total_qty = pack_qty * max(1, quantity)

        explicit_g = extract_explicit_weight_g(part)
        if explicit_g is not None:
            # Explicit weight in the name — treat it as the total content weight
            # (already covers all units in the pack), multiplied only by order qty.
            unit_w = explicit_g / pack_qty  # per-unit so LineItem is consistent
            total_w = explicit_g * max(1, quantity)
            result.notes.append(
                f"'{part}': explicit weight {explicit_g:.0f}g used (not category prior)"
            )
        else:
            unit_w = weights.get(cat, weights["default"])
            total_w = unit_w * total_qty

        result.items.append(
            LineItem(
                raw=part,
                category=cat,
                unit_weight_g=unit_w,
                quantity=total_qty,
                total_weight_g=total_w,
            )
        )

    result.estimated_content_g = sum(i.total_weight_g for i in result.items)
    result.estimated_total_g = result.estimated_content_g + tare_g
    return result
