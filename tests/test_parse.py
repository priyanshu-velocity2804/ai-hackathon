"""Unit tests for name parser."""

from engine.parse import parse_sku_name, extract_pack_qty, classify_category


def test_classify():
    assert classify_category("Claw Clip Set of 6") == "claw_clip"
    assert classify_category("Scrunchie Pack of 12 Pcs") == "scrunchie"
    assert classify_category("Random item") == "default"


def test_pack_qty():
    assert extract_pack_qty("Pack of 6") == 6
    assert extract_pack_qty("Set of 12 Pcs") == 12
    assert extract_pack_qty("6 Pcs") == 6
    assert extract_pack_qty("x12") == 12
    assert extract_pack_qty("Hair Tie") == 1


def test_single_item():
    r = parse_sku_name("Claw Clip Pack of 6", quantity=1)
    assert r.items[0].quantity == 6
    assert r.items[0].category == "claw_clip"
    assert r.estimated_content_g == 6 * 12  # 72 g
    assert r.estimated_total_g == 72 + 100  # + tare


def test_basket():
    r = parse_sku_name("Claw Clip Pack of 6, Scrunchie Pack of 3", quantity=1)
    assert len(r.items) == 2
    expected = 6 * 12 + 3 * 6  # 72 + 18 = 90
    assert r.estimated_content_g == expected


def test_quantity_multiplier():
    r = parse_sku_name("Claw Clip Pack of 6", quantity=2)
    assert r.items[0].quantity == 12
    assert r.estimated_content_g == 12 * 12
