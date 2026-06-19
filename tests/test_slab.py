"""Unit tests for slab.py — no DB required."""

import pytest
from engine.slab import slab, volumetric_kg, billable_kg, drives_slab, slab_from_dims


def test_slab_basic():
    assert slab(0.1) == 0.5
    assert slab(0.5) == 0.5
    assert slab(0.51) == 1.0
    assert slab(1.0) == 1.0
    assert slab(1.001) == 1.5


def test_volumetric():
    # 20×15×8 / 5000 = 0.48
    assert abs(volumetric_kg(20, 15, 8) - 0.48) < 1e-9


def test_billable():
    assert billable_kg(0.3, 0.48) == 0.48
    assert billable_kg(0.6, 0.48) == 0.6


def test_drives_slab():
    assert drives_slab(0.6, 0.48) is True
    assert drives_slab(0.3, 0.48) is False


def test_slab_from_dims():
    # PKG-SEJ-1: 20x15x8, dead=0 → vol=0.48 → slab=0.5
    assert slab_from_dims(0, 20, 15, 8) == 0.5
    # Sorter median ~0.562 → slab=1.0
    assert slab_from_dims(0.562, 0, 0, 0) == 1.0


def test_sejalimpex_cliff():
    """The known slab-cliff: declared 0.48 kg → 0.5 slab, sorter 0.562 → 1.0 slab."""
    declared_slab = slab_from_dims(0, 20, 15, 8)   # 0.5
    sorter_slab = slab(0.562)                        # 1.0
    assert declared_slab == 0.5
    assert sorter_slab == 1.0
    assert declared_slab != sorter_slab
