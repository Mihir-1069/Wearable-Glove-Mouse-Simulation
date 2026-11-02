"""
Unit tests for target acquisition assistance and ambiguity handling.
"""

import pytest
from wearable_mouse.config import Config
from wearable_mouse.cursor.cursor_engine import VirtualCursor
from wearable_mouse.cursor.target_assistance import TargetAssistanceEngine, Target, AcquisitionMode
from wearable_mouse.processing.uncertainty import UncertaintyState


def test_target_acquisition_modes():
    config = Config()
    engine = TargetAssistanceEngine(config)

    t1 = Target(id="t1", x=100, y=100, width=50, height=50, label="Target A")
    engine.set_targets([t1])

    # Cursor inside target bounds (x=120, y=120)
    cursor_inside = VirtualCursor(x=120, y=120)
    uncert = UncertaintyState(uncertainty_radius=20, confidence_score=0.9, tremor_energy=0.0, noise_estimate=0.0)

    # Mode A (Exact)
    res_a = engine.process(cursor_inside, uncert, mode=AcquisitionMode.A)
    assert res_a.selected_target == t1

    # Cursor 15px outside target bounds (x=165, y=120)
    cursor_outside = VirtualCursor(x=165, y=120)

    # Mode A (Exact) should NOT hit outside target
    res_a_outside = engine.process(cursor_outside, uncert, mode=AcquisitionMode.A)
    assert res_a_outside.selected_target is None

    # Mode B (Tolerance) should hit outside target because distance (15px) < uncertainty radius (20px)
    res_b_outside = engine.process(cursor_outside, uncert, mode=AcquisitionMode.B)
    assert res_b_outside.selected_target == t1


def test_ambiguity_detection():
    config = Config(ambiguity_ratio_threshold=0.8)
    engine = TargetAssistanceEngine(config)

    # Two targets right next to each other
    t1 = Target(id="t1", x=100, y=100, width=40, height=40, label="Target A")
    t2 = Target(id="t2", x=145, y=100, width=40, height=40, label="Target B")
    engine.set_targets([t1, t2])

    # Cursor positioned exactly between the two targets with a large uncertainty radius
    cursor_mid = VirtualCursor(x=142.5, y=120)
    large_uncert = UncertaintyState(uncertainty_radius=50, confidence_score=0.5, tremor_energy=0.1, noise_estimate=0.1)

    res = engine.process(cursor_mid, large_uncert, mode=AcquisitionMode.B)

    assert res.is_ambiguous is True
    assert res.selected_target is None  # Must suppress selection when ambiguous
    assert len(res.ambiguous_targets) == 2
