"""
Unit tests for Dynamic Sensitivity Suppression (Condition E).
"""

import numpy as np
import pytest
from wearable_mouse.config import Config
from wearable_mouse.cursor.cursor_engine import VirtualCursor
from wearable_mouse.cursor.target_assistance import Target
from wearable_mouse.cursor.dynamic_friction import DynamicSensitivitySuppression
from wearable_mouse.processing.uncertainty import UncertaintyState


def test_far_from_targets():
    friction_eng = DynamicSensitivitySuppression()
    target = Target(id="t1", x=800, y=500, width=100, height=100, label="Target 1")

    # Cursor far away at (100, 500)
    cursor = VirtualCursor(x=100, y=500, vx=100, vy=0)
    uncert = UncertaintyState(uncertainty_radius=20, confidence_score=0.9, tremor_energy=0.0, noise_estimate=0.0)

    f_state = friction_eng.compute_friction(cursor, [target], uncert)

    assert f_state.gain == 1.0
    assert f_state.friction_factor == 0.0


def test_approaching_vs_moving_away():
    friction_eng = DynamicSensitivitySuppression()
    target = Target(id="t1", x=500, y=500, width=100, height=100, label="Target 1")
    uncert = UncertaintyState(uncertainty_radius=20, confidence_score=0.9, tremor_energy=0.0, noise_estimate=0.0)

    # Cursor at (450, 550) - near target boundary
    # 1. Approaching target (vx > 0)
    cursor_approaching = VirtualCursor(x=450, y=550, vx=100, vy=0)
    f_approaching = friction_eng.compute_friction(cursor_approaching, [target], uncert)
    assert f_approaching.gain < 1.0
    assert f_approaching.directional_alignment > 0.0

    # 2. Moving away from target (vx < 0)
    cursor_away = VirtualCursor(x=450, y=550, vx=-100, vy=0)
    f_away = friction_eng.compute_friction(cursor_away, [target], uncert)
    # Must restore 100% normal gain immediately
    assert f_away.gain == 1.0
    assert f_away.friction_factor == 0.0


def test_inside_target_minimum_gain_floor():
    friction_eng = DynamicSensitivitySuppression()
    target = Target(id="t1", x=500, y=500, width=100, height=100, label="Target 1")
    uncert = UncertaintyState(uncertainty_radius=20, confidence_score=0.9, tremor_energy=0.0, noise_estimate=0.0)

    # Cursor inside target center at (550, 550) moving slowly
    cursor_inside = VirtualCursor(x=550, y=550, vx=20, vy=0)
    f_inside = friction_eng.compute_friction(cursor_inside, [target], uncert)

    # Must NOT freeze cursor (gain >= minimum_gain floor 0.25)
    assert f_inside.gain >= friction_eng.minimum_gain
    assert f_inside.gain > 0.0


def test_ambiguity_suppression():
    friction_eng = DynamicSensitivitySuppression()

    # Two targets placed very close together
    t1 = Target(id="t1", x=500, y=500, width=80, height=80, label="T1")
    t2 = Target(id="t2", x=590, y=500, width=80, height=80, label="T2")
    uncert = UncertaintyState(uncertainty_radius=30, confidence_score=0.8, tremor_energy=0.0, noise_estimate=0.0)

    # Cursor hovering static between them (vx=0, vy=0)
    cursor_mid = VirtualCursor(x=570, y=540, vx=0, vy=0)
    f_ambig = friction_eng.compute_friction(cursor_mid, [t1, t2], uncert)

    assert f_ambig.is_ambiguous is True
    # Friction factor must be suppressed compared to single target
    f_single = friction_eng.compute_friction(cursor_mid, [t1], uncert)
    assert f_ambig.friction_factor < f_single.friction_factor
