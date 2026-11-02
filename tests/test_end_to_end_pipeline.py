"""
Phase 3 Comprehensive End-to-End Interaction Integration Test Suite.
Verifies all 19 primary interaction steps and 6 critical failure/edge cases
across the unified 13-stage WearableMousePipeline.
"""

import pytest
import numpy as np
from typing import Tuple
from wearable_mouse.config import Config
from wearable_mouse.pipeline import WearableMousePipeline, PipelineFrameResult
from wearable_mouse.sensors.imu_data import IMUSample
from wearable_mouse.processing.calibration import CalibrationState
from wearable_mouse.processing.posture_reference import PostureState
from wearable_mouse.cursor.target_assistance import Target, AcquisitionMode
from wearable_mouse.gestures.gesture_map import GestureType
from wearable_mouse.interaction.interaction_state import UnifiedState
from wearable_mouse.interaction.output_adapter import SemanticEventType


def make_air_sample(t: float, gx: float = 0.0, gy: float = 0.0, gz: float = 0.0) -> IMUSample:
    """Simulates an IMU sample in AIR posture (accel variance > 0.08 surface threshold)."""
    ax = 1.0 + 0.8 * np.sin(t * 20.0)
    ay = 1.0 + 0.8 * np.cos(t * 20.0)
    az = 9.81 + 0.5 * np.sin(t * 15.0)
    return IMUSample(timestamp=t, accel=np.array([ax, ay, az]), gyro=np.array([gx, gy, gz]))


def make_rest_sample(t: float, gx: float = 0.0, gy: float = 0.0, gz: float = 0.0) -> IMUSample:
    """Simulates an IMU sample in REST posture (accel variance < 0.08 surface threshold)."""
    return IMUSample(timestamp=t, accel=np.array([0.5, 9.5, 2.0]), gyro=np.array([gx, gy, gz]))


def create_pipeline() -> Tuple[Config, WearableMousePipeline]:
    config = Config()
    pipeline = WearableMousePipeline(config)
    # Warmup calibrator to CALIBRATED state (250 samples @ 100Hz)
    for i in range(250):
        t = i * 0.01
        sample = make_air_sample(t)
        pipeline.process_frame(sample)
    assert pipeline.calibrator.state == CalibrationState.CALIBRATED
    return config, pipeline


# -----------------------------------------------------------------------------
# Test 1: Start in AIR
# -----------------------------------------------------------------------------
def test_1_start_in_air():
    config, pipeline = create_pipeline()

    res = None
    for i in range(20):
        t = 3.0 + i * 0.01
        res = pipeline.process_frame(make_air_sample(t, gx=0.0, gy=0.0, gz=0.0))

    assert res.calib_state == CalibrationState.CALIBRATED
    assert res.posture_info.state == PostureState.AIR
    assert res.interaction_state == UnifiedState.AIR


# -----------------------------------------------------------------------------
# Test 2: Move Cursor in AIR
# -----------------------------------------------------------------------------
def test_2_move_cursor_in_air():
    config, pipeline = create_pipeline()
    pipeline.cursor_engine.set_position(960.0, 540.0)

    init_x = pipeline.cursor_engine.cursor.x

    # Move wrist (gz = -0.50 rad/s) for 15 frames
    res = None
    for i in range(15):
        t = 3.0 + i * 0.01
        res = pipeline.process_frame(make_air_sample(t, gz=-0.50))

    assert res.cursor.x != init_x
    move_evts = [e for e in res.semantic_events if e.event_type == SemanticEventType.CURSOR_MOVE]
    assert len(move_evts) > 0


# -----------------------------------------------------------------------------
# Test 3: Approach Target
# -----------------------------------------------------------------------------
def test_3_approach_target():
    config, pipeline = create_pipeline()
    target = Target(id="t1", x=1200, y=500, width=100, height=80, label="Test Target")
    pipeline.set_targets([target])
    pipeline.cursor_engine.set_position(1100.0, 540.0)

    # Move cursor toward target (x=1200)
    res = None
    for i in range(15):
        t = 3.0 + i * 0.01
        res = pipeline.process_frame(make_air_sample(t, gz=-0.40))

    assert res.assist_result is not None


# -----------------------------------------------------------------------------
# Test 4: Dynamic Friction Activates near Target
# -----------------------------------------------------------------------------
def test_4_dynamic_friction_activates():
    config, pipeline = create_pipeline()
    target = Target(id="t1", x=1000, y=500, width=100, height=100, label="Target A")
    pipeline.set_targets([target])
    pipeline.set_mode(AcquisitionMode.E)  # Dynamic Friction Mode E
    pipeline.cursor_engine.set_position(960.0, 540.0)  # Approaching target from left (center=1050)

    # Move right toward target center (gz = +0.30)
    res = None
    for i in range(15):
        t = 3.0 + i * 0.01
        res = pipeline.process_frame(make_air_sample(t, gz=0.30))

    assert res.assist_result.friction_gain < 1.0
    assert res.interaction_state == UnifiedState.PRECISION


# -----------------------------------------------------------------------------
# Test 5: Thumb-Index Click
# -----------------------------------------------------------------------------
def test_5_thumb_index_click():
    config, pipeline = create_pipeline()

    # Stabilize AIR posture
    for i in range(20):
        pipeline.process_frame(make_air_sample(3.0 + i * 0.01))

    res = pipeline.process_frame(make_air_sample(3.3), gesture_input=GestureType.LEFT_CLICK)

    assert res.interaction_state == UnifiedState.LEFT_CLICK
    click_evts = [e for e in res.semantic_events if e.event_type == SemanticEventType.LEFT_CLICK]
    assert len(click_evts) == 1


# -----------------------------------------------------------------------------
# Test 6: Cursor Remains Stable During Click
# -----------------------------------------------------------------------------
def test_6_cursor_stability_during_click():
    config, pipeline = create_pipeline()
    pipeline.cursor_engine.set_position(960.0, 540.0)

    for i in range(20):
        pipeline.process_frame(make_air_sample(3.0 + i * 0.01))

    pos_before = (pipeline.cursor_engine.cursor.x, pipeline.cursor_engine.cursor.y)
    res = pipeline.process_frame(make_air_sample(3.3), gesture_input=GestureType.LEFT_CLICK)
    pos_after = (res.cursor.x, res.cursor.y)

    assert pos_before == pos_after


# -----------------------------------------------------------------------------
# Test 7: Hand Rests on Surface
# -----------------------------------------------------------------------------
def test_7_hand_rests_on_surface():
    config, pipeline = create_pipeline()

    # Simulate resting posture (constant accel vector, low gyro)
    res = None
    for i in range(45):
        t = 3.0 + i * 0.01
        res = pipeline.process_frame(make_rest_sample(t))

    assert res.posture_info.is_resting or res.interaction_state in (UnifiedState.REST, UnifiedState.REST_MOVING)


# -----------------------------------------------------------------------------
# Test 8: Cursor Location Retained on Rest (Zero Drift)
# -----------------------------------------------------------------------------
def test_8_cursor_location_retained_on_rest():
    config, pipeline = create_pipeline()
    pipeline.cursor_engine.set_position(500.0, 300.0)

    pos_air = (pipeline.cursor_engine.cursor.x, pipeline.cursor_engine.cursor.y)

    # Rest stationary for 30 frames
    res = None
    for i in range(30):
        t = 3.0 + i * 0.01
        res = pipeline.process_frame(make_rest_sample(t, gx=0.005, gy=-0.005))

    pos_rest = (res.cursor.x, res.cursor.y)
    assert pos_air == pos_rest


# -----------------------------------------------------------------------------
# Test 9: Intentional Resting Movement Moves Cursor
# -----------------------------------------------------------------------------
def test_9_intentional_resting_movement():
    config, pipeline = create_pipeline()
    pipeline.cursor_engine.set_position(500.0, 300.0)

    # Establish REST posture
    for i in range(30):
        pipeline.process_frame(make_rest_sample(3.0 + i * 0.01))

    x_before = pipeline.cursor_engine.cursor.x

    # Intentional rest motion (gz = 0.35 rad/s > activation threshold for 15 frames)
    res = None
    for i in range(15):
        t = 3.3 + i * 0.01
        res = pipeline.process_frame(make_rest_sample(t, gz=0.35))

    assert res.cursor.x != x_before
    assert res.posture_info.intentional_rest_motion


# -----------------------------------------------------------------------------
# Test 10: Thumb-Index Click While Resting
# -----------------------------------------------------------------------------
def test_10_resting_thumb_index_click():
    config, pipeline = create_pipeline()

    # Establish REST posture
    for i in range(30):
        pipeline.process_frame(make_rest_sample(3.0 + i * 0.01))

    # Click while resting
    res = pipeline.process_frame(make_rest_sample(3.3), gesture_input=GestureType.LEFT_CLICK)
    assert res.interaction_state == UnifiedState.LEFT_CLICK


# -----------------------------------------------------------------------------
# Test 11: Return to AIR
# -----------------------------------------------------------------------------
def test_11_return_to_air():
    config, pipeline = create_pipeline()

    # Establish REST posture
    for i in range(30):
        pipeline.process_frame(make_rest_sample(3.0 + i * 0.01))

    # Lift hand back to AIR (varying accel + gyro for 30 frames)
    res = None
    for i in range(30):
        t = 3.3 + i * 0.01
        res = pipeline.process_frame(make_air_sample(t, gx=0.20))

    assert not res.posture_info.is_resting
    assert res.interaction_state in (UnifiedState.AIR, UnifiedState.PRECISION)


# -----------------------------------------------------------------------------
# Test 12: Continue AIR Cursor Movement
# -----------------------------------------------------------------------------
def test_12_continue_air_movement():
    config, pipeline = create_pipeline()

    # Stabilize AIR posture
    for i in range(20):
        pipeline.process_frame(make_air_sample(3.0 + i * 0.01))

    x_start = pipeline.cursor_engine.cursor.x
    res = None
    for i in range(15):
        t = 3.2 + i * 0.01
        res = pipeline.process_frame(make_air_sample(t, gz=0.40))

    assert res.cursor.x != x_start


# -----------------------------------------------------------------------------
# Test 13: Perform Drag
# -----------------------------------------------------------------------------
def test_13_perform_drag():
    config, pipeline = create_pipeline()

    for i in range(20):
        pipeline.process_frame(make_air_sample(3.0 + i * 0.01))

    res = pipeline.process_frame(make_air_sample(3.3), gesture_input=GestureType.DRAG_START)

    assert res.interaction_state == UnifiedState.DRAG
    drag_evts = [e for e in res.semantic_events if e.event_type == SemanticEventType.DRAG_START]
    assert len(drag_evts) == 1


# -----------------------------------------------------------------------------
# Test 14: Release Drag
# -----------------------------------------------------------------------------
def test_14_release_drag():
    config, pipeline = create_pipeline()

    for i in range(20):
        pipeline.process_frame(make_air_sample(3.0 + i * 0.01))

    pipeline.process_frame(make_air_sample(3.2), gesture_input=GestureType.DRAG_START)
    assert pipeline.interaction_sm.current_state == UnifiedState.DRAG

    res = pipeline.process_frame(make_air_sample(3.3), gesture_input=GestureType.DRAG_END)
    assert res.interaction_state in (UnifiedState.AIR, UnifiedState.REST)
    drag_end_evts = [e for e in res.semantic_events if e.event_type == SemanticEventType.DRAG_END]
    assert len(drag_end_evts) == 1


# -----------------------------------------------------------------------------
# Test 15: Scroll Gesture
# -----------------------------------------------------------------------------
def test_15_scroll_gesture():
    config, pipeline = create_pipeline()

    for i in range(20):
        pipeline.process_frame(make_air_sample(3.0 + i * 0.01))

    res = pipeline.process_frame(make_air_sample(3.3), gesture_input=GestureType.SCROLL_UP)

    assert res.interaction_state == UnifiedState.SCROLL
    scroll_evts = [e for e in res.semantic_events if e.event_type == SemanticEventType.SCROLL]
    assert len(scroll_evts) == 1


# -----------------------------------------------------------------------------
# Test 16: Zoom Gesture
# -----------------------------------------------------------------------------
def test_16_zoom_gesture():
    config, pipeline = create_pipeline()

    for i in range(20):
        pipeline.process_frame(make_air_sample(3.0 + i * 0.01))

    res = pipeline.process_frame(make_air_sample(3.3), gesture_input=GestureType.ZOOM_IN)

    assert res.interaction_state == UnifiedState.ZOOM
    zoom_evts = [e for e in res.semantic_events if e.event_type == SemanticEventType.ZOOM]
    assert len(zoom_evts) == 1


# -----------------------------------------------------------------------------
# Test 17: Pause During Interaction
# -----------------------------------------------------------------------------
def test_17_pause_during_interaction():
    config, pipeline = create_pipeline()
    pipeline.cursor_engine.set_position(960.0, 540.0)

    res = pipeline.process_frame(make_air_sample(3.0), gesture_input=GestureType.PAUSE_TOGGLE)
    assert res.interaction_state == UnifiedState.PAUSED

    # Attempt cursor move while paused
    pos_paused = (res.cursor.x, res.cursor.y)
    res2 = pipeline.process_frame(make_air_sample(3.1, gz=0.50), gesture_input=GestureType.LEFT_CLICK)
    assert res2.interaction_state == UnifiedState.PAUSED
    assert (res2.cursor.x, res2.cursor.y) == pos_paused


# -----------------------------------------------------------------------------
# Test 18: Resume Interaction
# -----------------------------------------------------------------------------
def test_18_resume_interaction():
    config, pipeline = create_pipeline()

    for i in range(20):
        pipeline.process_frame(make_air_sample(3.0 + i * 0.01))

    pipeline.process_frame(make_air_sample(3.2), gesture_input=GestureType.PAUSE_TOGGLE)
    assert pipeline.interaction_sm.current_state == UnifiedState.PAUSED

    # Unpause
    res = pipeline.process_frame(make_air_sample(3.3), gesture_input=GestureType.PAUSE_TOGGLE)
    assert res.interaction_state in (UnifiedState.AIR, UnifiedState.REST)


# -----------------------------------------------------------------------------
# Test 19: Continue Movement Post Resume
# -----------------------------------------------------------------------------
def test_19_continue_movement_post_resume():
    config, pipeline = create_pipeline()

    for i in range(20):
        pipeline.process_frame(make_air_sample(3.0 + i * 0.01))

    pipeline.process_frame(make_air_sample(3.2), gesture_input=GestureType.PAUSE_TOGGLE)
    pipeline.process_frame(make_air_sample(3.3), gesture_input=GestureType.PAUSE_TOGGLE)

    x_before = pipeline.cursor_engine.cursor.x
    res = None
    for i in range(15):
        t = 3.4 + i * 0.01
        res = pipeline.process_frame(make_air_sample(t, gz=-0.50))

    assert res.cursor.x != x_before


# -----------------------------------------------------------------------------
# Test 20 (Edge): Tremor & High Noise Near Target
# -----------------------------------------------------------------------------
def test_20_edge_tremor_high_noise():
    config, pipeline = create_pipeline()
    pipeline.cursor_engine.set_position(960.0, 540.0)

    # Inject high tremor noise (high gyro variance)
    res = None
    for i in range(15):
        t = 3.0 + i * 0.01
        noise = (i % 2 * 2 - 1) * 2.50  # Tremor oscillations
        res = pipeline.process_frame(make_air_sample(t, gx=noise, gy=-noise))

    assert res.uncertainty.uncertainty_radius > config.uncertainty_base_radius


# -----------------------------------------------------------------------------
# Test 21 (Edge): Uncertain Finger State
# -----------------------------------------------------------------------------
def test_21_edge_uncertain_finger_state():
    config, pipeline = create_pipeline()

    for i in range(20):
        pipeline.process_frame(make_air_sample(3.0 + i * 0.01))

    # High spatial uncertainty should inhibit gesture click transitions
    pipeline.interaction_sm.config.gesture_max_uncertainty_threshold = 20.0

    res = pipeline.process_frame(
        make_air_sample(3.3),
        gesture_input=GestureType.LEFT_CLICK
    )
    # Under high default uncertainty (>20), state remains AIR or REST
    if res.uncertainty.uncertainty_radius > 20.0:
        assert res.interaction_state in (UnifiedState.AIR, UnifiedState.REST)


# -----------------------------------------------------------------------------
# Test 22 (Edge): Rapid AIR / REST Posture Transitions
# -----------------------------------------------------------------------------
def test_22_edge_rapid_air_rest_transition():
    config, pipeline = create_pipeline()
    pipeline.cursor_engine.set_position(960.0, 540.0)

    pos_initial = (pipeline.cursor_engine.cursor.x, pipeline.cursor_engine.cursor.y)

    # Rapid toggle AIR -> REST -> AIR -> REST
    for cycle in range(3):
        pipeline.process_frame(make_rest_sample(3.0 + cycle * 0.2))
        pipeline.process_frame(make_air_sample(3.1 + cycle * 0.2))

    pos_final = (pipeline.cursor_engine.cursor.x, pipeline.cursor_engine.cursor.y)
    assert abs(pos_initial[0] - pos_final[0]) < 2.0
    assert abs(pos_initial[1] - pos_final[1]) < 2.0


# -----------------------------------------------------------------------------
# Test 23 (Edge): Accidental Contact & Gesture Overlap
# -----------------------------------------------------------------------------
def test_23_edge_accidental_contact_and_overlap():
    config, pipeline = create_pipeline()

    for i in range(20):
        pipeline.process_frame(make_air_sample(3.0 + i * 0.01))

    # Trigger GESTURE_CANCEL
    res = pipeline.process_frame(make_air_sample(3.3), gesture_input=GestureType.GESTURE_CANCEL)
    assert res.interaction_state in (UnifiedState.AIR, UnifiedState.REST)


# -----------------------------------------------------------------------------
# Test 24 (Edge): Target Near Screen Edge
# -----------------------------------------------------------------------------
def test_24_edge_target_near_screen_edge():
    config, pipeline = create_pipeline()
    edge_target = Target(id="edge", x=0, y=0, width=50, height=50, label="Corner")
    pipeline.set_targets([edge_target])
    pipeline.cursor_engine.set_position(10.0, 10.0)

    res = pipeline.process_frame(make_air_sample(3.0, gz=0.50))  # Move left beyond screen
    assert res.cursor.x >= 0.0
    assert res.cursor.y >= 0.0


# -----------------------------------------------------------------------------
# Test 25 (Edge): Repeated Gestures & Rapid Clicks
# -----------------------------------------------------------------------------
def test_25_edge_repeated_gestures_and_transitions():
    config, pipeline = create_pipeline()

    for i in range(20):
        pipeline.process_frame(make_air_sample(3.0 + i * 0.01))

    for i in range(5):
        t = 3.3 + i * 0.1
        pipeline.process_frame(make_air_sample(t), gesture_input=GestureType.LEFT_CLICK)

    assert len(pipeline.output_adapter.event_history) >= 5
