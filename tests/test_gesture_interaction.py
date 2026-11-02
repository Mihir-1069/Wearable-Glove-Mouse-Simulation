"""
Comprehensive unit and integration test suite for the Gesture Interaction Engine.
Tests all 20 required scenarios (A through T) including single click, right click, double click,
drag start/hold/end, scroll, zoom, noisy debouncing, short accidental contact, cooldown, conflict resolution,
cancellation, posture transitions (AIR/REST), high uncertainty, and tremor inhibition.
"""

import pytest
import numpy as np
from wearable_mouse.config import Config
from wearable_mouse.sensors.imu_data import IMUSample, SensorData
from wearable_mouse.processing.posture_reference import PostureInfo, PostureState, RestMovementClassification
from wearable_mouse.processing.uncertainty import UncertaintyState
from wearable_mouse.gestures.gesture_engine import GestureEngine, SystemState, InteractionState
from wearable_mouse.gestures.gesture_map import GestureType, FingerName, FingerState, FingerPose


def setup_engine():
    config = Config()
    engine = GestureEngine(config)
    return config, engine


def make_sensor_data(timestamp: float, flex_dict: dict) -> SensorData:
    imu = IMUSample(timestamp=timestamp, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3))
    return SensorData(imu=imu, flex_sensors=flex_dict)


def make_posture_info(state: PostureState = PostureState.AIR) -> PostureInfo:
    return PostureInfo(
        state=state,
        classification=RestMovementClassification.AIR_MOVEMENT if state == PostureState.AIR else RestMovementClassification.REST_NOISE,
        is_resting=(state == PostureState.REST),
        is_transitioning=(state == PostureState.TRANSITION),
        reference_orientation=np.zeros(3),
        effective_gyro=np.zeros(3),
        intentional_rest_motion=False,
        transition_delta=(0.0, 0.0),
        transition_timestamp=0.0
    )


def make_uncertainty(radius: float = 15.0) -> UncertaintyState:
    return UncertaintyState(uncertainty_radius=radius, confidence_score=1.0, tremor_energy=0.0, noise_estimate=0.0)


# -----------------------------------------------------------------------------
# Test A: Single Left Click (thumb + index pinch)
# -----------------------------------------------------------------------------
def test_a_single_left_click():
    config, engine = setup_engine()
    posture = make_posture_info()
    uncert = make_uncertainty()

    # Step through debouncing (40ms) and min_stable_duration (80ms)
    pinch = {"thumb": 0.9, "index": 0.9, "middle": 0.0, "ring": 0.0, "little": 0.0}
    events = []

    for i in range(15):
        ts = i * config.dt
        data = make_sensor_data(ts, pinch)
        evt = engine.process_frame(data, ts, posture, uncert)
        if evt:
            events.append(evt)

    assert any(e.gesture == GestureType.LEFT_CLICK for e in events)


# -----------------------------------------------------------------------------
# Test B: Right Click (thumb + middle pinch)
# -----------------------------------------------------------------------------
def test_b_right_click():
    config, engine = setup_engine()
    posture = make_posture_info()
    uncert = make_uncertainty()

    right_pinch = {"thumb": 0.9, "index": 0.0, "middle": 0.9, "ring": 0.0, "little": 0.0}
    events = []

    for i in range(15):
        ts = i * config.dt
        data = make_sensor_data(ts, right_pinch)
        evt = engine.process_frame(data, ts, posture, uncert)
        if evt:
            events.append(evt)

    assert any(e.gesture == GestureType.RIGHT_CLICK for e in events)


# -----------------------------------------------------------------------------
# Test C: Double Click (two rapid thumb + index pinches)
# -----------------------------------------------------------------------------
def test_c_double_click():
    config, engine = setup_engine()
    posture = make_posture_info()
    uncert = make_uncertainty()

    pinch = {"thumb": 0.9, "index": 0.9, "middle": 0.0, "ring": 0.0, "little": 0.0}
    open_hand = {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "little": 0.0}

    events = []

    # First pinch (15 frames: ts=0.00s..0.14s) -> emits LEFT_CLICK at ts=0.12s
    for i in range(15):
        ts = i * config.dt
        data = make_sensor_data(ts, pinch)
        evt = engine.process_frame(data, ts, posture, uncert)
        if evt:
            events.append(evt)

    # Release pinch (5 frames: ts=0.15s..0.19s)
    for i in range(15, 20):
        ts = i * config.dt
        data = make_sensor_data(ts, open_hand)
        evt = engine.process_frame(data, ts, posture, uncert)
        if evt:
            events.append(evt)

    # Second pinch within double click window (15 frames: ts=0.20s..0.34s) -> emits DOUBLE_CLICK
    for i in range(20, 35):
        ts = i * config.dt
        data = make_sensor_data(ts, pinch)
        evt = engine.process_frame(data, ts, posture, uncert)
        if evt:
            events.append(evt)

    double_clicks = [e for e in events if e.gesture == GestureType.DOUBLE_CLICK]
    assert len(double_clicks) > 0


# -----------------------------------------------------------------------------
# Test D: Drag Start (sustained pinch > drag_hold_duration)
# -----------------------------------------------------------------------------
def test_d_drag_start():
    config, engine = setup_engine()
    posture = make_posture_info()
    uncert = make_uncertainty()

    pinch = {"thumb": 0.9, "index": 0.9, "middle": 0.0, "ring": 0.0, "little": 0.0}
    events = []

    # Hold pinch for 50 frames (0.50s > 0.35s drag_hold_duration)
    for i in range(50):
        ts = i * config.dt
        data = make_sensor_data(ts, pinch)
        evt = engine.process_frame(data, ts, posture, uncert)
        if evt:
            events.append(evt)

    drag_starts = [e for e in events if e.gesture == GestureType.DRAG_START]
    assert len(drag_starts) > 0
    assert engine.is_dragging is True


# -----------------------------------------------------------------------------
# Test E: Drag Hold (continuous DRAG_MOVE frames)
# -----------------------------------------------------------------------------
def test_e_drag_hold():
    config, engine = setup_engine()
    posture = make_posture_info()
    uncert = make_uncertainty()

    pinch = {"thumb": 0.9, "index": 0.9, "middle": 0.0, "ring": 0.0, "little": 0.0}
    events = []

    # Hold pinch for 70 frames
    for i in range(70):
        ts = i * config.dt
        data = make_sensor_data(ts, pinch)
        evt = engine.process_frame(data, ts, posture, uncert)
        if evt:
            events.append(evt)

    drag_moves = [e for e in events if e.gesture == GestureType.DRAG_MOVE]
    assert len(drag_moves) > 0


# -----------------------------------------------------------------------------
# Test F: Drag Release (DRAG_END on pinch release)
# -----------------------------------------------------------------------------
def test_f_drag_release():
    config, engine = setup_engine()
    posture = make_posture_info()
    uncert = make_uncertainty()

    pinch = {"thumb": 0.9, "index": 0.9, "middle": 0.0, "ring": 0.0, "little": 0.0}
    open_hand = {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "little": 0.0}
    events = []

    # Initiate drag (50 frames)
    for i in range(50):
        ts = i * config.dt
        data = make_sensor_data(ts, pinch)
        engine.process_frame(data, ts, posture, uncert)

    assert engine.is_dragging is True

    # Release pinch
    for i in range(50, 60):
        ts = i * config.dt
        data = make_sensor_data(ts, open_hand)
        evt = engine.process_frame(data, ts, posture, uncert)
        if evt:
            events.append(evt)

    drag_ends = [e for e in events if e.gesture == GestureType.DRAG_END]
    assert len(drag_ends) > 0
    assert engine.is_dragging is False


# -----------------------------------------------------------------------------
# Test G: Scroll (2-finger pose: index + middle flexed)
# -----------------------------------------------------------------------------
def test_g_scroll():
    config, engine = setup_engine()
    posture = make_posture_info()
    uncert = make_uncertainty()

    scroll_pose = {"thumb": 0.0, "index": 0.8, "middle": 0.8, "ring": 0.0, "little": 0.0}
    events = []

    for i in range(15):
        ts = i * config.dt
        data = make_sensor_data(ts, scroll_pose)
        evt = engine.process_frame(data, ts, posture, uncert)
        if evt:
            events.append(evt)

    assert any(e.gesture == GestureType.SCROLL_UP for e in events)


# -----------------------------------------------------------------------------
# Test H: Zoom (3-finger pose: index + middle + ring flexed)
# -----------------------------------------------------------------------------
def test_h_zoom():
    config, engine = setup_engine()
    posture = make_posture_info()
    uncert = make_uncertainty()

    zoom_pose = {"thumb": 0.0, "index": 0.8, "middle": 0.8, "ring": 0.8, "little": 0.0}
    events = []

    for i in range(15):
        ts = i * config.dt
        data = make_sensor_data(ts, zoom_pose)
        evt = engine.process_frame(data, ts, posture, uncert)
        if evt:
            events.append(evt)

    assert any(e.gesture == GestureType.ZOOM_IN for e in events)


# -----------------------------------------------------------------------------
# Test I: Noisy Finger Transition (analog noise debounced cleanly)
# -----------------------------------------------------------------------------
def test_i_noisy_finger_transition():
    config, engine = setup_engine()

    # Rapid high frequency noise oscillating around 0.65 / 0.75
    timestamps = [i * config.dt for i in range(10)]
    for i, ts in enumerate(timestamps):
        noisy_val = 0.65 if i % 2 == 0 else 0.75
        data = make_sensor_data(ts, {"thumb": 0.0, "index": noisy_val})
        pose = engine.finger_tracker.update(data.flex_sensors, ts)

    # Should remain debounced OPEN or UNKNOWN during fast jitter
    assert pose.finger_states[FingerName.INDEX] in (FingerState.OPEN, FingerState.UNKNOWN)


# -----------------------------------------------------------------------------
# Test J: Short Accidental Contact (contact < min_stable_duration ignored)
# -----------------------------------------------------------------------------
def test_j_short_accidental_contact():
    config, engine = setup_engine()
    posture = make_posture_info()
    uncert = make_uncertainty()

    pinch = {"thumb": 0.9, "index": 0.9, "middle": 0.0, "ring": 0.0, "little": 0.0}
    open_hand = {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "little": 0.0}
    events = []

    # Touch for only 3 frames (30ms < 80ms min_stable_duration)
    for i in range(3):
        ts = i * config.dt
        data = make_sensor_data(ts, pinch)
        evt = engine.process_frame(data, ts, posture, uncert)
        if evt:
            events.append(evt)

    # Immediately release
    for i in range(3, 10):
        ts = i * config.dt
        data = make_sensor_data(ts, open_hand)
        evt = engine.process_frame(data, ts, posture, uncert)
        if evt:
            events.append(evt)

    left_clicks = [e for e in events if e.gesture == GestureType.LEFT_CLICK]
    assert len(left_clicks) == 0


# -----------------------------------------------------------------------------
# Test K: Gesture Debounce (rapid chatter filtered out)
# -----------------------------------------------------------------------------
def test_k_gesture_debounce():
    config, engine = setup_engine()

    # Alternate flex high and low every single frame
    for i in range(6):
        ts = i * config.dt
        val = 0.9 if i % 2 == 0 else 0.1
        data = make_sensor_data(ts, {"thumb": val, "index": val})
        pose = engine.finger_tracker.update(data.flex_sensors, ts)

    # Confirmed state remains OPEN because candidate is reset every frame
    assert pose.finger_states[FingerName.THUMB] == FingerState.OPEN


# -----------------------------------------------------------------------------
# Test L: Gesture Cooldown (refractory period blocks rapid single-click re-trigger)
# -----------------------------------------------------------------------------
def test_l_gesture_cooldown():
    config, engine = setup_engine()
    posture = make_posture_info()
    uncert = make_uncertainty()

    pinch = {"thumb": 0.9, "index": 0.9, "middle": 0.0, "ring": 0.0, "little": 0.0}
    open_hand = {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "little": 0.0}
    events = []

    # First pinch (emits LEFT_CLICK at ts=0.12s)
    for i in range(15):
        ts = i * config.dt
        data = make_sensor_data(ts, pinch)
        evt = engine.process_frame(data, ts, posture, uncert)
        if evt:
            events.append(evt)

    # Release (5 frames)
    for i in range(15, 20):
        ts = i * config.dt
        data = make_sensor_data(ts, open_hand)
        engine.process_frame(data, ts, posture, uncert)

    # Change last_click_type to RIGHT_CLICK so double-click doesn't apply
    engine.last_click_type = GestureType.RIGHT_CLICK

    # Attempt second pinch during cooldown window
    for i in range(20, 35):
        ts = i * config.dt
        data = make_sensor_data(ts, pinch)
        evt = engine.process_frame(data, ts, posture, uncert)
        if evt:
            events.append(evt)

    # Second click attempt during cooldown should be blocked
    left_clicks = [e for e in events if e.gesture == GestureType.LEFT_CLICK]
    assert len(left_clicks) == 1


# -----------------------------------------------------------------------------
# Test M: Conflicting Gestures (higher priority 3-finger pre-empts 1-finger)
# -----------------------------------------------------------------------------
def test_m_conflicting_gestures():
    config, engine = setup_engine()
    posture = make_posture_info()
    uncert = make_uncertainty()

    zoom_pose = {"thumb": 0.0, "index": 0.8, "middle": 0.8, "ring": 0.8, "little": 0.0}
    events = []

    for i in range(15):
        ts = i * config.dt
        data = make_sensor_data(ts, zoom_pose)
        evt = engine.process_frame(data, ts, posture, uncert)
        if evt:
            events.append(evt)

    # Must resolve to ZOOM_IN, not LEFT_CLICK or RIGHT_CLICK
    assert all(e.gesture != GestureType.LEFT_CLICK for e in events)
    assert any(e.gesture == GestureType.ZOOM_IN for e in events)


# -----------------------------------------------------------------------------
# Test N: Gesture Cancellation (interrupted candidate emits GESTURE_CANCEL)
# -----------------------------------------------------------------------------
def test_n_gesture_cancellation():
    config, engine = setup_engine()
    posture = make_posture_info()
    uncert = make_uncertainty()

    pinch = {"thumb": 0.9, "index": 0.9, "middle": 0.0, "ring": 0.0, "little": 0.0}
    open_hand = {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "little": 0.0}
    events = []

    # Start pinch candidate for 7 frames (confirmed pose formed, but min_stable_duration not met yet)
    for i in range(7):
        ts = i * config.dt
        data = make_sensor_data(ts, pinch)
        evt = engine.process_frame(data, ts, posture, uncert)
        if evt:
            events.append(evt)

    # Intercept with open hand for 10 frames
    for i in range(7, 17):
        ts = i * config.dt
        data = make_sensor_data(ts, open_hand)
        evt = engine.process_frame(data, ts, posture, uncert)
        if evt:
            events.append(evt)

    cancels = [e for e in events if e.gesture == GestureType.GESTURE_CANCEL]
    assert len(cancels) > 0


# -----------------------------------------------------------------------------
# Test O: Gesture during REST (valid gesture allowed while resting)
# -----------------------------------------------------------------------------
def test_o_gesture_during_rest():
    config, engine = setup_engine()
    rest_posture = make_posture_info(PostureState.REST)
    uncert = make_uncertainty()

    pinch = {"thumb": 0.9, "index": 0.9, "middle": 0.0, "ring": 0.0, "little": 0.0}
    events = []

    for i in range(15):
        ts = i * config.dt
        data = make_sensor_data(ts, pinch)
        evt = engine.process_frame(data, ts, rest_posture, uncert)
        if evt:
            events.append(evt)

    assert any(e.gesture == GestureType.LEFT_CLICK for e in events)


# -----------------------------------------------------------------------------
# Test P: Gesture during AIR (valid gesture allowed in air)
# -----------------------------------------------------------------------------
def test_p_gesture_during_air():
    config, engine = setup_engine()
    air_posture = make_posture_info(PostureState.AIR)
    uncert = make_uncertainty()

    pinch = {"thumb": 0.9, "index": 0.9, "middle": 0.0, "ring": 0.0, "little": 0.0}
    events = []

    for i in range(15):
        ts = i * config.dt
        data = make_sensor_data(ts, pinch)
        evt = engine.process_frame(data, ts, air_posture, uncert)
        if evt:
            events.append(evt)

    assert any(e.gesture == GestureType.LEFT_CLICK for e in events)


# -----------------------------------------------------------------------------
# Test Q: Gesture during AIR->REST Transition (inhibited during freeze window)
# -----------------------------------------------------------------------------
def test_q_gesture_during_air_to_rest_transition():
    config, engine = setup_engine()
    transition_posture = make_posture_info(PostureState.TRANSITION)
    uncert = make_uncertainty()

    pinch = {"thumb": 0.9, "index": 0.9, "middle": 0.0, "ring": 0.0, "little": 0.0}
    events = []

    for i in range(15):
        ts = i * config.dt
        data = make_sensor_data(ts, pinch)
        evt = engine.process_frame(data, ts, transition_posture, uncert)
        if evt:
            events.append(evt)

    assert all(e.gesture != GestureType.LEFT_CLICK for e in events)


# -----------------------------------------------------------------------------
# Test R: Gesture during REST->AIR Transition (inhibited during freeze window)
# -----------------------------------------------------------------------------
def test_r_gesture_during_rest_to_air_transition():
    config, engine = setup_engine()
    transition_posture = make_posture_info(PostureState.TRANSITION)
    uncert = make_uncertainty()

    right_pinch = {"thumb": 0.9, "index": 0.0, "middle": 0.9, "ring": 0.0, "little": 0.0}
    events = []

    for i in range(15):
        ts = i * config.dt
        data = make_sensor_data(ts, right_pinch)
        evt = engine.process_frame(data, ts, transition_posture, uncert)
        if evt:
            events.append(evt)

    assert all(e.gesture != GestureType.RIGHT_CLICK for e in events)


# -----------------------------------------------------------------------------
# Test S: High Uncertainty Inhibition (uncertainty > 80px blocks gesture)
# -----------------------------------------------------------------------------
def test_s_high_uncertainty_inhibition():
    config, engine = setup_engine()
    posture = make_posture_info(PostureState.AIR)
    high_uncert = make_uncertainty(radius=100.0) # > 80px threshold

    pinch = {"thumb": 0.9, "index": 0.9, "middle": 0.0, "ring": 0.0, "little": 0.0}
    events = []

    for i in range(15):
        ts = i * config.dt
        data = make_sensor_data(ts, pinch)
        evt = engine.process_frame(data, ts, posture, high_uncert)
        if evt:
            events.append(evt)

    assert all(e.gesture != GestureType.LEFT_CLICK for e in events)


# -----------------------------------------------------------------------------
# Test T: Diagnostic Telemetry Generation
# -----------------------------------------------------------------------------
def test_t_diagnostic_telemetry():
    config, engine = setup_engine()
    posture = make_posture_info()
    uncert = make_uncertainty()

    pinch = {"thumb": 0.9, "index": 0.9, "middle": 0.0, "ring": 0.0, "little": 0.0}
    for i in range(15):
        ts = i * config.dt
        data = make_sensor_data(ts, pinch)
        engine.process_frame(data, ts, posture, uncert)

    diag = engine.get_diagnostic_state()
    assert diag is not None
    assert "THUMB" in diag.finger_states
    assert diag.finger_states["THUMB"] in ("TOUCHING", "CLOSED")
    assert diag.system_state == "ACTIVE"
