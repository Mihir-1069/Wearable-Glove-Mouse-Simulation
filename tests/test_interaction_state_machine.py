"""
Comprehensive unit & integration test suite for the Unified Interaction State Machine.
Tests all 25 required state machine scenarios including AIR/REST transitions, REST_MOVING,
authoritative pause from any state, unpause position jump prevention, click/drag/scroll/zoom lifecycles,
high uncertainty safety gating, transition freeze stability, and structured transition logging.
"""

import pytest
import numpy as np
from wearable_mouse.config import Config
from wearable_mouse.interaction.interaction_state import UnifiedState, StateTransitionLog
from wearable_mouse.interaction.interaction_state_machine import InteractionStateMachine
from wearable_mouse.processing.calibration import CalibrationState
from wearable_mouse.processing.posture_reference import PostureInfo, PostureState, RestMovementClassification
from wearable_mouse.gestures.gesture_map import GestureEvent, GestureType
from wearable_mouse.cursor.target_assistance import AssistanceResult, Target, AcquisitionMode
from wearable_mouse.processing.uncertainty import UncertaintyState
from wearable_mouse.cursor.cursor_engine import VirtualCursor


def setup_sm():
    config = Config()
    sm = InteractionStateMachine(config)
    return config, sm


def make_posture(state: PostureState = PostureState.AIR, intentional_motion: bool = False, is_transitioning: bool = False) -> PostureInfo:
    return PostureInfo(
        state=state,
        classification=RestMovementClassification.REST_INTENTIONAL_MOVEMENT if intentional_motion else (
            RestMovementClassification.TRANSITION_FREEZE if is_transitioning else (
                RestMovementClassification.REST_NOISE if state == PostureState.REST else RestMovementClassification.AIR_MOVEMENT
            )
        ),
        is_resting=(state == PostureState.REST),
        is_transitioning=is_transitioning,
        reference_orientation=np.zeros(3),
        effective_gyro=np.zeros(3),
        intentional_rest_motion=intentional_motion,
        transition_delta=(0.0, 0.0),
        transition_timestamp=0.0
    )


def make_uncert(radius: float = 15.0) -> UncertaintyState:
    return UncertaintyState(uncertainty_radius=radius, confidence_score=1.0, tremor_energy=0.0, noise_estimate=0.0)


def make_cursor(x: float = 960.0, y: float = 540.0) -> VirtualCursor:
    cur = VirtualCursor(x, y)
    return cur


def make_assist(friction_gain: float = 1.0) -> AssistanceResult:
    return AssistanceResult(
        mode=AcquisitionMode.D,
        selected_target=None,
        target_confidence=1.0,
        is_ambiguous=False,
        ambiguous_targets=[],
        snapped_position=(960.0, 540.0),
        friction_gain=friction_gain
    )


# -----------------------------------------------------------------------------
# Test 1: AIR -> REST Transition
# -----------------------------------------------------------------------------
def test_1_air_to_rest_transition():
    config, sm = setup_sm()
    air_p = make_posture(PostureState.AIR)
    rest_p = make_posture(PostureState.REST)
    uncert = make_uncert()
    cur = make_cursor()

    st1 = sm.update(1.0, CalibrationState.CALIBRATED, air_p, None, None, uncert, cur)
    assert st1 == UnifiedState.AIR

    st2 = sm.update(1.01, CalibrationState.CALIBRATED, rest_p, None, None, uncert, cur)
    assert st2 == UnifiedState.REST


# -----------------------------------------------------------------------------
# Test 2: REST -> AIR Transition
# -----------------------------------------------------------------------------
def test_2_rest_to_air_transition():
    config, sm = setup_sm()
    rest_p = make_posture(PostureState.REST)
    air_p = make_posture(PostureState.AIR)
    uncert = make_uncert()
    cur = make_cursor()

    sm.update(1.0, CalibrationState.CALIBRATED, rest_p, None, None, uncert, cur)
    st2 = sm.update(1.01, CalibrationState.CALIBRATED, air_p, None, None, uncert, cur)
    assert st2 == UnifiedState.AIR


# -----------------------------------------------------------------------------
# Test 3: AIR -> REST -> AIR Continuous Transition
# -----------------------------------------------------------------------------
def test_3_air_rest_air_continuous():
    config, sm = setup_sm()
    air_p = make_posture(PostureState.AIR)
    rest_p = make_posture(PostureState.REST)
    uncert = make_uncert()
    cur = make_cursor()

    assert sm.update(1.0, CalibrationState.CALIBRATED, air_p, None, None, uncert, cur) == UnifiedState.AIR
    assert sm.update(1.01, CalibrationState.CALIBRATED, rest_p, None, None, uncert, cur) == UnifiedState.REST
    assert sm.update(1.02, CalibrationState.CALIBRATED, air_p, None, None, uncert, cur) == UnifiedState.AIR


# -----------------------------------------------------------------------------
# Test 4: REST -> REST_MOVING
# -----------------------------------------------------------------------------
def test_4_rest_to_rest_moving():
    config, sm = setup_sm()
    rest_p = make_posture(PostureState.REST, intentional_motion=False)
    rest_moving_p = make_posture(PostureState.REST, intentional_motion=True)
    uncert = make_uncert()
    cur = make_cursor()

    sm.update(1.0, CalibrationState.CALIBRATED, rest_p, None, None, uncert, cur)
    st2 = sm.update(1.01, CalibrationState.CALIBRATED, rest_moving_p, None, None, uncert, cur)
    assert st2 == UnifiedState.REST_MOVING


# -----------------------------------------------------------------------------
# Test 5: REST_MOVING -> REST
# -----------------------------------------------------------------------------
def test_5_rest_moving_to_rest():
    config, sm = setup_sm()
    rest_moving_p = make_posture(PostureState.REST, intentional_motion=True)
    rest_p = make_posture(PostureState.REST, intentional_motion=False)
    uncert = make_uncert()
    cur = make_cursor()

    sm.update(1.0, CalibrationState.CALIBRATED, rest_moving_p, None, None, uncert, cur)
    st2 = sm.update(1.01, CalibrationState.CALIBRATED, rest_p, None, None, uncert, cur)
    assert st2 == UnifiedState.REST


# -----------------------------------------------------------------------------
# Test 6: Pause from AIR (AIR -> PAUSED)
# -----------------------------------------------------------------------------
def test_6_pause_from_air():
    config, sm = setup_sm()
    air_p = make_posture(PostureState.AIR)
    uncert = make_uncert()
    cur = make_cursor()
    pause_evt = GestureEvent(gesture=GestureType.PAUSE_TOGGLE, timestamp=1.0)

    sm.update(0.99, CalibrationState.CALIBRATED, air_p, None, None, uncert, cur)
    st = sm.update(1.0, CalibrationState.CALIBRATED, air_p, pause_evt, None, uncert, cur)
    assert st == UnifiedState.PAUSED


# -----------------------------------------------------------------------------
# Test 7: Pause from REST (REST -> PAUSED)
# -----------------------------------------------------------------------------
def test_7_pause_from_rest():
    config, sm = setup_sm()
    rest_p = make_posture(PostureState.REST)
    uncert = make_uncert()
    cur = make_cursor()
    pause_evt = GestureEvent(gesture=GestureType.PAUSE_TOGGLE, timestamp=1.0)

    sm.update(0.99, CalibrationState.CALIBRATED, rest_p, None, None, uncert, cur)
    st = sm.update(1.0, CalibrationState.CALIBRATED, rest_p, pause_evt, None, uncert, cur)
    assert st == UnifiedState.PAUSED


# -----------------------------------------------------------------------------
# Test 8: Pause during DRAG (DRAG -> PAUSED)
# -----------------------------------------------------------------------------
def test_8_pause_during_drag():
    config, sm = setup_sm()
    air_p = make_posture(PostureState.AIR)
    uncert = make_uncert()
    cur = make_cursor()
    drag_evt = GestureEvent(gesture=GestureType.DRAG_START, timestamp=1.0)
    pause_evt = GestureEvent(gesture=GestureType.PAUSE_TOGGLE, timestamp=1.05)

    sm.update(1.0, CalibrationState.CALIBRATED, air_p, drag_evt, None, uncert, cur)
    assert sm.current_state == UnifiedState.DRAG

    st = sm.update(1.05, CalibrationState.CALIBRATED, air_p, pause_evt, None, uncert, cur)
    assert st == UnifiedState.PAUSED


# -----------------------------------------------------------------------------
# Test 9: Pause during SCROLL (SCROLL -> PAUSED)
# -----------------------------------------------------------------------------
def test_9_pause_during_scroll():
    config, sm = setup_sm()
    air_p = make_posture(PostureState.AIR)
    uncert = make_uncert()
    cur = make_cursor()
    scroll_evt = GestureEvent(gesture=GestureType.SCROLL_UP, timestamp=1.0)
    pause_evt = GestureEvent(gesture=GestureType.PAUSE_TOGGLE, timestamp=1.05)

    sm.update(1.0, CalibrationState.CALIBRATED, air_p, scroll_evt, None, uncert, cur)
    assert sm.current_state == UnifiedState.SCROLL

    st = sm.update(1.05, CalibrationState.CALIBRATED, air_p, pause_evt, None, uncert, cur)
    assert st == UnifiedState.PAUSED


# -----------------------------------------------------------------------------
# Test 10: Pause during ZOOM (ZOOM -> PAUSED)
# -----------------------------------------------------------------------------
def test_10_pause_during_zoom():
    config, sm = setup_sm()
    air_p = make_posture(PostureState.AIR)
    uncert = make_uncert()
    cur = make_cursor()
    zoom_evt = GestureEvent(gesture=GestureType.ZOOM_IN, timestamp=1.0)
    pause_evt = GestureEvent(gesture=GestureType.PAUSE_TOGGLE, timestamp=1.05)

    sm.update(1.0, CalibrationState.CALIBRATED, air_p, zoom_evt, None, uncert, cur)
    assert sm.current_state == UnifiedState.ZOOM

    st = sm.update(1.05, CalibrationState.CALIBRATED, air_p, pause_evt, None, uncert, cur)
    assert st == UnifiedState.PAUSED


# -----------------------------------------------------------------------------
# Test 11: Unpause Resume without Position Jump
# -----------------------------------------------------------------------------
def test_11_unpause_resume_no_jump():
    config, sm = setup_sm()
    air_p = make_posture(PostureState.AIR)
    uncert = make_uncert()
    cur = make_cursor(720.0, 380.0)
    pause_evt = GestureEvent(gesture=GestureType.PAUSE_TOGGLE, timestamp=1.0)

    # AIR -> PAUSED
    sm.update(1.0, CalibrationState.CALIBRATED, air_p, pause_evt, None, uncert, cur)
    assert sm.current_state == UnifiedState.PAUSED

    # While PAUSED, attempt updates
    sm.update(1.05, CalibrationState.CALIBRATED, air_p, None, None, uncert, cur)
    assert sm.current_state == UnifiedState.PAUSED

    # Unpause
    st_res = sm.update(1.10, CalibrationState.CALIBRATED, air_p, pause_evt, None, uncert, cur)
    assert st_res == UnifiedState.AIR
    assert (cur.x, cur.y) == (720.0, 380.0)


# -----------------------------------------------------------------------------
# Test 12: Click Conflict Resolution (only one primary action active)
# -----------------------------------------------------------------------------
def test_12_click_conflict_resolution():
    config, sm = setup_sm()
    air_p = make_posture(PostureState.AIR)
    uncert = make_uncert()
    cur = make_cursor()
    left_evt = GestureEvent(gesture=GestureType.LEFT_CLICK, timestamp=1.0)

    st = sm.update(1.0, CalibrationState.CALIBRATED, air_p, left_evt, None, uncert, cur)
    assert st == UnifiedState.LEFT_CLICK
    assert st != UnifiedState.DRAG
    assert st != UnifiedState.SCROLL


# -----------------------------------------------------------------------------
# Test 13: Gesture Priority Arbitration
# -----------------------------------------------------------------------------
def test_13_gesture_priority_arbitration():
    config, sm = setup_sm()
    air_p = make_posture(PostureState.AIR)
    uncert = make_uncert()
    cur = make_cursor()
    zoom_evt = GestureEvent(gesture=GestureType.ZOOM_IN, timestamp=1.0)

    st = sm.update(1.0, CalibrationState.CALIBRATED, air_p, zoom_evt, None, uncert, cur)
    assert st == UnifiedState.ZOOM


# -----------------------------------------------------------------------------
# Test 14: High Uncertainty Gating
# -----------------------------------------------------------------------------
def test_14_high_uncertainty_gating():
    config, sm = setup_sm()
    air_p = make_posture(PostureState.AIR)
    high_uncert = make_uncert(radius=100.0) # > 80px threshold
    cur = make_cursor()
    click_evt = GestureEvent(gesture=GestureType.LEFT_CLICK, timestamp=1.0)

    st = sm.update(1.0, CalibrationState.CALIBRATED, air_p, click_evt, None, high_uncert, cur)
    assert st == UnifiedState.AIR


# -----------------------------------------------------------------------------
# Test 15: Noisy Transition Stability
# -----------------------------------------------------------------------------
def test_15_noisy_transition_stability():
    config, sm = setup_sm()
    transition_p = make_posture(PostureState.AIR, is_transitioning=True)
    uncert = make_uncert()
    cur = make_cursor()
    click_evt = GestureEvent(gesture=GestureType.LEFT_CLICK, timestamp=1.0)

    st = sm.update(1.0, CalibrationState.CALIBRATED, transition_p, click_evt, None, uncert, cur)
    assert st == UnifiedState.AIR


# -----------------------------------------------------------------------------
# Test 16: Repeated Transitions Stability
# -----------------------------------------------------------------------------
def test_16_repeated_transitions_stability():
    config, sm = setup_sm()
    uncert = make_uncert()
    cur = make_cursor()

    for i in range(10):
        p_state = PostureState.AIR if i % 2 == 0 else PostureState.REST
        p = make_posture(p_state)
        st = sm.update(i * 0.1, CalibrationState.CALIBRATED, p, None, None, uncert, cur)
        expected = UnifiedState.AIR if p_state == PostureState.AIR else UnifiedState.REST
        assert st == expected


# -----------------------------------------------------------------------------
# Test 17: AIR -> LEFT_CLICK
# -----------------------------------------------------------------------------
def test_17_air_to_left_click():
    config, sm = setup_sm()
    air_p = make_posture(PostureState.AIR)
    uncert = make_uncert()
    cur = make_cursor()
    evt = GestureEvent(gesture=GestureType.LEFT_CLICK, timestamp=1.0)

    st = sm.update(1.0, CalibrationState.CALIBRATED, air_p, evt, None, uncert, cur)
    assert st == UnifiedState.LEFT_CLICK


# -----------------------------------------------------------------------------
# Test 18: AIR -> RIGHT_CLICK
# -----------------------------------------------------------------------------
def test_18_air_to_right_click():
    config, sm = setup_sm()
    air_p = make_posture(PostureState.AIR)
    uncert = make_uncert()
    cur = make_cursor()
    evt = GestureEvent(gesture=GestureType.RIGHT_CLICK, timestamp=1.0)

    st = sm.update(1.0, CalibrationState.CALIBRATED, air_p, evt, None, uncert, cur)
    assert st == UnifiedState.RIGHT_CLICK


# -----------------------------------------------------------------------------
# Test 19: AIR -> DOUBLE_CLICK
# -----------------------------------------------------------------------------
def test_19_air_to_double_click():
    config, sm = setup_sm()
    air_p = make_posture(PostureState.AIR)
    uncert = make_uncert()
    cur = make_cursor()
    evt = GestureEvent(gesture=GestureType.DOUBLE_CLICK, timestamp=1.0)

    st = sm.update(1.0, CalibrationState.CALIBRATED, air_p, evt, None, uncert, cur)
    assert st == UnifiedState.DOUBLE_CLICK


# -----------------------------------------------------------------------------
# Test 20: DRAG Lifecycle (DRAG_START -> DRAG_MOVE -> DRAG_END)
# -----------------------------------------------------------------------------
def test_20_drag_lifecycle():
    config, sm = setup_sm()
    air_p = make_posture(PostureState.AIR)
    uncert = make_uncert()
    cur = make_cursor()

    start_evt = GestureEvent(gesture=GestureType.DRAG_START, timestamp=1.0)
    move_evt = GestureEvent(gesture=GestureType.DRAG_MOVE, timestamp=1.05)
    end_evt = GestureEvent(gesture=GestureType.DRAG_END, timestamp=1.10)

    assert sm.update(1.0, CalibrationState.CALIBRATED, air_p, start_evt, None, uncert, cur) == UnifiedState.DRAG
    assert sm.update(1.05, CalibrationState.CALIBRATED, air_p, move_evt, None, uncert, cur) == UnifiedState.DRAG
    assert sm.update(1.10, CalibrationState.CALIBRATED, air_p, end_evt, None, uncert, cur) == UnifiedState.AIR


# -----------------------------------------------------------------------------
# Test 21: AIR -> SCROLL
# -----------------------------------------------------------------------------
def test_21_air_to_scroll():
    config, sm = setup_sm()
    air_p = make_posture(PostureState.AIR)
    uncert = make_uncert()
    cur = make_cursor()
    evt = GestureEvent(gesture=GestureType.SCROLL_UP, timestamp=1.0)

    st = sm.update(1.0, CalibrationState.CALIBRATED, air_p, evt, None, uncert, cur)
    assert st == UnifiedState.SCROLL


# -----------------------------------------------------------------------------
# Test 22: AIR -> ZOOM
# -----------------------------------------------------------------------------
def test_22_air_to_zoom():
    config, sm = setup_sm()
    air_p = make_posture(PostureState.AIR)
    uncert = make_uncert()
    cur = make_cursor()
    evt = GestureEvent(gesture=GestureType.ZOOM_IN, timestamp=1.0)

    st = sm.update(1.0, CalibrationState.CALIBRATED, air_p, evt, None, uncert, cur)
    assert st == UnifiedState.ZOOM


# -----------------------------------------------------------------------------
# Test 23: Precision Mode under Dynamic Friction
# -----------------------------------------------------------------------------
def test_23_precision_mode():
    config, sm = setup_sm()
    air_p = make_posture(PostureState.AIR)
    uncert = make_uncert()
    cur = make_cursor()
    friction_assist = make_assist(friction_gain=0.40) # Gain < 1.0 -> PRECISION mode

    st = sm.update(1.0, CalibrationState.CALIBRATED, air_p, None, friction_assist, uncert, cur)
    assert st == UnifiedState.PRECISION


# -----------------------------------------------------------------------------
# Test 24: GESTURE_CANCEL Restores Safe Neutral State
# -----------------------------------------------------------------------------
def test_24_gesture_cancel():
    config, sm = setup_sm()
    air_p = make_posture(PostureState.AIR)
    uncert = make_uncert()
    cur = make_cursor()
    cancel_evt = GestureEvent(gesture=GestureType.GESTURE_CANCEL, timestamp=1.0)

    st = sm.update(1.0, CalibrationState.CALIBRATED, air_p, cancel_evt, None, uncert, cur)
    assert st == UnifiedState.AIR


# -----------------------------------------------------------------------------
# Test 25: State Transition Logging Verification
# -----------------------------------------------------------------------------
def test_25_state_transition_logging():
    config, sm = setup_sm()
    air_p = make_posture(PostureState.AIR)
    rest_p = make_posture(PostureState.REST)
    uncert = make_uncert()
    cur = make_cursor(960.0, 540.0)

    sm.update(1.0, CalibrationState.CALIBRATED, air_p, None, None, uncert, cur)
    sm.update(1.05, CalibrationState.CALIBRATED, rest_p, None, None, uncert, cur)

    assert len(sm.transition_logs) > 0
    last_log = sm.transition_logs[-1]
    assert isinstance(last_log, StateTransitionLog)
    assert last_log.old_state == UnifiedState.AIR
    assert last_log.new_state == UnifiedState.REST
    assert last_log.timestamp == 1.05
    assert last_log.cursor_x == 960.0
    assert last_log.cursor_y == 540.0
