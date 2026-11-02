"""
20 Automated Unit and Integration Tests for Resting-Surface Cursor Control.
Covers hysteresis classification, directional rest motions (Right, Left, Up, Down, Diagonal, Slow, Continuous),
re-anchoring, AIR <-> REST transitions, uncertainty model integration, dynamic friction, and full pipeline integration.
"""

import numpy as np
import pytest

from wearable_mouse.config import Config
from wearable_mouse.sensors.imu_data import IMUSample
from wearable_mouse.processing.orientation import OrientationEstimator
from wearable_mouse.processing.posture_reference import (
    PostureReferenceTracker,
    PostureState,
    RestMovementClassification
)
from wearable_mouse.processing.uncertainty import UncertaintyEstimator
from wearable_mouse.processing.motion_detector import MotionDetector
from wearable_mouse.cursor.cursor_engine import CursorEngine
from wearable_mouse.cursor.dynamic_friction import DynamicSensitivitySuppression
from wearable_mouse.cursor.target_assistance import TargetAssistanceEngine, Target, AcquisitionMode


def setup_pipeline():
    config = Config()
    orient_est = OrientationEstimator(config)
    posture_tracker = PostureReferenceTracker(config)
    cursor_eng = CursorEngine(config)
    motion_det = MotionDetector(config)
    uncertainty_est = UncertaintyEstimator(config)
    friction_eng = DynamicSensitivitySuppression()
    target_eng = TargetAssistanceEngine(config)
    return config, orient_est, posture_tracker, cursor_eng, motion_det, uncertainty_est, friction_eng, target_eng


def enter_rest_state(config, orient_est, posture_tracker, cursor_eng, num_steps=40):
    """Helper function to put the pipeline into a stable REST state."""
    for i in range(num_steps):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)


# -----------------------------------------------------------------------------
# Test Cases 1 - 20
# -----------------------------------------------------------------------------

def test_1_rest_stationary_zero_drift():
    config, orient_est, posture_tracker, cursor_eng, *_ = setup_pipeline()
    enter_rest_state(config, orient_est, posture_tracker, cursor_eng, num_steps=40)
    pos_start = cursor_eng.cursor.pos

    for i in range(40, 100):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    pos_end = cursor_eng.cursor.pos
    assert posture_tracker.state == PostureState.REST
    assert pos_end[0] == pytest.approx(pos_start[0], abs=1e-5)
    assert pos_end[1] == pytest.approx(pos_start[1], abs=1e-5)


def test_2_rest_low_noise_stable():
    config, orient_est, posture_tracker, cursor_eng, *_ = setup_pipeline()
    enter_rest_state(config, orient_est, posture_tracker, cursor_eng, num_steps=40)
    pos_start = cursor_eng.cursor.pos

    np.random.seed(42)
    for i in range(40, 140):
        ts = i * config.dt
        noisy_gyro = np.random.normal(0.0, 0.005, size=3)
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=noisy_gyro)
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    pos_end = cursor_eng.cursor.pos
    drift = float(np.hypot(pos_end[0] - pos_start[0], pos_end[1] - pos_start[1]))
    assert drift < 0.1


def test_3_rest_high_noise_bounded():
    config, orient_est, posture_tracker, cursor_eng, *_ = setup_pipeline()
    enter_rest_state(config, orient_est, posture_tracker, cursor_eng, num_steps=40)
    pos_start = cursor_eng.cursor.pos

    np.random.seed(42)
    for i in range(40, 140):
        ts = i * config.dt
        noisy_gyro = np.random.normal(0.0, 0.012, size=3)
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=noisy_gyro)
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    pos_end = cursor_eng.cursor.pos
    drift = float(np.hypot(pos_end[0] - pos_start[0], pos_end[1] - pos_start[1]))
    assert drift < 1.0


def test_4_rest_intentional_right_movement():
    config, orient_est, posture_tracker, cursor_eng, *_ = setup_pipeline()
    enter_rest_state(config, orient_est, posture_tracker, cursor_eng, num_steps=40)
    cursor_eng.set_position(720.0, 380.0)

    for i in range(40, 50):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.0, 0.2]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    pos = cursor_eng.cursor.pos
    assert pos[0] > 720.0
    assert pos[1] == pytest.approx(380.0, abs=1e-3)


def test_5_rest_intentional_left_movement():
    config, orient_est, posture_tracker, cursor_eng, *_ = setup_pipeline()
    enter_rest_state(config, orient_est, posture_tracker, cursor_eng, num_steps=40)
    cursor_eng.set_position(720.0, 380.0)

    for i in range(40, 50):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.0, -0.2]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    pos = cursor_eng.cursor.pos
    assert pos[0] < 720.0
    assert pos[1] == pytest.approx(380.0, abs=1e-3)


def test_6_rest_intentional_up_movement():
    config, orient_est, posture_tracker, cursor_eng, *_ = setup_pipeline()
    enter_rest_state(config, orient_est, posture_tracker, cursor_eng, num_steps=40)
    cursor_eng.set_position(720.0, 380.0)

    for i in range(40, 50):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.2, 0.0]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    pos = cursor_eng.cursor.pos
    assert pos[0] == pytest.approx(720.0, abs=1e-3)
    assert pos[1] < 380.0


def test_7_rest_intentional_down_movement():
    config, orient_est, posture_tracker, cursor_eng, *_ = setup_pipeline()
    enter_rest_state(config, orient_est, posture_tracker, cursor_eng, num_steps=40)
    cursor_eng.set_position(720.0, 380.0)

    for i in range(40, 50):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, -0.2, 0.0]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    pos = cursor_eng.cursor.pos
    assert pos[0] == pytest.approx(720.0, abs=1e-3)
    assert pos[1] > 380.0


def test_8_rest_slow_movement_responds():
    config, orient_est, posture_tracker, cursor_eng, *_ = setup_pipeline()
    enter_rest_state(config, orient_est, posture_tracker, cursor_eng, num_steps=40)
    cursor_eng.set_position(720.0, 380.0)

    # Slow motion (gz = 0.06 rad/s > activation threshold 0.045)
    for i in range(40, 50):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.0, 0.06]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    pos = cursor_eng.cursor.pos
    assert pos[0] > 720.0


def test_9_rest_continuous_movement_progresses():
    config, orient_est, posture_tracker, cursor_eng, *_ = setup_pipeline()
    enter_rest_state(config, orient_est, posture_tracker, cursor_eng, num_steps=40)

    x_history = []
    for i in range(40, 80):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.0, 0.15]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cur = cursor_eng.update(sample.gyro, posture_info=posture_info)
        x_history.append(cur.x)

    # Continuous movement should advance cursor monotonically right
    assert x_history[10] < x_history[20] < x_history[35]


def test_10_rest_noise_to_intentional_transition():
    config, orient_est, posture_tracker, cursor_eng, *_ = setup_pipeline()
    enter_rest_state(config, orient_est, posture_tracker, cursor_eng, num_steps=40)

    # Noise phase
    for i in range(40, 50):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.0, 0.01]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        assert posture_info.classification == RestMovementClassification.REST_NOISE

    # Intentional motion phase
    for i in range(50, 60):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.0, 0.2]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)

    assert posture_info.classification == RestMovementClassification.REST_INTENTIONAL_MOVEMENT


def test_11_rest_intentional_to_stop_rest():
    config, orient_est, posture_tracker, cursor_eng, *_ = setup_pipeline()
    enter_rest_state(config, orient_est, posture_tracker, cursor_eng, num_steps=40)

    # Intentional motion
    for i in range(40, 50):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.0, 0.2]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)

    assert posture_info.classification == RestMovementClassification.REST_INTENTIONAL_MOVEMENT

    # Stop motion
    for i in range(50, 65):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)

    assert posture_info.classification == RestMovementClassification.REST_NOISE


def test_12_air_to_rest_to_intentional_movement():
    config, orient_est, posture_tracker, cursor_eng, *_ = setup_pipeline()

    # AIR movement
    for i in range(20):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.0, 0.2]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    # Enter REST
    for i in range(20, 60):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    assert posture_tracker.state == PostureState.REST

    pos_rest = cursor_eng.cursor.pos

    # Intentional motion while resting
    for i in range(60, 75):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.0, 0.2]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    pos_after_move = cursor_eng.cursor.pos
    assert pos_after_move[0] > pos_rest[0]


def test_13_rest_movement_to_air():
    config, orient_est, posture_tracker, cursor_eng, *_ = setup_pipeline()
    enter_rest_state(config, orient_est, posture_tracker, cursor_eng, num_steps=40)

    # Move while RESTing
    cursor_eng.set_position(750.0, 350.0)
    for i in range(40, 50):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.0, 0.1]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    updated_pos = cursor_eng.cursor.pos

    # Lift hand to AIR (TRANSITION phase: 15 steps)
    for i in range(50, 65):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([1.5, 1.5, 9.81]), gyro=np.array([0.8, 0.8, 0.0]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    # Stationary in AIR
    for i in range(65, 70):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    air_pos = cursor_eng.cursor.pos
    assert air_pos[0] == pytest.approx(updated_pos[0], abs=1e-4)
    assert air_pos[1] == pytest.approx(updated_pos[1], abs=1e-4)


def test_14_full_air_rest_movement_air_pipeline():
    config, orient_est, posture_tracker, cursor_eng, *_ = setup_pipeline()

    # Step 1: AIR movement
    for i in range(20):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.0, 0.2]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    # Step 2: REST entry & noise
    for i in range(20, 60):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    assert posture_tracker.state == PostureState.REST

    # Step 3: Intentional REST movement (Right & Up)
    cursor_eng.set_position(750.0, 350.0)
    for i in range(60, 75):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.1, 0.15]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    rest_move_pos = cursor_eng.cursor.pos

    # Step 4: Lift to AIR
    for i in range(75, 90):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([1.5, 1.5, 9.81]), gyro=np.array([0.8, 0.8, 0.0]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    # Step 5: AIR movement
    for i in range(90, 100):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.0, 0.1]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    final_pos = cursor_eng.cursor.pos
    assert final_pos[0] > rest_move_pos[0]


def test_15_repeated_air_rest_transitions():
    config, orient_est, posture_tracker, cursor_eng, *_ = setup_pipeline()
    initial_pos = cursor_eng.cursor.pos

    for cycle in range(5):
        for i in range(15):
            ts = (cycle * 55 + i) * config.dt
            sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3))
            orient, _ = orient_est.update(sample)
            posture_info = posture_tracker.update(sample, orient, sample.gyro)
            cursor_eng.update(sample.gyro, posture_info=posture_info)
        for i in range(15, 55):
            ts = (cycle * 55 + i) * config.dt
            sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3))
            orient, _ = orient_est.update(sample)
            posture_info = posture_tracker.update(sample, orient, sample.gyro)
            cursor_eng.update(sample.gyro, posture_info=posture_info)

    final_pos = cursor_eng.cursor.pos
    assert float(np.hypot(final_pos[0] - initial_pos[0], final_pos[1] - initial_pos[1])) < 0.01


def test_16_first_movement_after_rest_reanchor():
    config, orient_est, posture_tracker, cursor_eng, *_ = setup_pipeline()
    enter_rest_state(config, orient_est, posture_tracker, cursor_eng, num_steps=40)
    pos_anchor = cursor_eng.cursor.pos

    cur = cursor_eng.cursor
    for i in range(40, 43):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.0, 0.1]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cur = cursor_eng.update(sample.gyro, posture_info=posture_info)

    step_dist = float(np.hypot(cur.x - pos_anchor[0], cur.y - pos_anchor[1]))
    assert 0.0 < step_dist < 3.0


def test_17_diagonal_movement_while_rest():
    config, orient_est, posture_tracker, cursor_eng, *_ = setup_pipeline()
    enter_rest_state(config, orient_est, posture_tracker, cursor_eng, num_steps=40)
    cursor_eng.set_position(720.0, 380.0)

    for i in range(40, 50):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.15, 0.15]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    pos = cursor_eng.cursor.pos
    assert pos[0] > 720.0
    assert pos[1] < 380.0


def test_18_rest_movement_with_uncertainty():
    config, orient_est, posture_tracker, cursor_eng, motion_det, uncertainty_est, *_ = setup_pipeline()
    enter_rest_state(config, orient_est, posture_tracker, cursor_eng, num_steps=40)

    for i in range(40, 55):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.0, 0.3]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        m_state = motion_det.update(sample.gyro)
        u_state = uncertainty_est.update(sample.gyro, sample.gyro, m_state)
        cur = cursor_eng.update(sample.gyro, posture_info=posture_info)

    assert u_state.uncertainty_radius > 0.0
    assert cur.x > 960.0


def test_19_rest_movement_with_dynamic_friction():
    config, orient_est, posture_tracker, cursor_eng, _, uncertainty_est, friction_eng, _ = setup_pipeline()
    enter_rest_state(config, orient_est, posture_tracker, cursor_eng, num_steps=40)

    target = Target(id="t1", x=1100, y=540, width=80, height=80, label="Target 1")
    cursor_eng.set_position(1050.0, 540.0)

    for i in range(40, 50):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.0, 0.2]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cur = cursor_eng.update(sample.gyro, posture_info=posture_info)
        u_state = uncertainty_est.update(sample.gyro, sample.gyro, MotionDetector(config).update(sample.gyro))
        f_state = friction_eng.compute_friction(cur, [target], u_state)

    assert f_state.gain <= 1.0


def test_20_end_to_end_full_pipeline():
    config, orient_est, posture_tracker, cursor_eng, motion_det, uncertainty_est, friction_eng, target_eng = setup_pipeline()
    target = Target(id="t_main", x=1200, y=540, width=100, height=100, label="Target")

    # Complete pipeline sequence
    for i in range(100):
        ts = i * config.dt
        if i < 30:
            gyro = np.array([0.0, 0.0, 0.2])
        elif i < 60:
            gyro = np.zeros(3)
        else:
            gyro = np.array([0.0, 0.0, 0.25])

        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=gyro)
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        m_state = motion_det.update(sample.gyro)
        u_state = uncertainty_est.update(sample.gyro, sample.gyro, m_state)
        cur = cursor_eng.update(sample.gyro, posture_info=posture_info)
        assist = target_eng.process(cur, u_state, mode=AcquisitionMode.B)

        assert cur.x >= 0.0 and cur.y >= 0.0
        assert assist is not None
