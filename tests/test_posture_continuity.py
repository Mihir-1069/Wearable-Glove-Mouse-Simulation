"""
Unit and Integration Tests for AIR <-> REST Posture Continuity, Re-Anchoring, and Noise/Drift Suppression.
"""

import numpy as np
import pytest

from wearable_mouse.config import Config
from wearable_mouse.sensors.imu_data import IMUSample
from wearable_mouse.processing.orientation import OrientationEstimator
from wearable_mouse.processing.posture_reference import PostureReferenceTracker, PostureState
from wearable_mouse.cursor.cursor_engine import CursorEngine


def setup_pipeline():
    config = Config()
    orient_est = OrientationEstimator(config)
    posture_tracker = PostureReferenceTracker(config)
    cursor_eng = CursorEngine(config)
    return config, orient_est, posture_tracker, cursor_eng


def test_1_air_to_rest_no_jump():
    config, orient_est, posture_tracker, cursor_eng = setup_pipeline()

    # Move cursor in AIR for 20 steps
    for i in range(20):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.0, 0.2]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    pos_before = cursor_eng.cursor.pos

    # Simulate transition into REST (40 steps of low-noise resting sample)
    for i in range(20, 60):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    pos_after = cursor_eng.cursor.pos

    # Cursor displacement across AIR -> REST transition must be approximately zero
    assert posture_tracker.state == PostureState.REST
    assert pos_after[0] == pytest.approx(pos_before[0], abs=1e-4)
    assert pos_after[1] == pytest.approx(pos_before[1], abs=1e-4)


def test_2_rest_to_air_no_jump():
    config, orient_est, posture_tracker, cursor_eng = setup_pipeline()

    # Start in REST state
    for i in range(40):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    assert posture_tracker.state == PostureState.REST
    pos_before_lift = cursor_eng.cursor.pos

    # Simulate lift-off rotation gesture during TRANSITION (15 steps)
    for i in range(40, 55):
        ts = i * config.dt
        # Posture change rotation (hand lifting up off desk)
        sample = IMUSample(timestamp=ts, accel=np.array([1.5, 1.5, 9.81]), gyro=np.array([0.8, 0.8, 0.0]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    # Now hand is in AIR state and stationary
    for i in range(55, 60):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    pos_after_lift = cursor_eng.cursor.pos

    # Position at completion of TRANSITION into AIR must not jump from the posture change itself
    assert posture_tracker.state == PostureState.AIR
    assert pos_after_lift[0] == pytest.approx(pos_before_lift[0], abs=1e-4)
    assert pos_after_lift[1] == pytest.approx(pos_before_lift[1], abs=1e-4)


def test_3_air_rest_air_continuity():
    config, orient_est, posture_tracker, cursor_eng = setup_pipeline()

    trajectory = []

    # Phase A: AIR movement (50 steps)
    for i in range(50):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.0, 0.1]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cur = cursor_eng.update(sample.gyro, posture_info=posture_info)
        trajectory.append(cur.pos)

    # Phase B: REST (40 steps)
    for i in range(50, 90):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cur = cursor_eng.update(sample.gyro, posture_info=posture_info)
        trajectory.append(cur.pos)

    # Phase C: Back to AIR (50 steps)
    for i in range(90, 140):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.0, 0.1]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cur = cursor_eng.update(sample.gyro, posture_info=posture_info)
        trajectory.append(cur.pos)

    # Verify no step discontinuities (step delta always bounded)
    trajectory_arr = np.array(trajectory)
    step_diffs = np.linalg.norm(np.diff(trajectory_arr, axis=0), axis=1)

    # Max step displacement should be smooth (< 10px per frame)
    assert np.max(step_diffs) < 10.0


def test_4_rest_noise_drift_suppression():
    config, orient_est, posture_tracker, cursor_eng = setup_pipeline()

    # Enter REST
    for i in range(40):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    pos_start_rest = cursor_eng.cursor.pos

    # Inject 200 frames (2.0s) of small sensor noise & gyro drift below intentional threshold
    np.random.seed(42)
    for i in range(40, 240):
        ts = i * config.dt
        noisy_gyro = np.random.normal(0.0, 0.005, size=3) + 0.002  # Slight noise and drift
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=noisy_gyro)
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    pos_end_rest = cursor_eng.cursor.pos
    drift_dist = float(np.hypot(pos_end_rest[0] - pos_start_rest[0], pos_end_rest[1] - pos_start_rest[1]))

    # Noise & drift while resting must be strictly suppressed (< 0.1 px total drift)
    assert drift_dist < 0.1


def test_5_intentional_movement_from_rest():
    config, orient_est, posture_tracker, cursor_eng = setup_pipeline()

    # Enter REST at screen center (960, 540)
    for i in range(40):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    assert posture_tracker.state == PostureState.REST
    start_x, start_y = cursor_eng.cursor.pos

    # Apply intentional rest movement (yaw gz = 0.2 rad/s right for 10 frames)
    for i in range(40, 50):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.0, 0.2]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        if i >= 42:
            assert posture_info.intentional_rest_motion is True
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    end_x, end_y = cursor_eng.cursor.pos

    # Cursor should have moved right (+X) smoothly
    assert end_x > start_x + 5.0
    assert end_y == pytest.approx(start_y, abs=1e-3)


def test_6_first_movement_after_reanchor():
    config, orient_est, posture_tracker, cursor_eng = setup_pipeline()

    # Enter REST and re-anchor
    for i in range(40):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    pos_rest = cursor_eng.cursor.pos

    # Apply intentional movement across persistence window (3 frames)
    cur = cursor_eng.cursor
    for i in range(40, 43):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.0, 0.1]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cur = cursor_eng.update(sample.gyro, posture_info=posture_info)

    single_step_dist = float(np.hypot(cur.x - pos_rest[0], cur.y - pos_rest[1]))

    # First movement step should be proportional and free of artificial spikes (< 3.0 px)
    assert single_step_dist > 0.0
    assert single_step_dist < 3.0


def test_7_repeated_transitions_no_drift():
    config, orient_est, posture_tracker, cursor_eng = setup_pipeline()

    initial_pos = cursor_eng.cursor.pos

    # Repeatedly cycle AIR -> REST -> AIR -> REST -> AIR with no intentional movement
    for cycle in range(5):
        # AIR phase (low noise)
        for i in range(15):
            ts = (cycle * 50 + i) * config.dt
            sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3))
            orient, _ = orient_est.update(sample)
            posture_info = posture_tracker.update(sample, orient, sample.gyro)
            cursor_eng.update(sample.gyro, posture_info=posture_info)

        # REST phase
        for i in range(15, 50):
            ts = (cycle * 50 + i) * config.dt
            sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3))
            orient, _ = orient_est.update(sample)
            posture_info = posture_tracker.update(sample, orient, sample.gyro)
            cursor_eng.update(sample.gyro, posture_info=posture_info)

    final_pos = cursor_eng.cursor.pos
    accumulated_error = float(np.hypot(final_pos[0] - initial_pos[0], final_pos[1] - initial_pos[1]))

    # Zero accumulated reference error across 5 full transition cycles
    assert accumulated_error < 0.01


def test_8_rest_intentional_movement_to_air():
    config, orient_est, posture_tracker, cursor_eng = setup_pipeline()

    # 1. Enter REST at screen center (960, 540)
    for i in range(40):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    assert posture_tracker.state == PostureState.REST

    # 2. Intentional movement while resting -> moves cursor from (720, 380)
    cursor_eng.set_position(720.0, 380.0)
    for i in range(40, 55):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.1, 0.2]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    updated_rest_pos = cursor_eng.cursor.pos
    assert updated_rest_pos[0] > 720.0
    assert updated_rest_pos[1] < 380.0

    # 3. Transition back to AIR (hand lift gesture during TRANSITION: 15 steps)
    for i in range(55, 70):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([1.5, 1.5, 9.81]), gyro=np.array([0.8, 0.8, 0.0]))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    # 4. Now hand is in AIR state and stationary
    for i in range(70, 75):
        ts = i * config.dt
        sample = IMUSample(timestamp=ts, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3))
        orient, _ = orient_est.update(sample)
        posture_info = posture_tracker.update(sample, orient, sample.gyro)
        cursor_eng.update(sample.gyro, posture_info=posture_info)

    air_start_pos = cursor_eng.cursor.pos

    # Cursor in AIR must start directly from the updated REST position (no snapback to 720, 380)
    assert posture_tracker.state == PostureState.AIR
    assert air_start_pos[0] == pytest.approx(updated_rest_pos[0], abs=1e-4)
    assert air_start_pos[1] == pytest.approx(updated_rest_pos[1], abs=1e-4)
