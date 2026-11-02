"""
Unit and integration tests for Human-in-the-Loop Test Mode infrastructure.
Tests KeyboardIMUAdapter, HumanTestLogger, and HumanTestManager.
"""

import os
import tempfile
import numpy as np
import pytest

from wearable_mouse.config import Config
from wearable_mouse.simulation.imu_simulator import IMUSimulator
from wearable_mouse.simulation.keyboard_imu_adapter import KeyboardIMUAdapter
from wearable_mouse.telemetry_logging.human_test_logger import HumanTestLogger, HumanTestTrialRecord
from wearable_mouse.interaction.human_test_manager import HumanTestManager, TargetLayout
from wearable_mouse.cursor.target_assistance import Target


def test_keyboard_imu_adapter_initial_state():
    sim = IMUSimulator()
    adapter = KeyboardIMUAdapter(sim)
    assert len(adapter.active_keys) == 0
    assert np.allclose(adapter.current_gyro, 0.0)


def test_keyboard_imu_adapter_keypresses():
    sim = IMUSimulator()
    adapter = KeyboardIMUAdapter(sim)

    # Press 'd' (move right -> wz > 0)
    adapter.key_down('d')
    assert 'd' in adapter.active_keys

    sample = adapter.update_step(dt=0.01)
    assert sample.gyro[2] != 0.0  # Gyro z affected
    assert adapter.current_gyro[2] > 0.0

    # Release 'd' and press 'a'
    adapter.key_up('d')
    adapter.key_down('a')
    assert 'd' not in adapter.active_keys
    assert 'a' in adapter.active_keys

    # Step adapter multiple times to let smoothing ramp wz negative
    for _ in range(10):
        adapter.update_step(dt=0.01)
    assert adapter.current_gyro[2] < 0.0

    # Clear keys
    adapter.clear_keys()
    assert len(adapter.active_keys) == 0
    assert np.allclose(adapter.current_gyro, 0.0)


def test_human_test_logger_buffering_and_ratings():
    logger = HumanTestLogger()
    record1 = HumanTestTrialRecord(
        trial_index=1,
        block_index=1,
        condition_code="B",
        condition_blinded_label="Condition 1",
        target_id="t_main",
        target_size=120,
        target_layout="ISOLATED",
        outcome="SUCCESS",
        acquisition_time_sec=1.5,
        cursor_path_length=250.0,
        corrective_movements=1,
        num_clicks=1,
        uncertainty_level=15.0,
        noise_level=0.01,
        tremor_level=0.02
    )

    record2 = HumanTestTrialRecord(
        trial_index=2,
        block_index=1,
        condition_code="B",
        condition_blinded_label="Condition 1",
        target_id="t_main",
        target_size=80,
        target_layout="ISOLATED",
        outcome="SUCCESS",
        acquisition_time_sec=1.2,
        cursor_path_length=200.0,
        corrective_movements=0,
        num_clicks=1,
        uncertainty_level=12.0,
        noise_level=0.01,
        tremor_level=0.02
    )

    logger.log_trial(record1)
    logger.log_trial(record2)

    assert len(logger.records) == 2
    assert logger.records[0].difficulty_rating is None

    # Attach block ratings
    logger.attach_block_ratings(block_index=1, difficulty=2, stability=4, confidence=5, fatigue=1)

    assert logger.records[0].difficulty_rating == 2
    assert logger.records[0].stability_rating == 4
    assert logger.records[0].confidence_rating == 5
    assert logger.records[0].fatigue_rating == 1
    assert logger.records[1].difficulty_rating == 2

    # Export to temp CSV file
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_output.csv")
        logger.export_csv(csv_path)

        assert os.path.exists(csv_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # Header + 2 rows = 3 lines
            assert len(lines) == 3
            assert "trial_index" in lines[0]
            assert "Condition 1" in lines[1]

    # Clear logger
    logger.clear()
    assert len(logger.records) == 0


def test_human_test_manager_layout_generation_and_trials():
    mgr = HumanTestManager()

    # Check default initial condition and target
    assert mgr.current_condition.code == "B"
    assert mgr.current_target_size == 120
    assert mgr.current_layout == TargetLayout.ISOLATED
    assert len(mgr.targets) == 1
    assert mgr.active_target.id == "t_main"

    # Reset metrics for trial
    mgr.reset_trial_metrics(cursor_pos=(960.0, 540.0), timestamp=0.0)

    # Simulate frame movement
    mgr.update_frame((970.0, 540.0))  # Moved 10px right
    mgr.update_frame((960.0, 540.0))  # Moved 10px left (reversal)
    assert mgr.trial_path_length == pytest.approx(20.0, abs=0.5)
    assert mgr.corrective_movements >= 1

    # Evaluate click on active target -> SUCCESS
    outcome, is_block_done = mgr.evaluate_click(
        cursor_pos=(960.0, 540.0),
        timestamp=1.0,
        selected_target=mgr.active_target,
        uncertainty_radius=10.0,
        noise_level=0.01,
        tremor_level=0.01
    )

    assert outcome == "SUCCESS"
    assert is_block_done is False
    assert mgr.total_trials_completed == 1
    assert mgr.trial_in_block == 1


def test_human_test_manager_block_completion_and_advance():
    mgr = HumanTestManager()
    mgr.trials_per_block = 3  # Shorten for testing

    for i in range(3):
        mgr.reset_trial_metrics((960.0, 540.0), timestamp=float(i))
        outcome, is_block_done = mgr.evaluate_click(
            cursor_pos=(960.0, 540.0),
            timestamp=float(i) + 0.5,
            selected_target=mgr.active_target,
            uncertainty_radius=10.0,
            noise_level=0.01,
            tremor_level=0.01
        )

    # 3rd click completes the block
    assert is_block_done is True
    assert mgr.trial_in_block == 3

    # Advance block
    mgr.advance_block()
    assert mgr.block_index == 2
    assert mgr.trial_in_block == 0
    # Condition cycled from B to C
    assert mgr.current_condition.code == "C"
