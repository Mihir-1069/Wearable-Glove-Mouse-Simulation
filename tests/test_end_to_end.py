"""
End-to-End Simulation Pipeline Integration Test.
"""

import numpy as np
import pytest
from wearable_mouse.config import Config
from wearable_mouse.simulation.imu_simulator import IMUSimulator
from wearable_mouse.simulation.scenarios import MotionScenario, ScenarioGenerator
from wearable_mouse.simulation.gesture_simulator import GestureSimulator
from wearable_mouse.processing.calibration import Calibrator, CalibrationState
from wearable_mouse.processing.filters import FilterPipeline, LowPassFilter, DeadZoneFilter, ExponentialSmoothingFilter
from wearable_mouse.processing.motion_detector import MotionDetector
from wearable_mouse.processing.uncertainty import UncertaintyEstimator
from wearable_mouse.cursor.cursor_engine import CursorEngine
from wearable_mouse.cursor.target_assistance import TargetAssistanceEngine, Target, AcquisitionMode
from wearable_mouse.gestures.gesture_engine import GestureEngine
from wearable_mouse.gestures.gesture_map import GestureType
from wearable_mouse.interaction.edge_feedback import EdgeDetector
from wearable_mouse.interaction.haptic_events import HapticManager, HapticEvent


def test_full_pipeline_simulation_run():
    config = Config(random_seed=42)

    sim = IMUSimulator(config)
    scenario_gen = ScenarioGenerator(sim)
    gesture_sim = GestureSimulator(config)
    calibrator = Calibrator(config)
    filters = FilterPipeline([
        LowPassFilter(config.lowpass_cutoff, config.sample_rate),
        DeadZoneFilter(config.deadzone_threshold),
        ExponentialSmoothingFilter(config.exp_smoothing_alpha)
    ])
    motion_detector = MotionDetector(config)
    uncertainty_est = UncertaintyEstimator(config)
    cursor_engine = CursorEngine(config)
    target_engine = TargetAssistanceEngine(config)
    gesture_engine = GestureEngine(config)
    edge_detector = EdgeDetector(config)
    haptic_manager = HapticManager()

    # Register target
    t1 = Target(id="t1", x=900, y=500, width=120, height=80, label="Center Target")
    target_engine.set_targets([t1])

    received_haptics = []
    haptic_manager.add_listener(lambda evt: received_haptics.append(evt))

    # Run 200 frame simulation steps (~2 seconds)
    for step in range(1, 201):
        sample = scenario_gen.run_scenario_step(MotionScenario.TARGET_CLICK_SEQUENCE, step)

        calib_state = calibrator.update(sample)
        calib_sample = calibrator.apply(sample)

        filt_gyro = filters.process(calib_sample.gyro)
        motion_state = motion_detector.update(filt_gyro)
        uncertainty = uncertainty_est.update(calib_sample.gyro, filt_gyro, motion_state)

        cursor = cursor_engine.update(filt_gyro)
        assist = target_engine.process(cursor, uncertainty, mode=AcquisitionMode.D)

        new_edges = edge_detector.update(cursor)
        for edge in new_edges:
            haptic_manager.trigger_edge_bump(edge.value, sample.timestamp)

    # Trigger a simulated gesture click
    evt = gesture_engine.process_gesture(GestureType.LEFT_CLICK, sim.current_time)
    assert evt is not None
    haptic_manager.trigger_click(sim.current_time)

    # Assertions
    assert sim.current_time > 0.0
    assert len(received_haptics) > 0
    assert cursor_engine.cursor.x >= 0.0 and cursor_engine.cursor.x <= config.screen_width
    assert cursor_engine.cursor.y >= 0.0 and cursor_engine.cursor.y <= config.screen_height
