"""
Hardware-in-the-Loop (HITL) Source Comparison Tests.
Verifies that telemetry from simulated source and serialized/replayed ESP32 hardware streams
pass through the exact same calibration, filtering, motion classification, uncertainty, and cursor engines.
"""

import os
import tempfile
import numpy as np
import pytest

from wearable_mouse.config import Config
from wearable_mouse.simulation.imu_simulator import IMUSimulator
from wearable_mouse.sensors.sensor_models import SimulatedIMUSensor
from wearable_mouse.sensors.esp32_serial_adapter import ESP32SerialAdapter
from wearable_mouse.sensors.recorder_replayer import SensorDataRecorder, SensorDataReplayer
from wearable_mouse.processing.calibration import Calibrator
from wearable_mouse.processing.filters import FilterPipeline, LowPassFilter, DeadZoneFilter, ExponentialSmoothingFilter
from wearable_mouse.processing.uncertainty import UncertaintyEstimator
from wearable_mouse.cursor.cursor_engine import CursorEngine


def test_source_compatibility_pipeline():
    config = Config()

    # 1. Generate baseline motion via Simulation
    sim = IMUSimulator(config)
    sim_sensor = SimulatedIMUSensor(sim)

    # Setup processing engines for Simulation pipeline
    calib_sim = Calibrator(config)
    filt_sim = FilterPipeline([
        LowPassFilter(config.lowpass_cutoff, config.sample_rate),
        DeadZoneFilter(config.deadzone_threshold),
        ExponentialSmoothingFilter(config.exp_smoothing_alpha)
    ])
    cursor_sim = CursorEngine(config)

    # 2. Record raw telemetry to CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "pipeline_recorded.csv")
        recorder = SensorDataRecorder(sim_sensor, csv_path)

        sim_cursor_positions = []
        for _ in range(50):
            data = recorder.read_sample()
            sample = data.imu
            c_state = calib_sim.update(sample)
            c_sample = calib_sim.apply(sample)
            f_gyro = filt_sim.process(c_sample.gyro)
            cur = cursor_sim.update(f_gyro, dt=config.dt)
            sim_cursor_positions.append((cur.x, cur.y))

        recorder.close()

        # 3. Replay recorded stream through a fresh identical processing pipeline
        replayer = SensorDataReplayer(csv_path)
        calib_rep = Calibrator(config)
        filt_rep = FilterPipeline([
            LowPassFilter(config.lowpass_cutoff, config.sample_rate),
            DeadZoneFilter(config.deadzone_threshold),
            ExponentialSmoothingFilter(config.exp_smoothing_alpha)
        ])
        cursor_rep = CursorEngine(config)

        rep_cursor_positions = []
        for _ in range(50):
            data = replayer.read_sample()
            sample = data.imu
            c_state = calib_rep.update(sample)
            c_sample = calib_rep.apply(sample)
            f_gyro = filt_rep.process(c_sample.gyro)
            cur = cursor_rep.update(f_gyro, dt=config.dt)
            rep_cursor_positions.append((cur.x, cur.y))

        # 4. Verify pipeline output equivalence (Replayed frames match simulation trajectory)
        assert len(sim_cursor_positions) == len(rep_cursor_positions)
        for p_sim, p_rep in zip(sim_cursor_positions, rep_cursor_positions):
            assert p_sim[0] == pytest.approx(p_rep[0], abs=1e-2)
            assert p_sim[1] == pytest.approx(p_rep[1], abs=1e-2)


def test_esp32_serial_mock_pipeline_run():
    config = Config()

    # Create mock ESP32 serial packet lines (100 Hz, 10ms intervals)
    lines = []
    lines.append("STATUS,MPU6050_OK\n")
    for i in range(30):
        ts_ms = 1000 + i * 10
        # Simulated rotation
        gx = 0.001
        gy = 0.05 * np.sin(i * 0.2)
        gz = 0.05 * np.cos(i * 0.2)
        lines.append(f"IMU,{ts_ms},0.0,0.0,9.81,{gx:.5f},{gy:.5f},{gz:.5f}\n")

    adapter = ESP32SerialAdapter(mock_lines=lines)
    adapter.connect()

    calibrator = Calibrator(config)
    uncertainty_est = UncertaintyEstimator(config)
    cursor_eng = CursorEngine(config)

    from wearable_mouse.processing.motion_detector import MotionDetector
    motion_det = MotionDetector(config)

    for i in range(30):
        data = adapter.read_sample()
        sample = data.imu

        c_state = calibrator.update(sample)
        c_sample = calibrator.apply(sample)
        m_state = motion_det.update(c_sample.gyro)
        u_state = uncertainty_est.update(c_sample.gyro, c_sample.gyro, m_state)
        cur = cursor_eng.update(c_sample.gyro, dt=config.dt)

        assert cur.x >= 0.0 and cur.y >= 0.0
        assert u_state.uncertainty_radius > 0.0

    assert adapter.total_packets_received == 30
    assert adapter.malformed_packet_count == 0
