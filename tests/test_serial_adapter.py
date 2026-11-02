"""
Unit and integration tests for ESP32 Serial Adapter, SensorDataRecorder, and SensorDataReplayer.
"""

import os
import tempfile
import numpy as np
import pytest

from wearable_mouse.sensors.imu_data import IMUSample, SensorData
from wearable_mouse.sensors.esp32_serial_adapter import ESP32SerialAdapter
from wearable_mouse.sensors.recorder_replayer import SensorDataRecorder, SensorDataReplayer
from wearable_mouse.sensors.sensor_models import SimulatedIMUSensor
from wearable_mouse.simulation.imu_simulator import IMUSimulator


def test_esp32_serial_adapter_valid_packet_parsing():
    adapter = ESP32SerialAdapter()
    raw_line = "IMU,1000,0.02,-0.01,9.81,0.00100,-0.00400,0.01200\n"

    sample = adapter.parse_packet_line(raw_line)
    assert sample is not None
    assert sample.timestamp == pytest.approx(1.0)
    assert sample.accel[0] == pytest.approx(0.02)
    assert sample.accel[1] == pytest.approx(-0.01)
    assert sample.accel[2] == pytest.approx(9.81)
    assert sample.gyro[0] == pytest.approx(0.001)
    assert sample.gyro[1] == pytest.approx(-0.004)
    assert sample.gyro[2] == pytest.approx(0.012)


def test_esp32_serial_adapter_malformed_packets():
    adapter = ESP32SerialAdapter()

    # Short packet
    assert adapter.parse_packet_line("IMU,1000,0.02") is None
    # Non-numeric field
    assert adapter.parse_packet_line("IMU,1000,abc,0.01,9.81,0.0,0.0,0.0") is None
    # Wrong prefix
    assert adapter.parse_packet_line("OTHER,1000,0.02,0.01,9.81,0.0,0.0,0.0") is None

    assert adapter.malformed_packet_count == 3


def test_esp32_serial_adapter_diagnostic_status_messages():
    adapter = ESP32SerialAdapter()

    msg1 = adapter.parse_packet_line("STATUS,MPU6050_OK\n")
    assert msg1 is None
    assert len(adapter.status_messages) == 1
    assert adapter.status_messages[0] == "STATUS,MPU6050_OK"

    msg2 = adapter.parse_packet_line("ERROR,MPU6050_NOT_FOUND\n")
    assert msg2 is None
    assert len(adapter.status_messages) == 2
    assert adapter.status_messages[1] == "ERROR,MPU6050_NOT_FOUND"


def test_esp32_serial_adapter_dropped_packet_detection():
    adapter = ESP32SerialAdapter()

    p1 = adapter.parse_packet_line("IMU,1000,0.0,0.0,9.81,0.0,0.0,0.0\n")
    assert p1 is not None
    assert adapter.dropped_packet_count == 0

    # Next frame jumps 50ms (missing ~4 frames)
    p2 = adapter.parse_packet_line("IMU,1050,0.0,0.0,9.81,0.0,0.0,0.0\n")
    assert p2 is not None
    assert adapter.dropped_packet_count >= 1


def test_esp32_serial_adapter_mock_stream_reading():
    mock_lines = [
        "STATUS,MPU6050_OK\n",
        "IMU,100,0.0,0.0,9.81,0.0,0.1,0.2\n",
        "IMU,110,0.0,0.0,9.81,0.0,0.1,0.2\n"
    ]
    adapter = ESP32SerialAdapter(mock_lines=mock_lines)
    adapter.connect()

    data1 = adapter.read_sample()
    assert data1.imu.timestamp == pytest.approx(0.1)
    assert data1.imu.gyro[1] == pytest.approx(0.1)

    data2 = adapter.read_sample()
    assert data2.imu.timestamp == pytest.approx(0.11)


def test_sensor_data_recorder_and_replayer():
    sim = IMUSimulator()
    sim_sensor = SimulatedIMUSensor(sim)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_recording.csv")
        recorder = SensorDataRecorder(sim_sensor, csv_path)

        # Record 10 samples
        recorded_samples = []
        for _ in range(10):
            data = recorder.read_sample()
            recorded_samples.append(data.imu)

        recorder.close()
        assert os.path.exists(csv_path)
        assert recorder.records_count == 10

        # Replay recorded CSV
        replayer = SensorDataReplayer(csv_path)
        assert len(replayer.samples) == 10

        for idx in range(10):
            replay_data = replayer.read_sample()
            orig = recorded_samples[idx]
            rep = replay_data.imu

            assert rep.timestamp == pytest.approx(orig.timestamp, abs=1e-3)
            assert np.allclose(rep.accel, orig.accel, atol=1e-3)
            assert np.allclose(rep.gyro, orig.gyro, atol=1e-4)
