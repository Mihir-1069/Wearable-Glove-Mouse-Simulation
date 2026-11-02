"""
Raw IMU Sensor Telemetry Recorder and Replayer module.
Enables recording live raw ESP32 serial sensor telemetry streams to CSV files
and replaying them frame-by-frame through the exact same processing pipeline.
"""

import csv
import time
import os
from typing import Optional, List
import numpy as np

from wearable_mouse.sensors.imu_data import IMUSample, SensorData
from wearable_mouse.sensors.sensor_models import AbstractSensor


class SensorDataRecorder(AbstractSensor):
    """
    Decorator wrapping any AbstractSensor source to record raw IMU frames to CSV.
    """

    def __init__(self, source_sensor: AbstractSensor, filepath: str):
        self.source_sensor = source_sensor
        self.filepath = filepath
        self.records_count = 0

        # Open CSV file and write header
        self.csv_file = open(self.filepath, mode="w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["timestamp", "ax", "ay", "az", "gx", "gy", "gz"])
        self.csv_file.flush()

    def read_sample(self) -> SensorData:
        """Fetches sample from underlying sensor, records frame to CSV, and returns data."""
        data = self.source_sensor.read_sample()
        imu = data.imu

        self.writer.writerow([
            f"{imu.timestamp:.4f}",
            f"{imu.accel[0]:.4f}",
            f"{imu.accel[1]:.4f}",
            f"{imu.accel[2]:.4f}",
            f"{imu.gyro[0]:.5f}",
            f"{imu.gyro[1]:.5f}",
            f"{imu.gyro[2]:.5f}"
        ])
        self.records_count += 1
        if self.records_count % 100 == 0:
            self.csv_file.flush()

        return data

    def reset(self) -> None:
        self.source_sensor.reset()

    def close(self) -> None:
        """Flushes and closes CSV file."""
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()

    def __del__(self):
        self.close()


class SensorDataReplayer(AbstractSensor):
    """
    Sensor source replaying raw recorded IMU CSV frames through the standard AbstractSensor interface.
    """

    def __init__(self, filepath: str, realtime_playback: bool = False):
        self.filepath = filepath
        self.realtime_playback = realtime_playback
        self.samples: List[IMUSample] = []
        self.current_idx = 0
        self.last_playback_time: Optional[float] = None
        self._load_csv()

    def _load_csv(self) -> None:
        """Loads CSV file into memory."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Recorded CSV file not found: {self.filepath}")

        self.samples.clear()
        with open(self.filepath, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = float(row["timestamp"])
                ax = float(row["ax"])
                ay = float(row["ay"])
                az = float(row["az"])
                gx = float(row["gx"])
                gy = float(row["gy"])
                gz = float(row["gz"])

                sample = IMUSample(
                    timestamp=ts,
                    accel=np.array([ax, ay, az]),
                    gyro=np.array([gx, gy, gz])
                )
                self.samples.append(sample)

        self.current_idx = 0

    def read_sample(self) -> SensorData:
        """
        Reads next recorded frame. Replay wraps around or holds last frame if finished.
        """
        if not self.samples:
            return SensorData(imu=IMUSample(timestamp=0.0))

        sample = self.samples[self.current_idx]

        # Optional real-time playback pacing
        if self.realtime_playback and self.current_idx > 0:
            prev_sample = self.samples[self.current_idx - 1]
            dt = max(0.0, sample.timestamp - prev_sample.timestamp)
            if self.last_playback_time is not None:
                elapsed = time.time() - self.last_playback_time
                if dt > elapsed:
                    time.sleep(dt - elapsed)

        self.last_playback_time = time.time()
        self.current_idx = (self.current_idx + 1) % len(self.samples)

        return SensorData(imu=sample)

    def reset(self) -> None:
        """Reset replay position to beginning."""
        self.current_idx = 0
        self.last_playback_time = None
