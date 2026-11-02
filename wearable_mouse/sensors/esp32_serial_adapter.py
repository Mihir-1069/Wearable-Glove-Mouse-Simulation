"""
ESP32 Serial IMU Sensor Adapter module.
Parses structured serial telemetry streams from ESP32 + MPU6050 hardware or Wokwi HITL simulation,
converting raw CSV packets into the standard SensorData interface.
"""

import sys
import time
import logging
from typing import Optional, List, Tuple
import numpy as np

from wearable_mouse.sensors.imu_data import IMUSample, SensorData
from wearable_mouse.sensors.sensor_models import AbstractSensor

logger = logging.getLogger(__name__)

try:
    import serial
except ImportError:
    serial = None


class ESP32SerialAdapter(AbstractSensor):
    """
    Serial Port Hardware Adapter for ESP32 + MPU6050 IMU telemetry.
    Implements AbstractSensor to seamlessly feed the existing calibration, filtering,
    uncertainty, and cursor pipelines.
    """

    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 115200,
        timeout: float = 1.0,
        mock_lines: Optional[List[str]] = None
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        self.connected = False
        self.mock_lines = mock_lines
        self.mock_idx = 0

        # Telemetry State & Diagnostics
        self.last_sample_time: float = 0.0
        self.last_raw_timestamp_ms: Optional[int] = None
        self.dropped_packet_count: int = 0
        self.malformed_packet_count: int = 0
        self.total_packets_received: int = 0
        self.last_valid_sample = IMUSample(timestamp=0.0)
        self.status_messages: List[str] = []

    def connect(self) -> bool:
        """Opens serial communication port."""
        if self.mock_lines is not None:
            self.connected = True
            logger.info("ESP32SerialAdapter connected in mock line stream mode.")
            return True

        if serial is None:
            logger.error("pyserial package is not installed. Install via `pip install pyserial`.")
            return False

        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            # Give ESP32 time to reset after DTR toggle
            time.sleep(1.0)
            self.serial_conn.reset_input_buffer()
            self.connected = True
            logger.info(f"ESP32SerialAdapter connected to {self.port} at {self.baudrate} baud.")
            return True
        except Exception as e:
            logger.error(f"Failed to open serial port {self.port}: {e}")
            self.connected = False
            return False

    def disconnect(self) -> None:
        """Closes serial connection."""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
            except Exception as e:
                logger.warning(f"Error closing serial port: {e}")
        self.connected = False

    def parse_packet_line(self, raw_line: str) -> Optional[IMUSample]:
        """
        Parses a single incoming string line.

        Protocol Format:
            IMU,timestamp_ms,ax,ay,az,gx,gy,gz
        Diagnostics:
            STATUS,... or ERROR,...
        """
        line = raw_line.strip()
        if not line:
            return None

        # Check for status/diagnostic messages
        if line.startswith("STATUS,") or line.startswith("ERROR,"):
            self.status_messages.append(line)
            logger.info(f"ESP32 Serial Message: {line}")
            return None

        # Parse IMU packet
        tokens = line.split(",")
        if len(tokens) != 8 or tokens[0] != "IMU":
            self.malformed_packet_count += 1
            logger.warning(f"Malformed IMU packet received ({len(tokens)} tokens): '{line}'")
            return None

        try:
            ts_ms = int(tokens[1])
            ax = float(tokens[2])
            ay = float(tokens[3])
            az = float(tokens[4])
            gx = float(tokens[5])
            gy = float(tokens[6])
            gz = float(tokens[7])
        except ValueError as e:
            self.malformed_packet_count += 1
            logger.warning(f"Numeric conversion failure in packet '{line}': {e}")
            return None

        # Detect timestamp gap / dropped frames (assuming 100 Hz = 10 ms interval)
        if self.last_raw_timestamp_ms is not None:
            delta_ms = ts_ms - self.last_raw_timestamp_ms
            if delta_ms > 25:  # Missed 2+ frames
                dropped = (delta_ms // 10) - 1
                self.dropped_packet_count += max(1, dropped)
                logger.debug(f"Detected dropped frame gap: delta={delta_ms}ms ({dropped} dropped)")

        self.last_raw_timestamp_ms = ts_ms
        self.total_packets_received += 1

        ts_sec = ts_ms / 1000.0
        accel = np.array([ax, ay, az], dtype=np.float64)
        gyro = np.array([gx, gy, gz], dtype=np.float64)

        sample = IMUSample(timestamp=ts_sec, accel=accel, gyro=gyro)
        self.last_valid_sample = sample
        return sample

    def read_sample(self) -> SensorData:
        """
        Reads the next valid IMU telemetry frame from serial port or mock stream.
        Fallback to last valid sample or zero sample if frame reading times out.
        """
        if not self.connected:
            self.connect()

        # Mock stream reading (for testing / simulated hardware stream)
        if self.mock_lines is not None:
            while self.mock_idx < len(self.mock_lines):
                line = self.mock_lines[self.mock_idx]
                self.mock_idx += 1
                parsed = self.parse_packet_line(line)
                if parsed is not None:
                    return SensorData(imu=parsed)

            # Reached end of mock stream, return last sample
            return SensorData(imu=self.last_valid_sample)

        # Real serial reading
        if self.serial_conn and self.serial_conn.is_open:
            try:
                raw_bytes = self.serial_conn.readline()
                line = raw_bytes.decode("utf-8", errors="ignore")
                parsed = self.parse_packet_line(line)
                if parsed is not None:
                    return SensorData(imu=parsed)
            except Exception as e:
                logger.error(f"Serial read error on port {self.port}: {e}")
                self.disconnect()

        return SensorData(imu=self.last_valid_sample)

    def reset(self) -> None:
        """Reset serial buffers and timestamp state."""
        self.last_raw_timestamp_ms = None
        self.dropped_packet_count = 0
        self.malformed_packet_count = 0
        self.total_packets_received = 0
        self.mock_idx = 0
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.reset_input_buffer()
            except Exception:
                pass
