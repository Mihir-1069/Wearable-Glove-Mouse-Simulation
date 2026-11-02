"""
Abstract sensor interface and concrete hardware/simulation implementations.
Ensures zero-code-change transition from software simulator to ESP32 serial stream.
"""

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
from wearable_mouse.sensors.imu_data import IMUSample, SensorData

if TYPE_CHECKING:
    from wearable_mouse.simulation.imu_simulator import IMUSimulator


class AbstractSensor(ABC):
    """Abstract base interface for motion sensors."""

    @abstractmethod
    def read_sample(self) -> SensorData:
        """Fetch the latest sensor telemetry frame."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset sensor state or simulator timeline."""
        pass


class SimulatedIMUSensor(AbstractSensor):
    """
    Adapter wrapping IMUSimulator into the standard AbstractSensor interface.
    """

    def __init__(self, simulator: 'IMUSimulator'):
        self.simulator = simulator

    def read_sample(self) -> SensorData:
        sample = self.simulator.step()
        return SensorData(imu=sample)

    def reset(self) -> None:
        self.simulator.reset()


class HardwareIMUSensor(AbstractSensor):
    """
    Hardware adapter connecting to ESP32 + MPU6050 via serial/Wokwi HITL simulation.
    """

    def __init__(self, port: str = "COM3", baudrate: int = 115200):
        from wearable_mouse.sensors.esp32_serial_adapter import ESP32SerialAdapter
        self.adapter = ESP32SerialAdapter(port=port, baudrate=baudrate)
        self.port = port
        self.baudrate = baudrate

    def connect(self) -> bool:
        return self.adapter.connect()

    def read_sample(self) -> SensorData:
        return self.adapter.read_sample()

    def reset(self) -> None:
        self.adapter.reset()

