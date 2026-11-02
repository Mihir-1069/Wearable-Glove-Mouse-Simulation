"""
Modular Signal Filtering Pipeline.
Provides interchangeable low-pass, dead-zone, moving average, and exponential smoothing filters.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from wearable_mouse.config import Config


class BaseFilter(ABC):
    """Abstract interface for motion signal filters."""

    @abstractmethod
    def process(self, vector: np.ndarray) -> np.ndarray:
        """Filter input 3D vector and return processed 3D vector."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal filter state."""
        pass


class LowPassFilter(BaseFilter):
    """
    1st-order IIR Low-Pass Filter based on RC cutoff frequency.
    Attenuates high-frequency sensor noise and rapid jitter.
    """

    def __init__(self, cutoff_freq: float = 12.0, sample_rate: float = 100.0):
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        dt = 1.0 / sample_rate
        rc = 1.0 / (2.0 * np.pi * cutoff_freq)
        self.alpha = dt / (rc + dt)
        self.state: Optional[np.ndarray] = None

    def process(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64)
        if self.state is None:
            self.state = vector.copy()
            return self.state

        self.state = self.alpha * vector + (1.0 - self.alpha) * self.state
        return self.state.copy()

    def reset(self) -> None:
        self.state = None


class DeadZoneFilter(BaseFilter):
    """
    Applies non-linear dead-zone thresholding.
    Zeroes out angular rates below threshold to prevent involuntary cursor drift during micro-movements.
    """

    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold

    def process(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64)
        norm = np.linalg.norm(vector)
        if norm < self.threshold:
            return np.zeros_like(vector)
        # Soft deadzone transition: scale remaining vector smoothly
        scale = (norm - self.threshold) / norm
        return vector * scale

    def reset(self) -> None:
        pass


class MovingAverageFilter(BaseFilter):
    """
    Sliding window simple moving average filter.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.buffer: List[np.ndarray] = []

    def process(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64)
        self.buffer.append(vector.copy())
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        return np.mean(self.buffer, axis=0)

    def reset(self) -> None:
        self.buffer.clear()


class ExponentialSmoothingFilter(BaseFilter):
    """
    Exponential smoothing filter (Single Exponential Smoothing).
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.state: Optional[np.ndarray] = None

    def process(self, vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float64)
        if self.state is None:
            self.state = vector.copy()
            return self.state

        self.state = self.alpha * vector + (1.0 - self.alpha) * self.state
        return self.state.copy()

    def reset(self) -> None:
        self.state = None


class FilterPipeline:
    """
    Chains multiple BaseFilter instances into a configurable sequential pipeline.
    """

    def __init__(self, filters: Optional[List[BaseFilter]] = None):
        self.filters: List[BaseFilter] = filters or []

    def add_filter(self, filter_instance: BaseFilter) -> 'FilterPipeline':
        self.filters.append(filter_instance)
        return self

    def process(self, vector: np.ndarray) -> np.ndarray:
        output = np.asarray(vector, dtype=np.float64)
        for f in self.filters:
            output = f.process(output)
        return output

    def reset(self) -> None:
        for f in self.filters:
            f.reset()
