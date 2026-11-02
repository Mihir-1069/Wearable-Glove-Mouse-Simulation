"""
Data Logger module.
Records frame-by-frame telemetry and exports CSV session logs for offline evaluation.
"""

import csv
from dataclasses import dataclass, asdict
from typing import List, Optional
import numpy as np


@dataclass
class LogFrame:
    timestamp: float
    raw_gyro_z: float
    raw_gyro_y: float
    filt_gyro_z: float
    filt_gyro_y: float
    cursor_x: float
    cursor_y: float
    cursor_speed: float
    motion_type: str
    is_stationary: bool
    uncertainty_radius: float
    confidence_score: float
    selected_target: str
    is_ambiguous: bool
    gesture_event: str
    haptic_event: str


class DataLogger:
    """
    Buffers telemetry frames and writes CSV export reports.
    """

    def __init__(self):
        self.frames: List[LogFrame] = []

    def log_frame(self, frame: LogFrame) -> None:
        self.frames.append(frame)

    def export_csv(self, filepath: str) -> None:
        """Export buffered frames to CSV file."""
        if not self.frames:
            return

        fieldnames = list(asdict(self.frames[0]).keys())
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for frame in self.frames:
                writer.writerow(asdict(frame))

    def clear(self) -> None:
        self.frames.clear()
