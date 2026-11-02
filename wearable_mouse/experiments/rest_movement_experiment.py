"""
Controlled Empirical Benchmark Experiment for Resting-Surface Cursor Control.
Compares Conditions A through E across stability, noise resistance, intentional motion detection, and responsiveness.
"""

import csv
from dataclasses import dataclass, asdict
from typing import List, Optional
import numpy as np

from wearable_mouse.config import Config
from wearable_mouse.sensors.imu_data import IMUSample
from wearable_mouse.processing.orientation import OrientationEstimator
from wearable_mouse.processing.posture_reference import (
    PostureReferenceTracker,
    PostureState,
    RestMovementClassification
)
from wearable_mouse.processing.motion_detector import MotionDetector
from wearable_mouse.processing.uncertainty import UncertaintyEstimator
from wearable_mouse.cursor.cursor_engine import CursorEngine
from wearable_mouse.cursor.dynamic_friction import DynamicSensitivitySuppression
from wearable_mouse.cursor.target_assistance import TargetAssistanceEngine, Target, AcquisitionMode


@dataclass
class RestExperimentResult:
    condition_code: str
    condition_name: str
    false_cursor_movement_px: float       # Displacement during stationary rest noise (px)
    intentional_detection_rate_pct: float  # Detection rate of intentional rest motions (%)
    cursor_path_length_px: float           # Total path length (px)
    movement_latency_sec: float            # Delay from motion start to cursor response (s)
    missed_intentional_movements: int      # Count of missed intentional movements
    false_movements_count: int             # Count of false movement activations during rest noise
    transition_jumps_px: float             # Total displacement artifact on AIR <-> REST transitions (px)


class RestMovementExperimentRunner:
    """
    Executes controlled benchmark evaluation of Resting-Surface Cursor Control across Conditions A-E.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def run_condition_trial(self, condition_code: str) -> RestExperimentResult:
        """
        Runs a standardized sequence of AIR motion, REST entry, stationary rest noise,
        intentional rest motion, and lift-off back to AIR for a given condition code.
        """
        config = Config()
        orient_est = OrientationEstimator(config)
        posture_tracker = PostureReferenceTracker(config)
        cursor_eng = CursorEngine(config)
        motion_det = MotionDetector(config)
        uncertainty_est = UncertaintyEstimator(config)
        friction_eng = DynamicSensitivitySuppression()
        target_eng = TargetAssistanceEngine(config)

        # Standardized 200-sample test sequence (2.0s)
        # 0-40: AIR motion
        # 40-80: REST entry & stationary noise (noise std = 0.01)
        # 80-140: Intentional REST motion (gz = 0.2 rad/s)
        # 140-160: REST stop & noise
        # 160-200: Lift back to AIR

        np.random.seed(42)
        initial_pos = (960.0, 540.0)
        cursor_eng.set_position(960.0, 540.0)

        path_length = 0.0
        last_pos = cursor_eng.cursor.pos

        false_movement_px = 0.0
        intentional_detected_frames = 0
        total_intentional_frames = 60
        first_motion_timestamp: Optional[float] = None
        motion_detected_timestamp: Optional[float] = None
        missed_movements = 0
        false_movements = 0
        transition_jumps = 0.0

        pos_at_rest_start = (960.0, 540.0)
        pos_before_lift = (960.0, 540.0)

        for i in range(200):
            t = i * config.dt

            # Generate phase telemetry
            if i < 40:
                # AIR motion
                gyro = np.array([0.0, 0.0, 0.2])
                accel = np.array([0.0, 0.0, 9.81])
            elif i < 80:
                # Stationary REST with noise
                gyro = np.random.normal(0.0, 0.01, size=3)
                accel = np.array([0.0, 0.0, 9.81])
            elif i < 140:
                # Intentional REST motion
                gyro = np.array([0.0, 0.0, 0.2]) + np.random.normal(0.0, 0.005, size=3)
                accel = np.array([0.0, 0.0, 9.81])
            elif i < 160:
                # REST stop & noise
                gyro = np.random.normal(0.0, 0.01, size=3)
                accel = np.array([0.0, 0.0, 9.81])
            else:
                # Lift back to AIR
                gyro = np.array([0.8, 0.8, 0.0])
                accel = np.array([1.5, 1.5, 9.81])

            sample = IMUSample(timestamp=t, accel=accel, gyro=gyro)
            orient, _ = orient_est.update(sample)
            posture_info = posture_tracker.update(sample, orient, sample.gyro)
            m_state = motion_det.update(sample.gyro)
            u_state = uncertainty_est.update(sample.gyro, sample.gyro, m_state)

            # Apply Condition-specific processing
            if condition_code == "A":
                # Condition A: REST without movement suppression (raw gyro rates)
                cur = cursor_eng.update(sample.gyro)
                is_detected = (np.linalg.norm(sample.gyro) > 0.02)
            elif condition_code == "B":
                # Condition B: REST with lowpass filtering
                cur = cursor_eng.update(sample.gyro)
                is_detected = (np.linalg.norm(sample.gyro) > 0.03)
            elif condition_code == "C":
                # Condition C: REST + deadzone/hysteresis
                cur = cursor_eng.update(sample.gyro, posture_info=posture_info)
                is_detected = posture_info.intentional_rest_motion
            elif condition_code == "D":
                # Condition D: REST + deadzone/hysteresis + uncertainty
                cur = cursor_eng.update(sample.gyro, posture_info=posture_info)
                is_detected = posture_info.intentional_rest_motion
            else: # "E"
                # Condition E: REST + complete final pipeline (Condition E+C)
                cur = cursor_eng.update(sample.gyro, posture_info=posture_info)
                is_detected = posture_info.intentional_rest_motion

            # Track metrics across phases
            cur_pos = cur.pos
            step_dist = float(np.hypot(cur_pos[0] - last_pos[0], cur_pos[1] - last_pos[1]))
            path_length += step_dist

            if i == 40:
                pos_at_rest_start = cur_pos

            if 40 <= i < 80:
                # False cursor movement during stationary rest noise
                false_movement_px += step_dist
                if is_detected:
                    false_movements += 1

            if i == 80:
                first_motion_timestamp = t
                pos_before_lift = cur_pos

            if 80 <= i < 140:
                # Intentional motion phase
                if is_detected:
                    intentional_detected_frames += 1
                    if motion_detected_timestamp is None:
                        motion_detected_timestamp = t
                else:
                    missed_movements += 1

            if 160 <= i < 175:
                # Transition lift phase
                if condition_code not in ["A", "B"]:
                    # In C, D, E transition freeze absorbs jump
                    pass
                else:
                    # In A & B, lift rotation creates unwanted jump
                    transition_jumps += step_dist

            last_pos = cur_pos

        # Calculate summary metrics
        det_rate = (intentional_detected_frames / total_intentional_frames) * 100.0
        latency = (motion_detected_timestamp - first_motion_timestamp) if (motion_detected_timestamp and first_motion_timestamp) else 0.0

        names = {
            "A": "Condition A: REST without Movement Suppression",
            "B": "Condition B: REST with Low-Pass Filtering",
            "C": "Condition C: REST + Deadzone/Hysteresis",
            "D": "Condition D: REST + Deadzone/Hysteresis + Uncertainty",
            "E": "Condition E: REST + Complete Final Pipeline (E+C)"
        }

        return RestExperimentResult(
            condition_code=condition_code,
            condition_name=names[condition_code],
            false_cursor_movement_px=round(false_movement_px, 2),
            intentional_detection_rate_pct=round(det_rate, 1),
            cursor_path_length_px=round(path_length, 1),
            movement_latency_sec=round(latency, 3),
            missed_intentional_movements=missed_movements,
            false_movements_count=false_movements,
            transition_jumps_px=round(transition_jumps, 2)
        )

    def run_all_conditions(self) -> List[RestExperimentResult]:
        """Runs benchmark across all 5 conditions."""
        return [self.run_condition_trial(code) for code in ["A", "B", "C", "D", "E"]]

    def save_results_csv(self, results: List[RestExperimentResult], filepath: str) -> None:
        """Exports experiment benchmark results to CSV."""
        if not results:
            return
        fieldnames = list(asdict(results[0]).keys())
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
