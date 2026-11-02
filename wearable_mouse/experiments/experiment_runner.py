"""
Controlled Experiment Runner & Performance Evaluator.
Executes rigorous benchmark trials comparing 4 specific control conditions across identical simulated motion streams:
- Condition A: Raw / Baseline Cursor (Unfiltered, exact point-and-click)
- Condition B: Filtered Cursor (Filtered, exact point-and-click)
- Condition C: Filtered + Uncertainty Tolerance (Filtered + Tolerance hitboxes + Ambiguity suppression)
- Condition D: Filtered + Uncertainty + Snapping (Filtered + Tolerance + Soft snapping)

Computes 6 quantitative metrics:
1. Acquisition Time (sec)
2. Success Rate (%)
3. Wrong Target Rate (%)
4. Misclick Rate (%)
5. Cursor Path Length (pixels)
6. Corrective Movements Count
"""

import csv
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import numpy as np

from wearable_mouse.config import Config
from wearable_mouse.simulation.imu_simulator import IMUSimulator
from wearable_mouse.processing.calibration import Calibrator
from wearable_mouse.processing.filters import FilterPipeline, LowPassFilter, DeadZoneFilter, ExponentialSmoothingFilter
from wearable_mouse.processing.motion_detector import MotionDetector
from wearable_mouse.processing.uncertainty import UncertaintyEstimator
from wearable_mouse.cursor.cursor_engine import CursorEngine
from wearable_mouse.cursor.target_assistance import TargetAssistanceEngine, Target, AcquisitionMode, AssistanceResult
from wearable_mouse.gestures.gesture_engine import GestureEngine
from wearable_mouse.gestures.gesture_map import GestureType


@dataclass
class ExperimentMetrics:
    condition_code: str               # "A", "B", "C", "D"
    condition_name: str               # Full readable name
    total_trials: int
    successful_selections: int
    wrong_selections: int
    misclicks: int
    success_rate: float               # %
    wrong_target_rate: float          # %
    misclick_rate: float              # %
    avg_acquisition_time_sec: float   # seconds
    total_path_length_pixels: float   # pixels
    corrective_movements_count: int   # count of direction flips


class ExperimentRunner:
    """
    Controlled experiment harness comparing Conditions A, B, C, and D across identical motion trajectories.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def create_standard_targets(self) -> List[Target]:
        """
        Generate a diverse target layout: isolated buttons, small targets (40x30px), and a close pair to test ambiguity.
        """
        return [
            Target(id="t1", x=250, y=200, width=140, height=90, label="Target 1 (Isolated)"),
            Target(id="t2_tiny", x=850, y=250, width=45, height=35, label="Target 2 (Tiny)"),
            Target(id="t3", x=1450, y=200, width=130, height=80, label="Target 3 (Isolated)"),
            Target(id="t4_tiny", x=450, y=700, width=40, height=30, label="Target 4 (Tiny)"),
            Target(id="t5", x=1250, y=720, width=150, height=90, label="Target 5 (Bottom-Right)"),
            # Close target pair to test ambiguity under spatial uncertainty:
            Target(id="t6_a", x=800, y=550, width=90, height=70, label="Pair A"),
            Target(id="t6_b", x=910, y=550, width=90, height=70, label="Pair B"),
        ]

    def run_benchmark(self, num_trials: int = 15) -> List[ExperimentMetrics]:
        """
        Executes identical simulated trajectories across Conditions A, B, C, and D.
        """
        conditions = [
            ("A", "Condition A: Raw / Baseline", AcquisitionMode.A, False),
            ("B", "Condition B: Filtered Cursor", AcquisitionMode.A, True),
            ("C", "Condition C: Filtered + Uncertainty Tolerance", AcquisitionMode.B, True),
            ("D", "Condition D: Filtered + Uncertainty + Snapping", AcquisitionMode.D, True),
        ]

        results: List[ExperimentMetrics] = []
        targets = self.create_standard_targets()

        for code, name, acq_mode, use_filtering in conditions:
            metrics = self._run_single_condition(
                code=code,
                name=name,
                acq_mode=acq_mode,
                use_filtering=use_filtering,
                targets=targets,
                num_trials=num_trials
            )
            results.append(metrics)

        return results

    def _run_single_condition(
        self,
        code: str,
        name: str,
        acq_mode: AcquisitionMode,
        use_filtering: bool,
        targets: List[Target],
        num_trials: int
    ) -> ExperimentMetrics:
        # Guarantee identical sensor inputs across conditions by using a fixed seed
        sim_config = Config()
        sim_config.random_seed = self.config.random_seed
        sim = IMUSimulator(sim_config)

        calibrator = Calibrator(sim_config)
        motion_detector = MotionDetector(sim_config)
        uncertainty_est = UncertaintyEstimator(sim_config)
        cursor_engine = CursorEngine(sim_config)
        target_engine = TargetAssistanceEngine(sim_config)
        target_engine.set_targets(targets)

        # Setup filter pipeline if enabled
        if use_filtering:
            filters = FilterPipeline([
                LowPassFilter(sim_config.lowpass_cutoff, sim_config.sample_rate),
                DeadZoneFilter(sim_config.deadzone_threshold),
                ExponentialSmoothingFilter(sim_config.exp_smoothing_alpha)
            ])
        else:
            filters = FilterPipeline([])

        total_path_length = 0.0
        corrective_movements = 0
        successful_selections = 0
        wrong_selections = 0
        misclicks = 0
        trial_times: List[float] = []

        last_vel_dir = np.zeros(2)

        # Execute trials
        for trial_idx in range(num_trials):
            target = targets[trial_idx % len(targets)]
            tcx, tcy = target.center

            # Start cursor at center screen
            cursor_engine.set_position(960.0, 540.0)
            prev_pos = (960.0, 540.0)

            # Pre-run 50 calibration frames
            for _ in range(50):
                sim.set_motion(np.zeros(3))
                s = sim.step()
                calibrator.update(s)

            start_time = sim.current_time
            trial_acquired = False
            trial_duration = 5.0

            # Phase 1: Hand Movement towards target (120 steps ~ 1.2s)
            for step in range(120):
                curr_time = sim.current_time
                cx, cy = cursor_engine.cursor.x, cursor_engine.cursor.y

                dx_to_target = tcx - cx
                dy_to_target = tcy - cy
                dist_to_target = np.hypot(dx_to_target, dy_to_target)

                # Angular velocity steering toward target with natural human aiming variability and tremor
                aim_noise = sim.rng.normal(0.0, 0.12, size=2)
                yaw_rate = np.clip(dx_to_target * 0.005 + aim_noise[0], -1.0, 1.0)
                pitch_rate = np.clip(-dy_to_target * 0.005 + aim_noise[1], -1.0, 1.0)

                sim.set_motion(np.array([0.0, pitch_rate, yaw_rate]))
                sample = sim.step()

                # Processing chain
                calib_sample = calibrator.apply(sample)
                filt_gyro = filters.process(calib_sample.gyro) if use_filtering else calib_sample.gyro
                motion_state = motion_detector.update(filt_gyro)
                uncertainty = uncertainty_est.update(calib_sample.gyro, filt_gyro, motion_state)

                # Cursor & Target Assistance
                cursor = cursor_engine.update(filt_gyro)
                assist_res = target_engine.process(cursor, uncertainty, mode=acq_mode)

                # Apply soft snapping if mode C or D
                if assist_res.snapped_position != (cursor.x, cursor.y):
                    cursor_engine.cursor.x, cursor_engine.cursor.y = assist_res.snapped_position

                curr_pos = (cursor_engine.cursor.x, cursor_engine.cursor.y)
                step_dist = np.hypot(curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
                total_path_length += step_dist

                # Detect trajectory direction changes (corrective movements / wobble)
                step_vel = np.array([curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1]])
                if np.linalg.norm(step_vel) > 1.0 and np.linalg.norm(last_vel_dir) > 1.0:
                    dot_prod = np.dot(step_vel / np.linalg.norm(step_vel), last_vel_dir / np.linalg.norm(last_vel_dir))
                    if dot_prod < -0.1:  # Direction reversal threshold
                        corrective_movements += 1
                if np.linalg.norm(step_vel) > 1.0:
                    last_vel_dir = step_vel / np.linalg.norm(step_vel)

                prev_pos = curr_pos

            # Phase 2: Dwell & Click Attempt (30 steps ~ 0.3s holding static position over target)
            for step in range(30):
                sim.set_motion(np.zeros(3))
                sample = sim.step()

                calib_sample = calibrator.apply(sample)
                filt_gyro = filters.process(calib_sample.gyro) if use_filtering else calib_sample.gyro
                motion_state = motion_detector.update(filt_gyro)
                uncertainty = uncertainty_est.update(calib_sample.gyro, filt_gyro, motion_state)

                cursor = cursor_engine.update(filt_gyro)
                assist_res = target_engine.process(cursor, uncertainty, mode=acq_mode)

                if assist_res.snapped_position != (cursor.x, cursor.y):
                    cursor_engine.cursor.x, cursor_engine.cursor.y = assist_res.snapped_position

            trial_duration = sim.current_time - start_time

            # Evaluate selection outcome for click
            final_cursor = cursor_engine.cursor
            final_uncertainty = uncertainty_est.update(np.zeros(3), np.zeros(3), motion_detector.update(np.zeros(3)))
            final_assist = target_engine.process(final_cursor, final_uncertainty, mode=acq_mode)

            if final_assist.selected_target and final_assist.selected_target.id == target.id:
                successful_selections += 1
                trial_times.append(trial_duration)
            elif final_assist.selected_target and final_assist.selected_target.id != target.id:
                wrong_selections += 1
                misclicks += 1
                trial_times.append(trial_duration + 1.0)  # Penalty for wrong target selection
            else:
                misclicks += 1
                trial_times.append(trial_duration + 1.5)  # Penalty for missed click

        # Compute aggregate quantitative metrics
        success_rate = (successful_selections / num_trials) * 100.0 if num_trials > 0 else 0.0
        wrong_target_rate = (wrong_selections / num_trials) * 100.0 if num_trials > 0 else 0.0
        misclick_rate = (misclicks / num_trials) * 100.0 if num_trials > 0 else 0.0
        avg_time = float(np.mean(trial_times)) if trial_times else 4.0

        return ExperimentMetrics(
            condition_code=code,
            condition_name=name,
            total_trials=num_trials,
            successful_selections=successful_selections,
            wrong_selections=wrong_selections,
            misclicks=misclicks,
            success_rate=round(success_rate, 2),
            wrong_target_rate=round(wrong_target_rate, 2),
            misclick_rate=round(misclick_rate, 2),
            avg_acquisition_time_sec=round(avg_time, 3),
            total_path_length_pixels=round(total_path_length, 1),
            corrective_movements_count=corrective_movements
        )

    def save_results_csv(self, metrics_list: List[ExperimentMetrics], filepath: str) -> None:
        """Export full metrics dataset to CSV."""
        if not metrics_list:
            return

        fieldnames = list(asdict(metrics_list[0]).keys())
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in metrics_list:
                writer.writerow(asdict(m))
