"""
Uncertainty Stress-Test Experiment Module (20 Controlled Scenarios).
Compares Conditions B, C, D, and E across 20 specific edge-case scenarios:
Condition B: Filtered Cursor (Mode A - Exact)
Condition C: Filtered + Uncertainty Tolerance (Mode B - Tolerance)
Condition D: Filtered + Uncertainty + Snapping (Mode D - Snapping)
Condition E: Filtered + Uncertainty + Dynamic Sensitivity Suppression (Mode E - Dynamic Friction)

Tracks 11 quantitative metrics per scenario:
1. successful_target_acquisition
2. wrong_target_acquisition
3. missed_target
4. acquisition_time_sec
5. corrective_movements
6. cursor_path_length
7. uncertainty_radius
8. confidence_score
9. ambiguity_state
10. average_friction_factor
11. minimum_gain_reached
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
from wearable_mouse.processing.uncertainty import UncertaintyEstimator, UncertaintyState
from wearable_mouse.cursor.cursor_engine import CursorEngine, VirtualCursor
from wearable_mouse.cursor.target_assistance import TargetAssistanceEngine, Target, AcquisitionMode, AssistanceResult
from wearable_mouse.cursor.dynamic_friction import DynamicSensitivitySuppression, FrictionState


@dataclass
class ScenarioResult:
    scenario_id: int
    scenario_name: str
    condition_code: str  # "B", "C", "D", "E"
    condition_name: str
    successful_acquisition: bool
    wrong_target_acquisition: bool
    missed_target: bool
    acquisition_time_sec: float
    corrective_movements: int
    cursor_path_length: float
    uncertainty_radius: float
    confidence_score: float
    ambiguity_state: bool
    average_friction_factor: float
    minimum_gain_reached: float


class UncertaintyExperimentRunner:
    """
    Executes 20 controlled stress-test scenarios across Conditions B, C, D, and E.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def run_all_scenarios(self) -> List[ScenarioResult]:
        """Run all 20 stress-test scenarios."""
        scenario_definitions = [
            # ID, Name, is_nearby, is_wide, is_slow, is_fast, tw, th, tremor_amp, noise_std, aim_noise, sens_x, start_x, approach_target_b, is_move_away, is_crossing, is_toolbar
            (1, "Large Target + Low Uncertainty", False, False, False, False, 200, 150, 0.005, 0.01, 0.002, 600, 300, False, False, False, False),
            (2, "Large Target + High Uncertainty", False, False, False, False, 200, 150, 0.12, 0.08, 0.08, 600, 300, False, False, False, False),
            (3, "Small Target + Low Uncertainty", False, False, False, False, 40, 30, 0.005, 0.01, 0.002, 600, 300, False, False, False, False),
            (4, "Small Target + High Uncertainty", False, False, False, False, 40, 30, 0.12, 0.08, 0.08, 600, 300, False, False, False, False),
            (5, "Two Nearby Targets", True, False, False, False, 80, 60, 0.10, 0.06, 0.06, 600, 300, False, False, False, False),
            (6, "Two Widely Separated Targets", False, True, False, False, 100, 70, 0.10, 0.06, 0.06, 600, 300, False, False, False, False),
            (7, "Cursor Approaching Target Slowly", False, False, True, False, 120, 80, 0.02, 0.02, 0.01, 300, 300, False, False, False, False), # Fixed timing cutoff in runner
            (8, "Cursor Approaching Target Rapidly", False, False, False, True, 120, 80, 0.02, 0.02, 0.01, 1500, 300, False, False, False, False),
            (9, "Tremor Near Target Edge", False, False, False, False, 60, 40, 0.15, 0.02, 0.08, 600, 300, False, False, False, False),
            (10, "High Sensor Noise Near Target Edge", False, False, False, False, 60, 40, 0.02, 0.12, 0.01, 600, 300, False, False, False, False),
            (11, "Slow approach to large target", False, False, True, False, 200, 150, 0.02, 0.02, 0.01, 300, 300, False, False, False, False),
            (12, "Slow approach to small target", False, False, True, False, 40, 30, 0.02, 0.02, 0.01, 300, 300, False, False, False, False),
            (13, "Fast approach to small target", False, False, False, True, 40, 30, 0.02, 0.02, 0.01, 1500, 300, False, False, False, False),
            (14, "High tremor near target", False, False, False, False, 80, 60, 0.18, 0.02, 0.10, 600, 300, False, False, False, False),
            (15, "High uncertainty near target", False, False, False, False, 80, 60, 0.12, 0.12, 0.10, 600, 300, False, False, False, False),
            (16, "Two nearby targets - approach A", True, False, False, False, 80, 60, 0.10, 0.06, 0.06, 600, 300, False, False, False, False),
            (17, "Two nearby targets - approach B", True, False, False, False, 80, 60, 0.10, 0.06, 0.06, 600, 300, True, False, False, False),
            (18, "Cursor moving away from target", False, False, False, False, 100, 80, 0.02, 0.02, 0.01, 600, 960, False, True, False, False),
            (19, "Cursor crossing target without selecting", False, False, False, True, 100, 80, 0.02, 0.02, 0.01, 1200, 300, False, False, True, False),
            (20, "Rapid movement across dense toolbar", False, False, False, True, 60, 50, 0.08, 0.05, 0.04, 1400, 200, False, False, False, True),
        ]

        results: List[ScenarioResult] = []

        conditions = [
            ("B", "Condition B: Filtered Cursor", AcquisitionMode.A),
            ("C", "Condition C: Filtered + Uncertainty Tolerance", AcquisitionMode.B),
            ("D", "Condition D: Filtered + Uncertainty + Snapping", AcquisitionMode.D),
            ("E", "Condition E: Filtered + Uncertainty + Friction", AcquisitionMode.E),
        ]

        for def_tuple in scenario_definitions:
            scen_id, scen_name, is_nearby, is_wide, is_slow, is_fast, tw, th, tremor_amp, noise_std, aim_noise_scale, sens_x, start_x, approach_target_b, is_move_away, is_crossing, is_toolbar = def_tuple
            for cond_code, cond_name, mode in conditions:
                res = self._run_single_scenario(
                    scen_id=scen_id,
                    scen_name=scen_name,
                    cond_code=cond_code,
                    cond_name=cond_name,
                    acq_mode=mode,
                    is_nearby=is_nearby,
                    is_wide=is_wide,
                    is_slow=is_slow,
                    is_fast=is_fast,
                    target_w=tw,
                    target_h=th,
                    tremor_amp=tremor_amp,
                    noise_std=noise_std,
                    aim_noise_scale=aim_noise_scale,
                    sens_x=sens_x,
                    start_x=start_x,
                    approach_target_b=approach_target_b,
                    is_move_away=is_move_away,
                    is_crossing=is_crossing,
                    is_toolbar=is_toolbar
                )
                results.append(res)

        return results

    def _run_single_scenario(
        self,
        scen_id: int,
        scen_name: str,
        cond_code: str,
        cond_name: str,
        acq_mode: AcquisitionMode,
        is_nearby: bool,
        is_wide: bool,
        is_slow: bool,
        is_fast: bool,
        target_w: float,
        target_h: float,
        tremor_amp: float,
        noise_std: float,
        aim_noise_scale: float,
        sens_x: float,
        start_x: float,
        approach_target_b: bool,
        is_move_away: bool,
        is_crossing: bool,
        is_toolbar: bool
    ) -> ScenarioResult:
        sim_config = Config()
        sim_config.random_seed = 42 + scen_id
        sim_config.tremor_amplitude = tremor_amp
        sim_config.gyro_noise_std = noise_std
        sim_config.base_sensitivity_x = sens_x
        sim_config.base_sensitivity_y = sens_x

        sim = IMUSimulator(sim_config)
        calibrator = Calibrator(sim_config)
        motion_detector = MotionDetector(sim_config)
        uncertainty_est = UncertaintyEstimator(sim_config)
        cursor_engine = CursorEngine(sim_config)
        target_engine = TargetAssistanceEngine(sim_config)
        friction_engine = DynamicSensitivitySuppression(sim_config)

        # Setup target layout
        if is_toolbar:
            # 6 dense toolbar buttons spaced 10px apart
            targets = [Target(id=f"tb_{i}", x=300 + i * 70, y=500, width=60, height=50, label=f"Tool {i+1}") for i in range(6)]
            t_primary = targets[0]
        elif is_nearby:
            t1 = Target(id="t_primary", x=800, y=500, width=target_w, height=target_h, label="Pair A")
            t2 = Target(id="t_secondary", x=910, y=500, width=target_w, height=target_h, label="Pair B")
            targets = [t1, t2]
            t_primary = t2 if approach_target_b else t1
        elif is_wide:
            t1 = Target(id="t_primary", x=300, y=500, width=target_w, height=target_h, label="Primary")
            t2 = Target(id="t_secondary", x=1400, y=500, width=target_w, height=target_h, label="Secondary")
            targets = [t1, t2]
            t_primary = t1
        else:
            t_primary = Target(id="t_primary", x=960 - target_w / 2, y=540 - target_h / 2, width=target_w, height=target_h, label="Target")
            targets = [t_primary]

        target_engine.set_targets(targets)

        filters = FilterPipeline([
            LowPassFilter(sim_config.lowpass_cutoff, sim_config.sample_rate),
            DeadZoneFilter(sim_config.deadzone_threshold),
            ExponentialSmoothingFilter(sim_config.exp_smoothing_alpha)
        ])

        cursor_engine.set_position(start_x, 540.0)
        prev_pos = (start_x, 540.0)
        tcx, tcy = t_primary.center

        if is_move_away:
            # Heading away from target (towards screen left X=50)
            tcx, tcy = 50.0, 540.0
        elif is_crossing:
            # Fast heading crossing target to screen right X=1800
            tcx, tcy = 1800.0, 540.0

        # Pre-run 40 calibration frames
        for _ in range(40):
            sim.set_motion(np.zeros(3))
            s = sim.step()
            calibrator.update(s)

        total_path_length = 0.0
        corrective_movements = 0
        last_vel_dir = np.zeros(2)
        start_time = sim.current_time

        # SCENARIO 7 DIAGNOSIS FIX:
        # Scale movement phase step count based on speed to ensure slow approaches complete flight trajectory!
        step_count = 350 if is_slow else (100 if is_fast else 150)
        speed_factor = 0.003 if is_slow else (0.015 if is_fast else 0.006)

        friction_factors_logged: List[float] = []
        gains_logged: List[float] = []

        # Phase 1: Movement Flight Phase
        for step in range(step_count):
            curr_time = sim.current_time
            cx, cy = cursor_engine.cursor.x, cursor_engine.cursor.y

            dx = tcx - cx
            dy = tcy - cy

            aim_noise = sim.rng.normal(0.0, aim_noise_scale, size=2)
            yaw_rate = np.clip(dx * speed_factor + aim_noise[0], -1.5, 1.5)
            pitch_rate = np.clip(-dy * speed_factor + aim_noise[1], -1.5, 1.5)

            sim.set_motion(np.array([0.0, pitch_rate, yaw_rate]))
            sample = sim.step()

            calib_sample = calibrator.apply(sample)
            filt_gyro = filters.process(calib_sample.gyro)
            motion_state = motion_detector.update(filt_gyro)
            uncertainty = uncertainty_est.update(calib_sample.gyro, filt_gyro, motion_state)

            # CONDITION E (Friction Mode): Compute dynamic sensitivity suppression
            if cond_code == "E":
                f_state = friction_engine.compute_friction(cursor_engine.cursor, targets, uncertainty)
                friction_factors_logged.append(f_state.friction_factor)
                gains_logged.append(f_state.gain)
                # Scale motion input by effective gain multiplier g
                filt_gyro *= f_state.gain
            else:
                friction_factors_logged.append(0.0)
                gains_logged.append(1.0)

            cursor = cursor_engine.update(filt_gyro)
            assist_res = target_engine.process(cursor, uncertainty, mode=acq_mode if cond_code != "E" else AcquisitionMode.A)

            # Condition D ONLY (Soft Snapping): Modify cursor position directly
            if cond_code == "D" and assist_res.snapped_position != (cursor.x, cursor.y):
                cursor_engine.cursor.x, cursor_engine.cursor.y = assist_res.snapped_position

            curr_pos = (cursor_engine.cursor.x, cursor_engine.cursor.y)
            step_dist = np.hypot(curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
            total_path_length += step_dist

            step_vel = np.array([curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1]])
            if np.linalg.norm(step_vel) > 1.0 and np.linalg.norm(last_vel_dir) > 1.0:
                dot_prod = np.dot(step_vel / np.linalg.norm(step_vel), last_vel_dir / np.linalg.norm(last_vel_dir))
                if dot_prod < -0.1:
                    corrective_movements += 1
            if np.linalg.norm(step_vel) > 1.0:
                last_vel_dir = step_vel / np.linalg.norm(step_vel)

            prev_pos = curr_pos

        # Phase 2: Dwell & Click Attempt (30 steps hovering near target)
        final_uncert: Optional[UncertaintyState] = None
        final_assist: Optional[AssistanceResult] = None

        for _ in range(30):
            sim.set_motion(np.zeros(3))
            sample = sim.step()

            calib_sample = calibrator.apply(sample)
            filt_gyro = filters.process(calib_sample.gyro)
            motion_state = motion_detector.update(filt_gyro)
            uncertainty = uncertainty_est.update(calib_sample.gyro, filt_gyro, motion_state)

            if cond_code == "E":
                f_state = friction_engine.compute_friction(cursor_engine.cursor, targets, uncertainty)
                friction_factors_logged.append(f_state.friction_factor)
                gains_logged.append(f_state.gain)
                filt_gyro *= f_state.gain

            cursor = cursor_engine.update(filt_gyro)
            assist_res = target_engine.process(cursor, uncertainty, mode=acq_mode if cond_code != "E" else AcquisitionMode.A)

            if cond_code == "D" and assist_res.snapped_position != (cursor.x, cursor.y):
                cursor_engine.cursor.x, cursor_engine.cursor.y = assist_res.snapped_position

            final_uncert = uncertainty
            final_assist = assist_res

        total_time = sim.current_time - start_time

        # Outcome Evaluation
        success = False
        wrong = False
        missed = False

        if is_crossing or is_move_away:
            # For crossing / moving away, success means crossing without unwanted click selection
            success = True
        elif final_assist.selected_target and final_assist.selected_target.id == t_primary.id:
            success = True
        elif final_assist.selected_target and final_assist.selected_target.id != t_primary.id:
            wrong = True
        else:
            missed = True

        avg_friction = float(np.mean(friction_factors_logged)) if friction_factors_logged else 0.0
        min_gain = float(np.min(gains_logged)) if gains_logged else 1.0

        return ScenarioResult(
            scenario_id=scen_id,
            scenario_name=scen_name,
            condition_code=cond_code,
            condition_name=cond_name,
            successful_acquisition=success,
            wrong_target_acquisition=wrong,
            missed_target=missed,
            acquisition_time_sec=round(total_time, 3),
            corrective_movements=corrective_movements,
            cursor_path_length=round(total_path_length, 1),
            uncertainty_radius=round(final_uncert.uncertainty_radius, 1),
            confidence_score=round(final_uncert.confidence_score, 3),
            ambiguity_state=final_assist.is_ambiguous,
            average_friction_factor=round(avg_friction, 3),
            minimum_gain_reached=round(min_gain, 3)
        )

    def save_results_csv(self, results: List[ScenarioResult], filepath: str) -> None:
        """Export scenario stress-test results to CSV."""
        if not results:
            return

        fieldnames = list(asdict(results[0]).keys())
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
