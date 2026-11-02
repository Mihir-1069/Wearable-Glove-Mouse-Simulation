"""
Human Test Manager module.
Orchestrates interactive trial blocks, target layout generation, condition cycling,
blinded trial masking, path metrics tracking, and trial outcome evaluation.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple, Dict
import numpy as np

from wearable_mouse.config import Config
from wearable_mouse.cursor.cursor_engine import VirtualCursor
from wearable_mouse.cursor.target_assistance import Target, AcquisitionMode
from wearable_mouse.telemetry_logging.human_test_logger import HumanTestLogger, HumanTestTrialRecord


class TargetLayout(Enum):
    ISOLATED = "ISOLATED"
    DENSE_GRID = "DENSE_GRID"
    TOOLBAR = "TOOLBAR"


@dataclass
class ConditionInfo:
    code: str                  # "B", "C", "E", "E+C"
    name: str                  # Readable name
    blinded_label: str         # "Condition 1", "Condition 2", etc.
    acq_mode: AcquisitionMode  # Underlying AcquisitionMode
    use_friction: bool         # Enable Condition E dynamic friction


class HumanTestManager:
    """
    Manages interactive human trial sequences, target layouts, and metrics collection.
    """

    def __init__(self, config: Optional[Config] = None, is_blinded: bool = True):
        self.config = config or Config()
        self.is_blinded = is_blinded
        self.logger = HumanTestLogger()

        # Define 4 Test Conditions
        self.conditions: List[ConditionInfo] = [
            ConditionInfo("B", "Condition B: Filtered Cursor", "Condition 1", AcquisitionMode.A, False),
            ConditionInfo("C", "Condition C: Filtered + Uncertainty Tolerance", "Condition 2", AcquisitionMode.B, False),
            ConditionInfo("E", "Condition E: Filtered + Dynamic Friction", "Condition 3", AcquisitionMode.A, True),
            ConditionInfo("E+C", "Condition E+C: Hybrid (Friction + Tolerance)", "Condition 4", AcquisitionMode.B, True),
        ]

        self.target_sizes = [120, 80, 40, 20]
        self.layouts = [TargetLayout.ISOLATED, TargetLayout.DENSE_GRID, TargetLayout.TOOLBAR]

        # Trial State
        self.current_condition_idx = 0
        self.current_layout_idx = 0
        self.current_size_idx = 0
        self.block_index = 1
        self.trial_in_block = 0
        self.trials_per_block = 5
        self.total_trials_completed = 0

        # Active Targets & Metrics
        self.targets: List[Target] = []
        self.active_target: Optional[Target] = None
        self.trial_start_time = 0.0
        self.trial_path_length = 0.0
        self.corrective_movements = 0
        self.num_clicks = 0
        self.last_pos = (960.0, 540.0)
        self.last_vel_dir = np.zeros(2)

        self._generate_next_layout()

    @property
    def current_condition(self) -> ConditionInfo:
        return self.conditions[self.current_condition_idx % len(self.conditions)]

    @property
    def current_target_size(self) -> int:
        return self.target_sizes[self.current_size_idx % len(self.target_sizes)]

    @property
    def current_layout(self) -> TargetLayout:
        return self.layouts[self.current_layout_idx % len(self.layouts)]

    def _generate_next_layout(self) -> None:
        """Generate targets for current layout and target size."""
        ts = self.current_target_size
        layout = self.current_layout
        self.targets.clear()

        if layout == TargetLayout.ISOLATED:
            # Single centered isolated target
            t = Target(id="t_main", x=960 - ts / 2, y=540 - ts / 2, width=ts, height=ts, label=f"Target ({ts}px)")
            self.targets.append(t)
            self.active_target = t

        elif layout == TargetLayout.DENSE_GRID:
            # 3x3 Grid of targets spaced closely (ts * 0.4 gap)
            gap = int(ts * 0.4)
            start_x = 960 - (3 * ts + 2 * gap) // 2
            start_y = 540 - (3 * ts + 2 * gap) // 2
            for row in range(3):
                for col in range(3):
                    tx = start_x + col * (ts + gap)
                    ty = start_y + row * (ts + gap)
                    t = Target(id=f"t_{row}_{col}", x=tx, y=ty, width=ts, height=ts, label=f"Grid ({row},{col})")
                    self.targets.append(t)
            # Pick center target as active
            self.active_target = self.targets[4]

        elif layout == TargetLayout.TOOLBAR:
            # 6 horizontal toolbar buttons
            gap = 8
            start_x = 960 - (6 * ts + 5 * gap) // 2
            for col in range(6):
                tx = start_x + col * (ts + gap)
                t = Target(id=f"tb_{col}", x=tx, y=500, width=ts, height=ts, label=f"Tool {col+1}")
                self.targets.append(t)
            # Pick 3rd toolbar button as active
            self.active_target = self.targets[2]

    def reset_trial_metrics(self, cursor_pos: Tuple[float, float], timestamp: float) -> None:
        """Reset path length and timers at the start of a new trial."""
        self.trial_start_time = timestamp
        self.trial_path_length = 0.0
        self.corrective_movements = 0
        self.num_clicks = 0
        self.last_pos = cursor_pos
        self.last_vel_dir = np.zeros(2)

    def update_frame(self, cursor_pos: Tuple[float, float]) -> None:
        """Update path length and direction reversal counts frame-by-frame."""
        cx, cy = cursor_pos
        px, py = self.last_pos
        dist = float(np.hypot(cx - px, cy - py))
        self.trial_path_length += dist

        step_vel = np.array([cx - px, cy - py])
        if np.linalg.norm(step_vel) > 1.0 and np.linalg.norm(self.last_vel_dir) > 0.5:
            dot_prod = np.dot(step_vel / np.linalg.norm(step_vel), self.last_vel_dir / np.linalg.norm(self.last_vel_dir))
            if dot_prod < -0.1:
                self.corrective_movements += 1
        if np.linalg.norm(step_vel) > 1.0:
            self.last_vel_dir = step_vel / np.linalg.norm(step_vel)

        self.last_pos = cursor_pos

    def evaluate_click(
        self,
        cursor_pos: Tuple[float, float],
        timestamp: float,
        selected_target: Optional[Target],
        uncertainty_radius: float,
        noise_level: float,
        tremor_level: float
    ) -> Tuple[str, bool]:
        """
        Evaluate click result against active target and return (outcome_string, is_block_complete).

        Returns:
            Tuple of (outcome_str, is_block_complete).
        """
        self.num_clicks += 1
        acq_time = max(0.1, timestamp - self.trial_start_time)

        if selected_target and self.active_target and selected_target.id == self.active_target.id:
            outcome = "SUCCESS"
        elif selected_target and self.active_target and selected_target.id != self.active_target.id:
            outcome = "WRONG_TARGET"
        else:
            outcome = "MISSED"

        # Log trial record
        cond = self.current_condition
        record = HumanTestTrialRecord(
            trial_index=self.total_trials_completed + 1,
            block_index=self.block_index,
            condition_code=cond.code,
            condition_blinded_label=cond.blinded_label,
            target_id=self.active_target.id if self.active_target else "none",
            target_size=self.current_target_size,
            target_layout=self.current_layout.value,
            outcome=outcome,
            acquisition_time_sec=round(acq_time, 3),
            cursor_path_length=round(self.trial_path_length, 1),
            corrective_movements=self.corrective_movements,
            num_clicks=self.num_clicks,
            uncertainty_level=round(uncertainty_radius, 1),
            noise_level=round(noise_level, 4),
            tremor_level=round(tremor_level, 4)
        )
        self.logger.log_trial(record)

        self.total_trials_completed += 1
        self.trial_in_block += 1

        is_block_complete = (self.trial_in_block >= self.trials_per_block)

        # Advance scenario variables for next trial
        self.current_size_idx = (self.current_size_idx + 1) % len(self.target_sizes)
        if self.current_size_idx == 0:
            self.current_layout_idx = (self.current_layout_idx + 1) % len(self.layouts)

        self._generate_next_layout()
        return outcome, is_block_complete

    def advance_block(self) -> None:
        """Advance to next block index and next condition."""
        self.block_index += 1
        self.trial_in_block = 0
        self.current_condition_idx = (self.current_condition_idx + 1) % len(self.conditions)
        self._generate_next_layout()
