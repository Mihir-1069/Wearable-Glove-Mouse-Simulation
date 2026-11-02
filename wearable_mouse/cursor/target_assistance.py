"""
Error-Tolerant Target Acquisition Engine.
Provides tolerance-based hit selection, ambiguity detection, and soft-snapping assistance.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple
import numpy as np
from wearable_mouse.config import Config
from wearable_mouse.cursor.cursor_engine import VirtualCursor
from wearable_mouse.processing.uncertainty import UncertaintyState


class AcquisitionMode(Enum):
    A = "A: Baseline (Exact)"
    B = "B: Tolerance Selection"
    C = "C: Soft Snapping"
    D = "D: Tolerance + Snapping"
    E = "E: Dynamic Sensitivity Suppression"


@dataclass
class Target:
    id: str
    x: float
    y: float
    width: float
    height: float
    label: str = ""

    @property
    def center(self) -> Tuple[float, float]:
        return self.x + self.width / 2.0, self.y + self.height / 2.0

    def contains_point(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height

    def distance_to_point(self, px: float, py: float) -> float:
        """Distance from point (px, py) to nearest edge or interior (0 if inside)."""
        cx, cy = self.center
        dx = max(abs(px - cx) - self.width / 2.0, 0.0)
        dy = max(abs(py - cy) - self.height / 2.0, 0.0)
        return float(np.hypot(dx, dy))


@dataclass
class AssistanceResult:
    mode: AcquisitionMode
    selected_target: Optional[Target]
    target_confidence: float
    is_ambiguous: bool
    ambiguous_targets: List[Target]
    snapped_position: Tuple[float, float]
    friction_gain: float = 1.0


class TargetAssistanceEngine:
    """
    Assists cursor interaction near UI target rectangles while preventing accidental snaps under ambiguity.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.targets: List[Target] = []
        from wearable_mouse.cursor.dynamic_friction import DynamicSensitivitySuppression
        self.friction_engine = DynamicSensitivitySuppression(self.config)

    def set_targets(self, targets: List[Target]) -> None:
        self.targets = targets

    def process(
        self,
        cursor: VirtualCursor,
        uncertainty: UncertaintyState,
        mode: AcquisitionMode = AcquisitionMode.D
    ) -> AssistanceResult:
        """
        Evaluate target acquisition under current cursor position, velocity, uncertainty, and mode.
        """
        if not self.targets:
            return AssistanceResult(
                mode=mode,
                selected_target=None,
                target_confidence=0.0,
                is_ambiguous=False,
                ambiguous_targets=[],
                snapped_position=(cursor.x, cursor.y)
            )

        cx, cy = cursor.x, cursor.y
        r_uncert = uncertainty.uncertainty_radius

        # Mode A: Baseline exact cursor matching
        if mode == AcquisitionMode.A:
            hit = next((t for t in self.targets if t.contains_point(cx, cy)), None)
            return AssistanceResult(
                mode=mode,
                selected_target=hit,
                target_confidence=1.0 if hit else 0.0,
                is_ambiguous=False,
                ambiguous_targets=[],
                snapped_position=(cx, cy)
            )

        # Candidate evaluation for Modes B, C, D
        scored_candidates: List[Tuple[Target, float, float]] = []

        for target in self.targets:
            dist = target.distance_to_point(cx, cy)
            # A target is candidate if within uncertainty radius + tolerance margin
            max_dist = r_uncert + self.config.target_tolerance_margin
            if dist <= max_dist:
                # Score formula: inverse distance weighted by uncertainty and velocity vector
                score = 1.0 - (dist / (max_dist + 1e-5))

                # Velocity vector alignment bonus: if moving towards target centroid, boost score
                tcx, tcy = target.center
                to_target = np.array([tcx - cx, tcy - cy])
                vel = np.array([cursor.vx, cursor.vy])
                if np.linalg.norm(to_target) > 1e-3 and np.linalg.norm(vel) > 10.0:
                    unit_target = to_target / np.linalg.norm(to_target)
                    unit_vel = vel / np.linalg.norm(vel)
                    alignment = max(0.0, float(np.dot(unit_target, unit_vel)))
                    score += 0.2 * alignment

                scored_candidates.append((target, score, dist))

        if not scored_candidates:
            return AssistanceResult(
                mode=mode,
                selected_target=None,
                target_confidence=0.0,
                is_ambiguous=False,
                ambiguous_targets=[],
                snapped_position=(cx, cy)
            )

        # Sort by candidate score descending
        scored_candidates.sort(key=lambda item: item[1], reverse=True)
        top_target, top_score, top_dist = scored_candidates[0]

        # Ambiguity Check: if top 2 candidates have close scores
        is_ambiguous = False
        ambiguous_targets: List[Target] = []
        if len(scored_candidates) >= 2:
            second_target, second_score, _ = scored_candidates[1]
            if top_score > 0.01 and (second_score / top_score) >= self.config.ambiguity_ratio_threshold:
                is_ambiguous = True
                ambiguous_targets = [top_target, second_target]

        selected_target = None if is_ambiguous else top_target
        target_conf = 0.0 if is_ambiguous else float(np.clip(top_score, 0.0, 1.0))

        # Calculate position (with optional soft snapping)
        snapped_x, snapped_y = cx, cy

        if mode in (AcquisitionMode.C, AcquisitionMode.D) and selected_target and not is_ambiguous:
            # Apply gravitational pull toward target center
            tcx, tcy = selected_target.center
            pull_strength = self.config.soft_snap_strength * target_conf
            snapped_x = (1.0 - pull_strength) * cx + pull_strength * tcx
            snapped_y = (1.0 - pull_strength) * cy + pull_strength * tcy

        friction_gain = 1.0
        if mode == AcquisitionMode.E:
            f_state = self.friction_engine.compute_friction(cursor, self.targets, uncertainty)
            friction_gain = f_state.gain

        return AssistanceResult(
            mode=mode,
            selected_target=selected_target,
            target_confidence=target_conf,
            is_ambiguous=is_ambiguous,
            ambiguous_targets=ambiguous_targets,
            snapped_position=(snapped_x, snapped_y),
            friction_gain=friction_gain
        )
