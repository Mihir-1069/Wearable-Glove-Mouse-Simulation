"""
Dynamic Sensitivity Suppression / Friction Module (Condition E).
Dynamically modulates cursor velocity gain g in [minimum_gain, 1.0] when approaching UI targets
in a relevant uncertainty region without modifying cursor coordinates directly (zero magnetic snapping).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from wearable_mouse.config import Config
from wearable_mouse.cursor.cursor_engine import VirtualCursor
from wearable_mouse.cursor.target_assistance import Target
from wearable_mouse.processing.uncertainty import UncertaintyState


@dataclass
class FrictionState:
    gain: float                      # Effective gain multiplier (minimum_gain to 1.0)
    friction_factor: float           # 1.0 - gain (0.0 to max_friction)
    primary_target: Optional[Target] # Target exerting primary friction
    is_ambiguous: bool               # True if multi-target conflict suppressed friction
    directional_alignment: float     # Cosine alignment with target heading (0.0 to 1.0)


class DynamicSensitivitySuppression:
    """
    Applies mathematical velocity-aware gain suppression when approaching target boundaries.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

        # Tunable Parameters (fully configurable)
        self.friction_start_distance: float = 90.0    # Base distance (px) from target boundary to engage friction
        self.minimum_gain: float = 0.25               # Minimum gain floor (prevents cursor freeze)
        self.maximum_friction: float = 0.75           # Max allowed friction intensity (1.0 - minimum_gain)
        self.uncertainty_weight: float = 0.5          # Weight for uncertainty radius expansion
        self.velocity_reference: float = 400.0        # Speed scale for velocity-aware damping (px/s)
        self.ambiguity_threshold: float = 0.75        # Proximity ratio threshold for multi-target ambiguity

    def compute_friction(
        self,
        cursor: VirtualCursor,
        targets: List[Target],
        uncertainty: UncertaintyState
    ) -> FrictionState:
        """
        Calculates gain multiplier g in [minimum_gain, 1.0] for the current frame.
        """
        if not targets:
            return FrictionState(
                gain=1.0,
                friction_factor=0.0,
                primary_target=None,
                is_ambiguous=False,
                directional_alignment=0.0
            )

        cx, cy = cursor.x, cursor.y
        vx, vy = cursor.vx, cursor.vy
        speed = float(np.hypot(vx, vy))

        r_uncert = uncertainty.uncertainty_radius

        # Calculate candidate target proximity and alignment
        candidate_scores: List[Tuple[Target, float, float, float]] = []

        for target in targets:
            dist = target.distance_to_point(cx, cy)
            d_start = self.friction_start_distance + self.uncertainty_weight * (r_uncert - self.config.uncertainty_base_radius)

            if dist <= d_start:
                # Directional Alignment Check
                tcx, tcy = target.center
                to_target = np.array([tcx - cx, tcy - cy])
                dist_to_center = float(np.linalg.norm(to_target))

                if dist_to_center > 1e-3 and speed > 5.0:
                    unit_target = to_target / dist_to_center
                    unit_vel = np.array([vx, vy]) / speed
                    alignment = float(np.dot(unit_target, unit_vel))
                else:
                    alignment = 1.0  # Hovering or stationary

                # If moving away from target (alignment <= 0), bypass friction for this target
                if alignment > 0.0:
                    # Spatial proximity factor s in [0, 1]
                    s = 1.0 - (dist / d_start)
                    # Cosine smooth transition
                    f_spatial = 0.5 * (1.0 - np.cos(np.pi * s))
                    candidate_scores.append((target, f_spatial, alignment, dist))

        if not candidate_scores:
            return FrictionState(
                gain=1.0,
                friction_factor=0.0,
                primary_target=None,
                is_ambiguous=False,
                directional_alignment=0.0
            )

        # Sort candidates by spatial friction intensity descending
        candidate_scores.sort(key=lambda item: item[1], reverse=True)
        top_target, top_spatial, top_align, top_dist = candidate_scores[0]

        # Multi-Target Ambiguity Guard
        is_ambiguous = False
        ambig_suppression = 1.0

        if len(candidate_scores) >= 2:
            sec_target, sec_spatial, _, _ = candidate_scores[1]
            if top_spatial > 0.01 and (sec_spatial / top_spatial) >= self.ambiguity_threshold:
                is_ambiguous = True
                # Suppress friction under ambiguity to prevent biased attraction to wrong target
                ratio = sec_spatial / top_spatial
                ambig_suppression = max(0.0, 1.0 - (ratio - self.ambiguity_threshold) / (1.0 - self.ambiguity_threshold))

        # Velocity-aware damping factor (fast movements bypass heavy braking)
        f_speed = 1.0 / (1.0 + (speed / self.velocity_reference) ** 1.2)

        # Uncertainty weighting factor
        f_uncert = 1.0 + 0.3 * (r_uncert / self.config.uncertainty_max_radius)

        # Composite Net Friction Intensity
        net_intensity = top_spatial * top_align * f_speed * f_uncert * ambig_suppression
        friction_factor = float(np.clip(net_intensity * self.maximum_friction, 0.0, self.maximum_friction))
        gain = float(np.clip(1.0 - friction_factor, self.minimum_gain, 1.0))

        return FrictionState(
            gain=gain,
            friction_factor=friction_factor,
            primary_target=top_target if not is_ambiguous else None,
            is_ambiguous=is_ambiguous,
            directional_alignment=top_align
        )
