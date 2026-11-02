"""
Finger State Tracker & Temporal Debouncer.
Processes raw analog flex and touch/pressure sensor inputs, applies Schmitt-trigger hysteresis
and temporal debouncing per finger, and outputs a stable FingerPose container.
"""

from typing import Dict, Optional
from wearable_mouse.config import Config
from wearable_mouse.gestures.gesture_map import FingerName, FingerState, FingerPose


class FingerTracker:
    """
    Tracks state of five fingers (thumb, index, middle, ring, little) with hysteresis and debouncing.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

        # Confirmed state per finger
        self.confirmed_states: Dict[FingerName, FingerState] = {
            f: FingerState.OPEN for f in FingerName
        }

        # Candidate state per finger and timestamp when candidate first appeared
        self.candidate_states: Dict[FingerName, FingerState] = {
            f: FingerState.OPEN for f in FingerName
        }
        self.candidate_timestamps: Dict[FingerName, float] = {
            f: 0.0 for f in FingerName
        }

        # History for noise/stability estimation
        self.last_flex: Dict[FingerName, float] = {f: 0.0 for f in FingerName}

    def reset(self) -> None:
        """Reset finger states to default OPEN."""
        for f in FingerName:
            self.confirmed_states[f] = FingerState.OPEN
            self.candidate_states[f] = FingerState.OPEN
            self.candidate_timestamps[f] = 0.0
            self.last_flex[f] = 0.0

    def update(
        self,
        flex_sensors: Optional[Dict[str, float]],
        timestamp: float
    ) -> FingerPose:
        """
        Process current flex_sensors frame and return a debounced FingerPose.

        Expected flex_sensors dictionary keys:
          - 'thumb', 'index', 'middle', 'ring', 'little' (float 0.0 to 1.0)
          - optional touch contacts: 'touch_thumb_index', 'touch_thumb_middle', etc. (float/bool 0.0 or 1.0)
        """
        flex_map: Dict[FingerName, float] = {}
        touch_contacts: Dict[str, bool] = {}
        total_jitter = 0.0

        for finger in FingerName:
            key = finger.value.lower()
            val = 0.0
            if flex_sensors and key in flex_sensors:
                val = float(flex_sensors[key])
            flex_map[finger] = val

            # Estimate noise jitter
            total_jitter += abs(val - self.last_flex[finger])
            self.last_flex[finger] = val

        # Extract touch sensor contacts if present
        if flex_sensors:
            for k, v in flex_sensors.items():
                if k.startswith("touch_"):
                    touch_contacts[k] = bool(v > 0.5)

        # Classify raw candidate state with Schmitt-trigger hysteresis
        for finger in FingerName:
            flex_val = flex_map[finger]
            curr_confirmed = self.confirmed_states[finger]

            # Determine raw state from sensor thresholds
            if flex_val >= self.config.flex_touch_threshold:
                raw_state = FingerState.TOUCHING
            elif flex_val >= self.config.flex_close_threshold:
                raw_state = FingerState.CLOSED
            elif flex_val <= self.config.flex_open_threshold:
                raw_state = FingerState.OPEN
            else:
                # In hysteresis deadband -> maintain current confirmed state
                raw_state = curr_confirmed

            # Override with touch contact sensors if explicitly active
            if finger == FingerName.THUMB or finger == FingerName.INDEX:
                if touch_contacts.get("touch_thumb_index", False):
                    raw_state = FingerState.TOUCHING
            if finger == FingerName.THUMB or finger == FingerName.MIDDLE:
                if touch_contacts.get("touch_thumb_middle", False):
                    raw_state = FingerState.TOUCHING

            # Apply temporal debouncing
            if raw_state != self.candidate_states[finger]:
                self.candidate_states[finger] = raw_state
                self.candidate_timestamps[finger] = timestamp

            # If candidate state has been stable for >= gesture_debounce_duration, confirm it
            if (timestamp - self.candidate_timestamps[finger]) >= self.config.gesture_debounce_duration:
                self.confirmed_states[finger] = self.candidate_states[finger]

        stability_score = max(0.0, 1.0 - (total_jitter * 2.0))

        return FingerPose(
            finger_states=self.confirmed_states.copy(),
            raw_flex=flex_map,
            touch_contacts=touch_contacts,
            stability_score=stability_score
        )
