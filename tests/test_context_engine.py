"""
Phase 5 App-Aware Context Engine Test Suite.
Verifies context resolution, 4-tier priority hierarchy, profile loading,
gesture re-mappings, context switching safety, diagnostic telemetry, and pipeline integration.
"""

import pytest
import numpy as np
from wearable_mouse.context.context_models import (
    AppCategory,
    SensitivityProfile,
    ContextProfile,
    Context
)
from wearable_mouse.context.app_detector import ApplicationDetector
from wearable_mouse.context.context_manager import ContextProfileManager
from wearable_mouse.pipeline import WearableMousePipeline
from wearable_mouse.sensors.imu_data import IMUSample
from wearable_mouse.gestures.gesture_map import GestureType


def create_calibrated_pipeline() -> WearableMousePipeline:
    pipeline = WearableMousePipeline()
    for i in range(250):
        t = i * 0.01
        ax = 1.0 + 0.8 * np.sin(t * 20.0)
        ay = 1.0 + 0.8 * np.cos(t * 20.0)
        az = 9.81 + 0.5 * np.sin(t * 15.0)
        sample = IMUSample(timestamp=t, accel=np.array([ax, ay, az]), gyro=np.array([0.0, 0.0, 0.0]))
        pipeline.process_frame(sample)
    return pipeline


# -----------------------------------------------------------------------------
# Test 1: Default Profile Resolution
# -----------------------------------------------------------------------------
def test_1_default_profile_resolution():
    detector = ApplicationDetector(mock_app="unknown_app.exe")
    mgr = ContextProfileManager(app_detector=detector)

    ctx = mgr.resolve_context()
    assert ctx.active_app == "unknown_app.exe"
    assert ctx.category == AppCategory.GENERAL
    assert ctx.profile_name == "general_default"
    assert ctx.is_user_override is False


# -----------------------------------------------------------------------------
# Test 2: Browser Profile Resolution
# -----------------------------------------------------------------------------
def test_2_browser_profile_resolution():
    detector = ApplicationDetector(mock_app="chrome.exe")
    mgr = ContextProfileManager(app_detector=detector)

    ctx = mgr.resolve_context()
    assert ctx.active_app == "chrome.exe"
    assert ctx.category == AppCategory.BROWSER
    assert ctx.profile_name == "browser_profile"
    assert ctx.sensitivity.scroll_sensitivity == 1.5


# -----------------------------------------------------------------------------
# Test 3: Presentation Profile Resolution
# -----------------------------------------------------------------------------
def test_3_presentation_profile_resolution():
    detector = ApplicationDetector(mock_app="powerpnt.exe")
    mgr = ContextProfileManager(app_detector=detector)

    ctx = mgr.resolve_context()
    assert ctx.active_app == "powerpnt.exe"
    assert ctx.category == AppCategory.PRESENTATION
    assert ctx.profile_name == "presentation_profile"
    assert ctx.sensitivity.speed_multiplier == 0.9


# -----------------------------------------------------------------------------
# Test 4: Creative Profile Resolution
# -----------------------------------------------------------------------------
def test_4_creative_profile_resolution():
    detector = ApplicationDetector(mock_app="photoshop.exe")
    mgr = ContextProfileManager(app_detector=detector)

    ctx = mgr.resolve_context()
    assert ctx.active_app == "photoshop.exe"
    assert ctx.category == AppCategory.CREATIVE
    assert ctx.profile_name == "creative_profile"
    assert ctx.sensitivity.precision_multiplier == 2.0


# -----------------------------------------------------------------------------
# Test 5: Unknown Application Safe Fallback
# -----------------------------------------------------------------------------
def test_5_unknown_application_fallback_to_general():
    detector = ApplicationDetector(mock_app="my_custom_tool_99.exe")
    mgr = ContextProfileManager(app_detector=detector)

    ctx = mgr.resolve_context()
    assert ctx.category == AppCategory.GENERAL
    assert ctx.profile_name == "general_default"


# -----------------------------------------------------------------------------
# Test 6: User Override Priority
# -----------------------------------------------------------------------------
def test_6_user_profile_override_priority():
    detector = ApplicationDetector(mock_app="chrome.exe")  # Normally browser
    mgr = ContextProfileManager(app_detector=detector)

    # Apply user override to presentation profile
    ok = mgr.set_user_override("presentation_profile")
    assert ok is True

    ctx = mgr.resolve_context()
    assert ctx.profile_name == "presentation_profile"
    assert ctx.is_user_override is True

    # Clear override
    mgr.clear_user_override()
    ctx_after = mgr.resolve_context()
    assert ctx_after.profile_name == "browser_profile"
    assert ctx_after.is_user_override is False


# -----------------------------------------------------------------------------
# Test 7: Invalid Profile Fallback
# -----------------------------------------------------------------------------
def test_7_invalid_profile_safe_fallback():
    mgr = ContextProfileManager()
    ok = mgr.set_user_override("non_existent_fake_profile")
    assert ok is False  # Rejected

    ctx = mgr.resolve_context()
    assert ctx.profile_name == "general_default"


# -----------------------------------------------------------------------------
# Test 8: Context Switching
# -----------------------------------------------------------------------------
def test_8_context_switching():
    detector = ApplicationDetector(mock_app="chrome.exe")
    mgr = ContextProfileManager(app_detector=detector)

    ctx1 = mgr.resolve_context()
    assert ctx1.category == AppCategory.BROWSER

    # Switch app
    detector.set_mock_app("code.exe")
    ctx2 = mgr.resolve_context()
    assert ctx2.category == AppCategory.PRODUCTIVITY
    assert ctx2.profile_name == "productivity_profile"


# -----------------------------------------------------------------------------
# Test 9: Context Switching During Movement (No Cursor Jump)
# -----------------------------------------------------------------------------
def test_9_context_switching_during_cursor_movement_no_jump():
    pipeline = create_calibrated_pipeline()
    pipeline.context_manager.app_detector.set_mock_app("chrome.exe")

    # Move cursor
    t = 3.0
    sample = IMUSample(t, np.array([1.0, 1.0, 9.81]), np.array([0.0, 0.0, 0.5]))
    res1 = pipeline.process_frame(sample)
    pos1_x, pos1_y = res1.cursor.x, res1.cursor.y

    # Switch app mid-motion
    pipeline.context_manager.app_detector.set_mock_app("photoshop.exe")
    res2 = pipeline.process_frame(sample)
    pos2_x, pos2_y = res2.cursor.x, res2.cursor.y

    # Distance between frames should be smooth, zero discontinuous jump
    dist = np.sqrt((pos2_x - pos1_x)**2 + (pos2_y - pos1_y)**2)
    assert dist < 10.0  # Smooth continuous motion delta


# -----------------------------------------------------------------------------
# Test 10: Context Switching During Gesture (No Accidental Duplication)
# -----------------------------------------------------------------------------
def test_10_context_switching_during_gesture_no_duplication():
    pipeline = create_calibrated_pipeline()
    pipeline.context_manager.app_detector.set_mock_app("chrome.exe")

    t = 3.0
    sample = IMUSample(t, np.array([1.0, 1.0, 9.81]), np.array([0.0, 0.0, 0.0]))
    res1 = pipeline.process_frame(sample, gesture_input=GestureType.LEFT_CLICK)

    # Switch app in next frame
    pipeline.context_manager.app_detector.set_mock_app("powerpnt.exe")
    res2 = pipeline.process_frame(sample)

    # Ensure no duplicate clicks triggered on res2
    res2_clicks = [e for e in res2.semantic_events if e.event_type.name == "LEFT_CLICK"]
    assert len(res2_clicks) == 0


# -----------------------------------------------------------------------------
# Test 11: Context Diagnostic Output
# -----------------------------------------------------------------------------
def test_11_context_diagnostic_output():
    detector = ApplicationDetector(mock_app="vlc.exe")
    mgr = ContextProfileManager(app_detector=detector)

    diag = mgr.get_diagnostic_info(current_state="REST")
    assert diag["active_app"] == "vlc.exe"
    assert diag["active_profile"] == "media_profile"
    assert diag["category"] == "MEDIA"
    assert diag["current_interaction_state"] == "REST"
    assert "SCROLL_UP" in diag["current_mapping"]


# -----------------------------------------------------------------------------
# Test 12: JSON Config Loading and Parsing
# -----------------------------------------------------------------------------
def test_12_json_config_loading_and_parsing():
    mgr = ContextProfileManager()
    assert len(mgr.profiles) >= 6
    assert "general_default" in mgr.profiles
    assert "browser_profile" in mgr.profiles
    assert "presentation_profile" in mgr.profiles
    assert "creative_profile" in mgr.profiles


# -----------------------------------------------------------------------------
# Test 13: End-to-End Pipeline Context Integration
# -----------------------------------------------------------------------------
def test_13_end_to_end_pipeline_with_context_engine():
    pipeline = create_calibrated_pipeline()
    pipeline.context_manager.app_detector.set_mock_app("firefox.exe")

    t = 3.0
    sample = IMUSample(t, np.array([1.0, 1.0, 9.81]), np.array([0.0, 0.0, 0.0]))
    res = pipeline.process_frame(sample)

    assert res.active_context is not None
    assert res.active_context.active_app == "firefox.exe"
    assert res.active_context.category == AppCategory.BROWSER
    assert res.active_context.profile_name == "browser_profile"


# -----------------------------------------------------------------------------
# Test 14: Custom App Detector Callback Integration
# -----------------------------------------------------------------------------
def test_14_custom_app_detector_callback():
    detector = ApplicationDetector()
    detector.register_custom_detector(lambda: "blender.exe")

    mgr = ContextProfileManager(app_detector=detector)
    ctx = mgr.resolve_context()

    assert ctx.active_app == "blender.exe"
    assert ctx.category == AppCategory.CREATIVE
    assert ctx.profile_name == "creative_profile"


# -----------------------------------------------------------------------------
# Test 15: Context Gesture Mapping Resolution
# -----------------------------------------------------------------------------
def test_15_context_gesture_mapping_resolution():
    mgr = ContextProfileManager()
    mapped_click = mgr.map_gesture("LEFT_CLICK")
    assert mapped_click == "LEFT_CLICK"
