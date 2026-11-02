"""
Phase 6 Personal Calibration, Profiles and Configuration Test Suite.
Verifies profile creation, JSON serialization/deserialization, validation & repair,
6-stage Calibration Wizard, system Config mapping, diagnostic output, and pipeline execution.
"""

import os
import pytest
import numpy as np
from wearable_mouse.profiles.profile_models import UserProfile, DefaultProfile, CustomProfile
from wearable_mouse.profiles.user_profile_manager import UserProfileManager
from wearable_mouse.calibration_wizard import CalibrationWizard, WizardStage
from wearable_mouse.pipeline import WearableMousePipeline
from wearable_mouse.sensors.imu_data import IMUSample


# -----------------------------------------------------------------------------
# Test 1: Profile Save and Load
# -----------------------------------------------------------------------------
def test_1_profile_save_and_load(tmp_path):
    mgr = UserProfileManager(profiles_dir=str(tmp_path))
    prof = mgr.create_profile(
        name="test_user_1",
        sensitivity=1.5,
        sensitivity_x=1400.0,
        sensitivity_y=1400.0,
        deadzone=0.03
    )

    save_path = mgr.save_profile(prof)
    assert os.path.exists(save_path)

    loaded = mgr.load_profile(save_path)
    assert loaded.name == "test_user_1"
    assert loaded.sensitivity == 1.5
    assert loaded.sensitivity_x == 1400.0
    assert loaded.deadzone == 0.03


# -----------------------------------------------------------------------------
# Test 2: Invalid Profile Validation and Repair
# -----------------------------------------------------------------------------
def test_2_invalid_profile_validation_and_repair():
    mgr = UserProfileManager()
    bad_data = {
        "name": "invalid_user",
        "sensitivity": -5.0,            # Negative sensitivity
        "deadzone": 0.9,               # Deadzone out of range > 0.5
        "rest_release_threshold": 0.1,  # Release > Activation (0.045)
        "rest_activation_threshold": 0.045
    }

    is_valid, errors, repaired = mgr.validate_profile(bad_data)
    assert is_valid is False
    assert len(errors) > 0

    # Verify repair clamping
    assert repaired.sensitivity > 0.0
    assert repaired.deadzone <= 0.5
    assert repaired.rest_release_threshold < repaired.rest_activation_threshold


# -----------------------------------------------------------------------------
# Test 3: Default Profile Fallback
# -----------------------------------------------------------------------------
def test_3_default_profile_fallback():
    mgr = UserProfileManager()
    prof = mgr.load_profile("non_existent_profile_file_12345.json")
    assert prof.name == "default_user"
    assert prof.sensitivity == 1.0
    assert prof.sensitivity_x == 1200.0


# -----------------------------------------------------------------------------
# Test 4: Calibration Wizard Stage Execution
# -----------------------------------------------------------------------------
def test_4_calibration_wizard_stage_execution():
    wizard = CalibrationWizard(profile_name="wiz_user")
    assert wizard.current_stage == WizardStage.STAGE_A_STATIONARY

    # Stage A
    res_a = None
    for i in range(100):
        sample = IMUSample(i*0.01, np.array([0.0, 0.0, 9.81]), np.array([0.01, -0.01, 0.0]))
        res_a = wizard.process_sample_stage_a(sample)

    assert res_a is not None
    assert res_a.success is True
    assert wizard.current_stage == WizardStage.STAGE_B_NEUTRAL_POSTURE


# -----------------------------------------------------------------------------
# Test 5: Calibration Value Computation
# -----------------------------------------------------------------------------
def test_5_calibration_value_computation():
    wizard = CalibrationWizard(profile_name="wiz_val_user")

    # Fast forward through stages
    # Stage C: peak gyro 2.0 rad/s
    res_c = wizard.process_sample_stage_c(gyro_magnitude_rads=2.0)
    assert res_c is not None
    assert abs(wizard.calibrated_sensitivity_x - 900.0) < 1e-3

    # Stage D: rest energy
    for i in range(50):
        res_d = wizard.process_sample_stage_d(0.02, 0.01, 0.20)

    assert wizard.current_stage == WizardStage.STAGE_E_GESTURE
    assert wizard.calibrated_rest_release < wizard.calibrated_rest_activation


# -----------------------------------------------------------------------------
# Test 6: Persistence Across Reloads
# -----------------------------------------------------------------------------
def test_6_persistence_across_reloads(tmp_path):
    mgr = UserProfileManager(profiles_dir=str(tmp_path))
    prof = mgr.create_profile("persist_user", sensitivity_x=1600.0, rest_activation_threshold=0.06)
    mgr.save_profile(prof)

    mgr2 = UserProfileManager(profiles_dir=str(tmp_path))
    loaded = mgr2.load_profile("persist_user")
    assert loaded.sensitivity_x == 1600.0
    assert loaded.rest_activation_threshold == 0.06


# -----------------------------------------------------------------------------
# Test 7: Profile Reset to Default
# -----------------------------------------------------------------------------
def test_7_profile_reset_to_default(tmp_path):
    mgr = UserProfileManager(profiles_dir=str(tmp_path))
    prof = mgr.create_profile("custom_user", sensitivity=2.0)
    mgr.active_profile = prof

    reset_prof = mgr.reset_to_default()
    assert reset_prof.name == "default_user"
    assert reset_prof.sensitivity == 1.0


# -----------------------------------------------------------------------------
# Test 8: Air/Rest Threshold Application
# -----------------------------------------------------------------------------
def test_8_air_rest_threshold_application():
    mgr = UserProfileManager()
    prof = UserProfile(
        name="rest_user",
        rest_accel_var_threshold=0.12,
        rest_activation_threshold=0.06,
        rest_release_threshold=0.03
    )

    cfg = mgr.apply_profile_to_config(prof)
    assert cfg.rest_accel_var_threshold == 0.12
    assert cfg.rest_activation_threshold == 0.06
    assert cfg.rest_release_threshold == 0.03


# -----------------------------------------------------------------------------
# Test 9: Gesture Parameter Application
# -----------------------------------------------------------------------------
def test_9_gesture_parameter_application():
    mgr = UserProfileManager()
    prof = UserProfile(
        name="gesture_user",
        gesture_debounce_duration=0.06,
        gesture_min_stable_duration=0.10,
        double_click_interval=0.40,
        uncertainty_tolerance=90.0
    )

    cfg = mgr.apply_profile_to_config(prof)
    assert cfg.gesture_debounce_duration == 0.06
    assert cfg.gesture_min_stable_duration == 0.10
    assert cfg.double_click_max_interval == 0.40
    assert cfg.gesture_max_uncertainty_threshold == 90.0


# -----------------------------------------------------------------------------
# Test 10: Profile Config Application
# -----------------------------------------------------------------------------
def test_10_profile_config_application():
    mgr = UserProfileManager()
    prof = UserProfile(
        name="scale_user",
        sensitivity=1.2,
        sensitivity_x=1000.0,
        sensitivity_y=1000.0
    )

    cfg = mgr.apply_profile_to_config(prof)
    assert cfg.base_sensitivity_x == 1200.0
    assert cfg.base_sensitivity_y == 1200.0


# -----------------------------------------------------------------------------
# Test 11: Profile Diagnostic Display
# -----------------------------------------------------------------------------
def test_11_profile_diagnostic_display():
    mgr = UserProfileManager()
    prof = UserProfile(name="diag_user", sensitivity=1.1)
    diag = mgr.get_diagnostic_display(prof)

    assert diag["profile_name"] == "diag_user"
    assert diag["status"] == "VALID"
    assert diag["motion_parameters"]["sensitivity_master"] == 1.1


# -----------------------------------------------------------------------------
# Test 12: Out of Range Parameter Clamping
# -----------------------------------------------------------------------------
def test_12_out_of_range_parameter_clamping():
    mgr = UserProfileManager()
    data = {"sensitivity": 50.0, "deadzone": -0.1}

    _, _, repaired = mgr.validate_profile(data)
    assert repaired.sensitivity <= 10.0
    assert repaired.deadzone >= 0.0


# -----------------------------------------------------------------------------
# Test 13: Missing Field Auto Fill
# -----------------------------------------------------------------------------
def test_13_missing_field_auto_fill():
    mgr = UserProfileManager()
    partial_data = {"name": "partial_user", "sensitivity": 1.5}

    _, _, repaired = mgr.validate_profile(partial_data)
    assert repaired.name == "partial_user"
    assert repaired.sensitivity == 1.5
    assert repaired.sensitivity_x == 1200.0  # Auto-filled from default
    assert repaired.screen_width == 1920


# -----------------------------------------------------------------------------
# Test 14: Wizard Complete Flow
# -----------------------------------------------------------------------------
def test_14_wizard_complete_flow(tmp_path):
    wizard = CalibrationWizard(profile_name="full_wiz_user")
    wizard.profile_manager = UserProfileManager(profiles_dir=str(tmp_path))

    # Stage A
    for i in range(100):
        wizard.process_sample_stage_a(IMUSample(i*0.01, np.array([0,0,9.81]), np.array([0,0,0])))

    # Stage B
    for i in range(50):
        wizard.process_sample_stage_b(IMUSample(i*0.01, np.array([0,0,9.81]), np.array([0,0,0])))

    # Stage C
    wizard.process_sample_stage_c(1.8)

    # Stage D
    for i in range(50):
        wizard.process_sample_stage_d(0.02, 0.01, 0.25)

    # Stage E
    for i in range(20):
        wizard.process_sample_stage_e(0.20, 0.80)

    # Stage F Confirmation
    profile, save_path = wizard.process_stage_f_confirmation()

    assert wizard.current_stage == WizardStage.COMPLETED
    assert os.path.exists(save_path)
    assert profile.name == "full_wiz_user"


# -----------------------------------------------------------------------------
# Test 15: End-to-End Pipeline With User Profile
# -----------------------------------------------------------------------------
def test_15_end_to_end_pipeline_with_user_profile():
    mgr = UserProfileManager()
    user_prof = UserProfile(name="pipeline_user", sensitivity=1.2, deadzone=0.03)
    custom_cfg = mgr.apply_profile_to_config(user_prof)

    pipeline = WearableMousePipeline(config=custom_cfg)
    assert pipeline.config.deadzone_threshold == 0.03
    assert pipeline.config.base_sensitivity_x == 1440.0

    # Run frame tick
    sample = IMUSample(1.0, np.array([0.0, 0.0, 9.81]), np.array([0.0, 0.0, 0.0]))
    res = pipeline.process_frame(sample)
    assert res is not None
