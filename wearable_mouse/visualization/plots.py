"""
Offline Matplotlib Plotting Helpers for Telemetry & Controlled Experiments.
"""

from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
from wearable_mouse.telemetry_logging.data_logger import LogFrame
from wearable_mouse.experiments.experiment_runner import ExperimentMetrics


def plot_telemetry_summary(frames: List[LogFrame], save_path: Optional[str] = None) -> None:
    """
    Plots raw vs filtered signals, cursor path, and uncertainty radius timeline.
    """
    if not frames:
        print("No telemetry frames to plot.")
        return

    timestamps = [f.timestamp for f in frames]
    raw_gz = [f.raw_gyro_z for f in frames]
    filt_gz = [f.filt_gyro_z for f in frames]
    cursor_x = [f.cursor_x for f in frames]
    cursor_y = [f.cursor_y for f in frames]
    uncertainties = [f.uncertainty_radius for f in frames]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)

    # 1. Raw vs Filtered Gyro Z
    axes[0].plot(timestamps, raw_gz, label="Raw Gyro Z", alpha=0.5, color="red")
    axes[0].plot(timestamps, filt_gz, label="Filtered Gyro Z", color="blue", linewidth=1.5)
    axes[0].set_title("Gyroscope Z Motion Signal (Raw vs Filtered)")
    axes[0].set_ylabel("Rad/s")
    axes[0].legend()
    axes[0].grid(True)

    # 2. Cursor Trajectory (2D Path)
    axes[1].plot(cursor_x, cursor_y, label="Cursor Path", color="green")
    axes[1].set_title("Virtual Cursor Trajectory (Pixels)")
    axes[1].set_xlabel("X (px)")
    axes[1].set_ylabel("Y (px)")
    axes[1].invert_yaxis()
    axes[1].grid(True)

    # 3. Dynamic Uncertainty Radius
    axes[2].plot(timestamps, uncertainties, label="Uncertainty Radius (px)", color="purple")
    axes[2].set_title("Target Acquisition Uncertainty Radius")
    axes[2].set_xlabel("Timestamp (s)")
    axes[2].set_ylabel("Radius (px)")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_experiment_results(results: List[ExperimentMetrics], save_path: Optional[str] = None) -> None:
    """
    Generates a 6-panel grid comparing quantitative metrics across Conditions A, B, C, and D.
    """
    if not results:
        return

    labels = [r.condition_code for r in results]
    colors = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71']

    success_rates = [r.success_rate for r in results]
    wrong_rates = [r.wrong_target_rate for r in results]
    misclick_rates = [r.misclick_rate for r in results]
    avg_times = [r.avg_acquisition_time_sec for r in results]
    path_lengths = [r.total_path_length_pixels for r in results]
    corrective_moves = [r.corrective_movements_count for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Controlled Target Acquisition Experiment Comparison", fontsize=14, fontweight="bold")

    # 1. Success Rate (%)
    axes[0, 0].bar(labels, success_rates, color=colors)
    axes[0, 0].set_title("Success Rate (%)")
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Wrong Target Rate (%)
    axes[0, 1].bar(labels, wrong_rates, color=colors)
    axes[0, 1].set_title("Wrong Target Rate (%)")
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # 3. Misclick Rate (%)
    axes[0, 2].bar(labels, misclick_rates, color=colors)
    axes[0, 2].set_title("Misclick Rate (%)")
    axes[0, 2].set_ylim(0, 105)
    axes[0, 2].grid(axis='y', linestyle='--', alpha=0.7)

    # 4. Acquisition Time (s)
    axes[1, 0].bar(labels, avg_times, color=colors)
    axes[1, 0].set_title("Avg Acquisition Time (s)")
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # 5. Cursor Path Length (px)
    axes[1, 1].bar(labels, path_lengths, color=colors)
    axes[1, 1].set_title("Total Path Length (px)")
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # 6. Corrective Movements
    axes[1, 2].bar(labels, corrective_moves, color=colors)
    axes[1, 2].set_title("Corrective Movements Count")
    axes[1, 2].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Experiment comparison chart saved to {save_path}")
    else:
        plt.show()
