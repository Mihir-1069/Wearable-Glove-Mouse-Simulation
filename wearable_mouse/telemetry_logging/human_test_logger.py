"""
Human Test Telemetry Logger module.
Buffers interactive human trial telemetry and subjective feedback survey ratings (1-5 scales)
and exports CSV results.
"""

import csv
from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class HumanTestTrialRecord:
    trial_index: int
    block_index: int
    condition_code: str             # "B", "C", "E", "E+C"
    condition_blinded_label: str     # "Condition X"
    target_id: str
    target_size: int                # 20, 40, 80, 120 (px)
    target_layout: str              # "ISOLATED", "DENSE_GRID", "TOOLBAR"
    outcome: str                    # "SUCCESS", "MISSED", "WRONG_TARGET"
    acquisition_time_sec: float
    cursor_path_length: float
    corrective_movements: int
    num_clicks: int
    uncertainty_level: float
    noise_level: float
    tremor_level: float
    difficulty_rating: Optional[int] = None   # 1-5
    stability_rating: Optional[int] = None    # 1-5
    confidence_rating: Optional[int] = None   # 1-5
    fatigue_rating: Optional[int] = None      # 1-5


class HumanTestLogger:
    """
    Manages interactive trial data buffering and CSV export.
    """

    def __init__(self):
        self.records: List[HumanTestTrialRecord] = []

    def log_trial(self, record: HumanTestTrialRecord) -> None:
        self.records.append(record)

    def attach_block_ratings(
        self,
        block_index: int,
        difficulty: int,
        stability: int,
        confidence: int,
        fatigue: int
    ) -> None:
        """Attach subjective 1-5 ratings to all records in block_index."""
        for rec in self.records:
            if rec.block_index == block_index:
                rec.difficulty_rating = difficulty
                rec.stability_rating = stability
                rec.confidence_rating = confidence
                rec.fatigue_rating = fatigue

    def export_csv(self, filepath: str) -> None:
        """Export all trial records and subjective survey scores to CSV."""
        if not self.records:
            return

        fieldnames = list(asdict(self.records[0]).keys())
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self.records:
                writer.writerow(asdict(rec))

    def clear(self) -> None:
        self.records.clear()
