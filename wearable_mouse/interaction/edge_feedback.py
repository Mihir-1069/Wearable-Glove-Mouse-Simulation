"""
Screen Edge Detection module.
Detects when the cursor comes into contact with virtual screen boundaries
to trigger haptic tactile feedback.
"""

from enum import Enum, auto
from typing import List, Optional
from wearable_mouse.config import Config
from wearable_mouse.cursor.cursor_engine import VirtualCursor


class ScreenEdge(Enum):
    EDGE_LEFT = "EDGE_LEFT"
    EDGE_RIGHT = "EDGE_RIGHT"
    EDGE_TOP = "EDGE_TOP"
    EDGE_BOTTOM = "EDGE_BOTTOM"


class EdgeDetector:
    """
    Monitors VirtualCursor coordinates against screen boundary thresholds.
    """

    def __init__(self, config: Optional[Config] = None, margin_pixels: float = 2.0):
        self.config = config or Config()
        self.margin = margin_pixels
        self.active_edges: List[ScreenEdge] = []

    def update(self, cursor: VirtualCursor) -> List[ScreenEdge]:
        """
        Check cursor position and return list of newly hit ScreenEdge boundaries.
        """
        hits: List[ScreenEdge] = []

        if cursor.x <= self.margin:
            hits.append(ScreenEdge.EDGE_LEFT)
        elif cursor.x >= self.config.screen_width - self.margin:
            hits.append(ScreenEdge.EDGE_RIGHT)

        if cursor.y <= self.margin:
            hits.append(ScreenEdge.EDGE_TOP)
        elif cursor.y >= self.config.screen_height - self.margin:
            hits.append(ScreenEdge.EDGE_BOTTOM)

        # Detect newly entered edge collisions
        new_hits = [e for e in hits if e not in self.active_edges]
        self.active_edges = hits
        return new_hits
