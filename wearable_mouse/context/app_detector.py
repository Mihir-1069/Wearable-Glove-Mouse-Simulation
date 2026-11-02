"""
Application Detector.
Identifies active system window/application, supporting live detection or mock simulation.
"""

from typing import Optional, Callable


class ApplicationDetector:
    """
    Detects active foreground application name.
    Supports setting mock active application for testing and simulation.
    """

    def __init__(self, mock_app: str = "unknown"):
        self._mock_app: str = mock_app
        self._custom_detector_cb: Optional[Callable[[], str]] = None

    def set_mock_app(self, app_name: str) -> None:
        """Set active mock application name (e.g., 'chrome.exe', 'powerpnt.exe')."""
        self._mock_app = app_name

    def register_custom_detector(self, cb: Callable[[], str]) -> None:
        """Register custom system process detector callback (e.g. win32gui / psutil wrapper)."""
        self._custom_detector_cb = cb

    def get_active_app(self) -> str:
        """
        Return name of active application process.
        """
        if self._custom_detector_cb is not None:
            try:
                res = self._custom_detector_cb()
                if res:
                    return res
            except Exception:
                pass
        return self._mock_app
