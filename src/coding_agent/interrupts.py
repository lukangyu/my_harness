from __future__ import annotations


class TaskInterrupted(BaseException):
    """Raised when the user intentionally interrupts the current agent task."""
