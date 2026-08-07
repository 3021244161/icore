"""
icore.models.exceptions - Model-layer exception hierarchy.

v0.5: These classes are unified into ``icore.exceptions`` so every
icore error inherits from ``ICoreError`` (giving each a stable
``code`` / ``http_status`` / ``retryable`` triple that the API layer
can translate into a structured JSON response).

This module re-exports the unified classes for backward-compatibility
with existing imports such as::

    from icore.models.exceptions import ModelNotFoundError
"""

from __future__ import annotations

from icore.exceptions import (
    ICoreError,
    ModelAPIError,
    ModelNotFoundError,
    ModelTimeoutError,
    ModelUnhealthyError,
    ModelUnsupportedError,
    NoAvailableModelError,
)

#: Legacy alias kept so ``except ModelError`` keeps working.
ModelError = ICoreError


__all__ = [
    "ModelError",
    "ModelNotFoundError",
    "ModelUnhealthyError",
    "NoAvailableModelError",
    "ModelAPIError",
    "ModelTimeoutError",
    "ModelUnsupportedError",
]
