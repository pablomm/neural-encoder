"""Utilities for loading and preprocessing measurements."""

import lazy_loader as lazy
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={"preprocessing": ["MeasurementPreprocessor"]},
)

if TYPE_CHECKING:
    from .preprocessing import MeasurementPreprocessor

__all__ = ["MeasurementPreprocessor"]
