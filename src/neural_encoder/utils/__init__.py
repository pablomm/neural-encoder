"""Utilities for loading and preprocessing measurements."""

import lazy_loader as lazy
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "preprocessing": ["MeasurementPreprocessor"],
        "split_views": ["split_views"],
        "confound_removal": ["ConfoundRemover"],
    },
)

if TYPE_CHECKING:
    from .split_views import split_views
    from .preprocessing import MeasurementPreprocessor
    from .confound_removal import ConfoundRemover

__all__ = ["MeasurementPreprocessor", "split_views", "ConfoundRemover"]
