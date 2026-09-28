"""Utilities for loading and preprocessing measurements."""

import lazy_loader as lazy
from typing import TYPE_CHECKING

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "preprocessing": ["MeasurementPreprocessor"],
        "split_views": ["split_views"],
        "confound_removal": ["ConfoundRemover"],
        "evaluation": ["retrieval_metrics", "compute_rsa", "evaluate_pair", "evaluate_views"],
    },
)

if TYPE_CHECKING:
    from .split_views import split_views
    from .preprocessing import MeasurementPreprocessor
    from .confound_removal import ConfoundRemover
    from .evaluation import retrieval_metrics, compute_rsa, evaluate_pair, evaluate_views

__all__ = ["MeasurementPreprocessor", "split_views", "ConfoundRemover",
           "retrieval_metrics", "compute_rsa", "evaluate_pair", "evaluate_views"]
